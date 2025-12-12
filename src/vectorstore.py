import os
import re
import torch
import numpy as np
from typing import List, Any, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.embedding import EmbeddingPipeline
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

class PgVectorStore:
    def __init__(self, 
                embedding_model: str = "all-mpnet-base-v2", 
                chunk_size: int = 1024, 
                chunk_overlap: int = 256,
                use_reranker: bool = True):
        
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_reranker = use_reranker

        # ------------------------------------------------------------
        # FIX: Proper model loading without meta tensor issues
        # ------------------------------------------------------------
        print(f"[INFO] Loading embedding model '{embedding_model}'...")
        
        # Load model directly without device specification first
        self.model = SentenceTransformer(embedding_model)
        
        # Then move to CPU if needed
        if torch.cuda.is_available():
            print("[INFO] CUDA available, using GPU")
            self.model = self.model.cuda()
        else:
            print("[INFO] Using CPU")
            self.model = self.model.cpu()
        
        self.model.eval()
        
        # Test encoding to ensure model works
        try:
            test_emb = self.model.encode(["test"], convert_to_numpy=True)
            if np.isnan(test_emb).any():
                raise ValueError("Model producing NaN values")
            print(f"[INFO] Model test successful, embedding dimension: {test_emb.shape}")
        except Exception as e:
            print(f"[ERROR] Model initialization failed: {e}")
            raise

        # ------------------------------------------------------------
        # Reranker (unchanged)
        # ------------------------------------------------------------
        if use_reranker:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print(f"[INFO] Loaded reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Database config and initialization
        self.db_config = {
            'host': os.getenv('PGVECTOR_HOST', 'localhost'),
            'port': int(os.getenv('PGVECTOR_PORT', '5432')),
            'database': os.getenv('PGVECTOR_DATABASE', 'postgres'),
            'user': os.getenv('PGVECTOR_USER', 'postgres'),
            'password': os.getenv('PGVECTOR_PASSWORD', 'postgres'),
            'sslmode': 'require'
        }

        self._init_database()
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def _get_connection(self):
        return psycopg2.connect(**self.db_config)

    def _init_database(self):
        """Initialize documents, chunks, embeddings tables"""
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Documents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    name TEXT,
                    type TEXT,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            # Chunks table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    text TEXT NOT NULL,
                    page INTEGER,
                    chunk_id INTEGER,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            # Embeddings table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
                    embedding vector(768)
                );
            """)
            # Index for vector search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_idx
                ON embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            conn.commit()
            print("[INFO] Database initialized with tables: documents, chunks, embeddings")
        finally:
            cur.close()
            conn.close()

    def build_from_documents(self, documents: List[Any]):
        """Insert documents, chunks, and embeddings"""
        print(f"[INFO] Building vector store from {len(documents)} documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        conn = self._get_connection()
        cur = conn.cursor()
        try:
            # Insert documents
            doc_map = {}
            for idx, doc in enumerate(documents):
                cur.execute(
                    "INSERT INTO documents (name, type, source) VALUES (%s, %s, %s) RETURNING id",
                    (doc.metadata.get("name", "unknown"), 
                    doc.metadata.get("type", "unknown"), 
                    doc.metadata.get("source", "unknown"))
                )
                doc_id = cur.fetchone()[0]
                doc_map[doc.metadata.get("source", f"doc_{idx}")] = doc_id

            # Insert chunks ONE BY ONE to get IDs reliably
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                document_id = doc_map.get(chunk.metadata.get("source"))
                cur.execute(
                    "INSERT INTO chunks (document_id, text, page, chunk_id) VALUES (%s, %s, %s, %s) RETURNING id",
                    (document_id, 
                    clean_text(chunk.page_content), 
                    chunk.metadata.get("page"), 
                    chunk.metadata.get("chunk_id", i))
                )
                chunk_id = cur.fetchone()[0]
                chunk_ids.append(chunk_id)
            
            print(f"[INFO] Inserted {len(chunk_ids)} chunks")

            # Insert embeddings in BATCHES
            embedding_data = [(chunk_id, emb.tolist()) for chunk_id, emb in zip(chunk_ids, embeddings)]
            
            batch_size = 1000
            for i in range(0, len(embedding_data), batch_size):
                batch = embedding_data[i:i + batch_size]
                execute_values(
                    cur,
                    "INSERT INTO embeddings (chunk_id, embedding) VALUES %s",
                    batch
                )
                print(f"[INFO] Inserted embedding batch {i//batch_size + 1}/{(len(embedding_data)-1)//batch_size + 1}")
            
            conn.commit()
            print(f"[INFO] Successfully inserted {len(chunks)} chunks and {len(embeddings)} embeddings")
        finally:
            cur.close()
            conn.close()

    def search(self, query_embedding: np.ndarray, top_k: int = 15) -> List[Dict]:
        """Search vector embeddings"""
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            query_vec = query_embedding[0].tolist()
            cur.execute("""
                SELECT ch.id, ch.text, d.name, d.type, d.source, 1 - (e.embedding <=> %s::vector) as similarity
                FROM embeddings e
                JOIN chunks ch ON e.chunk_id = ch.id
                JOIN documents d ON ch.document_id = d.id
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
            """, (query_vec, query_vec, top_k))
            results = []
            for row in cur.fetchall():
                results.append({
                    "chunk_id": row[0],
                    "distance": 1 - row[5],
                    "metadata": {
                        "text": row[1],
                        "document_name": row[2],
                        "document_type": row[3],
                        "source": row[4]
                    }
                })
            return results
        finally:
            cur.close()
            conn.close()

    def query(self, query_text: str, top_k: int = 15, initial_k: int = 50) -> List[Dict]:
        # Clean and validate query text
        query_text = query_text.strip()
        if not query_text:
            print("[ERROR] Empty query text")
            return []
        
        # Encode query with explicit numpy conversion
        query_emb = self.model.encode([query_text], convert_to_numpy=True).astype('float32')
        
        # Check for NaN values
        if np.isnan(query_emb).any():
            print(f"[ERROR] NaN detected in embedding for query: {query_text[:100]}")
            return []
        
        initial_results = self.search(query_emb, top_k=initial_k)
        
        if self.use_reranker and initial_results:
            pairs = [(query_text, r["metadata"]["text"]) for r in initial_results]
            scores = self.reranker.predict(pairs)
            for i, score in enumerate(scores):
                initial_results[i]["rerank_score"] = float(score)
            final_results = sorted(initial_results, key=lambda x: x.get("rerank_score", -999), reverse=True)[:top_k]
        else:
            final_results = initial_results[:top_k]
        
        return final_results

    def get_stats(self) -> Dict:
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM documents")
            total_docs = cur.fetchone()[0]
            return {"total_chunks": total_chunks, "total_sources": total_docs}
        finally:
            cur.close()
            conn.close()

    def clear(self):
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute("TRUNCATE embeddings, chunks, documents CASCADE;")
            conn.commit()
            print("[INFO] Cleared all data from vector store")
        finally:
            cur.close()
            conn.close()


# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents
    
    docs = load_all_documents("data")
    store = PgVectorStore()
    store.build_from_documents(docs)
    
    print(store.query("What is attention mechanism?", top_k=15, initial_k=50))
    print(store.get_stats())