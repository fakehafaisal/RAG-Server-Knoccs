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
from rank_bm25 import BM25Okapi

load_dotenv()

def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

class PgVectorStore:
    def __init__(self, 
                embedding_model: str = "all-mpnet-base-v2", chunk_size: int = 1024, chunk_overlap: int = 256,
                use_reranker: bool = True, debug: bool = False):
        
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_reranker = use_reranker
        self.debug = debug

        # ------------------------------------------------------------
        # FIX: Proper model loading without meta tensor issues
        # ------------------------------------------------------------
        print(f"Loading embedding model '{embedding_model}'...")
        
        # Load model directly without device specification first
        self.model = SentenceTransformer(embedding_model)
        
        self.model = self.model.cpu()
        self.model.eval()
        
        # Test encoding to ensure model works
        # try:
        #     test_emb = self.model.encode(["test"], convert_to_numpy=True)
        #     if np.isnan(test_emb).any():
        #         raise ValueError("Model producing NaN values")
        #     # print(f"[INFO] Model test successful, embedding dimension: {test_emb.shape}")
        # except Exception as e:
        #     print(f"[ERROR] Model initialization failed: {e}")
        #     raise

        # ------------------------------------------------------------
        # Reranker with error handling
        # ------------------------------------------------------------
        if use_reranker:
            try:
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("Loaded reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as e:
                print(f"[WARNING] Failed to load reranker: {e}")
                print("[WARNING] Continuing without reranker")
                self.use_reranker = False
                self.reranker = None

        # Database config and initialization
        self.db_config = {
            'host': os.getenv('PGVECTOR_HOST', 'localhost'),
            'port': int(os.getenv('PGVECTOR_PORT', '5432')),
            'database': os.getenv('PGVECTOR_DATABASE', 'postgres'),
            'user': os.getenv('PGVECTOR_USER', 'postgres'),
            'password': os.getenv('PGVECTOR_PASSWORD', 'postgres'),
            'sslmode': 'require'}

        self._init_database()
        # print(f"[INFO] Loaded embedding model: {embedding_model}")

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
        print(f"Building vector store from {len(documents)} documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        conn = self._get_connection()
        cur = conn.cursor()
        try:
            # Insert documents and map chunk -> document
            doc_map = {}
            for idx, doc in enumerate(documents):
                source = doc.metadata.get("source", f"doc_{idx}")
                cur.execute(
                    "INSERT INTO documents (name, type, source) VALUES (%s, %s, %s) RETURNING id",
                    (doc.metadata.get("name", "unknown"), doc.metadata.get("type", "unknown"), source)
                )
                doc_id = cur.fetchone()[0]
                # Store mapping with BOTH source and doc_idx as fallback
                doc_map[source] = doc_id
                doc_map[f"doc_{idx}"] = doc_id  # Fallback key
                
                if self.debug:
                    print(f"[DEBUG] Mapped document '{source}' to doc_id={doc_id}")

            # Prepare chunk data
            chunk_data = []
            unmapped_count = 0
            for i, chunk in enumerate(chunks):
                # Get document identifier from chunk metadata
                chunk_doc_ref = chunk.metadata.get("document")
                document_id = doc_map.get(chunk_doc_ref)
                
                if document_id is None:
                    # Fallback: try to match by source
                    source = chunk.metadata.get("source")
                    document_id = doc_map.get(source)
                
                if document_id is None:
                    unmapped_count += 1
                    if self.debug:
                        print(f"[WARNING] Chunk {i} couldn't be mapped to document. Ref: '{chunk_doc_ref}', Source: '{chunk.metadata.get('source')}'")
                
                chunk_data.append((
                    document_id,
                    clean_text(chunk.page_content),
                    chunk.metadata.get("page"),
                    chunk.metadata.get("chunk_id", i)
                ))
            
            if unmapped_count > 0:
                print(f"[WARNING] {unmapped_count} chunks could not be mapped to documents!")
            
            # Insert chunks and get IDs - USE DIFFERENT APPROACH
            chunk_ids = []
            print(f"[INFO] Inserting {len(chunk_data)} chunks...")
            
            # Method 1: Try execute_values with page_size
            try:
                from psycopg2.extras import execute_values
                result = execute_values(
                    cur,
                    "INSERT INTO chunks (document_id, text, page, chunk_id) VALUES %s RETURNING id",
                    chunk_data,
                    page_size=1000,
                    fetch=True
                )
                # execute_values with fetch=True returns the results directly
                if result:
                    chunk_ids = [row[0] for row in result]
                else:
                    # Fallback: fetch from cursor
                    chunk_ids = [row[0] for row in cur.fetchall()]
                
                print(f"[INFO] Inserted {len(chunk_ids)} chunks via batch insert")
                
            except Exception as e:
                print(f"[WARNING] Batch insert failed: {e}")
                print("[INFO] Falling back to individual inserts...")
                # Fallback: insert one by one
                chunk_ids = []
                for chunk_tuple in chunk_data:
                    cur.execute(
                        "INSERT INTO chunks (document_id, text, page, chunk_id) VALUES (%s, %s, %s, %s) RETURNING id",
                        chunk_tuple
                    )
                    chunk_ids.append(cur.fetchone()[0])
                print(f"[INFO] Inserted {len(chunk_ids)} chunks via individual inserts")

            # Verify chunk_ids length matches
            if len(chunk_ids) != len(embeddings):
                raise ValueError(f"Mismatch: {len(chunk_ids)} chunks inserted but {len(embeddings)} embeddings generated!")

            # Insert embeddings
            # print(f"[INFO] Inserting {len(chunk_ids)} embeddings...")
            embedding_data = [(chunk_id, emb.tolist()) for chunk_id, emb in zip(chunk_ids, embeddings)]
            execute_values(
                cur,
                "INSERT INTO embeddings (chunk_id, embedding) VALUES %s",
                embedding_data,
                page_size=1000
            )
            
            conn.commit()
            print(f"[SUCCESS] Successfully inserted {len(chunks)} chunks and {len(chunk_ids)} embeddings")
            
        except Exception as e:
            conn.rollback()
            print(f"[ERROR] Build failed: {e}")
            raise
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

    def query(self, query_text: str, top_k: int = 15, initial_k: int = 50, use_hybrid: bool = True) -> List[Dict]:
        """Hybrid retrieval: BM25 + Dense embeddings + Reranking"""
        query_text = query_text.strip()
        if not query_text:
            print("[ERROR] Empty query text")
            return []
        
        if use_hybrid:
            # HYBRID: Combine BM25 and dense retrieval
            bm25_results = self.search_bm25(query_text, top_k=initial_k // 2)
            dense_results = self.search_dense(query_text, top_k=initial_k // 2)
            
            # Merge and deduplicate by chunk_id
            seen_ids = set()
            combined_results = []
            
            for r in bm25_results + dense_results:
                chunk_id = r.get("chunk_id")
                if chunk_id not in seen_ids:
                    combined_results.append(r)
                    seen_ids.add(chunk_id)
            
            initial_results = combined_results[:initial_k]
        else:
            # Dense only
            initial_results = self.search_dense(query_text, top_k=initial_k)
        
        # Rerank
        if self.use_reranker and self.reranker and initial_results:
            pairs = [(query_text, r["metadata"]["text"]) for r in initial_results]
            scores = self.reranker.predict(pairs)
            for i, score in enumerate(scores):
                initial_results[i]["rerank_score"] = float(score)
            final_results = sorted(initial_results, key=lambda x: x.get("rerank_score", -999), reverse=True)[:top_k]
        else:
            final_results = initial_results[:top_k]
        
        return final_results

    def search_dense(self, query_text: str, top_k: int = 15) -> List[Dict]:
        """Dense vector search (your existing search method)"""
        query_emb = self.model.encode([query_text], convert_to_numpy=True).astype('float32')
        
        if np.isnan(query_emb).any():
            print(f"[ERROR] NaN in embedding")
            return []
        
        return self.search(query_emb, top_k=top_k)

    def search_bm25(self, query_text: str, top_k: int = 15) -> List[Dict]:
        """BM25 keyword search (good for names, dates, exact matches)"""
        conn = self._get_connection()
        cur = conn.cursor()
        
        try:
            # Fetch all chunks with their text
            cur.execute("""
                SELECT ch.id, ch.text, d.name, d.type, d.source
                FROM chunks ch
                JOIN documents d ON ch.document_id = d.id""")
            
            rows = cur.fetchall()
            if not rows:
                return []
            
            # Prepare for BM25
            chunk_ids = [row[0] for row in rows]
            texts = [row[1] for row in rows]
            metadata = [{
                "text": row[1],
                "document_name": row[2],
                "document_type": row[3],
                "source": row[4]
            } for row in rows]
            
            # Tokenize (simple whitespace + lowercase)
            tokenized_corpus = [doc.lower().split() for doc in texts]
            tokenized_query = query_text.lower().split()
            
            # BM25 scoring
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(tokenized_query)
            
            # Get top-k
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only return if there's some match
                    results.append({
                        "chunk_id": chunk_ids[idx],
                        "bm25_score": float(scores[idx]),
                        "distance": 1 - (scores[idx] / max(scores)),  # Normalize
                        "metadata": metadata[idx]
                    })
            
            return results
            
        finally:
            cur.close()
            conn.close()

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