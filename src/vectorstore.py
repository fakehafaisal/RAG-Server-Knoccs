import os
import numpy as np
from typing import List, Any, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.embedding import EmbeddingPipeline
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import re
import torch

def clean_text(text: str) -> str:
    """Remove NUL bytes and dangerous control characters that PostgreSQL hates"""
    if not text:
        return ""
    # Remove NUL (0x00) and other ASCII control chars except \n \t \r
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

load_dotenv()

class PgVectorStore:
    def __init__(self, 
                 embedding_model: str = "all-mpnet-base-v2", 
                 chunk_size: int = 512, 
                 chunk_overlap: int = 128,
                 use_reranker: bool = True,
                 table_name: str = "document_embeddings"):
        
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)

        if torch.cuda.is_available():                 # ← new
            self.model = self.model.cuda()            # ← new (moves model to GPU)
        self.model.eval()
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_name = table_name
        
        # Database connection from environment variables (Supabase compatible)
        self.db_config = {
            'host': os.getenv('PGVECTOR_HOST', 'localhost'),
            'port': int(os.getenv('PGVECTOR_PORT', '5432')),
            'database': os.getenv('PGVECTOR_DATABASE', 'postgres'),
            'user': os.getenv('PGVECTOR_USER', 'postgres'),
            'password': os.getenv('PGVECTOR_PASSWORD', 'postgres'),
            'sslmode': 'require'  # Supabase requires SSL
        }
        
        # Add cross-encoder reranker for better relevance
        self.use_reranker = use_reranker
        if use_reranker:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print(f"[INFO] Loaded reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        print(f"[INFO] Loaded embedding model: {embedding_model}")
        
        # Initialize database
        self._init_database()

    def _get_connection(self):
        """Create a new database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"[ERROR] Failed to connect to database: {e}")
            print("[INFO] Make sure your Supabase credentials are correct in .env")
            raise

    def _init_database(self):
        """Initialize database with pgvector extension and create tables"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Enable pgvector extension (should already be enabled in Supabase)
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create table with vector column (768 dimensions for all-mpnet-base-v2)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding vector(768),
                    source TEXT,
                    chunk_id INTEGER,
                    page INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create index for faster similarity search
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                ON {self.table_name} 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            conn.commit()
            cur.close()
            conn.close()
            print(f"[INFO] Database initialized with table '{self.table_name}'")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize database: {e}")
            print("[INFO] Make sure pgvector extension is enabled in Supabase Dashboard")
            raise

    def build_from_documents(self, documents: List[Any]):
        """Build vector store from documents"""
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        
        # Prepare data for insertion
        data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            data.append((
                clean_text(chunk.page_content),
                embedding.tolist(),  # Convert numpy array to list for PostgreSQL
                chunk.metadata.get('source', 'unknown'),
                chunk.metadata.get('chunk_id', i),
                chunk.metadata.get('page', None)
            ))
        
        # Insert into database in batches
        self._insert_embeddings(data)
        print(f"[INFO] Vector store built with {len(data)} chunks")

    def _insert_embeddings(self, data: List[tuple], batch_size: int = 1000):
        """Insert embeddings in batches"""
        conn = self._get_connection()
        cur = conn.cursor()
        
        try:
            # Clear existing data
            cur.execute(f"TRUNCATE TABLE {self.table_name};")
            
            # Insert in batches
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {self.table_name} (text, embedding, source, chunk_id, page)
                    VALUES %s
                    """,
                    batch
                )
                print(f"[INFO] Inserted batch {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}")
            
            conn.commit()
            print(f"[INFO] Successfully inserted {len(data)} embeddings")
            
        except Exception as e:
            conn.rollback()
            print(f"[ERROR] Failed to insert embeddings: {e}")
            raise
        finally:
            cur.close()
            conn.close()

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Dict]:
        """Search for similar embeddings using cosine similarity"""
        conn = self._get_connection()
        cur = conn.cursor()
        
        try:
            # Convert query embedding to list
            query_vec = query_embedding[0].tolist()
            
            # Use cosine distance (1 - cosine similarity)
            cur.execute(f"""
                SELECT id, text, source, chunk_id, page,
                       1 - (embedding <=> %s::vector) as similarity
                FROM {self.table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_vec, query_vec, top_k))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "index": row[0],
                    "distance": 1 - row[5],  # Convert back to distance
                    "metadata": {
                        "text": row[1],
                        "source": row[2],
                        "chunk_id": row[3],
                        "page": row[4]
                    }
                })
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            return []
        finally:
            cur.close()
            conn.close()

    def rerank_results(self, query_text: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank results using cross-encoder for better relevance"""
        if not self.use_reranker or not results:
            return results[:top_k]
        
        # Prepare pairs for reranking
        pairs = [(query_text, r["metadata"]["text"]) for r in results if r["metadata"]]
        
        # Get reranking scores
        scores = self.reranker.predict(pairs)
        
        # Combine scores with results
        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)
        
        # Sort by rerank score (higher is better)
        reranked = sorted(results, key=lambda x: x.get("rerank_score", -999), reverse=True)
        
        print(f"[INFO] Reranked {len(results)} results, returning top {top_k}")
        return reranked[:top_k]

    def query(self, query_text: str, top_k: int = 5, initial_k: int = 50) -> List[Dict]:

        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        
        # Stage 1: Vector search
        initial_results = self.search(query_emb, top_k=initial_k)
        
        # Stage 2: Rerank
        if self.use_reranker:
            final_results = self.rerank_results(query_text, initial_results, top_k=top_k)
        else:
            final_results = initial_results[:top_k]
        
        return final_results

    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        conn = self._get_connection()
        cur = conn.cursor()
        
        try:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            total_chunks = cur.fetchone()[0]
            
            cur.execute(f"SELECT COUNT(DISTINCT source) FROM {self.table_name}")
            total_sources = cur.fetchone()[0]
            
            return {
                "total_chunks": total_chunks,
                "total_sources": total_sources
            }
        finally:
            cur.close()
            conn.close()

    def clear(self):
        """Clear all data from the vector store"""
        conn = self._get_connection()
        cur = conn.cursor()
        
        try:
            cur.execute(f"TRUNCATE TABLE {self.table_name};")
            conn.commit()
            print(f"[INFO] Cleared all data from {self.table_name}")
        finally:
            cur.close()
            conn.close()


# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents
    
    docs = load_all_documents("data")
    store = PgVectorStore()
    store.build_from_documents(docs)
    
    print(store.query("What is attention mechanism?", top_k=5, initial_k=50))
    print(store.get_stats())