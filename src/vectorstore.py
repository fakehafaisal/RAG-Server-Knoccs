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

        print(f"Loading embedding model '{embedding_model}'...")
        
        self.model = SentenceTransformer(embedding_model)
        self.model = self.model.cpu()
        self.model.eval()

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

    def _get_connection(self):
        return psycopg2.connect(**self.db_config)

    def _init_database(self):
        """Initialize all 6 tables with proper foreign keys, indexes, and pgvector extension"""
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # =====================================================================
            # 1. COMPANIES TABLE
            # =====================================================================
            cur.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    company_id SERIAL PRIMARY KEY,
                    company_name VARCHAR(255) NOT NULL UNIQUE,
                    domain VARCHAR(255) UNIQUE,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # =====================================================================
            # 2. DOCUMENTS TABLE
            # =====================================================================
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id SERIAL PRIMARY KEY,
                    company_id INTEGER NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
                    name VARCHAR(255),
                    type VARCHAR(50),
                    source TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            cur.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS company_id INTEGER
                REFERENCES companies(company_id) ON DELETE CASCADE;
            """)
            
            cur.execute("""
                UPDATE documents SET company_id = 1 WHERE company_id IS NULL;
            """)
            
            cur.execute("""
                ALTER TABLE documents ALTER COLUMN company_id SET NOT NULL;
            """)
            
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_company_id ON documents(company_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_company_created ON documents(company_id, created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_company_type ON documents(company_id, type);")
            
            # =====================================================================
            # 3. CHUNKS TABLE
            # =====================================================================
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id SERIAL PRIMARY KEY,
                    doc_id INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    company_id INTEGER NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
                    text TEXT NOT NULL,
                    section VARCHAR(255),
                    chunk_number INTEGER,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            cur.execute("""
                ALTER TABLE chunks
                ADD COLUMN IF NOT EXISTS company_id INTEGER
                REFERENCES companies(company_id) ON DELETE CASCADE;
            """)
            
            cur.execute("""
                UPDATE chunks SET company_id = 1 WHERE company_id IS NULL;
            """)
            
            cur.execute("""
                ALTER TABLE chunks ALTER COLUMN company_id SET NOT NULL;
            """)
            
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_company_id ON chunks(company_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_company_created ON chunks(company_id, created_at);")
            
            # =====================================================================
            # 4. EMBEDDINGS TABLE
            # =====================================================================
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    emb_id SERIAL PRIMARY KEY,
                    chunk_id INTEGER NOT NULL UNIQUE REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                    company_id INTEGER NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
                    embedding vector(768),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            cur.execute("""
                ALTER TABLE embeddings
                ADD COLUMN IF NOT EXISTS company_id INTEGER
                REFERENCES companies(company_id) ON DELETE CASCADE;
            """)
            
            cur.execute("""
                UPDATE embeddings SET company_id = 1 WHERE company_id IS NULL;
            """)
            
            cur.execute("""
                ALTER TABLE embeddings ALTER COLUMN company_id SET NOT NULL;
            """)
            
            cur.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_company_id ON embeddings(company_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_company_created ON embeddings(company_id, created_at);")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_vector
                ON embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            # =====================================================================
            # 5. INTERNAL_EMPLOYEES TABLE
            # =====================================================================
            cur.execute("""
                CREATE TABLE IF NOT EXISTS internal_employees (
                    emp_id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    role VARCHAR(100),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            cur.execute("CREATE INDEX IF NOT EXISTS idx_internal_emp_email ON internal_employees(email);")
            
            # =====================================================================
            # 6. EXTERNAL_EMPLOYEES TABLE
            # =====================================================================
            cur.execute("""
                CREATE TABLE IF NOT EXISTS external_employees (
                    ext_emp_id SERIAL PRIMARY KEY,
                    company_id INTEGER NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    role VARCHAR(100),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            cur.execute("CREATE INDEX IF NOT EXISTS idx_external_emp_company ON external_employees(company_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_external_emp_email ON external_employees(email);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_external_emp_company_email ON external_employees(company_id, email);")
            
            conn.commit()
            # print("[INFO] Database initialized with 6 tables:")
            # print("       - companies")
            # print("       - documents")
            # print("       - chunks")
            # print("       - embeddings")
            # print("       - internal_employees")
            # print("       - external_employees")
            
        except Exception as e:
            conn.rollback()
            print(f"[ERROR] Database initialization failed: {e}")
            raise
        finally:
            cur.close()
            conn.close()

    def build_from_documents(self, documents: List[Any]):
        """Insert documents, chunks, and embeddings with company_id"""
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
                company_id = doc.metadata.get('company_id')
                source = doc.metadata.get("source", f"doc_{idx}")
                name = doc.metadata.get("name", "unknown")
                doc_type = doc.metadata.get("type", "unknown")
                
                if company_id is None:
                    print(f"[WARNING] Document '{name}' has no company_id!")
                    continue
                
                cur.execute(
                    "INSERT INTO documents (company_id, name, type, source) VALUES (%s, %s, %s, %s) RETURNING doc_id",
                    (company_id, name, doc_type, source)
                )
                doc_id = cur.fetchone()[0]
                doc_map[source] = doc_id
                doc_map[f"doc_{idx}"] = doc_id
                
                if self.debug:
                    print(f"[DEBUG] Mapped document '{source}' (company_id={company_id}) to doc_id={doc_id}")

            # Prepare chunk data with company_id
            chunk_data = []
            unmapped_count = 0
            for i, chunk in enumerate(chunks):
                chunk_doc_ref = chunk.metadata.get("document")
                document_id = doc_map.get(chunk_doc_ref)
                
                if document_id is None:
                    source = chunk.metadata.get("source")
                    document_id = doc_map.get(source)
                
                if document_id is None:
                    unmapped_count += 1
                    if self.debug:
                        print(f"[WARNING] Chunk {i} couldn't be mapped to document. Ref: '{chunk_doc_ref}'")
                    continue
                
                company_id = chunk.metadata.get('company_id')
                section = chunk.metadata.get('section', 'Unknown')
                chunk_id = chunk.metadata.get('chunk_id', i)
                
                chunk_data.append((
                    document_id,
                    company_id,
                    clean_text(chunk.page_content),
                    section,
                    chunk_id
                ))
            
            if unmapped_count > 0:
                print(f"[WARNING] {unmapped_count} chunks could not be mapped to documents!")
            
            # Insert chunks and get IDs
            chunk_ids = []
            print(f"[INFO] Inserting {len(chunk_data)} chunks...")
            
            try:
                result = execute_values(
                    cur,
                    "INSERT INTO chunks (doc_id, company_id, text, section, chunk_number) VALUES %s RETURNING chunk_id",
                    chunk_data,
                    page_size=1000,
                    fetch=True
                )
                if result:
                    chunk_ids = [row[0] for row in result]
                else:
                    chunk_ids = [row[0] for row in cur.fetchall()]
                
                print(f"[INFO] Inserted {len(chunk_ids)} chunks via batch insert")
                
            except Exception as e:
                print(f"[WARNING] Batch insert failed: {e}")
                print("[INFO] Falling back to individual inserts...")
                chunk_ids = []
                for chunk_tuple in chunk_data:
                    cur.execute(
                        "INSERT INTO chunks (doc_id, company_id, text, section, chunk_number) VALUES (%s, %s, %s, %s, %s) RETURNING chunk_id",
                        chunk_tuple
                    )
                    chunk_ids.append(cur.fetchone()[0])
                print(f"[INFO] Inserted {len(chunk_ids)} chunks via individual inserts")

            # Verify chunk_ids length matches
            if len(chunk_ids) != len(embeddings):
                raise ValueError(f"Mismatch: {len(chunk_ids)} chunks inserted but {len(embeddings)} embeddings generated!")

            # Prepare embedding data with company_id
            embedding_data = []
            for i, (chunk_id, emb) in enumerate(zip(chunk_ids, embeddings)):
                company_id = chunk_data[i][1]
                embedding_data.append((chunk_id, company_id, emb.tolist()))
            
            print(f"[INFO] Inserting {len(embedding_data)} embeddings...")
            execute_values(
                cur,
                "INSERT INTO embeddings (chunk_id, company_id, embedding) VALUES %s",
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

    def search(self, query_embedding: np.ndarray, company_id: int, top_k: int = 15) -> List[Dict]:
        """Search vector embeddings for a specific company"""
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            query_vec = query_embedding[0].tolist()
            cur.execute("""
                SELECT ch.chunk_id, ch.text, d.name, d.type, d.source, 1 - (e.embedding <=> %s::vector) as similarity
                FROM embeddings e
                JOIN chunks ch ON e.chunk_id = ch.chunk_id
                JOIN documents d ON ch.doc_id = d.doc_id
                WHERE e.company_id = %s
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
            """, (query_vec, company_id, query_vec, top_k))
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

    def query(self, query_text: str, company_id: int, top_k: int = 15, initial_k: int = 50, use_hybrid: bool = True) -> List[Dict]:
        """Hybrid retrieval: BM25 + Dense embeddings + Reranking for a specific company"""
        query_text = query_text.strip()
        if not query_text:
            print("[ERROR] Empty query text")
            return []
        
        if company_id is None:
            print("[ERROR] company_id is required for scoped queries")
            return []
        
        if use_hybrid:
            bm25_results = self.search_bm25(query_text, company_id, top_k=initial_k // 2)
            dense_results = self.search_dense(query_text, company_id, top_k=initial_k // 2)
            
            seen_ids = set()
            combined_results = []
            
            for r in bm25_results + dense_results:
                chunk_id = r.get("chunk_id")
                if chunk_id not in seen_ids:
                    combined_results.append(r)
                    seen_ids.add(chunk_id)
            
            initial_results = combined_results[:initial_k]
        else:
            initial_results = self.search_dense(query_text, company_id, top_k=initial_k)
        
        if self.use_reranker and self.reranker and initial_results:
            pairs = [(query_text, r["metadata"]["text"]) for r in initial_results]
            scores = self.reranker.predict(pairs)
            for i, score in enumerate(scores):
                initial_results[i]["rerank_score"] = float(score)
            final_results = sorted(initial_results, key=lambda x: x.get("rerank_score", -999), reverse=True)[:top_k]
        else:
            final_results = initial_results[:top_k]
        
        return final_results

    def search_dense(self, query_text: str, company_id: int, top_k: int = 15) -> List[Dict]:
        """Dense vector search for a specific company"""
        query_emb = self.model.encode([query_text], convert_to_numpy=True).astype('float32')
        
        if np.isnan(query_emb).any():
            print(f"[ERROR] NaN in embedding")
            return []
        
        return self.search(query_emb, company_id, top_k=top_k)

    def search_bm25(self, query_text: str, company_id: int, top_k: int = 15) -> List[Dict]:
        """BM25 keyword search for a specific company"""
        conn = self._get_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("""
                SELECT ch.chunk_id, ch.text, d.name, d.type, d.source
                FROM chunks ch
                JOIN documents d ON ch.doc_id = d.doc_id
                WHERE ch.company_id = %s""", (company_id,))
            
            rows = cur.fetchall()
            if not rows:
                if self.debug:
                    print(f"[DEBUG] No chunks found for company_id={company_id}")
                return []
            
            chunk_ids = [row[0] for row in rows]
            texts = [row[1] for row in rows]
            metadata = [{
                "text": row[1],
                "document_name": row[2],
                "document_type": row[3],
                "source": row[4]
            } for row in rows]
            
            tokenized_corpus = [doc.lower().split() for doc in texts]
            tokenized_query = query_text.lower().split()
            
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(tokenized_query)
            
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    results.append({
                        "chunk_id": chunk_ids[idx],
                        "bm25_score": float(scores[idx]),
                        "distance": 1 - (scores[idx] / max(scores)),
                        "metadata": metadata[idx]
                    })
            
            return results
            
        finally:
            cur.close()
            conn.close()

    def get_stats(self, company_id: int = None) -> Dict:
        """Get statistics, optionally scoped to a specific company"""
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            if company_id is not None:
                cur.execute("SELECT COUNT(*) FROM chunks WHERE company_id = %s", (company_id,))
                total_chunks = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM documents WHERE company_id = %s", (company_id,))
                total_docs = cur.fetchone()[0]
                return {
                    "company_id": company_id,
                    "total_chunks": total_chunks,
                    "total_sources": total_docs
                }
            else:
                cur.execute("SELECT COUNT(*) FROM chunks")
                total_chunks = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM documents")
                total_docs = cur.fetchone()[0]
                return {"total_chunks": total_chunks, "total_sources": total_docs}
        finally:
            cur.close()
            conn.close()

    def clear(self):
        """Clear all data from vector store with proper timeout handling"""
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute("SET statement_timeout = '60s';")
            
            print("[INFO] Clearing embeddings...")
            cur.execute("DELETE FROM embeddings;")
            
            print("[INFO] Clearing chunks...")
            cur.execute("DELETE FROM chunks;")
            
            print("[INFO] Clearing documents...")
            cur.execute("DELETE FROM documents;")
            
            cur.execute("ALTER SEQUENCE documents_doc_id_seq RESTART WITH 1;")
            cur.execute("ALTER SEQUENCE chunks_chunk_id_seq RESTART WITH 1;")
            cur.execute("ALTER SEQUENCE embeddings_emb_id_seq RESTART WITH 1;")
            
            conn.commit()
            print("[INFO] Successfully cleared all data from vector store")
            
        except psycopg2.errors.QueryCanceled:
            conn.rollback()
            print("[ERROR] Query timeout while clearing. Retrying with longer timeout...")
            import time
            time.sleep(2)
            self.clear()
            
        except Exception as e:
            conn.rollback()
            print(f"[ERROR] Failed to clear vector store: {e}")
            raise
            
        finally:
            cur.close()
            conn.close()


# Example usage
if __name__ == "__main__":
    from src.data_loader import load_all_documents
    
    docs = load_all_documents("data")
    store = PgVectorStore()
    store.build_from_documents(docs)
    
    print(store.query("What is the company policy?", company_id=1, top_k=15, initial_k=50))
    print(store.get_stats(company_id=1))