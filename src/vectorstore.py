# import os
# import faiss
# import numpy as np
# import pickle
# from typing import List, Any
# from sentence_transformers import SentenceTransformer
# from src.embedding import EmbeddingPipeline

# class FaissVectorStore:
#     def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-mpnet-base-v2", chunk_size: int = 2000, chunk_overlap: int = 300):
#         self.persist_dir = persist_dir
#         os.makedirs(self.persist_dir, exist_ok=True)
#         self.index = None
#         self.metadata = []
#         self.embedding_model = embedding_model
#         self.model = SentenceTransformer(embedding_model)
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         print(f"[INFO] Loaded embedding model: {embedding_model}")

#     def build_from_documents(self, documents: List[Any]):
#         print(f"[INFO] Building vector store from {len(documents)} raw documents...")
#         emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
#         chunks = emb_pipe.chunk_documents(documents)
#         embeddings = emb_pipe.embed_chunks(chunks)
#         metadatas = [{"text": chunk.page_content} for chunk in chunks]
#         self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
#         self.save()
#         print(f"[INFO] Vector store built and saved to {self.persist_dir}")

#     def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
#         dim = embeddings.shape[1]
#         if self.index is None:
#             self.index = faiss.IndexFlatL2(dim)
#         self.index.add(embeddings)
#         if metadatas:
#             self.metadata.extend(metadatas)
#         print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

#     def save(self):
#         faiss_path = os.path.join(self.persist_dir, "faiss.index")
#         meta_path = os.path.join(self.persist_dir, "metadata.pkl")
#         faiss.write_index(self.index, faiss_path)
#         with open(meta_path, "wb") as f:
#             pickle.dump(self.metadata, f)
#         print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

#     def load(self):
#         faiss_path = os.path.join(self.persist_dir, "faiss.index")
#         meta_path = os.path.join(self.persist_dir, "metadata.pkl")
#         self.index = faiss.read_index(faiss_path)
#         with open(meta_path, "rb") as f:
#             self.metadata = pickle.load(f)
#         print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

#     def search(self, query_embedding: np.ndarray, top_k: int = 20):
#         D, I = self.index.search(query_embedding, top_k)
#         results = []
#         for idx, dist in zip(I[0], D[0]):
#             meta = self.metadata[idx] if idx < len(self.metadata) else None
#             results.append({"index": idx, "distance": dist, "metadata": meta})
#         return results

#     def query(self, query_text: str, top_k: int = 20):
#         print(f"[INFO] Querying vector store for: '{query_text}'")
#         query_emb = self.model.encode([query_text]).astype('float32')
#         return self.search(query_emb, top_k=top_k)

# # Example usage
# if __name__ == "__main__":
#     from data_loader import load_all_documents
#     docs = load_all_documents("data")
#     store = FaissVectorStore("faiss_store")
#     store.build_from_documents(docs)
#     store.load()
#     print(store.query("What is attention mechanism?", top_k=20))

import os
import faiss
import numpy as np
import pickle
from typing import List, Any, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", 
                 embedding_model: str = "all-mpnet-base-v2", 
                 chunk_size: int = 512, 
                 chunk_overlap: int = 128,
                 use_reranker: bool = True):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Add cross-encoder reranker for better relevance
        self.use_reranker = use_reranker
        if use_reranker:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print(f"[INFO] Loaded reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        
        # Store rich metadata
        metadatas = [
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get('source', 'unknown'),
                "chunk_id": chunk.metadata.get('chunk_id', i),
                "page": chunk.metadata.get('page', None)
            } 
            for i, chunk in enumerate(chunks)
        ]
        
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        
        self.index.add(embeddings)
        
        if metadatas:
            self.metadata.extend(metadatas)
        
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 20):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        
        return results

    def rerank_results(self, query_text: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank results using cross-encoder for better relevance
        """
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

    def query(self, query_text: str, top_k: int = 5, initial_k: int = 20):
        """
        Query with two-stage retrieval:
        1. Get initial_k results from vector search
        2. Rerank to get top_k best results
        """
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

# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents
    
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    
    print(store.query("What is attention mechanism?", top_k=5, initial_k=20))