from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-mpnet-base-v2", chunk_size: int = 512, chunk_overlap: int = 128):
        """
        Improved chunking strategy:
        - Smaller chunks (512 tokens) for better precision
        - 25% overlap to preserve context across boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")
        print(f"[INFO] Chunk size: {chunk_size}, Overlap: {chunk_overlap}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Smart chunking with better separators for technical content
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            # Better separators for technical/business documents
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        
        # Add chunk metadata for better tracking
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            # Preserve source information
            if 'source' not in chunk.metadata:
                chunk.metadata['source'] = 'unknown'
        
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)