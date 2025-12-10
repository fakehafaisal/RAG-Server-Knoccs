# import os
# from dotenv import load_dotenv
# from src.vectorstore import FaissVectorStore
# from langchain_groq import ChatGroq

# load_dotenv()

# class RAGSearch:
#     def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-mpnet-base-v2", llm_model: str = "llama-3.3-70b-versatile"):
#         self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
#         # Load or build vectorstore
#         faiss_path = os.path.join(persist_dir, "faiss.index")
#         meta_path = os.path.join(persist_dir, "metadata.pkl")
#         if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
#             from .data_loader import load_all_documents
#             docs = load_all_documents("data")
#             self.vectorstore.build_from_documents(docs)
#         else:
#             self.vectorstore.load()
#         groq_api_key = os.getenv("GROQ_API_KEY")
#         self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
#         print(f"[INFO] Groq LLM initialized: {llm_model}")

#     def search_and_summarize(self, query: str, top_k: int = 20) -> str:
#         results = self.vectorstore.query(query, top_k=top_k)
#         texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
#         context = "\n\n".join(texts)
#         if not context:
#             return "No relevant documents found."
#         prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
#         response = self.llm.invoke([prompt])
#         return response.content

# # Example usage
# if __name__ == "__main__":
#     rag_search = RAGSearch()
#     query = "What is attention mechanism?"
#     summary = rag_search.search_and_summarize(query, top_k=3)
#     print("Summary:", summary)

import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq
from typing import List, Dict

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", 
                 embedding_model: str = "all-mpnet-base-v2", 
                 llm_model: str = "llama-3.3-70b-versatile",
                 use_query_expansion: bool = True):
        
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model, use_reranker=True)
        
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from .data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model, temperature=0.1)
        self.use_query_expansion = use_query_expansion
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def expand_query(self, query: str) -> List[str]:
        """
        Generate related queries to improve retrieval coverage
        """
        if not self.use_query_expansion:
            return [query]
        
        expansion_prompt = f"""Given the following question, generate 2 alternative phrasings or related questions that would help retrieve relevant information. Keep them concise.

Original question: {query}

Alternative questions (one per line):"""
        
        try:
            response = self.llm.invoke(expansion_prompt)
            alternatives = [line.strip() for line in response.content.strip().split('\n') if line.strip()]
            # Return original + alternatives (max 3 total)
            return [query] + alternatives[:2]
        except:
            return [query]

    def search_and_summarize(self, query: str, top_k: int = 4, initial_k: int = 15) -> str:
        """
        Enhanced RAG pipeline with query expansion and better prompting
        """
        # Step 1: Query expansion
        queries = self.expand_query(query)
        print(f"[INFO] Expanded query into {len(queries)} variations")
        
        # Step 2: Retrieve for all query variations and deduplicate
        all_results = []
        seen_texts = set()
        
        for q in queries:
            results = self.vectorstore.query(q, top_k=top_k, initial_k=initial_k)
            for r in results:
                text = r["metadata"].get("text", "")
                if text and text not in seen_texts:
                    all_results.append(r)
                    seen_texts.add(text)
        
        # Take top results
        all_results = all_results[:top_k]
        
        if not all_results:
            return "No relevant documents found in the knowledge base."
        
        # Step 3: Build context with metadata
        context_parts = []
        for i, r in enumerate(all_results, 1):
            meta = r["metadata"]
            text = meta.get("text", "")
            source = meta.get("source", "unknown")
            
            context_parts.append(f"[Document {i}] (Source: {source})\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # Step 4: Enhanced prompt for better synthesis
        prompt = f"""You are a helpful AI assistant with access to a knowledge base. Answer the user's question based on the provided context.

**Instructions:**
1. Synthesize information from the documents into a clear, cohesive answer
2. Write naturally - avoid repetition and over-citing
3. Structure your answer logically (use paragraphs, not bullet points unless asked)
4. For technical topics, explain clearly and concisely
5. Only mention document sources if there are conflicting views or to highlight key points
6. If information is incomplete, briefly state what's missing at the end
7. Keep your answer focused and concise - quality over quantity

**Question:** {query}

**Context from Knowledge Base:**
{context}

**Your Answer:**"""

        # Step 5: Generate response
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def search_only(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Return raw search results without LLM summarization (useful for debugging)
        """
        return self.vectorstore.query(query, top_k=top_k, initial_k=top_k*4)

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    
    query = "Tell me how to derive the Poisson loss function, tell me the steps in easy and concise words"
    summary = rag_search.search_and_summarize(query, top_k=5)
    print("="*80)
    print("ANSWER:")
    print("="*80)
    print(summary)