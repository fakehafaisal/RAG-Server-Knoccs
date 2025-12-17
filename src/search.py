import os
from dotenv import load_dotenv
from src.vectorstore import PgVectorStore
from langchain_openai import ChatOpenAI  # Changed from ChatGroq
from typing import List, Dict

load_dotenv()

class RAGSearch:
    def __init__(self, embedding_model: str = "all-mpnet-base-v2", llm_model: str = "gpt-5",
                use_query_expansion: bool = False,
                debug: bool = False, vectorstore: PgVectorStore = None):
    
        if vectorstore:
            self.vectorstore = vectorstore
        else:
            self.vectorstore = PgVectorStore(embedding_model=embedding_model, use_reranker=True)
        
        openai_api_key = os.getenv("OPENAI_API_KEY")  # Changed from GROQ_API_KEY
        self.llm = ChatOpenAI(api_key=openai_api_key, model=llm_model, temperature=0.1)  # Changed to ChatOpenAI
        self.use_query_expansion = use_query_expansion
        self.debug = debug
        print(f"LLM initialized: {llm_model}")


    def search_and_summarize(self, query: str, top_k: int = 15, initial_k: int = 50) -> str:
        # Step 1: Query expansion
        queries = query
        # print(f"[INFO] Expanded query into {len(queries)} variations")
        
        # Step 2: Retrieve for all query variations and deduplicate
        all_results = []
        seen_chunk_ids = set()  # Use chunk_id instead 
        
        # for q in queries:
        results = self.vectorstore.query(queries, top_k=top_k, initial_k=initial_k, use_hybrid=True)
        if self.debug:
            # print(f"[DEBUG] Query '{q[:50]}...' returned {len(results)} results")
            pass
            
        for r in results:
            chunk_id = r.get("chunk_id")
            # Use chunk_id for deduplication instead of text
            if chunk_id and chunk_id not in seen_chunk_ids:
                all_results.append(r)
                seen_chunk_ids.add(chunk_id)
        
        if self.debug:
            # print(f"[DEBUG] Total unique results after deduplication: {len(all_results)}")
            pass
        
        # Sort by score (rerank_score if available, else distance)
        all_results.sort(
            key=lambda x: x.get("rerank_score", -x.get("distance", 999)), 
            reverse=True
        )
        
        # Take top results
        all_results = all_results[:top_k]
        
        if self.debug and all_results:
            print(f"[DEBUG] Top result score: {all_results[0].get('rerank_score', all_results[0].get('distance'))}")
            print(f"[DEBUG] Top result preview: {all_results[0]['metadata'].get('text', '')[:200]}...")
        
        if not all_results:
            return "I couldn't find any relevant documents in the knowledge base for your query."
        
        # Step 3: Build context with metadata (include doc_name and doc_type)
        context_parts = []
        for i, r in enumerate(all_results, 1):
            meta = r["metadata"]
            text = meta.get("text", "")
            source = meta.get("source", "unknown")
            doc_name = meta.get("document_name", "unknown")
            doc_type = meta.get("document_type", "unknown")
            
            # Add score info in debug mode
            score_info = ""
            if self.debug:
                score = r.get("rerank_score", r.get("distance", "N/A"))
                score_info = f" [Score: {score:.3f}]"
            
            # DON'T use [Document 1], just add the text with source info
            context_parts.append(
                f"Source: {source} | Type: {doc_type}{score_info}\n{text}"
            )
                
        context = "\n\n".join(context_parts)
        
        # Step 4: Knoccs-specific prompt
        prompt = f"""You are Knoccs AI Assistant, helping staff find information from client documents, agreements, and communications.

**Guidelines:**
- Answer based strictly on the provided documents
- Do NOT use phrases like "According to Document 1" or "Document 2 states"
- If information isn't in the documents, say "I couldn't find this information in the available documents"
- For agreements or contracts, mention the specific document name/source at the end
- Write naturally as if synthesizing information from your knowledge base
- Present information clearly and professionally for business use
- If multiple documents contain relevant info, synthesize them coherently


**Query:** {query}

**Retrieved Documents:**
{context}

**Response:**"""

        # Step 5: Generate response
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def search_only(self, query: str, top_k: int = 15) -> List[Dict]:
        """Return raw search results without LLM summarization"""
        return self.vectorstore.query(query, top_k=top_k, initial_k=top_k*4)

    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        return self.vectorstore.get_stats()




# Example usage
if __name__ == "__main__":
    # Initialize with debug mode
    rag_search = RAGSearch(debug=True, use_query_expansion=False)
    
    print("Knowledge base stats:", rag_search.get_stats())
    
    # Test query
    query = "Can you give me a summary on the reflection on muscular design by brenda laurel?"
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print('='*80)
    
    summary = rag_search.search_and_summarize(query, top_k=15, initial_k=50)
    print("\nANSWER:")
    print(summary)
    
    # DeepEval evaluation (optional)
    print(f"\n{'='*80}")
    print("DEEPEVAL EVALUATION")
    print('='*80)