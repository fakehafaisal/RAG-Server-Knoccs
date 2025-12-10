# from src.data_loader import load_all_documents
# from src.vectorstore import FaissVectorStore
# from src.search import RAGSearch

# # Example usage
# if __name__ == "__main__":
    
#     docs = load_all_documents("data")
#     store = FaissVectorStore("faiss_store")
#     store.build_from_documents(docs)
#     # store.load()
#     #print(store.query("What is attention mechanism?", top_k=3))
#     rag_search = RAGSearch()
#     query = "What is scrum?"
#     summary = rag_search.search_and_summarize(query, top_k=20)
#     print("Summary:", summary)

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

def build_knowledge_base():
    """
    Build the vector store from documents in the data folder
    """
    print("\n" + "="*80)
    print("BUILDING KNOWLEDGE BASE")
    print("="*80)
    
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store", chunk_size=512, chunk_overlap=128, use_reranker=True)
    store.build_from_documents(docs)
    
    print("\n[SUCCESS] Knowledge base built successfully!")
    print("="*80 + "\n")

def query_knowledge_base(query: str, top_k: int = 5):
    """
    Query the knowledge base and get an answer
    """
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    rag_search = RAGSearch(use_query_expansion=True)
    summary = rag_search.search_and_summarize(query, top_k=top_k)
    
    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(summary)
    print("="*80 + "\n")
    
    return summary

def interactive_mode():
    """
    Interactive question-answering mode
    """
    print("\n" + "="*80)
    print("INTERACTIVE RAG KNOWLEDGE BASE")
    print("="*80)
    print("Type your questions below. Type 'quit' or 'exit' to stop.\n")
    
    rag_search = RAGSearch(use_query_expansion=True)
    
    while True:
        query = input("Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        print("\n" + "-"*80)
        print("Searching knowledge base...")
        print("-"*80)
        
        summary = rag_search.search_and_summarize(query, top_k=5)
        
        print("\n" + "-"*80)
        print("ANSWER:")
        print("-"*80)
        print(summary)
        print("-"*80 + "\n")

# Example usage
if __name__ == "__main__":
    import sys
    
    # Check if we need to rebuild the knowledge base
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        build_knowledge_base()
    
    # Check for interactive mode
    elif len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    
    # Single query mode
    elif len(sys.argv) > 1 and sys.argv[1] == "--query":
        if len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            query_knowledge_base(query)
        else:
            print("Usage: python app.py --query 'your question here'")
    
    # Default: run example queries
    else:
        print("\nUsage:")
        print("  python app.py --build              # Build knowledge base from data folder")
        print("  python app.py --interactive        # Interactive Q&A mode")
        print("  python app.py --query 'question'   # Ask a single question")
        print("\nRunning example queries...\n")
        
        # Example queries
        example_queries = [
            "What is Scrum?",
            "Tell me how to derive the Poisson loss function in easy steps",
            "What are the key principles of agile development?"
        ]
        
        for query in example_queries:
            query_knowledge_base(query, top_k=5)