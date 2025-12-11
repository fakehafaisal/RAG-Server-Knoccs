from src.data_loader import load_all_documents
from src.vectorstore import PgVectorStore
from src.search import RAGSearch

def build_knowledge_base():
    print("\n" + "="*80)
    print("BUILDING KNOWLEDGE BASE")
    print("="*80)
    
    docs = load_all_documents("data")
    store = PgVectorStore(chunk_size=512, chunk_overlap=128, use_reranker=True)
    store.build_from_documents(docs)
    
    stats = store.get_stats()
    print(f"\n[SUCCESS] Knowledge base built successfully!")
    print(f"[INFO] Total chunks: {stats['total_chunks']}")
    print(f"[INFO] Total sources: {stats['total_sources']}")
    print("="*80 + "\n")

def query_knowledge_base(query: str, top_k: int = 4):
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
    print("\n" + "="*80)
    print("INTERACTIVE RAG KNOWLEDGE BASE")
    print("="*80)
    print("Type your questions below. Type 'quit' or 'exit' to stop.\n")
    
    rag_search = RAGSearch(use_query_expansion=True)
    
    # Show stats
    stats = rag_search.get_stats()
    print(f"[INFO] Knowledge base contains {stats['total_chunks']} chunks from {stats['total_sources']} sources\n")
    
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
        
        summary = rag_search.search_and_summarize(query, top_k=4)
        
        print("\n" + "-"*80)
        print("ANSWER:")
        print("-"*80)
        print(summary)
        print("-"*80 + "\n")

def show_stats():
    print("\n" + "="*80)
    print("KNOWLEDGE BASE STATISTICS")
    print("="*80)
    
    rag_search = RAGSearch(use_query_expansion=False)
    stats = rag_search.get_stats()
    
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total sources: {stats['total_sources']}")
    print("="*80 + "\n")

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
    
    # Show stats
    elif len(sys.argv) > 1 and sys.argv[1] == "--stats":
        show_stats()
    
    # Default: run example queries
    else:
        print("\nUsage:")
        print("  python app.py --build              # Build knowledge base from data folder")
        print("  python app.py --interactive        # Interactive Q&A mode")
        print("  python app.py --query 'question'   # Ask a single question")
        print("  python app.py --stats              # Show knowledge base statistics")
        print("\nRunning example queries...\n")
        
        # Example queries
        example_queries = [
            "What is Scrum?",
            "Tell me how to derive the Poisson loss function in easy steps",
            "What are the key principles of agile development?"
        ]
        
        for query in example_queries:
            query_knowledge_base(query, top_k=4)