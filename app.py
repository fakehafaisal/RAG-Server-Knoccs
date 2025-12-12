from src.data_loader import load_all_documents
from src.vectorstore import PgVectorStore
from src.search import RAGSearch

# -------------------------
# GLOBAL SINGLETONS - DON'T BUILD HERE!
# -------------------------
store = None
rag_search = None

def initialize_rag():
    """Initialize RAG system (call once)"""
    global store, rag_search
    
    if store is None:
        store = PgVectorStore(chunk_size=1024, chunk_overlap=256, use_reranker=True)
    
    if rag_search is None:
        rag_search = RAGSearch(use_query_expansion=False, vectorstore=store)
    
    return store, rag_search

# -------------------------
# FUNCTIONS
# -------------------------
def rebuild_knowledge_base():
    print("\n" + "="*80)
    print("REBUILDING KNOWLEDGE BASE")
    print("="*80)
    
    global store, rag_search
    
    # Initialize if needed
    if store is None:
        store = PgVectorStore(chunk_size=1024, chunk_overlap=256, use_reranker=True)
    
    # Clear existing data
    store.clear()
    
    # Load and build
    docs = load_all_documents("data")
    store.build_from_documents(docs)
    
    # Reinitialize RAG
    rag_search = RAGSearch(use_query_expansion=False, vectorstore=store)
    
    stats = store.get_stats()
    print(f"\n[SUCCESS] Knowledge base rebuilt successfully!")
    print(f"[INFO] Total chunks: {stats['total_chunks']}")
    print(f"[INFO] Total sources: {stats['total_sources']}")
    print("="*80 + "\n")


def query_knowledge_base(query: str, top_k: int = 15):
    global store, rag_search
    initialize_rag()
    
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    summary = rag_search.search_and_summarize(query, top_k=top_k)
    
    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(summary)
    print("="*80 + "\n")
    
    return summary


def interactive_mode():
    global store, rag_search
    initialize_rag()
    
    print("\n" + "="*80)
    print("INTERACTIVE RAG KNOWLEDGE BASE")
    print("="*80)
    print("Type your questions below. Type 'quit' or 'exit' to stop.\n")
    
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
        
        summary = rag_search.search_and_summarize(query, top_k=15)
        
        print("\n" + "-"*80)
        print("ANSWER:")
        print("-"*80)
        print(summary)
        print("-"*80 + "\n")


def show_stats():
    global store, rag_search
    initialize_rag()
    
    print("\n" + "="*80)
    print("KNOWLEDGE BASE STATISTICS")
    print("="*80)
    
    stats = rag_search.get_stats()
    
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total sources: {stats['total_sources']}")
    print("="*80 + "\n")


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        rebuild_knowledge_base()
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--query":
        if len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            query_knowledge_base(query)
        else:
            print("Usage: python app.py --query 'your question here'")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--stats":
        show_stats()
    
    else:
        print("\nUsage:")
        print("  python app.py --build              # Rebuild knowledge base from data folder")
        print("  python app.py --interactive        # Interactive Q&A mode")
        print("  python app.py --query 'question'   # Ask a single question")
        print("  python app.py --stats              # Show knowledge base statistics")