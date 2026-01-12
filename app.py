# from src.data_loader import load_all_documents
# from src.vectorstore import PgVectorStore
# from src.search import RAGSearch

# # -------------------------
# # GLOBAL SINGLETONS
# # -------------------------
# store = None
# rag_search = None

# def initialize_rag(debug=False):
#     """Initialize RAG system (call once)"""
#     global store, rag_search
    
#     if store is None:
#         store = PgVectorStore(
#             chunk_size=1024, 
#             chunk_overlap=256, 
#             use_reranker=True,
#             debug=debug
#         )
    
#     if rag_search is None:
#         rag_search = RAGSearch(
#             use_query_expansion=False, 
#             debug=debug
#         )
    
#     return store, rag_search

# # -------------------------
# # FUNCTIONS
# # -------------------------
# def rebuild_knowledge_base(debug=True):
#     """Rebuild knowledge base from data folder"""
#     print("\n" + "="*80)
#     print("BUILDING KNOWLEDGE BASE")
#     print("="*80)
    
#     global store, rag_search
    
#     # Initialize if needed
#     if store is None:
#         store = PgVectorStore(
#             chunk_size=1024, 
#             chunk_overlap=256, 
#             use_reranker=True,
#             debug=debug
#         )
    
#     # Clear existing data
#     store.clear()
    
#     # Load and build
#     docs = load_all_documents("data")
#     print(f"Loaded {len(docs)} documents")
    
#     store.build_from_documents(docs)
    
#     # Reinitialize RAG
#     rag_search = RAGSearch(
#         use_query_expansion=False,
#         debug=debug
#     )
    
#     stats = store.get_stats()
#     print(f"\n[SUCCESS] Knowledge base rebuilt successfully!")
#     print(f"[INFO] Total chunks: {stats['total_chunks']}")
#     print(f"[INFO] Total sources: {stats['total_sources']}")
#     print("="*80 + "\n")


# def query_knowledge_base(query: str, top_k: int = 15, debug=False):
#     """Query knowledge base with automatic company extraction"""
#     global store, rag_search
#     initialize_rag(debug=debug)
    
#     print("\n" + "="*80)
#     print(f"QUERY: {query}")
#     print("="*80)
    
#     # Company is auto-extracted inside search_and_summarize
#     summary = rag_search.search_and_summarize(query, top_k=top_k)
    
#     print("\n" + "="*80)
#     print("ANSWER:")
#     print("="*80)
#     print(summary)
#     print("="*80 + "\n")
    
#     return summary


# def interactive_mode(debug=False):
#     """Interactive Q&A mode with automatic company extraction"""
#     global store, rag_search
#     initialize_rag(debug=debug)
    
#     print("\n" + "="*80)
#     print("INTERACTIVE RAG KNOWLEDGE BASE")
#     print("="*80)
#     print("Type your questions below (include company name). Type 'quit' or 'exit' to stop.\n")
#     print("Examples:")
#     print("  - What is Pulse Unity's main business?")
#     print("  - Tell me about vantage nexus HR policies")
#     print("  - What agreements does BeaconPioneer have?\n")
    
#     stats = rag_search.get_stats()
#     print(f"[INFO] Knowledge base contains {stats['total_chunks']} chunks from {stats['total_sources']} sources\n")
    
#     while True:
#         query = input("Your question: ").strip()
        
#         if query.lower() in ['quit', 'exit', 'q']:
#             print("\nGoodbye!")
#             break
#         if not query:
#             continue
        
#         print("\n" + "-"*80)
#         print("Searching knowledge base...")
#         print("-"*80)
        
#         summary = rag_search.search_and_summarize(query, top_k=15)
        
#         print("\n" + "-"*80)
#         print("ANSWER:")
#         print("-"*80)
#         print(summary)
#         print("-"*80 + "\n")


# def show_stats(company_name: str = None):
#     """Show knowledge base statistics"""
#     global store, rag_search
#     initialize_rag()
    
#     print("\n" + "="*80)
#     print("KNOWLEDGE BASE STATISTICS")
#     print("="*80)
    
#     stats = rag_search.get_stats()
    
#     print(f"Total chunks: {stats['total_chunks']}")
#     print(f"Total sources: {stats['total_sources']}")
#     print("="*80 + "\n")


# # -------------------------
# # MAIN
# # -------------------------
# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) > 1 and sys.argv[1] == "--build":
#         debug = "--debug" in sys.argv
#         rebuild_knowledge_base(debug=debug)
    
#     elif len(sys.argv) > 1 and sys.argv[1] == "--interactive":
#         debug = "--debug" in sys.argv
#         interactive_mode(debug=debug)
    
#     elif len(sys.argv) > 1 and sys.argv[1] == "--query":
#         if len(sys.argv) > 2:
#             debug = "--debug" in sys.argv
#             query_parts = [arg for arg in sys.argv[2:] if arg != "--debug"]
#             query = " ".join(query_parts)
#             query_knowledge_base(query, debug=debug)
#         else:
#             print("Usage: python app.py --query 'your question here' [--debug]")
    
#     elif len(sys.argv) > 1 and sys.argv[1] == "--stats":
#         show_stats()
    
#     else:
#         print("\nUsage:")
#         print("  python app.py --build [--debug]")
#         print("    Rebuild knowledge base from data folder")
#         print()
#         print("  python app.py --interactive [--debug]")
#         print("    Interactive Q&A mode (company auto-extracted from query)")
#         print()
#         print("  python app.py --query 'question' [--debug]")
#         print("    Ask a single question (include company name in query)")
#         print("    Example: python app.py --query 'What is Pulse Unity's main business?'")
#         print()
#         print("  python app.py --stats")
#         print("    Show knowledge base statistics")
#         print()
#         print("Note: Always include a company name in your query (e.g., 'Pulse Unity', 'BeaconPioneer')")

from src.data_loader import load_all_documents
from src.vectorstore import PgVectorStore
from src.search import RAGSearch

# -------------------------
# GLOBAL SINGLETONS
# -------------------------
store = None
rag_search = None

def initialize_rag(debug=False):
    """Initialize RAG system (call once)"""
    global store, rag_search
    
    if store is None:
        store = PgVectorStore(
            chunk_size=1024, 
            chunk_overlap=256, 
            use_reranker=True,
            debug=debug
        )
    
    if rag_search is None:
        rag_search = RAGSearch(
            use_query_expansion=False, 
            debug=debug
        )
    
    return store, rag_search

# -------------------------
# FUNCTIONS
# -------------------------
def rebuild_knowledge_base(debug=True):
    """Rebuild knowledge base from data folder"""
    print("\n" + "="*80)
    print("BUILDING KNOWLEDGE BASE")
    print("="*80)
    
    global store, rag_search
    
    # Initialize if needed
    if store is None:
        store = PgVectorStore(
            chunk_size=1024, 
            chunk_overlap=256, 
            use_reranker=True,
            debug=debug
        )
    
    # Clear existing data
    store.clear()
    
    # Load and build
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents")
    
    store.build_from_documents(docs)
    
    # Reinitialize RAG
    rag_search = RAGSearch(
        use_query_expansion=False,
        debug=debug
    )
    
    stats = store.get_stats()
    print(f"\n[SUCCESS] Knowledge base rebuilt successfully!")
    print(f"[INFO] Total chunks: {stats['total_chunks']}")
    print(f"[INFO] Total sources: {stats['total_sources']}")
    print("="*80 + "\n")


def query_knowledge_base(query: str, top_k: int = 15, debug=False):
    """Query knowledge base with conversation history"""
    global store, rag_search
    initialize_rag(debug=debug)
    
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    # No conversation history for single query
    response, _ = rag_search.search_and_summarize(query, top_k=top_k, conversation_history=[])
    
    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(response)
    print("="*80 + "\n")
    
    return response


def interactive_mode(debug=False):
    """Interactive Q&A mode with conversation history"""
    global store, rag_search
    initialize_rag(debug=debug)
    
    print("\n" + "="*80)
    print("INTERACTIVE RAG KNOWLEDGE BASE")
    print("="*80)
    print("Type your questions below (include company name). Type 'quit' or 'exit' to stop.")
    print("Conversation history is maintained during this session.\n")
    print("Examples:")
    print("  - What is Pulse Unity's main business?")
    print("  - Tell me about vantage nexus HR policies")
    print("  - Which company does Heidi Smith work at?\n")
    
    stats = rag_search.get_stats()
    print(f"[INFO] Knowledge base contains {stats['total_chunks']} chunks from {stats['total_sources']} sources\n")
    
    # Initialize conversation history for this session
    conversation_history = []
    
    while True:
        query = input("Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! Conversation history has been cleared.")
            break
        if not query:
            continue
        
        print("\n" + "-"*80)
        print("Searching knowledge base...")
        print("-"*80)
        
        # Search and get response WITH conversation history
        response, conversation_history = rag_search.search_and_summarize(
            query, 
            top_k=15,
            conversation_history=conversation_history
        )
        
        print("\n" + "-"*80)
        print("ANSWER:")
        print("-"*80)
        print(response)
        print("-"*80 + "\n")


def show_stats(company_id: int = None):
    """Show knowledge base statistics"""
    global store, rag_search
    initialize_rag()
    
    print("\n" + "="*80)
    print("KNOWLEDGE BASE STATISTICS")
    print("="*80)
    
    stats = rag_search.get_stats(company_id=company_id)
    
    if company_id:
        print(f"Company ID: {stats.get('company_id')}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total sources: {stats['total_sources']}")
    print("="*80 + "\n")


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        debug = "--debug" in sys.argv
        rebuild_knowledge_base(debug=debug)
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        debug = "--debug" in sys.argv
        interactive_mode(debug=debug)
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--query":
        if len(sys.argv) > 2:
            debug = "--debug" in sys.argv
            query_parts = [arg for arg in sys.argv[2:] if arg != "--debug"]
            query = " ".join(query_parts)
            query_knowledge_base(query, debug=debug)
        else:
            print("Usage: python app.py --query 'your question here' [--debug]")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--stats":
        show_stats()
    
    else:
        print("\nUsage:")
        print("  python app.py --build [--debug]")
        print("    Rebuild knowledge base from data folder")
        print()
        print("  python app.py --interactive [--debug]")
        print("    Interactive Q&A mode (company auto-extracted from query)")
        print("    Conversation history is retained during the chat session")
        print()
        print("  python app.py --query 'question' [--debug]")
        print("    Ask a single question (include company name in query)")
        print("    Example: python app.py --query 'What is Pulse Unity's main business?'")
        print()
        print("  python app.py --stats")
        print("    Show knowledge base statistics")
        print()
        # print("Note: Always include a company name in your query (e.g., 'Pulse Unity', 'BeaconPioneer')")