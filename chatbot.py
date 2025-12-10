# import streamlit as st
# from src.search import RAGSearch

# # Page config
# st.set_page_config(
#     page_title="Knoccs Knowledge Base",
#     page_icon="💬",
#     layout="centered"
# )

# # Initialize RAG search (only once)
# @st.cache_resource
# def load_rag():
#     return RAGSearch()

# rag_search = load_rag()

# # Title
# st.title("💬 Knoccs Knowledge Base Chatbot")
# st.markdown("Ask me anything about the documents in our knowledge base!")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Chat input
# if prompt := st.chat_input("What would you like to know?"):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # Generate response
#     with st.chat_message("assistant"):
#         with st.spinner("Searching knowledge base..."):
#             response = rag_search.search_and_summarize(prompt, top_k=20)
#         st.markdown(response)
    
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})

# # Sidebar with info
# with st.sidebar:
#     st.header("About")
#     st.info("This chatbot uses RAG (Retrieval Augmented Generation) to answer questions based on Knoccs documentation.")
    
#     if st.button("Clear Chat History"):
#         st.session_state.messages = []
#         st.rerun()

import streamlit as st
from src.search import RAGSearch
from src.vectorstore import PgVectorStore
from src.data_loader import load_all_documents

# Page config
st.set_page_config(
    page_title="Knoccs Knowledge Base",
    page_icon="💬",
    layout="centered"
)

# Initialize RAG search (only once)
@st.cache_resource
def load_rag():
    """Load RAG search system with improved settings"""
    return RAGSearch(use_query_expansion=True)

@st.cache_resource
def check_kb_exists():
    """Check if knowledge base has data"""
    try:
        store = PgVectorStore()
        stats = store.get_stats()
        return stats['total_chunks'] > 0, stats
    except:
        return False, None

# Check if KB exists
kb_exists, kb_stats = check_kb_exists()

# Title
st.title("💬 Knoccs Knowledge Base Chatbot")
st.markdown("Ask me anything about the documents in our knowledge base!")

# Sidebar with info and controls
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Top-k slider for result count
    top_k = st.slider(
        "Number of documents to retrieve",
        min_value=8,
        max_value=20,
        value=10,
        help="More documents = more context but potentially more repetition"
    )
    
    # Query expansion toggle
    use_expansion = st.checkbox(
        "Use query expansion",
        value=True,
        help="Generates alternative phrasings for better retrieval"
    )
    
    # Debug mode
    show_sources = st.checkbox(
        "Show retrieved sources",
        value=False,
        help="Display the actual chunks being retrieved"
    )
    
    st.divider()
    
    st.header("📚 About")
    st.info("""
    This chatbot uses **RAG (Retrieval Augmented Generation)** with:
    - 🔍 Smart chunking (512 tokens)
    - 🎯 Cross-encoder reranking
    - 🔄 Query expansion
    - 🤖 Llama 3.3 70B via Groq
    - 🐘 PostgreSQL + pgvector
    """)
    
    st.divider()
    
    # Knowledge base management
    st.header("🛠️ Knowledge Base")
    
    if kb_exists and kb_stats:
        st.success("✅ Knowledge base loaded")
        st.metric("Total Chunks", f"{kb_stats['total_chunks']:,}")
        st.metric("Total Sources", kb_stats['total_sources'])
        
        if st.button("🔄 Rebuild Knowledge Base", help="Rebuild from data/ folder"):
            with st.spinner("Rebuilding knowledge base..."):
                try:
                    docs = load_all_documents("data")
                    store = PgVectorStore(chunk_size=512, chunk_overlap=128, use_reranker=True)
                    store.build_from_documents(docs)
                    st.success("Knowledge base rebuilt successfully!")
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error rebuilding: {str(e)}")
    else:
        st.warning("⚠️ Knowledge base is empty or not initialized")
        
        if st.button("🏗️ Build Knowledge Base", help="Build from data/ folder"):
            with st.spinner("Building knowledge base... This may take a few minutes."):
                try:
                    docs = load_all_documents("data")
                    if not docs:
                        st.error("No documents found in data/ folder!")
                    else:
                        store = PgVectorStore(chunk_size=512, chunk_overlap=128, use_reranker=True)
                        store.build_from_documents(docs)
                        stats = store.get_stats()
                        st.success(f"Knowledge base built with {stats['total_chunks']:,} chunks from {stats['total_sources']} sources!")
                        st.cache_resource.clear()
                        st.rerun()
                except Exception as e:
                    st.error(f"Error building: {str(e)}")
                    if "connection" in str(e).lower():
                        st.error("Make sure PostgreSQL is running and pgvector extension is installed!")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
if not kb_exists:
    st.error("⚠️ Please build the knowledge base first using the sidebar button.")
    st.info("Make sure PostgreSQL is running with pgvector extension installed.")
    st.stop()

# Load RAG search
try:
    rag_search = load_rag()
except Exception as e:
    st.error(f"Error loading RAG system: {str(e)}")
    if "connection" in str(e).lower():
        st.error("Cannot connect to PostgreSQL. Please check your database connection.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                # Update query expansion setting
                rag_search.use_query_expansion = use_expansion
                
                # Get raw results for debugging if needed
                if show_sources:
                    raw_results = rag_search.search_only(prompt, top_k=top_k)
                
                response = rag_search.search_and_summarize(prompt, top_k=top_k)
                st.markdown(response)
                
                # Show sources if debug mode is on
                if show_sources:
                    with st.expander("🔍 View Retrieved Sources"):
                        for i, result in enumerate(raw_results, 1):
                            st.markdown(f"**Source {i}** (Score: {result.get('rerank_score', 'N/A'):.3f})")
                            st.markdown(f"*File: {result['metadata'].get('source', 'unknown')}*")
                            st.text_area(f"Content {i}", result['metadata'].get('text', ''), height=150, key=f"source_{i}_{len(st.session_state.messages)}")
                            st.divider()
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("*Powered by Llama 3.3 70B (Groq) • PostgreSQL + pgvector • Built with Streamlit*")