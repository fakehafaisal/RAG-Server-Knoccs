import streamlit as st
from src.search import RAGSearch

# Page config
st.set_page_config(
    page_title="Knoccs Knowledge Base",
    page_icon="💬",
    layout="centered"
)

# Initialize RAG search (only once)
@st.cache_resource
def load_rag():
    return RAGSearch()

rag_search = load_rag()

# Title
st.title("💬 Knoccs Knowledge Base Chatbot")
st.markdown("Ask me anything about the documents in our knowledge base!")

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
        with st.spinner("Searching knowledge base..."):
            response = rag_search.search_and_summarize(prompt, top_k=5)
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.info("This chatbot uses RAG (Retrieval Augmented Generation) to answer questions based on Knoccs documentation.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()