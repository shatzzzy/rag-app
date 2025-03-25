import os
import shutil
import uuid
import streamlit as st

def create_temp_directory():
    """Create a temporary directory for file storage."""
    temp_dir = f"./temp_{str(uuid.uuid4())}"
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def clean_temp_directory(dir_path):
    """Remove a temporary directory and its contents."""
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            return True
        except Exception as e:
            st.error(f"Error removing temporary directory: {str(e)}")
            return False
    return True

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    if 'documents' not in st.session_state:
        st.session_state.documents = None
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = create_temp_directory()
    if 'submit_query' not in st.session_state:
        st.session_state.submit_query = False
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "gpt-3.5-turbo"
    if 'current_embedding_model' not in st.session_state:
        st.session_state.current_embedding_model = "text-embedding-ada-002"
    if 'document_info' not in st.session_state:
        st.session_state.document_info = None
    if 'show_vector_db' not in st.session_state:
        st.session_state.show_vector_db = False
    if 'pca_plot' not in st.session_state:
        st.session_state.pca_plot = None
    if 'embedding_stats' not in st.session_state:
        st.session_state.embedding_stats = None
    if 'selected_embedding' not in st.session_state:
        st.session_state.selected_embedding = None

def reset_full_session():
    """Reset the entire session state and clean up resources."""
    # Clean up temp directory
    clean_temp_directory(st.session_state.temp_dir)
    
    # Reset session state
    st.session_state.conversation_history = []
    st.session_state.document_processed = False
    st.session_state.documents = None
    st.session_state.embeddings = None
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.temp_dir = create_temp_directory()
    st.session_state.submit_query = False
    st.session_state.current_query = ""
    st.session_state.document_info = None
    st.session_state.current_model = "gpt-3.5-turbo"
    st.session_state.current_embedding_model = "text-embedding-ada-002"
    st.session_state.show_vector_db = False
    st.session_state.pca_plot = None
    st.session_state.embedding_stats = None
    st.session_state.selected_embedding = None
    
    st.success("Session fully reset. Please upload a new document to start.")

def reset_conversation_only():
    """Reset only the conversation history, keeping document and embeddings."""
    # Keep documents and embeddings, but reset conversation
    st.session_state.conversation_history = []
    st.session_state.submit_query = False
    st.session_state.current_query = ""
    st.session_state.show_vector_db = False
    st.session_state.pca_plot = None
    st.session_state.selected_embedding = None
    
    st.success("Conversation reset. Document and embeddings retained.")

def handle_query_submit():
    """Handle query submission from text input."""
    st.session_state.submit_query = True
    st.session_state.current_query = st.session_state.query_input