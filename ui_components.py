import streamlit as st
from config import api_key as config_api_key

def apply_custom_css():
    """Apply custom CSS styling to the application."""
    st.markdown("""
    <style>
        .main .block-container {padding-top: 2rem;}
        .stTextInput > div > div > input {min-height: 50px;}
        .chat-message {
            padding: 1rem; 
            border-radius: 0.5rem; 
            margin-bottom: 1rem; 
            display: flex;
            flex-direction: column;
            color: #000000;
        }
        .chat-message.user {
            background-color: #f0f2f6;
        }
        .chat-message.assistant {
            background-color: #e6f7ff;
        }
        .chat-header {
            font-size: 0.8rem; 
            color: #555; 
            margin-bottom: 0.5rem;
        }
        .chat-content {
            display: flex;
            color: #000000;
        }
        .chat-icon {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }
        .user-icon {
            background-color: #e1e1e1;
        }
        .assistant-icon {
            background-color: #3a86ff;
            color: white;
        }
        .message-text {
            flex-grow: 1;
            color: #000000;
        }
        .document-chunk {
            background-color: #fff9e6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 3px solid #ffcc00;
            margin-bottom: 0.5rem;
            color: #000000;
        }
        .relevance-score {
            float: right;
            font-size: 0.8rem;
            background-color: #ffcc00;
            color: #333;
            padding: 0.2rem 0.5rem;
            border-radius: 1rem;
        }
        .model-toggle {
            background-color: #f0f2f6;
            padding: 0.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .debug-info {
            font-size: 0.8rem;
            color: #666;
            margin-top: 0.5rem;
        }
        .highlight {
            background-color: #ffff00;
            padding: 0 0.2rem;
        }
        .sidebar .stButton button {
            width: 100%;
        }
        .comparison-container {
            display: flex;
            margin-top: 1rem;
            gap: 1rem;
        }
        .comparison-column {
            flex: 1;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
        }
        .comparison-title {
            font-weight: bold;
            margin-bottom: 0.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #dee2e6;
            color: #000000;
        }
        .retrieved-chunks-container {
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
        }
        .retrieved-chunks-title {
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: #000000;
        }
        .reset-button {
            margin-top: 0.5rem;
        }
        .reset-full {
            background-color: #ff6b6b !important;
            color: white !important;
        }
        .reset-chat {
            background-color: #4dabf7 !important;
            color: white !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #e6f7ff;
            border-radius: 4px 4px 0 0;
            border-right: 1px solid #e6f7ff;
            border-left: 1px solid #e6f7ff;
            border-top: 1px solid #e6f7ff;
        }
    </style>
    """, unsafe_allow_html=True)

def setup_sidebar():
    """Set up the sidebar with all controls and inputs."""
    st.sidebar.title("📚 Document RAG Assistant")
    st.sidebar.markdown("---")
    
    # Get OpenAI API key
    api_key = config_api_key
    if api_key is None:
        api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
    
    # Model selection
    st.sidebar.subheader("Model Settings")
    model_name = st.sidebar.selectbox(
        "Select OpenAI Model:",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"],
        help="Choose which OpenAI model to use for generating responses"
    )
    
    # Embedding model selection
    embedding_model = st.sidebar.selectbox(
        "Select Embedding Model:",
        ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
        help="Choose which OpenAI embedding model to use for document vectorization"
    )
    
    # Chunking settings
    st.sidebar.subheader("Chunking Settings")
    chunk_size = st.sidebar.slider(
        "Chunk Size (characters):", 
        min_value=100, 
        max_value=4000, 
        value=1000,
        help="Size of text chunks in characters"
    )
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap (%):", 
        min_value=0, 
        max_value=50, 
        value=10,
        help="Percentage of overlap between chunks"
    )
    
    # Retrieval settings
    st.sidebar.subheader("Retrieval Settings")
    top_k = st.sidebar.slider(
        "Number of chunks to retrieve:", 
        min_value=1, 
        max_value=10, 
        value=4,
        help="How many document chunks to retrieve for each query"
    )
    
    similarity_metric = st.sidebar.selectbox(
        "Similarity Metric:",
        ["Cosine Similarity", "Euclidean Distance", "Dot Product"],
        help="Method used to calculate similarity between query and document chunks"
    )
    
    temperature = st.sidebar.slider(
        "Temperature:", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.0, 
        step=0.1,
        help="Controls randomness in generation (0=deterministic, 1=creative)"
    )
    
    # Advanced Options Expander
    with st.sidebar.expander("Advanced Options"):
        show_debug_info = st.checkbox("Show Debug Information", value=False)
        
        # Add embedding workers control
        import multiprocessing
        max_cpu = multiprocessing.cpu_count()
        embedding_workers = st.slider(
            "Embedding Workers:", 
            min_value=1, 
            max_value=max_cpu, 
            value=min(4, max_cpu),
            help="Number of parallel workers for generating embeddings. Higher values may improve speed but increase API usage."
        )
    
    # Document upload
    st.sidebar.subheader("Upload Document")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file (PDF, DOCX, TXT, CSV):", 
        type=["pdf", "docx", "txt", "csv"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with ❤️ with LangChain and OpenAI")


    return {
        "api_key": api_key,
        "model_name": model_name,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "similarity_metric": similarity_metric,
        "temperature": temperature,
        "show_debug_info": show_debug_info,
        "embedding_workers": embedding_workers,
        "uploaded_file": uploaded_file
    }

def display_conversation_history():
    """Display the conversation history."""
    for message in st.session_state.conversation_history:
        role = message["role"]
        
        if role == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div class="chat-header">You</div>
                <div class="chat-content">
                    <div class="chat-icon user-icon">👤</div>
                    <div class="message-text">{message.get("content", "")}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Check if this is a comparison response
            if "comparison" in message and message["comparison"]:
                # Display both RAG and direct LLM responses side by side
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="chat-header">AI Assistant</div>
                    <div class="chat-content">
                        <div class="chat-icon assistant-icon">🤖</div>
                        <div class="message-text">
                            <div class="comparison-container">
                                <div class="comparison-column">
                                    <div class="comparison-title">RAG Response:</div>
                                    {message.get("rag_response", "")}
                                </div>
                                <div class="comparison-column">
                                    <div class="comparison-title">Direct LLM Response:</div>
                                    {message.get("direct_response", "")}
                                </div>
                            </div>
                            {message.get("debug_info", "")}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display retrieved chunks if available
                if "retrieved_chunks" in message:
                    st.markdown(f"""
                    <div class="retrieved-chunks-container">
                        <div class="retrieved-chunks-title">Retrieved Document Chunks:</div>
                    """, unsafe_allow_html=True)
                    
                    for i, (chunk, score) in enumerate(message["retrieved_chunks"]):
                        st.markdown(f"<div class='document-chunk'><span class='relevance-score'>Relevance: {score:.2f}</span>{chunk}</div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                # Regular assistant response
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="chat-header">AI Assistant</div>
                    <div class="chat-content">
                        <div class="chat-icon assistant-icon">🤖</div>
                        <div class="message-text">{message.get("content", "")}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display retrieved chunks if available
                if "retrieved_chunks" in message:
                    st.markdown(f"""
                    <div class="retrieved-chunks-container">
                        <div class="retrieved-chunks-title">Retrieved Document Chunks:</div>
                    """, unsafe_allow_html=True)
                    
                    for i, (chunk, score) in enumerate(message["retrieved_chunks"]):
                        st.markdown(f"<div class='document-chunk'><span class='relevance-score'>Relevance: {score:.2f}</span>{chunk}</div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

def display_document_info(filename, total_chars, num_chunks, chunk_size, chunk_overlap, embedding_model):
    """Display document information after processing."""
    doc_info_col1, doc_info_col2 = st.columns(2)
    with doc_info_col1:
        st.success(f"✅ Document processed successfully!")
        st.info(f"📄 **Document**: {filename}")
        st.info(f"📊 **Document Size**: {total_chars:,} characters")
    with doc_info_col2:
        st.info(f"🧩 **Chunks**: {num_chunks} chunks")
        st.info(f"⚙️ **Chunk Size**: {chunk_size} chars with {chunk_overlap}% overlap")
        st.info(f"🔢 **Embedding Model**: {embedding_model}")

def display_sample_chunks(chunks, max_samples=3):
    """Display sample chunks from the document."""
    with st.expander("View Sample Chunks"):
        for i, doc in enumerate(chunks[:max_samples]):
            st.markdown(f"**Chunk {i+1}**")
            st.markdown(f"<div class='document-chunk'>{doc.page_content}</div>", unsafe_allow_html=True)
            st.markdown("---")
        if len(chunks) > max_samples:
            st.write(f"... and {len(chunks) - max_samples} more chunks")

def display_onboarding():
    """Display onboarding information when no document is uploaded."""
    st.info("👈 Please upload a document in the sidebar to get started")
    
    st.markdown("""
    ### 📋 How to use this Document RAG Assistant:
    
    1. **Upload a Document** - Use the sidebar to upload a PDF, DOCX, TXT, or CSV file.
    2. **Adjust Settings** - Customize the model, embedding model, chunk size, and retrieval parameters.
    3. **Ask Questions** - Chat with your document to extract information.
    
    ### ✨ Key Features:
    
    - **Multiple File Formats** - Support for PDF, DOCX, TXT, and CSV files
    - **Model Selection** - Choose from various OpenAI models and embedding models
    - **Customizable Chunking** - Adjust chunk size and overlap
    - **Flexible Retrieval** - Select different similarity metrics and control the number of retrieved chunks
    - **Debug Information** - View behind-the-scenes processing details
    - **RAG vs Direct LLM Comparison** - Compare responses with and without document context
    - **Vector DB Visualization** - View your document chunks in embedding space
    - **Two Reset Options** - Reset everything or just the conversation history
    
    ### 📚 Example Questions to Ask:
    
    Once you upload a document, you can ask questions like:
    - "What is the main topic of this document?"
    - "Summarize the key points in section 2."
    - "What recommendations are made in the conclusion?"
    - "What data is presented about [specific topic]?"
    """)