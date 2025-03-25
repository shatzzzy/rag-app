import streamlit as st
import openai
import time
from utils import initialize_session_state, reset_full_session, reset_conversation_only, handle_query_submit
from ui_components import (
    apply_custom_css, setup_sidebar, display_conversation_history, 
    display_document_info, display_sample_chunks, display_onboarding
)
from document_processor import process_document
from embeddings import retrieve_relevant_documents
from simple_vector_db_view import display_vector_db_view
from config import api_key as config_api_key

# Set page configuration
st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
initialize_session_state()

# Apply custom CSS
apply_custom_css()

# Set up sidebar and get inputs
sidebar_inputs = setup_sidebar()

# Use API key from config or sidebar
api_key = config_api_key if config_api_key else sidebar_inputs["api_key"]
if api_key:
    openai.api_key = api_key
else:
    st.warning("Please enter your OpenAI API key in the sidebar.")

# Add reset buttons to sidebar with clear labels
st.sidebar.markdown("### Reset Options")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🗑️ Reset Everything", help="Reset the entire session, removes document and all data"):
        reset_full_session()
        st.rerun()
with col2:
    if st.button("🧹 Reset Chat Only", help="Reset only the conversation, keeps document and embeddings"):
        reset_conversation_only()
        st.rerun()

# Main content area
st.title("Document RAG Assistant")

# Process document if uploaded and not already processed
if sidebar_inputs["uploaded_file"] is not None and not st.session_state.document_processed:
    split_docs, embeddings_list = process_document(
        sidebar_inputs["uploaded_file"],
        st.session_state.temp_dir,
        sidebar_inputs["chunk_size"],
        sidebar_inputs["chunk_overlap"],
        sidebar_inputs["embedding_model"],
        max_workers=sidebar_inputs.get("embedding_workers", 4)
    )
    
    if split_docs and embeddings_list:
        # Store in session state
        st.session_state.documents = split_docs
        st.session_state.embeddings = embeddings_list
        st.session_state.document_processed = True
        
        # Store the current model for reference
        st.session_state.current_model = sidebar_inputs["model_name"]
        st.session_state.current_embedding_model = sidebar_inputs["embedding_model"]
        
        # Store document info for later reference
        st.session_state.document_info = {
            "filename": sidebar_inputs["uploaded_file"].name,
            "total_chars": sum(len(doc.page_content) for doc in split_docs),
            "num_chunks": len(split_docs),
            "chunk_size": sidebar_inputs["chunk_size"],
            "chunk_overlap": sidebar_inputs["chunk_overlap"],
            "embedding_model": sidebar_inputs["embedding_model"]
        }
        
        # Display document info
        display_document_info(
            sidebar_inputs["uploaded_file"].name,
            sum(len(doc.page_content) for doc in split_docs),
            len(split_docs),
            sidebar_inputs["chunk_size"],
            sidebar_inputs["chunk_overlap"],
            sidebar_inputs["embedding_model"]
        )
        
        # Display sample chunks
        display_sample_chunks(split_docs)

# Add a button to view vector database visualization
if st.session_state.document_processed:
    st.markdown("---")
    st.markdown("### Vector Database Visualization")
    st.write("View the document chunks in the embedding space and analyze the raw embedding values.")
    
    # Add tab for Vector DB visualization
    if st.button("🔍 View Vector DB Visualization"):
        st.session_state.show_vector_db = True
    
    # Display Vector DB visualization if button was clicked
    if st.session_state.get("show_vector_db", False):
        display_vector_db_view()

# Conversation Area
if st.session_state.document_processed:
    st.markdown("### Chat with your document")
    
    # Display conversation history
    display_conversation_history()
    
    # Query input
    query_input = st.text_input(
        "Ask a question about your document:", 
        key="query_input", 
        placeholder="Enter your question here...", 
        on_change=handle_query_submit
    )
    
    # Add option to compare RAG vs direct LLM
    compare_responses = st.checkbox("Compare RAG with direct LLM response", value=True)
    
    # Process the query
    if st.session_state.submit_query and st.session_state.current_query:
        query = st.session_state.current_query
        
        # Reset submit flag
        st.session_state.submit_query = False
        
        # Add user query to conversation history
        st.session_state.conversation_history.append({"role": "user", "content": query})
        
        with st.spinner('Thinking...'):
            # Start timing for query processing
            query_start_time = time.time()
            
            try:
                # Retrieve relevant documents
                retrieved_docs_with_scores = retrieve_relevant_documents(
                    query,
                    st.session_state.embeddings,
                    st.session_state.documents,
                    sidebar_inputs["embedding_model"],
                    sidebar_inputs["similarity_metric"],
                    k=sidebar_inputs["top_k"]
                )
                
                # Record query processing time
                query_processing_time = time.time() - query_start_time
                
                # Start timing for response generation
                response_start_time = time.time()
                
                # Check if we got any documents back
                if not retrieved_docs_with_scores:
                    # Handle case where no relevant documents were found
                    response = "I couldn't find any relevant information in the document to answer your question. Please try rephrasing your query or ask about a different topic covered in the document."
                    
                    # Add response to conversation history without retrieved chunks
                    st.session_state.conversation_history.append({
                        "role": "assistant", 
                        "content": response,
                        "retrieved_chunks": []
                    })
                    
                    # Force refresh
                    st.rerun()
                
                # Prepare context for the model
                context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, (doc, _) in enumerate(retrieved_docs_with_scores)])
                
                # Prepare retrieved chunks for display
                retrieved_chunks = [(doc.page_content, score) for doc, score in retrieved_docs_with_scores]
                
                # Create prompt for the RAG response
                rag_prompt = f"""You are an AI assistant answering questions based on the provided document chunks. 
                
                Answer the following question using ONLY the information from the provided document chunks. 
                If the information is not available in the chunks, say "I don't have enough information to answer this question based on the document."
                Don't use prior knowledge. Be concise and informative.
                
                Document chunks:
                {context}
                
                Question: {query}
                
                Answer:"""
                
                # Generate RAG response - USING ORIGINAL OPENAI API FORMAT
                rag_completion = openai.ChatCompletion.create(
                    model=sidebar_inputs["model_name"],
                    messages=[
                        {"role": "system", "content": "You are a helpful document assistant."},
                        {"role": "user", "content": rag_prompt}
                    ],
                    temperature=sidebar_inputs["temperature"],
                    max_tokens=1000
                )
                
                rag_response = rag_completion['choices'][0]['message']['content']
                
                # If comparison is enabled, also generate a direct LLM response
                if compare_responses:
                    # Create prompt for direct LLM response (without document context)
                    direct_prompt = f"""You are an AI assistant answering questions based on your knowledge.
                    
                    Answer the following question using your own knowledge. Be concise and informative.
                    
                    Question: {query}
                    
                    Answer:"""
                    
                    # Generate direct LLM response - USING ORIGINAL OPENAI API FORMAT
                    direct_completion = openai.ChatCompletion.create(
                        model=sidebar_inputs["model_name"],
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": direct_prompt}
                        ],
                        temperature=sidebar_inputs["temperature"],
                        max_tokens=1000
                    )
                    
                    direct_response = direct_completion['choices'][0]['message']['content']
                
                # Record response generation time
                response_generation_time = time.time() - response_start_time
                
                # Format the debug info
                debug_info = f"""
                <div class='debug-info'>
                    <p>🔍 Query processed in {query_processing_time:.2f} seconds</p>
                    <p>⚡ Response generated in {response_generation_time:.2f} seconds</p>
                    <p>🧠 Model used: {sidebar_inputs["model_name"]}</p>
                    <p>🔤 Embedding model used: {sidebar_inputs["embedding_model"]}</p>
                    <p>📊 Top-{sidebar_inputs["top_k"]} chunks retrieved using {sidebar_inputs["similarity_metric"]}</p>
                    <p>🔢 Chunk size: {sidebar_inputs["chunk_size"]} chars with {sidebar_inputs["chunk_overlap"]}% overlap</p>
                    <p>🌡️ Temperature: {sidebar_inputs["temperature"]}</p>
                </div>
                """
                
                # Add response to conversation history
                if compare_responses:
                    # Add comparison response
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "comparison": True,
                        "rag_response": rag_response,
                        "direct_response": direct_response,
                        "retrieved_chunks": retrieved_chunks,
                        "debug_info": debug_info if sidebar_inputs.get("show_debug_info", False) else ""
                    })
                else:
                    # Add regular RAG response
                    st.session_state.conversation_history.append({
                        "role": "assistant", 
                        "content": rag_response + (debug_info if sidebar_inputs.get("show_debug_info", False) else ""),
                        "retrieved_chunks": retrieved_chunks
                    })
                
                # Force refresh
                st.rerun()
                
            except Exception as e:
                # Handle errors gracefully
                error_message = f"An error occurred: {str(e)}"
                st.session_state.conversation_history.append({
                    "role": "assistant", 
                    "content": error_message,
                    "retrieved_chunks": []
                })
                
                import traceback
                st.error(traceback.format_exc())
                
                # Force refresh
                st.rerun()

# Display onboarding information if no document is processed
if not st.session_state.document_processed:
    display_onboarding()