import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader, 
    CSVLoader
)
import traceback
import time
import multiprocessing

from embeddings import generate_embeddings

def save_uploaded_file(uploaded_file, temp_dir):
    """Save an uploaded file to a temporary directory."""
    try:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        st.success(f"✅ File saved: {uploaded_file.name}")
        return temp_path
    except Exception as e:
        st.error(f"Error saving uploaded file: {str(e)}")
        return None

def load_document(file_path):
    """Load a document based on its file extension."""
    try:
        # Display loading message
        loading_msg = st.empty()
        loading_msg.text(f"Loading document: {os.path.basename(file_path)}...")
        
        # Start timing
        start_time = time.time()
        
        if file_path.endswith('.pdf'):
            loading_msg.text("Loading PDF document...")
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loading_msg.text("Loading DOCX document...")
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith('.txt'):
            loading_msg.text("Loading TXT document...")
            loader = TextLoader(file_path)
        elif file_path.endswith('.csv'):
            loading_msg.text("Loading CSV document...")
            loader = CSVLoader(file_path)
        else:
            st.error(f"Unsupported file format: {file_path}")
            return None
        
        # Load the document
        documents = loader.load()
        loading_time = time.time() - start_time
        
        # Store timing data
        st.session_state.timing_data['Document Loading'] = loading_time
        
        loading_msg.success(f"✅ Document loaded in {loading_time:.2f}s")
        
        return documents
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        st.error(traceback.format_exc())
        return None

def chunk_document(documents, chunk_size, chunk_overlap_percentage):
    """Split documents into chunks based on size and overlap."""
    try:
        if not documents:
            return []
        
        # Status indicator    
        status = st.empty()
        status.text("Chunking document...")
            
        # Start timing
        start_time = time.time()
        
        overlap_chars = int(chunk_size * (chunk_overlap_percentage / 100))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_chars,
            length_function=len,
        )
        
        # Split the documents
        split_docs = text_splitter.split_documents(documents)
        chunking_time = time.time() - start_time
        
        # Store timing data
        st.session_state.timing_data['Document Chunking'] = chunking_time
        
        status.success(f"✅ Document chunked into {len(split_docs)} chunks in {chunking_time:.2f}s")
        
        return split_docs
    except Exception as e:
        st.error(f"Error chunking document: {str(e)}")
        st.error(traceback.format_exc())
        return []

def process_document(uploaded_file, temp_dir, chunk_size, chunk_overlap, embedding_model, max_workers=4):
    """Process an uploaded document: save, load, chunk, and generate embeddings."""
    with st.status("Processing document...") as status:
        try:
            # Save the uploaded file to a temporary directory
            status.update(label="Saving uploaded file...")
            temp_path = save_uploaded_file(uploaded_file, temp_dir)
            if not temp_path:
                status.update(label="Failed to save file", state="error")
                return None, None
            
            # Load the document
            status.update(label="Loading document...")
            documents = load_document(temp_path)
            if not documents:
                status.update(label="Failed to load document", state="error")
                return None, None
                
            # Calculate document statistics
            total_chars = sum(len(doc.page_content) for doc in documents)
            status.update(label=f"Document loaded: {total_chars:,} characters")
            
            # Split documents into chunks
            status.update(label="Chunking document...")
            split_docs = chunk_document(documents, chunk_size, chunk_overlap)
            if not split_docs:
                status.update(label="Failed to chunk document", state="error")
                return None, None
            
            status.update(label=f"Document chunked into {len(split_docs)} chunks")
                
            # Calculate optimal number of workers for embedding generation
            num_workers = min(
                len(split_docs),  # Don't use more workers than chunks
                multiprocessing.cpu_count(),  # Don't exceed CPU count
                max_workers  # Use the provided max_workers
            )
            
            # Generate embeddings for all document chunks with multi-threading
            status.update(label=f"Generating embeddings using {num_workers} parallel workers...")
            
            # Start timing for embeddings
            embedding_start_time = time.time()
            
            embeddings_list = generate_embeddings(split_docs, embedding_model, max_workers=num_workers)
            
            # End timing for embeddings
            embedding_time = time.time() - embedding_start_time
            
            # Store timing data
            st.session_state.timing_data['Embedding Generation'] = embedding_time
            
            # Store current embedding model for metrics
            st.session_state.current_embedding_model = embedding_model
            
            if not embeddings_list:
                status.update(label="Failed to generate embeddings", state="error")
                return None, None
            
            status.update(label="Document processing complete!", state="complete")
            return split_docs, embeddings_list
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            st.error(traceback.format_exc())
            status.update(label=f"Error: {str(e)}", state="error")
            return None, None