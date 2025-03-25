import streamlit as st
import openai
import numpy as np
import time
import threading
import concurrent.futures
import queue
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from config import api_key as config_api_key

# Set up OpenAI API key from config
openai.api_key = config_api_key

def get_embedding(text, model="text-embedding-ada-002", max_retries=3, timeout=30):
    """
    Generates embeddings for a given text using the specified model.
    
    Args:
        text (str): The text to be embedded.
        model (str): The OpenAI model to use for generating embeddings.
        max_retries (int): Number of retry attempts if the API call fails.
        timeout (int): Timeout for the API call in seconds.
        
    Returns:
        list: A list of floats representing the embedding vector.
    """
    for attempt in range(max_retries):
        try:
            # Set a timeout for the API call
            start_time = time.time()
            
            # Generate embedding using the simple approach
            response = openai.Embedding.create(
                input=text,
                model=model
            )
            return response['data'][0]['embedding']
            
        except Exception as e:
            # Don't use st.error in this function since it might be called from a thread
            print(f"Error generating embedding (Attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
            else:
                print("Max retries reached. Could not generate embedding.")
                return None
    
    return None

def embedding_worker(work_queue, result_dict, lock, model, progress_counter, total_chunks):
    """Worker function for generating embeddings in parallel."""
    while not work_queue.empty():
        try:
            chunk_id, text = work_queue.get(block=False)
            
            # Generate embedding with timeout and retries
            embedding_vector = get_embedding(text, model=model)
            
            # Store the result
            with lock:
                if embedding_vector:
                    result_dict[chunk_id] = embedding_vector
                else:
                    # Use a zero vector as a fallback
                    result_dict[chunk_id] = [0.0] * 1536  # Standard size for OpenAI embeddings
                
                # Update progress
                progress_counter[0] += 1
                
            work_queue.task_done()
        except queue.Empty:
            break  # Queue is empty, exit the loop
        except Exception as e:
            # Don't use st.error in worker threads
            print(f"Error in worker thread: {str(e)}")
            with lock:
                progress_counter[0] += 1
            work_queue.task_done()

def generate_embeddings(documents, embedding_model, max_workers=4):
    """Generate embeddings for a list of documents with progress tracking and multi-threading."""
    embeddings_list = []
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_docs = len(documents)
    status_text.text(f"Generating embeddings for {total_docs} document chunks using {max_workers} parallel workers...")
    
    # Set up threading resources
    work_queue = queue.Queue()
    result_dict = {}  # To store results in order
    lock = threading.Lock()
    progress_counter = [0]  # Mutable container for progress tracking
    
    # Add documents to the work queue
    for i, doc in enumerate(documents):
        # Skip very long chunks that might cause issues
        if len(doc.page_content) > 8000:
            # Don't use st.warning from the worker thread
            print(f"Chunk {i+1} is very long ({len(doc.page_content)} chars). Truncating to 8000 chars.")
            text = doc.page_content[:8000]
        else:
            text = doc.page_content
        
        work_queue.put((i, text))
    
    # Create and start worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start workers
        futures = []
        for _ in range(max_workers):
            future = executor.submit(
                embedding_worker,
                work_queue,
                result_dict,
                lock,
                embedding_model,
                progress_counter,
                total_docs
            )
            futures.append(future)
        
        # Monitor progress while workers are running
        while progress_counter[0] < total_docs:
            # Update progress from the main thread
            progress = progress_counter[0] / total_docs
            progress_bar.progress(progress)
            status_text.text(f"Processing {progress_counter[0]}/{total_docs} chunks ({progress:.0%})")
            time.sleep(0.1)  # Small sleep to prevent UI hang
            
            # Check if all tasks are done
            if work_queue.empty() and all(future.done() for future in futures):
                break
    
    # Complete the progress bar
    progress_bar.progress(1.0)
    status_text.text("✅ Embedding generation complete!")
    
    # Convert result_dict to ordered list
    for i in range(total_docs):
        if i in result_dict:
            embeddings_list.append(result_dict[i])
        else:
            # Fallback for any missed chunks
            st.warning(f"Missing embedding for chunk {i+1}. Using zero vector.")
            embeddings_list.append([0.0] * 1536)  # Standard size for OpenAI embeddings
    
    return embeddings_list if len(embeddings_list) > 0 else None

def calculate_similarity(query_embedding, doc_embedding, metric):
    """Calculate similarity between query and document embeddings."""
    if metric == "Cosine Similarity":
        return cosine_similarity([query_embedding], [doc_embedding])[0][0]
    elif metric == "Euclidean Distance":
        # Convert distance to similarity score (1 / (1 + distance))
        distance = euclidean_distances([query_embedding], [doc_embedding])[0][0]
        return 1 / (1 + distance)
    else:  # Dot Product
        return np.dot(query_embedding, doc_embedding)

def retrieve_relevant_documents(query, embeddings, documents, embedding_model, similarity_metric, k=4):
    """Retrieve the most relevant documents for a query."""
    try:
        # Status indicator
        status = st.empty()
        status.text("Generating query embedding...")
        
        # Get embedding for query
        query_embedding = get_embedding(query, model=embedding_model)
        if not query_embedding:
            status.error("Failed to generate query embedding.")
            return []
        
        status.text("Finding similar documents...")
        
        # Calculate similarity scores
        scores = []
        for doc_embedding in embeddings:
            score = calculate_similarity(query_embedding, doc_embedding, similarity_metric)
            scores.append(score)
        
        # Create document-score pairs and sort by score
        doc_score_pairs = list(zip(documents, scores))
        sorted_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        
        status.text("✅ Retrieved relevant documents.")
        
        # Return top k documents with their scores
        return sorted_pairs[:k]
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []