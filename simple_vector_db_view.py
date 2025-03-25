import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gc
import io
import base64

def create_pca_plot(embeddings, limit=None):
    """Create PCA plot as a static image to avoid memory leaks"""
    try:
        # Limit the number of points if needed
        if limit and len(embeddings) > limit:
            embeddings_subset = embeddings[:limit]
        else:
            embeddings_subset = embeddings
            
        # Apply PCA
        reducer = PCA(n_components=2, random_state=42)
        reduced_data = reducer.fit_transform(embeddings_subset)
        explained_variance = reducer.explained_variance_ratio_.sum()
        
        # Create a new figure (explicitly with plt.Figure to avoid leaks)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Create scatter plot
        scatter = ax.scatter(
            reduced_data[:, 0], 
            reduced_data[:, 1],
            c=np.arange(len(reduced_data)),
            cmap='viridis',
            alpha=0.7,
            s=100
        )
        
        # Add chunk numbers as labels
        for i, (x, y) in enumerate(reduced_data):
            ax.text(x, y, str(i), fontsize=8, ha='center', va='center', color='black')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Chunk Index')
        
        # Set title and labels
        ax.set_title(f"Document Chunks in Embedding Space (Explained variance: {explained_variance:.2%})")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        
        # Remove gridlines
        ax.grid(False)
        
        # Convert plot to PNG image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)  # Explicitly close the figure
        
        # Clear the matplotlib memory
        plt.clf()
        plt.close('all')
        
        # Convert to base64 for displaying
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        
        # Clean up
        buf.close()
        gc.collect()  # Force garbage collection
        
        return {
            'image': img_str,
            'explained_variance': explained_variance,
            'points_shown': len(reduced_data),
            'total_points': len(embeddings)
        }
    except Exception as e:
        st.error(f"Error creating PCA plot: {e}")
        return None

def create_histogram_image(values):
    """Create histogram as a static image to avoid memory leaks"""
    try:
        # Create a new figure
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        
        # Create histogram
        ax.hist(values, bins=30, alpha=0.7, color='blue')
        mean_val = np.mean(values)
        median_val = np.median(values)
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_val:.4f}')
        ax.set_title(f"Distribution of Embedding Values")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend()
        
        # Convert plot to PNG image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)  # Explicitly close the figure
        
        # Clear the matplotlib memory
        plt.clf()
        plt.close('all')
        
        # Convert to base64 for displaying
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        
        # Clean up
        buf.close()
        gc.collect()  # Force garbage collection
        
        return img_str
    except Exception as e:
        st.error(f"Error creating histogram: {e}")
        return None

def display_vector_db_view():
    """Display a visualization of the vector database (embeddings)"""
    try:
        if "documents" not in st.session_state or "embeddings" not in st.session_state:
            st.warning("Please upload and process a document first to view embeddings.")
            return
            
        if not st.session_state.documents or not st.session_state.embeddings:
            st.warning("No document or embeddings found. Please process a document first.")
            return
        
        # Add custom CSS for black text
        st.markdown("""
        <style>
            /* Make all text in Vector DB View black */
            div[data-testid="stVerticalBlock"] h3,
            div[data-testid="stVerticalBlock"] h4,
            div[data-testid="stVerticalBlock"] p,
            div[data-testid="stVerticalBlock"] li,
            div[data-testid="stVerticalBlock"] span,
            div[data-testid="stVerticalBlock"] div,
            div[data-testid="stVerticalBlock"] .streamlit-expanderHeader {
                color: black !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔍 Vector Database Visualization")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["2D Projection", "Raw Embeddings & Stats"])
        
        with tab1:
            st.markdown("#### PCA Visualization of Document Chunks")
            
            # Add a generate button to avoid automatic recomputation
            if st.button("Generate PCA Visualization", key="generate_pca"):
                with st.spinner("Generating PCA visualization..."):
                    # Get embeddings
                    embeddings = np.array(st.session_state.embeddings)
                    
                    # Create plot with limit
                    plot_data = create_pca_plot(embeddings, limit=100)
                    
                    if plot_data:
                        # Store results in session state
                        st.session_state.pca_plot = plot_data
            
            # Display the plot if it exists
            if "pca_plot" in st.session_state and st.session_state.pca_plot:
                plot_data = st.session_state.pca_plot
                
                # Display the image using HTML
                st.markdown(f"""
                <div style="text-align: center;">
                    <img src="data:image/png;base64,{plot_data['image']}" style="max-width: 100%;">
                </div>
                """, unsafe_allow_html=True)
                
                # Display additional information
                if plot_data['points_shown'] < plot_data['total_points']:
                    st.info(f"Showing first {plot_data['points_shown']} of {plot_data['total_points']} chunks for better performance.")
                
                st.markdown(f"**Explained variance:** {plot_data['explained_variance']:.2%}")
                
                st.markdown("""
                📌 **How to interpret this visualization:**
                - Each point represents a document chunk in the embedding space
                - Points close together are semantically similar
                - The numbers indicate the chunk indices (order in the document)
                - Connected chunks in the original document may appear in different locations if they cover different topics
                """)
        
        with tab2:
            st.markdown("#### Embedding Statistics and Raw Values")
            
            try:
                # Calculate statistics only once
                if "embedding_stats" not in st.session_state:
                    with st.spinner("Calculating embedding statistics..."):
                        embeddings = np.array(st.session_state.embeddings)
                        all_values = embeddings.flatten()
                        
                        st.session_state.embedding_stats = {
                            "mean": np.mean(all_values),
                            "median": np.median(all_values),
                            "std": np.std(all_values),
                            "min": np.min(all_values),
                            "max": np.max(all_values),
                            "dimensions": embeddings.shape[1],
                            "total_embeddings": embeddings.shape[0],
                            "total_values": len(all_values)
                        }
                
                # Display statistics
                stats = st.session_state.embedding_stats
                stats_df = pd.DataFrame({
                    "Statistic": ["Mean", "Median", "Standard Deviation", "Min", "Max", "Dimensions", "Total Embeddings", "Total Values"],
                    "Value": [
                        f"{stats['mean']:.6f}",
                        f"{stats['median']:.6f}",
                        f"{stats['std']:.6f}",
                        f"{stats['min']:.6f}",
                        f"{stats['max']:.6f}",
                        f"{stats['dimensions']}",
                        f"{stats['total_embeddings']}",
                        f"{stats['total_values']:,}"
                    ]
                })
                
                st.dataframe(stats_df)
                
                # Raw embeddings with safe selection
                st.markdown("#### View Raw Embedding Values")
                
                # Safely get the format function
                def format_chunk(idx):
                    try:
                        content = st.session_state.documents[idx].page_content
                        preview = content[:30].replace('\n', ' ')
                        return f"Chunk {idx}: {preview}..."
                    except:
                        return f"Chunk {idx}"
                
                # Create a dropdown to select a chunk
                chunk_options = list(range(len(st.session_state.documents)))
                chunk_index = st.selectbox(
                    "Select chunk to view raw embedding values:",
                    chunk_options,
                    format_func=format_chunk,
                    key="chunk_selector" 
                )
                
                # Add a button to generate the embedding view
                if st.button("View Selected Embedding", key="view_embedding"):
                    if chunk_index is not None and chunk_index < len(st.session_state.embeddings):
                        with st.spinner("Loading embedding values..."):
                            # Get the selected embedding
                            embedding = st.session_state.embeddings[chunk_index]
                            
                            # Create a DataFrame for the raw values
                            raw_df = pd.DataFrame({
                                "Dimension": range(len(embedding)),
                                "Value": embedding
                            })
                            
                            # Store in session state
                            st.session_state.selected_embedding = {
                                "index": chunk_index,
                                "data": raw_df,
                                "histogram": create_histogram_image(embedding)
                            }
                
                # Display the embedding data if it exists
                if "selected_embedding" in st.session_state:
                    sel_data = st.session_state.selected_embedding
                    
                    st.markdown(f"**Raw embedding values for Chunk {sel_data['index']}:**")
                    st.dataframe(sel_data['data'], height=300)
                    
                    if sel_data['histogram']:
                        st.markdown("#### Distribution of Values in This Embedding")
                        
                        # Display the histogram using HTML
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <img src="data:image/png;base64,{sel_data['histogram']}" style="max-width: 100%;">
                        </div>
                        """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error in embedding statistics view: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    except Exception as e:
        st.error(f"Error in vector database view: {str(e)}")
        import traceback
        st.error(traceback.format_exc())