"""
RAGveda - Streamlit Application

A comprehensive GraphRAG application for querying spiritual texts using semantic search.
Supports uploading CSV datasets and performing intelligent document retrieval.

Author: RAGveda Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import time
from typing import List, Dict, Any

from modules.main_processor import RAGProcessor
from modules.utils import create_sample_query_suggestions

# Configure Streamlit page
st.set_page_config(
    page_title="RAGveda - Spiritual Text Query System",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    
    .query-box {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B35;
        margin: 1rem 0;
    }
    
    .result-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #E9ECEF;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        z-index: 1;
        position: relative;
    }
    
    .summary-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        z-index: 10;
        position: relative;
    }
    
    .summary-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .summary-content {
        font-size: 1.2rem;
        line-height: 1.8;
        text-align: justify;
    }
    
    .similarity-score {
        background-color: #28A745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .dataset-info {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'embeddings_generated' not in st.session_state:
        st.session_state.embeddings_generated = False
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'last_query' not in st.session_state:
        st.session_state.last_query = None
    if 'last_results' not in st.session_state:
        st.session_state.last_results = None
    if 'show_details' not in st.session_state:
        st.session_state.show_details = False

def display_header():
    """Display the main application header."""
    st.markdown('<h1 class="main-header">🕉️ RAGveda</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #6C757D; margin-bottom: 2rem;">'
        'Intelligent Semantic Search for Spiritual Texts</p>',
        unsafe_allow_html=True
    )

def display_sidebar():
    """Display the sidebar with upload-only dataset option."""
    st.sidebar.markdown('<h2 class="sub-header">📚 Upload Dataset</h2>', unsafe_allow_html=True)
    return handle_file_upload()

def handle_existing_dataset():
    """Handle selection of existing datasets."""
    dataset_path = Path("c:/Users/parth/OneDrive/Desktop/one/RAGveda/dataset")
    
    if not dataset_path.exists():
        st.sidebar.error("Dataset directory not found!")
        return None, None, None
    
    # List available CSV files
    csv_files = list(dataset_path.glob("*.csv"))
    
    if not csv_files:
        st.sidebar.warning("No CSV files found in dataset directory!")
        return None, None, None
    
    # Create selection options
    file_options = {f.stem: str(f) for f in csv_files}
    
    selected_file = st.sidebar.selectbox(
        "Select Dataset:",
        options=list(file_options.keys()),
        help="Choose from available datasets"
    )
    
    if selected_file:
        file_path = file_options[selected_file]
        
        # Display dataset info and column selection
        try:
            df_preview = pd.read_csv(file_path, nrows=5)
            st.sidebar.markdown('<div class="dataset-info">', unsafe_allow_html=True)
            st.sidebar.write(f"**Dataset:** {selected_file}")
            st.sidebar.write(f"**Columns:** {', '.join(df_preview.columns)}")
            
            # Column selection for main content
            st.sidebar.markdown("**📝 Content Column Selection:**")
            content_column = st.sidebar.selectbox(
                "Select main content column:",
                options=df_preview.columns.tolist(),
                index=df_preview.columns.tolist().index('translation') if 'translation' in df_preview.columns else 0,
                help="Choose which column contains the main text content for search"
            )
            
            st.sidebar.write(f"**Preview:**")
            st.sidebar.dataframe(df_preview, use_container_width=True)
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
            return file_path, selected_file, content_column
            
        except Exception as e:
            st.sidebar.error(f"Error reading dataset: {str(e)}")
            return None, None, None
    
    return None, None, None

def handle_file_upload():
    """Handle file upload functionality."""
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Dataset",
        type=['csv'],
        help="Upload a CSV file with columns: chapter, verse, translation, question (optional)"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # Display file info and column selection
        try:
            df_preview = pd.read_csv(temp_path, nrows=5)
            st.sidebar.markdown('<div class="dataset-info">', unsafe_allow_html=True)
            st.sidebar.write(f"**File:** {uploaded_file.name}")
            st.sidebar.write(f"**Size:** {uploaded_file.size} bytes")
            st.sidebar.write(f"**Columns:** {', '.join(df_preview.columns)}")
            
            # Column selection for main content
            st.sidebar.markdown("**📝 Content Column Selection:**")
            content_column = st.sidebar.selectbox(
                "Select main content column:",
                options=df_preview.columns.tolist(),
                index=df_preview.columns.tolist().index('translation') if 'translation' in df_preview.columns else 0,
                help="Choose which column contains the main text content for search",
                key="upload_content_column"
            )
            
            st.sidebar.write(f"**Preview:**")
            st.sidebar.dataframe(df_preview, use_container_width=True)
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
            return temp_path, uploaded_file.name, content_column
            
        except Exception as e:
            st.sidebar.error(f"Error reading uploaded file: {str(e)}")
            return None, None, None
        
    return None, None, None

def process_dataset(file_path: str, dataset_name: str, content_column: str = "translation"):
    """Process the selected dataset."""
    try:
        # Initialize processor if not exists
        if st.session_state.processor is None:
            with st.spinner("Initializing RAG processor..."):
                st.session_state.processor = RAGProcessor()
        
        processor = st.session_state.processor
        
        # Load dataset
        with st.spinner("Loading dataset..."):
            df = processor.load_dataset(file_path)
            st.success(f"✅ Loaded {len(df)} records from {dataset_name}")
        
        # Process documents
        with st.spinner("Processing documents..."):
            processor.process_documents(df, content_column=content_column)
            st.session_state.documents_processed = True
            st.success(f"✅ Processed {len(processor.documents)} documents")
        
        # Generate embeddings
        with st.spinner("Generating embeddings... This may take a few minutes."):
            embeddings = processor.generate_embeddings()
            st.session_state.embeddings_generated = True
            st.success(f"✅ Generated embeddings with shape: {embeddings.shape}")
        
        st.session_state.current_dataset = dataset_name
        
        # Display dataset statistics
        stats = processor.get_stats()
        display_dataset_stats(stats)
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing dataset: {str(e)}")
        return False

def display_dataset_stats(stats: Dict[str, Any]):
    """Display dataset statistics."""
    st.markdown('<h3 class="sub-header">📊 Dataset Statistics</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents", stats.get('num_documents', 0))
    
    with col2:
        st.metric("Dataset Type", stats.get('dataset_type', 'Unknown'))
    
    with col3:
        st.metric("Embedding Dims", stats.get('embedding_dimensions', 0))
    
    with col4:
        st.metric("Rows", stats.get('num_rows', 0))

def display_query_interface():
    """Display the query interface."""
    st.markdown('<h3 class="sub-header">🔍 Query Interface</h3>', unsafe_allow_html=True)
    
    if not st.session_state.embeddings_generated:
        st.warning("⚠️ Please process a dataset first to enable querying.")
        return
    
    processor = st.session_state.processor
    
    # Query input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter your query:",
            placeholder="Ask anything about the spiritual texts...",
            help="Enter a question or topic you'd like to explore"
        )
    
    with col2:
        top_k = st.number_input(
            "Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of results to return"
        )
    
    # Sample queries
    sample_queries = processor.get_sample_queries()
    st.markdown("**💡 Sample Queries:**")
    
    cols = st.columns(len(sample_queries))
    for i, sample_query in enumerate(sample_queries):
        with cols[i % len(cols)]:
            if st.button(sample_query, key=f"sample_{i}", width='stretch'):
                query = sample_query
                st.rerun()
    
    # Process query
    if query and st.button("🔍 Search", type="primary", width='stretch'):
        # Reset details on new search
        st.session_state.show_details = False
        process_query(query, top_k)
        # Persist and re-render results after search
        st.rerun()

    # If we have previous results, show them to survive reruns (e.g., checkbox toggles)
    if st.session_state.last_results:
        display_query_results(st.session_state.last_query, st.session_state.last_results)

def process_query(query: str, top_k: int):
    """Process and display query results."""
    try:
        processor = st.session_state.processor
        
        with st.spinner("Searching for relevant content..."):
            results = processor.query(query, top_k)
        
        # Persist latest results and query for subsequent reruns
        st.session_state.last_query = query
        st.session_state.last_results = results

        # Add to query history
        st.session_state.query_history.append({
            'query': query,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results_count': len(results)
        })
        
        # Display results immediately
        display_query_results(query, results)
        
    except Exception as e:
        st.error(f"❌ Error processing query: {str(e)}")

def display_query_results(query: str, results: List[Dict[str, Any]]):
    """Display query results in a formatted manner."""
    st.markdown('<h3 class="sub-header">📋 Search Results</h3>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="query-box"><strong>Query:</strong> "{query}"</div>', unsafe_allow_html=True)
    
    if not results:
        st.warning("No results found for your query.")
        return
    
    # Generate and display summary
    summary = st.session_state.processor.get_summary(query, results)
    st.markdown(f"""
    <div class="summary-card">
        <div class="summary-title">📝 Answer</div>
        <div class="summary-content">{summary}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Option to show detailed results (state persists via key)
    st.checkbox(
        "🔍 Show detailed results",
        help="Toggle to view individual search results with full content",
        key="show_details"
    )
    show_details = st.session_state.get("show_details", False)
    
    if show_details:
        st.write(f"Found **{len(results)}** relevant results:")
        
        for i, result in enumerate(results):
            similarity_score = result['similarity_score']
            content = result['content']
            metadata = result.get('metadata', {})
            
            # Create result card
            st.markdown(f"""
            <div class="result-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h4 style="margin: 0; color: #2E86AB;">Result #{i+1}</h4>
                    <span class="similarity-score">Similarity: {similarity_score:.3f}</span>
                </div>
                <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1rem;">{content}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display metadata if available
            if metadata:
                with st.expander(f"📝 Metadata for Result #{i+1}"):
                    st.json(metadata)
    else:
        st.info("💡 Check the box above to view detailed individual results if needed.")

def display_query_history():
    """Display query history in sidebar."""
    if st.session_state.query_history:
        st.sidebar.markdown('<h3 class="sub-header">📜 Query History</h3>', unsafe_allow_html=True)
        
        for i, query_info in enumerate(reversed(st.session_state.query_history[-5:])):  # Show last 5
            with st.sidebar.expander(f"Query {len(st.session_state.query_history) - i}"):
                st.write(f"**Query:** {query_info['query'][:50]}...")
                st.write(f"**Time:** {query_info['timestamp']}")
                st.write(f"**Results:** {query_info['results_count']}")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Display sidebar
    file_path, dataset_name, content_column = display_sidebar()
    
    # Display query history
    display_query_history()
    
    # Main content area
    if file_path and dataset_name:
        # Check if we need to process a new dataset
        if (st.session_state.current_dataset != dataset_name or 
            not st.session_state.embeddings_generated):
            
            st.markdown('<h3 class="sub-header">⚙️ Processing Dataset</h3>', unsafe_allow_html=True)
            
            if st.button("🚀 Process Dataset", type="primary", width='stretch'):
                success = process_dataset(file_path, dataset_name, content_column)
                if success:
                    st.rerun()
        
        # Display query interface if dataset is processed
        if st.session_state.embeddings_generated:
            display_query_interface()
    
    else:
        # Welcome message
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background-color: #F8F9FA; border-radius: 15px; margin: 2rem 0;">
            <h2 style="color: #2E86AB; margin-bottom: 1rem;">Welcome to RAGveda! 🙏</h2>
            <p style="font-size: 1.2rem; color: #6C757D; margin-bottom: 2rem;">
                Your intelligent companion for exploring spiritual texts through semantic search.
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 300px;">
                    <h4 style="color: #FF6B35; margin-bottom: 1rem;">📚 Upload Dataset</h4>
                    <p>Upload a CSV matching your dataset format, then choose the content column.</p>
                </div>
                <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 300px;">
                    <h4 style="color: #FF6B35; margin-bottom: 1rem;">🔍 Semantic Search</h4>
                    <p>Ask questions and find relevant passages using AI-powered search.</p>
                </div>
                <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 300px;">
                    <h4 style="color: #FF6B35; margin-bottom: 1rem;">🧠 Intelligent Retrieval</h4>
                    <p>Get contextually relevant answers with similarity scoring.</p>
                </div>
            </div>
            <p style="margin-top: 2rem; color: #6C757D;">
                👈 Start by uploading a CSV from the sidebar
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #6C757D; font-size: 0.9rem;">'
        'RAGveda - Bridging Ancient Wisdom with Modern AI | Built with ❤️ using Streamlit</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
