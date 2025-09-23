"""
RAGVeda Main Application.
Production-level Streamlit application for Neo4j-based RAG system.
"""

import streamlit as st
import pandas as pd
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

# Import modules
from modules.config import Config
from modules.neo4j_manager import Neo4jManager
from modules.document_processor import DocumentProcessor
from modules.retrieval import Retrieval
from modules.llm_chain import LLMChain
from modules.llm_chain import LLMChain as _LLMChain
from modules.ui_components import UIComponents


class RAGVedaApp:
    """Main application class for RAGVeda."""
    
    def __init__(self):
        """Initialize the application."""
        self.ui = UIComponents()
        self.processor = DocumentProcessor()
        self.neo4j_manager = None
        self.llm_chain = None
        self.retrieval = Retrieval()
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'current_file' not in st.session_state:
            st.session_state.current_file = None
        if 'embeddings_created' not in st.session_state:
            st.session_state.embeddings_created = False
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'top_k' not in st.session_state:
            st.session_state.top_k = Config.DEFAULT_TOP_K
        if 'neo4j_manager' not in st.session_state:
            st.session_state.neo4j_manager = None
        if 'llm_chain' not in st.session_state:
            st.session_state.llm_chain = None
    
    def validate_configuration(self) -> bool:
        """
        Validate application configuration.
        
        Returns:
            True if configuration is valid
        """
        errors = Config.validate()
        
        if errors:
            st.sidebar.error("Configuration Issues:")
            for error in errors:
                st.sidebar.error(f"• {error}")
            
            st.sidebar.markdown("""
            ### Setup Instructions:
            1. Create a `.env` file in the project root
            2. Add the following variables:
            ```
            NEO4J_URL=neo4j+s://your-instance.databases.neo4j.io
            NEO4J_USERNAME=neo4j
            NEO4J_PASSWORD=your-password
            GROQ_API_KEY=your-groq-api-key
            ```
            """)
            return False
        
        return True
    
    def handle_file_upload(self) -> Optional[Dict[str, Any]]:
        """
        Handle file upload and preview.
        
        Returns:
            Dictionary with file info or None
        """
        uploaded_file = self.ui.render_file_upload_section()
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            
            try:
                # Read and display preview
                df_preview = pd.read_csv(temp_path, nrows=5)
                df_full = pd.read_csv(temp_path)
                
                # Display preview
                self.ui.render_dataset_preview(df_full, uploaded_file.name, uploaded_file.size)
                
                # Column selection
                content_column = self.ui.render_column_selector(
                    df_preview.columns.tolist(),
                    'translation' if 'translation' in df_preview.columns else None
                )
                
                # Display data preview
                st.sidebar.markdown("### 📊 Data Preview")
                st.sidebar.dataframe(df_preview, width='stretch')
                
                # Process button
                if st.sidebar.button("🚀 Process & Create Embeddings", type="primary", width='stretch'):
                    return {
                        'path': temp_path,
                        'filename': uploaded_file.name,
                        'content_column': content_column
                    }
                    
            except Exception as e:
                st.sidebar.error(f"Error reading file: {str(e)}")
        
        return None
    
    def process_dataset(self, file_info: Dict[str, Any]) -> bool:
        """
        Process uploaded dataset and create embeddings.
        
        Args:
            file_info: Dictionary with file information
            
        Returns:
            True if successful
        """
        try:
            # Initialize Neo4j manager if needed
            if st.session_state.neo4j_manager is None:
                st.session_state.neo4j_manager = Neo4jManager()
            
            # Process CSV to chunks
            with st.spinner("📖 Processing document..."):
                final_docs, df = self.processor.process_csv_to_chunks(
                    file_info['path'],
                    file_info['content_column'],
                    file_info['filename']
                )
                self.ui.render_status_message(f"Created {len(final_docs)} document chunks", "success")
            
            # Create vector store
            with st.spinner("🔗 Creating embeddings in Neo4j..."):
                index_name = Config.get_index_name(file_info['filename'])
                st.session_state.neo4j_manager.create_vector_store(final_docs, index_name)
                self.ui.render_status_message(f"Created vector index: {index_name}", "success")
            
            # Create file relationships
            with st.spinner("🔗 Creating file relationships..."):
                st.session_state.neo4j_manager.create_file_relationships(file_info['filename'])
                self.ui.render_status_message(f"Created relationships for {file_info['filename']}", "success")
            
            # Update session state
            st.session_state.current_file = file_info['filename']
            st.session_state.embeddings_created = True
            
            # Clear chat history when switching files
            st.session_state.chat_history = []
            
            return True
            
        except Exception as e:
            self.ui.render_status_message(f"Error processing dataset: {str(e)}", "error")
            return False
    
    def handle_chat_interaction(self):
        """Handle chat interaction with the user."""
        if not st.session_state.embeddings_created:
            self.ui.render_status_message(
                "Please upload and process a CSV file first to start chatting.", 
                "warning"
            )
            return
        
        # Display current file info
        st.info(f"📄 Currently querying: **{st.session_state.current_file}** | Top-K: {st.session_state.top_k}")
        
        # Initialize LLM chain if needed
        if st.session_state.llm_chain is None:
            st.session_state.llm_chain = LLMChain()
        
        # Render chat history
        for msg in st.session_state.chat_history:
            self.ui.render_chat_message(
                msg['role'], 
                msg['content'], 
                msg.get('references')
            )
        
        # Chat input
        user_input = st.chat_input("Ask a question about the uploaded content...")
        
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            # Display user message
            self.ui.render_chat_message('user', user_input)
            
            # Generate response
            with st.chat_message('assistant'):
                with st.spinner("Thinking..."):
                    # Step 1: Rewrite vague user queries into precise retrieval queries using the LLM.
                    # Why: Improves retrieval quality by resolving ambiguity before vector search.
                    if not hasattr(st.session_state.llm_chain, "rewrite_query"):
                        # Reinitialize if the method isn't present (old session instance)
                        st.session_state.llm_chain = _LLMChain()
                    if not hasattr(st.session_state.llm_chain, "rewrite_query"):
                        rewrite = {"rewritten_query": user_input, "did_rewrite": False, "notes": "rewrite unavailable"}
                    else:
                        try:
                            rewrite = st.session_state.llm_chain.rewrite_query(
                                user_input,
                                chat_history=st.session_state.chat_history,
                                source_name=st.session_state.current_file,
                                fallback_on_error=True,
                            )
                        except Exception:
                            rewrite = {"rewritten_query": user_input, "did_rewrite": False, "notes": "rewrite failed"}
                    rewritten_query = rewrite.get("rewritten_query", user_input) or user_input

                    # Step 2: Retrieve documents using the rewritten query but keep the original
                    # human-friendly question for the final answer generation.
                    docs = st.session_state.neo4j_manager.retrieve_with_filename_filter(
                        rewritten_query,
                        st.session_state.current_file,
                        st.session_state.top_k
                    )
                    
                    if not docs:
                        response = {
                            "text": "No relevant content found in the current dataset.",
                            "refrence": []
                        }
                    else:
                        # Step 3: Generate response using LLM chain (support new and old signatures)
                        try:
                            response = st.session_state.llm_chain.graph_qa_chain(
                                user_input,
                                docs,
                                source_name=st.session_state.current_file,
                                fallback_on_error=True
                            )
                        except TypeError:
                            # The existing LLMChain instance is likely from a previous run.
                            # Reinitialize it to pick up the new signature and retry.
                            st.session_state.llm_chain = _LLMChain()
                            try:
                                response = st.session_state.llm_chain.graph_qa_chain(
                                    user_input,
                                    docs,
                                    source_name=st.session_state.current_file,
                                    fallback_on_error=True
                                )
                            except TypeError:
                                # Final fallback: call without source_name
                                response = st.session_state.llm_chain.graph_qa_chain(
                                    user_input,
                                    docs,
                                    fallback_on_error=True
                                )
                
                # Display response as main text with references dropdown
                self.ui.render_chat_message('assistant', response)
            
            # Add assistant message to history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
    
    def run(self):
        """Run the main application."""
        # Page configuration
        st.set_page_config(
            page_title=Config.APP_TITLE,
            page_icon=Config.APP_ICON,
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        self.initialize_session_state()
        
        # Apply custom CSS
        self.ui.apply_custom_css()
        
        # Render header
        self.ui.render_header(
            title="RAGVeda Neo4j Chatbot",
            subtitle="Intelligent Semantic Search with Neo4j AuraDB"
        )
        
        # Sidebar
        self.ui.render_sidebar_header("📚 Upload Dataset")
        
        # Validate configuration
        if not self.validate_configuration():
            return
        
        st.sidebar.success("✅ Configuration validated")
        
        # Settings
        settings = self.ui.render_settings_section(st.session_state.top_k)
        st.session_state.top_k = settings['top_k']
        
        # Handle file upload
        file_info = self.handle_file_upload()
        
        if file_info:
            # Process the dataset
            success = self.process_dataset(file_info)
            if success:
                st.balloons()
                st.rerun()
        
        # Main content area
        if st.session_state.embeddings_created:
            # Chat interface
            st.markdown('<h2 class="sub-header">💬 Chat</h2>', unsafe_allow_html=True)
            self.handle_chat_interaction()
        else:
            # Welcome screen
            self.ui.render_welcome_screen()
        
        # Footer
        self.ui.render_footer()


def main():
    """Main entry point."""
    app = RAGVedaApp()
    app.run()


if __name__ == "__main__":
    main()
