"""
RAGVeda Main Application.
Production-level Streamlit application for Neo4j-based RAG system.
"""

import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Import modules
from modules.config import Config
from modules.neo4j_manager import Neo4jManager
from modules.document_processor import DocumentProcessor
from modules.retrieval import Retrieval
from modules.llm_chain import LLMChain
from modules.llm_chain import LLMChain as _LLMChain
from modules.memory_manager import MemoryManager
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
            st.session_state.neo4j_manager = Neo4jManager()
        if 'memory_manager' not in st.session_state:
            st.session_state.memory_manager = MemoryManager() if Config.MEMORY_ENABLED else None
        if 'llm_chain' not in st.session_state:
            st.session_state.llm_chain = None
        if 'auth_applied' not in st.session_state:
            # Require explicit Save of API Configuration before enabling upload/chat
            st.session_state.auth_applied = False
    
    def validate_configuration(self) -> bool:
        """
        Validate application configuration.
        
        Returns:
            True if configuration is valid
        """
        errors = Config.validate()
        
        if errors:
            st.error("Configuration Issues:")
            for error in errors:
                st.error(f"• {error}")
            st.info("Use the sidebar 'API Configuration' to provide or update these values, then click Save.")
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
                
                # Display data preview in main area
                st.markdown("### 📊 Data Preview")
                st.dataframe(df_preview, width='stretch')
                
                # Process button
                if st.button("🚀 Process & Create Embeddings", type="primary"):
                    return {
                        'path': temp_path,
                        'filename': uploaded_file.name,
                        'content_column': content_column
                    }
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
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
            
            # Clear chat history and memory when switching files
            st.session_state.chat_history = []
            if st.session_state.memory_manager:
                st.session_state.memory_manager.reset_session()
            
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
                    # Step 1: Skip query rewriting for simple queries to improve speed
                    rewritten_query = user_input
                    if len(user_input.split()) > 5:  # Only rewrite complex queries
                        if not hasattr(st.session_state.llm_chain, "rewrite_query"):
                            st.session_state.llm_chain = _LLMChain()
                        if hasattr(st.session_state.llm_chain, "rewrite_query"):
                            try:
                                rewrite = st.session_state.llm_chain.rewrite_query(
                                    user_input,
                                    chat_history=st.session_state.chat_history,
                                    source_name=st.session_state.current_file,
                                    fallback_on_error=True,
                                )
                                rewritten_query = rewrite.get("rewritten_query", user_input) or user_input
                            except Exception:
                                rewritten_query = user_input

                    # Step 2: Retrieve documents
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
                        # Step 3: Get memory context if available
                        memory_context = None
                        if st.session_state.memory_manager and st.session_state.memory_manager.is_available():
                            memory_context = st.session_state.memory_manager.get_memory_context()
                        
                        # Step 4: Generate response using LLM chain with memory context
                        try:
                            response = st.session_state.llm_chain.graph_qa_chain(
                                user_input,
                                docs,
                                source_name=st.session_state.current_file,
                                memory_context=memory_context,
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
                                    memory_context=memory_context,
                                    fallback_on_error=True
                                )
                            except TypeError:
                                # Final fallback: call without source_name and memory_context
                                response = st.session_state.llm_chain.graph_qa_chain(
                                    user_input,
                                    docs,
                                    fallback_on_error=True
                                )
                
                # Display response as main text with references dropdown
                self.ui.render_chat_message('assistant', response)
            
            # Save conversation turn to memory
            if st.session_state.memory_manager and st.session_state.memory_manager.is_available():
                response_text = response.get('text', str(response)) if isinstance(response, dict) else str(response)
                st.session_state.memory_manager.save_conversation_turn(user_input, response_text)
            
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
            title="RAGVeda ",
            subtitle="Intelligent Semantic Search with Neo4j AuraDB"
        )
        # Sidebar: API Keys Inputs
        creds = self.ui.render_api_key_inputs()
        if creds:
            # Validate required inputs before applying
            missing = []
            if not (creds.get("NEO4J_URL") or "").strip():
                missing.append("Neo4j URL")
            if not (creds.get("NEO4J_PASSWORD") or "").strip():
                missing.append("Neo4j Password")
            if not (creds.get("GROQ_API_KEY") or "").strip():
                missing.append("Groq API Key")
            if missing:
                st.sidebar.error("Please provide: " + ", ".join(missing))
            else:
                # Apply runtime credentials and reinitialize services
                Config.NEO4J_URL = creds.get("NEO4J_URL") or Config.NEO4J_URL
                Config.NEO4J_USERNAME = creds.get("NEO4J_USERNAME") or Config.NEO4J_USERNAME
                Config.NEO4J_PASSWORD = creds.get("NEO4J_PASSWORD") or Config.NEO4J_PASSWORD
                Config.GROQ_API_KEY = creds.get("GROQ_API_KEY") or Config.GROQ_API_KEY
                # Environment variables for any downstream libs
                if creds.get("NEO4J_URL") is not None:
                    os.environ["NEO4J_URL"] = creds["NEO4J_URL"]
                if creds.get("NEO4J_USERNAME") is not None:
                    os.environ["NEO4J_USERNAME"] = creds["NEO4J_USERNAME"]
                if creds.get("NEO4J_PASSWORD") is not None:
                    os.environ["NEO4J_PASSWORD"] = creds["NEO4J_PASSWORD"]
                if creds.get("GROQ_API_KEY") is not None:
                    os.environ["GROQ_API_KEY"] = creds["GROQ_API_KEY"]
                # Hugging Face token 
                if creds.get("HF_TOKEN"):
                    os.environ["HUGGINGFACEHUB_API_TOKEN"] = creds["HF_TOKEN"]
                    os.environ["HUGGING_FACE_HUB_TOKEN"] = creds["HF_TOKEN"]
                # Reinitialize stateful services to pick up new creds
                st.session_state.neo4j_manager = Neo4jManager()
                st.session_state.llm_chain = None  # recreated lazily
                if Config.MEMORY_ENABLED:
                    st.session_state.memory_manager = MemoryManager()
                st.session_state.auth_applied = True
                st.sidebar.success("Configuration saved. Reloading…")
                st.rerun()
        
        # Gate the rest of the UI until auth is applied in this session
        if not st.session_state.auth_applied:
            st.info("Please complete API Configuration in the sidebar and click 'Save Configuration' to proceed.")
            self.ui.render_welcome_screen()
            self.ui.render_footer()
            return
        
        # Validate configuration (for cases where .env is already present)
        if not self.validate_configuration():
            return

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
