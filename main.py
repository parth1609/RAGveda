"""
RAGVeda LangGraph Streamlit Application.
Main entry point for the LangGraph-powered RAGVeda application.
"""

import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Import LangGraph components
from langgraph_ragveda import (
    build_document_processing_graph,
    build_query_processing_graph,
    build_complete_ragveda_graph,
    RAGVedaState,
    initialize_services,
    check_api_config
)

# Import UI components and configuration
from modules.config import Config
from modules.ui_components import UIComponents


class LangGraphRAGVedaApp:
    """LangGraph-powered RAGVeda application."""
    
    def __init__(self):
        """Initialize the application."""
        self.ui = UIComponents()
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        # Graph instances
        if 'document_graph' not in st.session_state:
            st.session_state.document_graph = build_document_processing_graph()
        if 'query_graph' not in st.session_state:
            st.session_state.query_graph = build_query_processing_graph()
        if 'complete_graph' not in st.session_state:
            st.session_state.complete_graph = build_complete_ragveda_graph()
            
        # State tracking
        if 'current_file' not in st.session_state:
            st.session_state.current_file = None
        if 'embeddings_created' not in st.session_state:
            st.session_state.embeddings_created = False
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'top_k' not in st.session_state:
            st.session_state.top_k = Config.DEFAULT_TOP_K
        if 'auth_applied' not in st.session_state:
            st.session_state.auth_applied = False
        if 'services_initialized' not in st.session_state:
            st.session_state.services_initialized = False
        if 'graph_state' not in st.session_state:
            st.session_state.graph_state = {}
            
    def check_and_initialize_services(self) -> bool:
        """Check configuration and initialize services using LangGraph."""
        # Create initial state for configuration check
        state = {"messages": []}
        
        # Check API configuration
        config_result = check_api_config(state)
        
        if not config_result.get("api_config_valid", False):
            st.error("Configuration Issues:")
            if config_result.get("error_message"):
                st.error(config_result["error_message"])
            st.info("Use the sidebar 'API Configuration' to provide or update these values.")
            return False
            
        # Initialize services if not already done
        if not st.session_state.services_initialized:
            with st.spinner("🔧 Initializing services..."):
                service_result = initialize_services(state)
                
                if service_result.get("neo4j_connected", False):
                    st.session_state.services_initialized = True
                    st.session_state.graph_state.update(service_result)
                    st.success("✅ Services initialized successfully!")
                    return True
                else:
                    st.error(f"Failed to initialize services: {service_result.get('error_message', 'Unknown error')}")
                    return False
        
        return True
    
    def handle_file_upload(self) -> Optional[Dict[str, Any]]:
        """Handle file upload and preview."""
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
                st.markdown("### 📊 Data Preview")
                st.dataframe(df_preview, use_container_width=True)
                
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
    
    def process_document_with_graph(self, file_info: Dict[str, Any]) -> bool:
        """Process document using LangGraph document processing pipeline."""
        try:
            # Prepare initial state
            initial_state = {
                "uploaded_file_path": file_info['path'],
                "filename": file_info['filename'],
                "content_column": file_info['content_column'],
                "messages": [],
                # Include service instances from session state
                **st.session_state.graph_state
            }
            
            # Create progress container
            progress_container = st.container()
            
            with progress_container:
                # Process through the document graph with status updates
                with st.spinner("📖 Loading CSV file..."):
                    result = st.session_state.document_graph.invoke(initial_state)
                    
                    # Check for errors at each step
                    if result.get("error_message"):
                        st.error(f"Error: {result['error_message']}")
                        return False
                    
                    # Display progress based on current_step
                    current_step = result.get("current_step", "")
                    
                    if current_step == "csv_loaded":
                        st.success("✅ CSV file loaded")
                    
                    if current_step in ["csv_parsed", "documents_grouped", "documents_chunked"]:
                        st.success(f"✅ Documents processed: {len(result.get('final_documents', []))} chunks created")
                    
                    if current_step == "embeddings_generated":
                        st.success(f"✅ Embeddings created and stored in Neo4j")
                        st.success(f"✅ Index name: {result.get('index_name', 'N/A')}")
                    
                    if current_step == "relationships_created":
                        st.success("✅ File relationships created in Neo4j")
            
            # Update session state
            st.session_state.current_file = file_info['filename']
            st.session_state.embeddings_created = True
            st.session_state.graph_state.update(result)
            
            # Clear chat history
            st.session_state.chat_history = []
            
            return True
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return False
    
    def handle_chat_with_graph(self):
        """Handle chat interaction using LangGraph query processing pipeline."""
        if not st.session_state.embeddings_created:
            self.ui.render_status_message(
                "Please upload and process a CSV file first to start chatting.", 
                "warning"
            )
            return
        
        # Display current file info
        st.info(f"📄 Currently querying: **{st.session_state.current_file}** | Top-K: {st.session_state.top_k}")
        
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
            
            # Generate response using LangGraph
            with st.chat_message('assistant'):
                with st.spinner("Thinking..."):
                    # Prepare state for query processing
                    query_state = {
                        "user_query": user_input,
                        "filename": st.session_state.current_file,
                        "top_k": st.session_state.top_k,
                        "messages": [],
                        # Include service instances
                        **st.session_state.graph_state
                    }
                    
                    # Process through query graph
                    result = st.session_state.query_graph.invoke(query_state)
                    
                    # Extract response
                    llm_response = result.get("llm_response", {})
                    response_text = llm_response.get("text", "Unable to generate response")
                    references = result.get("references", [])
                    
                    # Display response
                    response = {
                        "text": response_text,
                        "refrence": references  # Note: keeping typo for compatibility
                    }
                    
                    st.markdown(response_text)
                    
                    # Display sources if available
                    if references:
                        with st.expander("📚 Sources"):
                            for ref in references:
                                st.markdown(f"• {ref}")
            
            # Add assistant message to history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'references': references
            })
    
    def visualize_graph(self):
        """Visualize the LangGraph structure."""
        with st.expander("🔍 View LangGraph Structure"):
            tab1, tab2, tab3 = st.tabs(["Document Processing", "Query Processing", "Complete Graph"])
            
            with tab1:
                st.markdown("### Document Processing Graph")
                doc_graph = build_document_processing_graph()
                mermaid_code = doc_graph.get_graph().draw_mermaid()
                st.code(mermaid_code, language="mermaid")
                
            with tab2:
                st.markdown("### Query Processing Graph")
                query_graph = build_query_processing_graph()
                mermaid_code = query_graph.get_graph().draw_mermaid()
                st.code(mermaid_code, language="mermaid")
                
            with tab3:
                st.markdown("### Complete RAGVeda Graph")
                complete_graph = build_complete_ragveda_graph()
                mermaid_code = complete_graph.get_graph().draw_mermaid()
                st.code(mermaid_code, language="mermaid")
    
    def run(self):
        """Run the main application."""
        # Page configuration
        st.set_page_config(
            page_title="RAGVeda LangGraph",
            page_icon="🕉️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        self.initialize_session_state()
        
        # Apply custom CSS
        self.ui.apply_custom_css()
        
        # Render header
        self.ui.render_header(
            title="RAGVeda LangGraph",
            subtitle="Intelligent Semantic Search powered by LangGraph & Neo4j"
        )
        
        # Sidebar: API Keys Inputs
        creds = self.ui.render_api_key_inputs()
        if creds:
            # Validate and apply credentials
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
                # Apply runtime credentials
                Config.NEO4J_URL = creds.get("NEO4J_URL") or Config.NEO4J_URL
                Config.NEO4J_USERNAME = creds.get("NEO4J_USERNAME") or Config.NEO4J_USERNAME
                Config.NEO4J_PASSWORD = creds.get("NEO4J_PASSWORD") or Config.NEO4J_PASSWORD
                Config.GROQ_API_KEY = creds.get("GROQ_API_KEY") or Config.GROQ_API_KEY
                
                # Set environment variables
                if creds.get("NEO4J_URL"):
                    os.environ["NEO4J_URL"] = creds["NEO4J_URL"]
                if creds.get("NEO4J_USERNAME"):
                    os.environ["NEO4J_USERNAME"] = creds["NEO4J_USERNAME"]
                if creds.get("NEO4J_PASSWORD"):
                    os.environ["NEO4J_PASSWORD"] = creds["NEO4J_PASSWORD"]
                if creds.get("GROQ_API_KEY"):
                    os.environ["GROQ_API_KEY"] = creds["GROQ_API_KEY"]
                if creds.get("HF_TOKEN"):
                    os.environ["HUGGINGFACEHUB_API_TOKEN"] = creds["HF_TOKEN"]
                    os.environ["HUGGING_FACE_HUB_TOKEN"] = creds["HF_TOKEN"]
                
                # Reset service initialization flag to reinitialize with new credentials
                st.session_state.services_initialized = False
                st.session_state.auth_applied = True
                st.sidebar.success("Configuration saved. Reinitializing services...")
                st.rerun()
        
        # Check authentication status
        if not st.session_state.auth_applied:
            st.info("Please complete API Configuration in the sidebar and click 'Save Configuration' to proceed.")
            self.ui.render_welcome_screen()
            self.ui.render_footer()
            return
        
        # Check and initialize services
        if not self.check_and_initialize_services():
            return
        
        # Settings
        settings = self.ui.render_settings_section(st.session_state.top_k)
        st.session_state.top_k = settings['top_k']
        
        # Graph visualization
        self.visualize_graph()
        
        # Handle file upload
        file_info = self.handle_file_upload()
        
        if file_info:
            # Process the document using LangGraph
            success = self.process_document_with_graph(file_info)
            if success:
                st.balloons()
                st.rerun()
        
        # Main content area
        if st.session_state.embeddings_created:
            # Chat interface
            st.markdown('<h2 class="sub-header">💬 Chat with LangGraph</h2>', unsafe_allow_html=True)
            self.handle_chat_with_graph()
        else:
            # Welcome screen
            self.ui.render_welcome_screen()
        
        # Footer
        self.ui.render_footer()


def main():
    """Main entry point for LangGraph application."""
    app = LangGraphRAGVedaApp()
    app.run()


if __name__ == "__main__":
    main()
