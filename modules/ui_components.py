"""
UI Components Module.
Provides reusable Streamlit UI components and styling.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path


class UIComponents:
    """Provides UI components and styling for Streamlit app."""
    
    @staticmethod
    def apply_custom_css():
        """Apply custom CSS styling to the app."""
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
            
            .chat-container {
                background-color: #f7f7f7;
                border-radius: 10px;
                padding: 1rem;
                margin-bottom: 1rem;
            }
            
            .dataset-info {
                background-color: #E3F2FD;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
            
            .status-success {
                background-color: #d4edda;
                color: #155724;
                padding: 0.5rem 1rem;
                border-radius: 5px;
                margin: 0.5rem 0;
            }
            
            .status-error {
                background-color: #f8d7da;
                color: #721c24;
                padding: 0.5rem 1rem;
                border-radius: 5px;
                margin: 0.5rem 0;
            }
            
            .reference-chip {
                background-color: #6c757d;
                color: white;
                padding: 0.25rem 0.5rem;
                border-radius: 15px;
                font-size: 0.85rem;
                display: inline-block;
                margin: 0.2rem;
            }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_header(title: str = "RAGVeda Neo4j Chatbot", subtitle: str = None):
        """
        Render application header.
        
        Args:
            title: Main title
            subtitle: Optional subtitle
        """
        st.markdown(f'<h1 class="main-header">🕉️ {title}</h1>', unsafe_allow_html=True)
        if subtitle:
            st.markdown(
                f'<p style="text-align: center; font-size: 1.2rem; color: #6C757D; margin-bottom: 2rem;">{subtitle}</p>',
                unsafe_allow_html=True
            )
    
    @staticmethod
    def render_sidebar_header(text: str):
        """
        Render sidebar section header.
        
        Args:
            text: Header text
        """
        st.sidebar.markdown(f'<h2 class="sub-header">{text}</h2>', unsafe_allow_html=True)
    
    @staticmethod
    def render_file_upload_section() -> Optional[Any]:
        """
        Render file upload section.
        
        Returns:
            Uploaded file object or None
        """
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV Dataset",
            type=['csv'],
            help="Upload a CSV file with your data (e.g., Gita_questions.csv format)"
        )
        return uploaded_file
    
    @staticmethod
    def render_dataset_preview(df: pd.DataFrame, filename: str, filesize: int):
        """
        Render dataset preview in sidebar.
        
        Args:
            df: DataFrame preview
            filename: Name of the file
            filesize: Size of the file in bytes
        """
        st.sidebar.markdown('<div class="dataset-info">', unsafe_allow_html=True)
        st.sidebar.write(f"**File:** {filename}")
        st.sidebar.write(f"**Size:** {filesize:,} bytes")
        st.sidebar.write(f"**Rows:** {len(df)}")
        st.sidebar.write(f"**Columns:** {', '.join(df.columns)}")
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def render_column_selector(columns: List[str], default_column: str = None) -> str:
        """
        Render column selector.
        
        Args:
            columns: List of column names
            default_column: Default selection
            
        Returns:
            Selected column name
        """
        st.sidebar.markdown("### 📝 Select Main Content Column")
        st.sidebar.info("Choose the column that contains the main text content for RAG")
        
        default_idx = 0
        if default_column and default_column in columns:
            default_idx = columns.index(default_column)
        
        return st.sidebar.selectbox(
            "Content Column:",
            options=columns,
            index=default_idx,
            help="This column will be used for creating embeddings and search"
        )
    
    @staticmethod
    def render_settings_section(default_top_k: int = 5) -> Dict[str, Any]:
        """
        Render settings section.
        
        Args:
            default_top_k: Default value for top-k
            
        Returns:
            Dictionary of settings
        """
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Settings")
        
        settings = {
            'top_k': st.sidebar.slider(
                "Top-K Results",
                min_value=1,
                max_value=10,
                value=default_top_k,
                help="Number of documents to retrieve"
            )
        }
        
        return settings
    
    @staticmethod
    def render_chat_message(role: str, content: Any, references: List[str] = None):
        """
        Render a chat message.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            references: Optional list of references
        """
        with st.chat_message(role):
            # If dict with 'text', render text as main answer and refs in dropdown
            if isinstance(content, dict) and 'text' in content:
                st.markdown(content['text'])
                refs = references if references is not None else content.get('refrence')
                if refs and role == 'assistant':
                    with st.expander("📚 Sources"):
                        for ref in refs:
                            st.markdown(f'<span class="reference-chip">{ref}</span>', unsafe_allow_html=True)
                # Optional raw JSON toggle for debugging
                # with st.expander("🔎 Raw JSON"):
                #     st.json(content.get('refrence'))
            else:
                st.markdown(content)
                if references and role == 'assistant':
                    with st.expander("📚 Sources"):
                        for ref in references:
                            st.markdown(f'<span class="reference-chip">{ref}</span>', unsafe_allow_html=True)
    
    @staticmethod
    def render_status_message(message: str, status: str = "info"):
        """
        Render status message.
        
        Args:
            message: Message to display
            status: 'success', 'error', 'warning', or 'info'
        """
        if status == "success":
            st.success(f"✅ {message}")
        elif status == "error":
            st.error(f"❌ {message}")
        elif status == "warning":
            st.warning(f"⚠️ {message}")
        else:
            st.info(f"ℹ️ {message}")
    
    @staticmethod
    def render_welcome_screen():
        """Render welcome screen when no data is loaded."""
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background-color: #F8F9FA; border-radius: 15px; margin: 2rem 0;">
            <h2 style="color: #2E86AB; margin-bottom: 1rem;">Welcome to RAGVeda! 🙏</h2>
            <p style="font-size: 1.2rem; color: #6C757D; margin-bottom: 2rem;">
                Your intelligent companion for exploring texts through semantic search with Neo4j AuraDB.
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 300px;">
                    <h4 style="color: #FF6B35; margin-bottom: 1rem;">📚 Upload CSV</h4>
                    <p>Upload any CSV file with structured content</p>
                </div>
                <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 300px;">
                    <h4 style="color: #FF6B35; margin-bottom: 1rem;">🎯 Select Column</h4>
                    <p>Choose the main content column for processing</p>
                </div>
                <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 300px;">
                    <h4 style="color: #FF6B35; margin-bottom: 1rem;">💬 Ask Questions</h4>
                    <p>Chat with your data using natural language</p>
                </div>
            </div>
            <p style="margin-top: 2rem; color: #6C757D;">
                👈 Start by uploading a CSV file from the sidebar
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_footer():
        """Render application footer."""
        st.markdown("---")
        st.markdown(
            '<p style="text-align: center; color: #6C757D; font-size: 0.9rem;">'
            'RAGVeda - Powered by Neo4j AuraDB & Groq | Built with ❤️ using Streamlit</p>',
            unsafe_allow_html=True
        )
    
    @staticmethod
    def render_progress_bar(value: int, max_value: int, text: str = ""):
        """
        Render progress bar.
        
        Args:
            value: Current value
            max_value: Maximum value
            text: Optional text to display
        """
        progress = value / max_value
        st.progress(progress, text=text)
