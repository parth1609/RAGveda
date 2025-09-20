"""
RAGVeda - Neo4j Streamlit Chatbot Application
A comprehensive GraphRAG application with Neo4j AuraDB integration
"""

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path
import tempfile
from typing import List, Dict, Any, Optional
import time
import re

# Langchain imports
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="RAGVeda - Neo4j Chatbot",
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
    
    .answer-block {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .answer-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .answer-content {
        font-size: 1.2rem;
        line-height: 1.8;
        text-align: justify;
    }
    
    .result-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #E9ECEF;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    
    .status-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'embeddings_created' not in st.session_state:
        st.session_state.embeddings_created = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'neo4j_connected' not in st.session_state:
        st.session_state.neo4j_connected = False
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = None
    if 'chat' not in st.session_state:
        st.session_state.chat = []  # list of {role: 'user'|'assistant', content: str, refs: List[str]}
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 5

def check_neo4j_credentials():
    """Check if Neo4j credentials are available."""
    neo4j_url = os.getenv("NEO4J_URL")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not all([neo4j_url, neo4j_username, neo4j_password]):
        return False, "Missing Neo4j credentials. Please check your .env file."
    
    return True, "Credentials found"

def display_header():
    """Display the main application header."""
    st.markdown('<h1 class="main-header">🕉️ RAGVeda Neo4j Chatbot</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #6C757D; margin-bottom: 2rem;">'
        'Intelligent Semantic Search with Neo4j AuraDB Integration</p>',
        unsafe_allow_html=True
    )

def display_sidebar():
    """Display the sidebar with file upload and configuration."""
    st.sidebar.markdown('<h2 class="sub-header">📚 Upload Dataset</h2>', unsafe_allow_html=True)
    
    # Check Neo4j credentials
    creds_valid, creds_message = check_neo4j_credentials()
    if not creds_valid:
        st.sidebar.error(f"⚠️ {creds_message}")
        st.sidebar.markdown("""
        ### Setup Instructions:
        1. Create a `.env` file in the project root
        2. Add the following variables:
        ```
        NEO4J_URL=your_neo4j_url
        NEO4J_USERNAME=your_username
        NEO4J_PASSWORD=your_password
        ```
        """)
        return None, None, None
    else:
        st.sidebar.success("✅ Neo4j credentials configured")
    
    # Retrieval settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")
    st.session_state.top_k = st.sidebar.slider("Top-K Results", min_value=1, max_value=10, value=st.session_state.top_k)
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Dataset",
        type=['csv'],
        help="Upload a CSV file with your data. Must have consistent format like Gita_questions.csv"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        try:
            # Read and display preview
            df_preview = pd.read_csv(temp_path, nrows=5)
            
            st.sidebar.markdown('<div class="dataset-info">', unsafe_allow_html=True)
            st.sidebar.write(f"**File:** {uploaded_file.name}")
            st.sidebar.write(f"**Size:** {uploaded_file.size:,} bytes")
            st.sidebar.write(f"**Columns:** {', '.join(df_preview.columns)}")
            
            # Column selection for main content
            st.sidebar.markdown("### 📝 Select Main Content Column")
            st.sidebar.info("Choose the column that contains the main text content for RAG")
            
            content_column = st.sidebar.selectbox(
                "Content Column:",
                options=df_preview.columns.tolist(),
                index=df_preview.columns.tolist().index('translation') 
                    if 'translation' in df_preview.columns else 0,
                help="This column will be used for creating embeddings and search"
            )
            
            # Display preview
            st.sidebar.markdown("### 📊 Data Preview")
            st.sidebar.dataframe(df_preview, use_container_width=True)
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
            # Process button
            if st.sidebar.button("🚀 Process & Create Embeddings", type="primary", use_container_width=True):
                if st.session_state.current_file != uploaded_file.name:
                    st.session_state.embeddings_created = False
                    st.session_state.vectorstore = None
                st.session_state.current_file = uploaded_file.name
                return temp_path, uploaded_file.name, content_column
            
            return None, None, None
            
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)}")
            return None, None, None
    
    return None, None, None

def group_documents_by_chunk(docs: List[Document], chunk_size: int = 6) -> List[Document]:
    """Group documents into chunks of specified size."""
    grouped_docs = []
    
    for i in range(0, len(docs), chunk_size):
        chunk_docs = docs[i:i + chunk_size]
        
        # Combine content
        combined_content = "\n\n".join([doc.page_content for doc in chunk_docs])
        
        # Combine metadata
        combined_metadata = {
            'row_start': i,
            'row_end': min(i + chunk_size - 1, len(docs) - 1),
        }
        
        # Add chapter and verse info if available
        if chunk_docs[0].metadata:
            first_meta = chunk_docs[0].metadata
            if 'chapter' in first_meta:
                combined_metadata['chapters'] = [doc.metadata.get('chapter') for doc in chunk_docs]
            if 'verse' in first_meta:
                combined_metadata['verses'] = [doc.metadata.get('verse') for doc in chunk_docs]
        
        grouped_docs.append(Document(
            page_content=combined_content,
            metadata=combined_metadata
        ))
    
    return grouped_docs

def get_llm() -> Optional[ChatGroq]:
    """Return a Groq chat model if GROQ_API_KEY is configured, else None."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Try Streamlit secrets fallback
        try:
            api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
        except Exception:
            api_key = None
    if not api_key:
        return None
    try:
        return ChatGroq(model="gemma2-9b-it", temperature=0, groq_api_key=api_key)
    except Exception:
        return None

def format_context(docs: List[Document]) -> str:
    """Format retrieved docs into a context string for the LLM."""
    parts = []
    for i, d in enumerate(docs, start=1):
        parts.append(f"[Doc {i}]\n{d.page_content}")
    return "\n\n".join(parts)

def extract_refs(docs: List[Document]) -> List[str]:
    """Extract 'Chapter N Verse M' references from docs content/metadata."""
    pat = re.compile(r"Chapter\s+(\d+)\s+Verse\s+(\d+)", re.IGNORECASE)
    refs_set = []
    seen = set()
    for d in docs:
        # From content
        for ch, ve in pat.findall(d.page_content or ""):
            key = (int(ch), int(ve))
            if key not in seen:
                seen.add(key)
                refs_set.append(f"Chapter {int(ch)} Verse {int(ve)}")
        # From metadata if present
        ch = d.metadata.get("chapter") if isinstance(d.metadata, dict) else None
        ve = d.metadata.get("verse") if isinstance(d.metadata, dict) else None
        if ch is not None and ve is not None:
            key = (int(ch), int(ve))
            if key not in seen:
                seen.add(key)
                refs_set.append(f"Chapter {int(ch)} Verse {int(ve)}")
    return refs_set

def graph_qa_chain(question: str, top_k: int) -> Dict[str, Any]:
    """Retrieve, ask LLM for concise JSON answer, and attach authoritative references."""
    if not st.session_state.vectorstore:
        return {"text": "Vector store is not initialized. Please upload and process a CSV first.", "refrence": []}
    retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    docs: List[Document] = retriever.get_relevant_documents(question)
    context = format_context(docs)
    refs = extract_refs(docs)

    llm = get_llm()
    if llm is None:
        # Fallback: heuristic concise answer from top documents
        snippet = " ".join((docs[0].page_content.split(". ")[:3] if docs else ["No context found."]))
        return {"text": snippet.strip(), "refrence": refs}

    parser = JsonOutputParser()
    format_instructions = (
        '{\n  "text": "<concise answer in 3–5 sentences>"\n}\n'  # exact JSON schema
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Use the provided context from the Bhagavad Gita to answer "
         "the user's question accurately and concisely in 3–5 sentences. "
         "Do not include verse text in your answer. If the context is insufficient, say you are uncertain and ask for clarification. "
         "Follow the output format instructions exactly."),
        ("human",
         "Question:\n{question}\n\nContext:\n{context}\n\nOutput format:\n{format_instructions}")
    ])

    try:
        result: Dict[str, Any] = (prompt | llm | parser).invoke({
            "question": question,
            "context": context,
            "format_instructions": format_instructions
        })
        # Ensure we strictly attach clean references
        result["refrence"] = refs
        # Guard: if model didn't return dict
        if not isinstance(result, dict) or "text" not in result:
            result = {"text": str(result), "refrence": refs}
        return result
    except Exception as e:
        # On any LLM failure, provide a sensible fallback
        snippet = " ".join((docs[0].page_content.split(". ")[:3] if docs else ["No context found."]))
        return {"text": snippet.strip(), "refrence": refs, "_warning": f"LLM error: {e}"}

def process_csv_and_create_embeddings(file_path: str, filename: str, content_column: str):
    """Process CSV file and create embeddings in Neo4j."""
    try:
        with st.spinner("📖 Loading dataset..."):
            df = pd.read_csv(file_path)
            st.success(f"✅ Loaded {len(df)} rows from {filename}")
        
        # Create content column with all metadata
        with st.spinner("🔄 Processing documents..."):
            # Build content with metadata
            if 'chapter' in df.columns and 'verse' in df.columns:
                df['content'] = (
                    "Chapter " + df['chapter'].astype(str) + 
                    " Verse " + df['verse'].astype(str) + 
                    " — " + df[content_column].astype(str).str.strip()
                )
            else:
                df['content'] = df[content_column].astype(str).str.strip()
            
            # Create document loader
            loader = DataFrameLoader(
                df,
                page_content_column="content",
            )
            
            # Load documents
            docs = list(loader.lazy_load())
            
            # Add filename to metadata for each document
            for doc in docs:
                doc.metadata['filename'] = filename
                # Preserve all original columns as metadata
                for col in df.columns:
                    if col != 'content' and col in df.columns:
                        # Get the row index from the document
                        if hasattr(doc, 'metadata') and 'row' in doc.metadata:
                            row_idx = doc.metadata['row']
                            doc.metadata[col] = df.iloc[row_idx][col]
            
            st.success(f"✅ Created {len(docs)} documents")
        
        # Group documents into chunks (following graphrag pattern)
        with st.spinner("📦 Creating document chunks..."):
            # Use RecursiveCharacterTextSplitter for better chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Split documents
            final_docs = []
            for doc in docs:
                splits = text_splitter.split_documents([doc])
                for split in splits:
                    split.metadata['filename'] = filename  # Ensure filename is preserved
                final_docs.extend(splits)
            
            st.success(f"✅ Created {len(final_docs)} chunks")
        
        # Initialize embedding model
        with st.spinner("🤖 Initializing embedding model..."):
            if st.session_state.embedding_model is None:
                st.session_state.embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            embedding_model = st.session_state.embedding_model
            st.success("✅ Embedding model ready")
        
        # Create Neo4j vector store (following exact graphrag pattern)
        with st.spinner("🔗 Connecting to Neo4j and creating embeddings..."):
            # Dynamic index name based on filename
            index_name = filename.replace('.csv', '').replace(' ', '_').lower() + "_idx"
            
            # Create vector store
            vectorstore = Neo4jVector.from_documents(
                documents=final_docs,
                embedding=embedding_model,
                url=os.getenv("NEO4J_URL"),
                username=os.getenv("NEO4J_USERNAME") or "neo4j",
                password=os.getenv("NEO4J_PASSWORD"),
                index_name=index_name,
                node_label="Chunk",
                text_node_property="text",
                embedding_node_property="embedding",
            )
            
            st.success(f"✅ Ingested {len(final_docs)} chunks into Neo4j vector store: {index_name}")
        
        # Create File nodes and relationships (following exact graphrag pattern)
        with st.spinner("🔗 Creating File-Chunk relationships..."):
            # Create constraint for unique file names
            vectorstore.query(
                "CREATE CONSTRAINT file_name_unique IF NOT EXISTS FOR (f:File) REQUIRE f.name IS UNIQUE"
            )
            
            # Set filename for any chunks that might not have it
            vectorstore.query(f"""
                MATCH (c:Chunk) 
                WHERE c.filename IS NULL 
                SET c.filename = '{filename}'
            """)
            
            # Create File node and connect chunks
            vectorstore.query(f"""
                MATCH (c:Chunk) 
                WHERE c.filename IS NOT NULL
                MERGE (f:File {{name: c.filename}})
                MERGE (f)-[:HAS_CHUNK]->(c)
            """)
            
            st.success(f"✅ Linked Chunk nodes to File node: {filename}")
        
        # Store in session state
        st.session_state.vectorstore = vectorstore
        st.session_state.embeddings_created = True
        st.session_state.neo4j_connected = True
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        return False

def perform_rag_query(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform RAG query on the vector store."""
    if not st.session_state.vectorstore:
        return []
    
    try:
        # Create retriever (following exact graphrag pattern)
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        
        # Get relevant documents
        results = retriever.get_relevant_documents(query)
        
        # Format results
        formatted_results = []
        for i, doc in enumerate(results, start=1):
            formatted_results.append({
                'rank': i,
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': 1.0 - (i * 0.1)  # Approximate score
            })
        
        return formatted_results
        
    except Exception as e:
        st.error(f"Error performing query: {str(e)}")
        return []

def display_chat_ui():
    """ChatGPT-like interface that shows the JSON 'text' as answer and references in a Sources expander."""
    st.markdown('<h2 class="sub-header">💬 Chat</h2>', unsafe_allow_html=True)
    if not st.session_state.embeddings_created:
        st.warning("⚠️ Please upload and process a CSV file first to start chatting.")
        return
    if st.session_state.current_file:
        st.info(f"📄 Currently querying: **{st.session_state.current_file}** | Top-K: {st.session_state.top_k}")

    # Render chat history
    for msg in st.session_state.chat:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
            if msg['role'] == 'assistant' and msg.get('refs'):
                with st.expander("Sources"):
                    for r in msg['refs']:
                        st.write(r)

    # Chat input
    user_input = st.chat_input("Ask a question about the uploaded content…")
    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                qa = graph_qa_chain(user_input, st.session_state.top_k)
            answer_text = qa.get("text", "(No answer)")
            refs = qa.get("refrence", [])
            st.markdown(answer_text)
            if refs:
                with st.expander("Sources"):
                    for r in refs:
                        st.write(r)
        st.session_state.chat.append({"role": "assistant", "content": answer_text, "refs": refs})

def display_query_history():
    """Display query history in sidebar."""
    if st.session_state.query_history:
        st.sidebar.markdown("### 📜 Query History")
        for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.sidebar.expander(f"Query {len(st.session_state.query_history) - i}"):
                st.write(f"**Query:** {item['query'][:50]}...")
                st.write(f"**Time:** {item['timestamp']}")
                st.write(f"**Results:** {item['results_count']}")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Display sidebar and get file info
    file_path, filename, content_column = display_sidebar()
    
    # Process file if provided
    if file_path and filename and content_column:
        success = process_csv_and_create_embeddings(file_path, filename, content_column)
        if success:
            st.balloons()
            st.rerun()
    
    # Display query interface
    if st.session_state.embeddings_created:
        display_chat_ui()
    else:
        # Welcome message
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background-color: #F8F9FA; border-radius: 15px; margin: 2rem 0;">
            <h2 style="color: #2E86AB; margin-bottom: 1rem;">Welcome to RAGVeda Neo4j Chatbot! 🙏</h2>
            <p style="font-size: 1.2rem; color: #6C757D; margin-bottom: 2rem;">
                Your intelligent companion for exploring texts through semantic search with Neo4j AuraDB.
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 300px;">
                    <h4 style="color: #FF6B35; margin-bottom: 1rem;">📚 Upload CSV</h4>
                    <p>Upload any CSV file with the same format as Gita_questions.csv</p>
                </div>
                <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 300px;">
                    <h4 style="color: #FF6B35; margin-bottom: 1rem;">🎯 Select Column</h4>
                    <p>Choose the main content column for RAG processing</p>
                </div>
                <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 300px;">
                    <h4 style="color: #FF6B35; margin-bottom: 1rem;">🔗 Neo4j Storage</h4>
                    <p>Embeddings stored securely in Neo4j AuraDB with File-Chunk relationships</p>
                </div>
            </div>
            <p style="margin-top: 2rem; color: #6C757D;">
                👈 Start by uploading a CSV file from the sidebar
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display query history
    display_query_history()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #6C757D; font-size: 0.9rem;">'
        'RAGVeda Neo4j - Powered by Neo4j AuraDB & HuggingFace | Built with ❤️ using Streamlit</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
