# RAGVeda

A sophisticated Retrieval-Augmented Generation (RAG) system that graph database and  LLM to create an intelligent document processing and question-answering platform. The system processes PDF documents, creates knowledge graphs, and enables natural language querying of document content.

## 🌟 Features

### Document Processing
- PDF document ingestion and automatic chunking
- Intelligent text splitting with customizable chunk sizes
- Entity extraction and relationship mapping
- Automatic knowledge graph generation

### Knowledge Graph Integration
- Neo4j-based graph database implementation
- Entity relationship modeling
- Graph-based information retrieval
- Interactive graph visualization

### Question Answering
- Natural language query processing
- Context-aware response generation
- Graph-based information retrieval
- Custom prompt templates for improved responses

### User Interface
- Streamlit-based web interface
- Interactive document upload
- Real-time query processing
- Error handling and user feedback

## 🛠 Technical Stack

- **Frontend**: Streamlit
- **Database**: Neo4j Graph Database
- **LLM**: Google Gemini Pro
- **Document Processing**: LangChain
- **Language**: Python 3.8+

## 📋 Prerequisites

1. Python 3.8 or higher
2. Neo4j Database (local instance or cloud)
3. Google Cloud account with Gemini API access
4. Git (for version control)

## 🚀 Setup Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd graph-rag-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```env
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
GEMINI_API_KEY=your_gemini_api_key
```

### 3. Neo4j Setup


## 💻 Usage Guide

### Starting the Application

```bash
streamlit run app.py
```

### Document Processing Flow

1. **Upload Documents**
   - Use the file uploader in the UI
   - Support for PDF documents
   - Multiple file upload capability

2. **Knowledge Graph Creation**
   - Automatic document chunking
   - Entity extraction
   - Relationship mapping
   - Graph database population

3. **Querying Documents**
   - Enter natural language questions
   - View responses based on document context
   - Explore graph visualizations

## 🔧 Technical Details

### Document Processing Pipeline

```python
Document Upload → Text Extraction → Chunking → Entity Recognition → Graph Creation
```

### Knowledge Graph Schema

```cypher

```

### Query Processing Flow

```python
User Query → LLM Processing → Graph Search → Context Retrieval → Response Generation
```

## 📁 Project Structure

```
graph-rag-system/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # Documentation

```

### Core Components

- `app.py`: Main application logic and Streamlit interface
- `GraphRAG`: Core class implementing RAG functionality
- `Neo4jGraph`: Graph database interface
- `GraphQAChain`: Question-answering chain implementation

## 🤝 Contributing Guidelines




