# RAGVeda

The Retrieval-Augmented Generation (RAG) system that graph database and  LLM to create an intelligent document processing and question-answering platform. The system processes PDF documents, creates knowledge graphs, and enables natural language querying of document content.

## The Graph Data Structure
<img src="/output/op1.png" alt="op1"> <img src="/output/op2.png" alt="op2">

## 🌟 Features


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


## 🛠 Technical Stack

- **Frontend**: Streamlit
- **Database**: Neo4j Graph Database
- **LLM**: Groq with "mixtral-8x7b-32768"


## 🚀 Setup Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd RAGveda

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
groq_api_key=your_groq_api_key
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


### Knowledge Graph Schema

```cypher

```

### Query Processing Flow

```python
User Query → LLM Processing → Graph Search → Context Retrieval → Response Generation
```



### Core Components

- `app.py`: Main application logic and Streamlit interface
- `GraphRAG`: Core class implementing RAG functionality
- `Neo4jGraph`: Graph database interface
- `GraphQAChain`: Question-answering chain implementation





