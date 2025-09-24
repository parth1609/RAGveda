# RAGVeda

A Retrieval-Augmented Generation (RAG) system that leverages Neo4j graph database and LLMs to create an intelligent document processing and question-answering platform. The system processes documents, creates vector embeddings, and enables natural language querying with context-aware responses.

## 🌟 Features

### Core Capabilities
- **Document Processing**: Process and chunk documents with configurable settings
- **Vector Search**: Semantic search with sentence-transformers embeddings
- **LLM Integration**: Powered by Groq LLM for high-performance inference
- **Query Rewriting**: Automatically refines vague queries for better retrieval
- **Context-Aware Responses**: Maintains conversation context for follow-up questions

### Knowledge Graph Integration
- Neo4j-based vector database
- Document and chunk relationship modeling
- Efficient similarity search with cosine distance
- Automatic file-chunk relationship management

## 🛠 Technical Stack

- **Frontend**: Streamlit
- **Vector Database**: Neo4j with vector search
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: Groq with `gemma2-9b-it`
- **Language**: Python 3.8+

## � Dependencies

```bash
streamlit
pandas
langchain
langchain-neo4j
langchain-huggingface
langchain-groq
python-dotenv
neo4j
sentence-transformers
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Neo4j AuraDB or local Neo4j instance
- Groq API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RAGveda.git
   cd RAGveda
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your credentials:
   ```env
   NEO4J_URI=your_neo4j_uri
   NEO4J_USERNAME=your_username
   NEO4J_PASSWORD=your_password
   GROQ_API_KEY=your_groq_api_key
   ```

### Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run main.py
   ```

2. Open your browser to `http://localhost:8501`

3. Upload a document and start asking questions!

## 🔧 Configuration

Customize the application by modifying `modules/config.py`:
- Adjust chunking parameters (size, overlap)
- Configure embedding model settings
- Set default number of retrieved documents
- Tune similarity thresholds

## 🤖 How It Works

1. **Document Processing**:
   - Documents are split into chunks with configurable sizes
   - Each chunk is embedded using sentence-transformers
   - Chunks are stored in Neo4j with metadata and relationships

2. **Query Processing**:
   - User queries are automatically rewritten for better retrieval
   - Queries are embedded and used for semantic search
   - Top-k most relevant chunks are retrieved

3. **Response Generation**:
   - Retrieved context is formatted into a prompt
   - LLM generates a response using the provided context
   - Response includes relevant document references

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) for the LLM orchestration framework
- [Neo4j](https://neo4j.com/) for the graph database
- [Groq](https://groq.com/) for high-performance LLM inference
- [Hugging Face](https://huggingface.co/) for the sentence-transformers
