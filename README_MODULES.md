# RAGVeda - Production Modular Architecture

A production-ready modular RAG application with Streamlit, Neo4j, and Groq LLM.

## 📁 Module Structure

```
modules/
├── config.py              # Configuration management
├── neo4j_manager.py       # Neo4j operations
├── document_processor.py  # CSV processing
├── retrieval.py          # Document retrieval
├── llm_chain.py          # LLM & QA chain
└── ui_components.py      # UI components

main.py                   # Application orchestrator
```

## 🚀 Quick Start

1. **Install dependencies**
```bash
pip install -r requirements_minimal.txt
```

2. **Configure .env**
```
NEO4J_URL=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
GROQ_API_KEY=your-groq-api-key
```

3. **Run application**
```bash
streamlit run main.py
```

## 🔑 Key Features

- **Modular Architecture**: Clean separation of concerns
- **File-Scoped RAG**: Each CSV creates isolated search scope
- **Neo4j Integration**: Vector storage with Cypher filtering
- **Groq LLM**: Fast inference with Gemma2-9b
- **ChatGPT UI**: Conversational interface with sources

## 📊 Data Format

CSV files with columns:
- `chapter`, `verse` (optional)
- Main content column (user-selectable)

## 🏗️ Architecture

1. **Document Processing**: CSV → Documents → Chunks → Embeddings
2. **Storage**: Neo4j AuraDB with vector index per file
3. **Retrieval**: Cypher-filtered similarity search
4. **Generation**: Groq LLM with JSON output parsing
5. **UI**: Streamlit chat interface with references
