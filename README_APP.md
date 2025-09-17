# RAGveda - Spiritual Text Query System

A comprehensive GraphRAG (Retrieval-Augmented Generation) application built with Streamlit for querying spiritual texts using semantic search. This application converts the Jupyter notebook functionality into a user-friendly web interface.

## 🌟 Features

- **Intelligent Semantic Search**: Query spiritual texts using natural language
- **Multiple Dataset Support**: Upload custom CSV files or use existing datasets
- **Real-time Processing**: Dynamic document processing and embedding generation
- **Beautiful UI**: Modern, responsive Streamlit interface with custom styling
- **Query History**: Track and revisit previous searches
- **Similarity Scoring**: View relevance scores for search results
- **Auto-detection**: Automatically detect dataset types (Gita, Patanjali Yoga Sutras, etc.)

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- 4GB+ RAM (recommended for embedding generation)
- Internet connection (for downloading models on first run)

### Dependencies
All required packages are listed in `requirements.txt`. Key dependencies include:
- `streamlit` - Web application framework
- `langchain` - Document processing and text splitting
- `langchain-huggingface` - HuggingFace embeddings integration
- `sentence-transformers` - Semantic embeddings
- `pandas` - Data manipulation
- `scikit-learn` - Similarity calculations
- `numpy` - Numerical operations

## 🚀 Installation & Setup

### 1. Clone or Download
```bash
# If using git
git clone <repository-url>
cd RAGveda

# Or download and extract the files to your desired directory
```

### 2. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
# Test if all imports work
python -c "import streamlit, langchain, sentence_transformers; print('All dependencies installed successfully!')"
```

## 🎯 Usage

### Starting the Application
```bash
# Navigate to the RAGveda directory
cd c:\Users\parth\OneDrive\Desktop\one\RAGveda

# Run the Streamlit application
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using Existing Datasets

1. **Select Dataset Source**: Choose "Use Existing Dataset" in the sidebar
2. **Pick Dataset**: Select from available CSV files (Gita_questions.csv, PYS_English_Questions.csv)
3. **Process Dataset**: Click "🚀 Process Dataset" to load and generate embeddings
4. **Start Querying**: Use the query interface to ask questions

### Uploading Custom Datasets

1. **Select Upload Option**: Choose "Upload New Dataset" in the sidebar
2. **Upload CSV**: Select your CSV file with the required format
3. **Verify Format**: Check the preview to ensure correct columns
4. **Process**: Click "🚀 Process Dataset" to begin processing

### CSV Format Requirements

Your CSV file should contain these columns:

#### For Spiritual Texts (like Bhagavad Gita):
- `chapter` (required): Chapter number
- `verse` (required): Verse number  
- `translation` (required): English translation text
- `question` (optional): Related question
- `sanskrit` (optional): Original Sanskrit text
- `speaker` (optional): Who is speaking

#### For Yoga Sutras:
- `chapter` (required): Chapter number
- `verse` (required): Sutra number
- `translation` (required): English translation
- `question` (required): Related question
- `sanskrit` (optional): Original Sanskrit

#### Generic Format:
- `translation` (required): Main text content
- Additional columns will be preserved as metadata

### Sample Queries

#### For Bhagavad Gita:
- "What is dharma according to Krishna?"
- "Tell me about Arjuna's dilemma"
- "What does the Gita say about duty?"
- "Explain the concept of karma yoga"

#### For Patanjali Yoga Sutras:
- "What is yoga according to Patanjali?"
- "How to control the mind?"
- "What are the eight limbs of yoga?"
- "Explain meditation techniques"

## 🏗️ Architecture

### Core Components

1. **GraphRAGProcessor** (`graphrag_processor.py`)
   - Document loading and processing
   - Embedding generation using HuggingFace models
   - Semantic similarity search
   - Query processing and retrieval

2. **Streamlit App** (`app.py`)
   - User interface and interaction
   - File upload and dataset management
   - Query interface and results display
   - Session state management

### Processing Pipeline

1. **Data Loading**: CSV files are loaded and validated
2. **Document Processing**: Text is formatted and split into chunks
3. **Embedding Generation**: Documents are converted to vector embeddings
4. **Query Processing**: User queries are embedded and compared with documents
5. **Result Retrieval**: Most similar documents are returned with scores

### Embedding Model

- **Default Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Language**: English
- **Performance**: Balanced speed and accuracy for semantic search

## 🔧 Configuration

### Customizing Embedding Model

Edit `graphrag_processor.py` to change the embedding model:

```python
# In GraphRAGProcessor.__init__()
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"  # Higher accuracy
# or
embedding_model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # Faster
```

### Adjusting Chunk Size

Modify document processing parameters:

```python
# In process_documents() method
chunk_size = 1500  # Larger chunks for more context
chunk_overlap = 300  # More overlap for better continuity
```

### UI Customization

The Streamlit interface can be customized by modifying the CSS in `app.py`:

```python
# Edit the custom CSS in the st.markdown() call
st.markdown("""
<style>
    .main-header {
        color: #YOUR_COLOR;  # Change header color
    }
</style>
""", unsafe_allow_html=True)
```

## 📊 Performance Notes

### First Run
- Initial model download: ~100MB (sentence-transformers model)
- Embedding generation: 2-5 minutes for 700+ documents
- Subsequent runs: Much faster (embeddings cached in memory)

### Memory Usage
- Base application: ~200MB
- With embeddings loaded: ~500MB-1GB (depending on dataset size)
- Recommendation: 4GB+ RAM for smooth operation

### Query Speed
- Typical query response: <1 second
- Depends on dataset size and number of results requested

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install --upgrade -r requirements.txt
   ```

2. **Memory Issues**
   ```python
   # Reduce chunk size or use smaller embedding model
   chunk_size = 500  # Smaller chunks
   ```

3. **Slow Performance**
   ```python
   # Use faster embedding model
   embedding_model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
   ```

4. **CSV Format Issues**
   - Ensure CSV has required columns
   - Check for special characters or encoding issues
   - Verify data types (chapter/verse should be numeric)

### Error Messages

- **"Missing required columns"**: Check CSV format matches requirements
- **"Failed to initialize embeddings"**: Check internet connection and model availability
- **"No documents available"**: Ensure dataset processing completed successfully

## 🔮 Future Enhancements

### Planned Features
- **Multi-language Support**: Support for Sanskrit, Hindi, and other languages
- **Advanced Filtering**: Filter by chapter, speaker, or topic
- **Export Results**: Save search results to PDF or CSV
- **Batch Processing**: Process multiple datasets simultaneously
- **API Integration**: RESTful API for programmatic access

### Potential Improvements
- **Caching**: Persistent storage of embeddings
- **Advanced Search**: Boolean operators and phrase matching
- **Visualization**: Interactive charts and graphs
- **Authentication**: User accounts and personalized history

## 📝 License

This project is open source. Please refer to the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📞 Support

For questions or issues:
1. Check this documentation
2. Review the troubleshooting section
3. Check existing issues in the repository
4. Create a new issue with detailed information

---

**RAGveda** - Bridging Ancient Wisdom with Modern AI 🕉️
