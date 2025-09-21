"""
Configuration module for RAGVeda application.
Centralized configuration for all application settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration."""
    
    # Neo4j Configuration
    NEO4J_URL = os.getenv("NEO4J_URL")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
    # LLM Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL = "gemma2-9b-it"
    LLM_TEMPERATURE = 0
    
    # Embedding Configuration
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE = 'cuda' if os.getenv("USE_GPU", "false").lower() == "true" else 'cpu'
    
    # Document Processing Configuration
    GROUP_SIZE = 6  # Number of documents to group together
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 200
    
    # Retrieval Configuration
    DEFAULT_TOP_K = 8
    MAX_CONTEXT_CHARS = 6000
    
    # Neo4j Index Configuration
    NODE_LABEL = "Chunk"
    TEXT_PROPERTY = "text"
    EMBEDDING_PROPERTY = "embedding"
    
    # Application Settings
    APP_TITLE = "RAGVeda - Neo4j Chatbot"
    APP_ICON = "🕉️"
    
    @classmethod
    def validate(cls):
        """Validate required configuration."""
        errors = []
        
        if not cls.NEO4J_URL:
            errors.append("NEO4J_URL is not set")
        if not cls.NEO4J_PASSWORD:
            errors.append("NEO4J_PASSWORD is not set")
        if not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is not set")
            
        return errors
    
    @classmethod
    def get_index_name(cls, filename: str) -> str:
        """Generate index name from filename."""
        base_name = Path(filename).stem
        return f"{base_name.replace(' ', '_').lower()}_idx"
