"""
Utility functions and configurations for RAGveda application.

Contains helper functions, constants, and common utilities used across modules.
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

def detect_dataset_type(columns: List[str]) -> str:
    """
    Auto-detect dataset type based on column names.
    
    Args:
        columns: List of column names from DataFrame
        
    Returns:
        Detected dataset type ('gita', 'pys', or 'generic')
    """
    columns_lower = [col.lower() for col in columns]
    
    if "sanskrit" in columns_lower and "speaker" in columns_lower:
        return "gita"
    elif "sanskrit" in columns_lower and "question" in columns_lower:
        return "pys"
    else:
        return "generic"

def validate_columns(columns: List[str], dataset_type: str) -> List[str]:
    """
    Validate required columns based on dataset type.
    
    Args:
        columns: List of column names from DataFrame
        dataset_type: Type of dataset
        
    Returns:
        List of missing required columns
    """
    # Create case-insensitive column mapping
    columns_lower = {col.lower(): col for col in columns}
    
    if dataset_type == "gita":
        required_cols_lower = ["chapter", "verse", "translation"]
        if "question" in columns_lower:
            required_cols_lower.append("question")
    elif dataset_type == "pys":
        required_cols_lower = ["chapter", "verse", "translation", "question"]
    else:
        # Generic validation - at least translation column
        required_cols_lower = ["translation"]
    
    missing_cols = [col for col in required_cols_lower if col not in columns_lower]
    return missing_cols

def create_sample_query_suggestions(dataset_type: str) -> List[str]:
    """
    Create sample query suggestions based on dataset type.
    
    Args:
        dataset_type: Type of dataset
        
    Returns:
        List of sample queries
    """
    if dataset_type == "gita":
        return [
            "What is dharma according to Krishna?",
            "Tell me about Arjuna's dilemma",
            "What does the Gita say about duty?",
            "Explain the concept of karma yoga",
            "What is the nature of the soul?"
        ]
    elif dataset_type == "pys":
        return [
            "What is yoga according to Patanjali?",
            "How to control the mind?",
            "What are the eight limbs of yoga?",
            "Explain meditation techniques",
            "What is samadhi?"
        ]
    else:
        return [
            "Search for relevant content",
            "Find similar passages",
            "Explore key concepts"
        ]

def format_content(row: Dict[str, Any], dataset_type: str, content_column: str) -> str:
    """
    Format content based on dataset type and available columns.
    
    Args:
        row: DataFrame row as dictionary
        dataset_type: Type of dataset
        content_column: Name of the main content column
        
    Returns:
        Formatted content string
    """
    if dataset_type in ["gita", "pys"] and "chapter" in row and "verse" in row:
        return (
            f"Chapter {row['chapter']} Verse {row['verse']} — "
            f"{str(row[content_column]).strip()}"
        )
    else:
        return str(row[content_column]).strip()

def get_stats(documents: List[Any], dataset_type: str, embedding_model: str, 
              processed_data: Any, embeddings_shape: tuple = None) -> Dict[str, Any]:
    """
    Get statistics about the processed data.
    
    Args:
        documents: List of processed documents
        dataset_type: Type of dataset
        embedding_model: Name of embedding model used
        processed_data: Processed DataFrame
        embeddings_shape: Shape of embeddings array
        
    Returns:
        Dictionary containing various statistics
    """
    stats = {
        "num_documents": len(documents),
        "dataset_type": dataset_type,
        "embedding_model": embedding_model,
        "embeddings_generated": embeddings_shape is not None,
        "embedding_dimensions": embeddings_shape[1] if embeddings_shape else None
    }
    
    if processed_data is not None:
        stats.update({
            "num_rows": len(processed_data),
            "columns": list(processed_data.columns)
        })
    
    return stats
