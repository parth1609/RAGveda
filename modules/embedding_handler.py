"""
Embedding handler module for RAGveda application.

Handles embedding generation and similarity calculations.
"""

import numpy as np
from typing import List, Optional
import logging

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from .utils import DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingHandler:
    """Handles embedding generation and similarity operations."""
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initialize the embedding handler.
        
        Args:
            model_name: Name of the HuggingFace embedding model to use
        """
        self.model_name = model_name
        self.embeddings = None
        self.document_embeddings = None
        
        # Initialize embedding model
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the HuggingFace embeddings model."""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Initialized embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise
    
    def generate_embeddings(self, documents: List[Document]) -> np.ndarray:
        """
        Generate embeddings for the documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            NumPy array of document embeddings
        """
        if not documents:
            raise ValueError("No documents provided for embedding generation")
        
        try:
            # Extract text content from documents
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            embeddings_list = self.embeddings.embed_documents(texts)
            
            # Convert to numpy array
            self.document_embeddings = np.array(embeddings_list)
            logger.info(f"Generated embeddings with shape: {self.document_embeddings.shape}")
            
            return self.document_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string to embed
            
        Returns:
            NumPy array of query embedding
        """
        try:
            query_embedding = self.embeddings.embed_query(query)
            return np.array(query_embedding).reshape(1, -1)
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise
    
    def get_embeddings(self) -> Optional[np.ndarray]:
        """Get the stored document embeddings."""
        return self.document_embeddings
    
    def set_embeddings(self, embeddings: np.ndarray):
        """Set the document embeddings."""
        self.document_embeddings = embeddings
    
    def get_model_name(self) -> str:
        """Get the embedding model name."""
        return self.model_name
    
    def get_embedding_dimensions(self) -> Optional[int]:
        """Get the dimensionality of embeddings."""
        if self.document_embeddings is not None:
            return self.document_embeddings.shape[1]
        return None
