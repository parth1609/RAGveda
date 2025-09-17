"""
Main processor module for RAGveda application.

Orchestrates all components and provides a simple interface.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import pickle

from .document_processor import DocumentProcessor
from .embedding_handler import EmbeddingHandler
from .query_processor import QueryProcessor
from .utils import get_stats, create_sample_query_suggestions

logger = logging.getLogger(__name__)


class RAGProcessor:
    """
    Main processor that orchestrates all RAG operations.
    
    This is a simplified interface that manages document processing,
    embedding generation, and query processing.
    """
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the RAG processor.
        
        Args:
            embedding_model_name: Name of the HuggingFace embedding model
        """
        self.doc_processor = DocumentProcessor()
        self.embedding_handler = EmbeddingHandler(embedding_model_name)
        self.query_processor = None
        self.documents = []
        
        logger.info("RAG Processor initialized")
    
    def load_dataset(self, file_path: str, dataset_type: str = "auto") -> pd.DataFrame:
        """
        Load CSV dataset.
        
        Args:
            file_path: Path to CSV file
            dataset_type: Type of dataset
            
        Returns:
            Loaded DataFrame
        """
        return self.doc_processor.load_csv_dataset(file_path, dataset_type)
    
    def process_documents(self, df: pd.DataFrame, content_column: str = "translation",
                         chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Process documents from DataFrame.
        
        Args:
            df: Input DataFrame
            content_column: Main content column name
            chunk_size: Text chunk size
            chunk_overlap: Chunk overlap size
        """
        self.documents = self.doc_processor.process_documents(
            df, chunk_size, chunk_overlap, content_column
        )
        
        # Initialize query processor with documents
        self.query_processor = QueryProcessor(self.embedding_handler, self.documents)
    
    def generate_embeddings(self) -> np.ndarray:
        """Generate embeddings for all documents."""
        if not self.documents:
            raise ValueError("No documents to embed. Process documents first.")
        
        return self.embedding_handler.generate_embeddings(self.documents)
    
    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the document collection.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with scores
        """
        if not self.query_processor:
            raise ValueError("Documents not processed. Process documents first.")
        
        return self.query_processor.query_documents(query, top_k)
    
    def get_summary(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate summary from query results."""
        if not self.query_processor:
            raise ValueError("Query processor not initialized.")
        
        return self.query_processor.generate_summary(query, results)
    
    def get_sample_queries(self) -> List[str]:
        """Get sample queries based on dataset type."""
        dataset_type = self.doc_processor.get_dataset_type()
        return create_sample_query_suggestions(dataset_type or "generic")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        dataset_type = self.doc_processor.get_dataset_type()
        processed_data = self.doc_processor.get_processed_data()
        embedding_model = self.embedding_handler.get_model_name()
        embeddings = self.embedding_handler.get_embeddings()
        embeddings_shape = embeddings.shape if embeddings is not None else None
        
        return get_stats(
            self.documents, dataset_type, embedding_model, 
            processed_data, embeddings_shape
        )
    
    def save_data(self, save_path: str):
        """
        Save processed data to disk.
        
        Args:
            save_path: Directory to save data
        """
        try:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save documents
            with open(save_dir / "documents.pkl", "wb") as f:
                pickle.dump(self.documents, f)
            
            # Save embeddings
            embeddings = self.embedding_handler.get_embeddings()
            if embeddings is not None:
                np.save(save_dir / "embeddings.npy", embeddings)
            
            # Save processed DataFrame
            processed_data = self.doc_processor.get_processed_data()
            if processed_data is not None:
                processed_data.to_csv(save_dir / "processed_data.csv", index=False)
            
            # Save metadata
            stats = self.get_stats()
            with open(save_dir / "metadata.pkl", "wb") as f:
                pickle.dump(stats, f)
            
            logger.info(f"Data saved to {save_dir}")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def load_data(self, load_path: str):
        """
        Load previously saved data.
        
        Args:
            load_path: Directory to load data from
        """
        try:
            load_dir = Path(load_path)
            
            # Load documents
            with open(load_dir / "documents.pkl", "rb") as f:
                self.documents = pickle.load(f)
            
            # Load embeddings
            embeddings_path = load_dir / "embeddings.npy"
            if embeddings_path.exists():
                embeddings = np.load(embeddings_path)
                self.embedding_handler.set_embeddings(embeddings)
            
            # Load processed DataFrame
            processed_data_path = load_dir / "processed_data.csv"
            if processed_data_path.exists():
                processed_data = pd.read_csv(processed_data_path)
                self.doc_processor.processed_data = processed_data
            
            # Reinitialize query processor
            if self.documents:
                self.query_processor = QueryProcessor(self.embedding_handler, self.documents)
            
            logger.info(f"Data loaded from {load_dir}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
