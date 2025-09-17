"""
Document processing module for RAGveda application.

Handles CSV loading, validation, and document creation.
"""

import pandas as pd
from typing import List, Optional
from pathlib import Path
import logging

from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .utils import detect_dataset_type, validate_columns, format_content

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading and processing operations."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.dataset_name = None
        self.processed_data = None
    
    def load_csv_dataset(self, file_path: str, dataset_type: str = "auto") -> pd.DataFrame:
        """
        Load and validate CSV dataset.
        
        Args:
            file_path: Path to the CSV file
            dataset_type: Type of dataset ("gita", "pys", or "auto" for auto-detection)
            
        Returns:
            Loaded and validated DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
            
            # Auto-detect dataset type if needed
            if dataset_type == "auto":
                dataset_type = detect_dataset_type(df.columns.tolist())
            
            # Validate required columns
            missing_cols = validate_columns(df.columns.tolist(), dataset_type)
            if missing_cols:
                raise ValueError(f"Missing required columns (case-insensitive): {missing_cols}")
            
            self.dataset_name = dataset_type
            logger.info(f"Dataset type detected/set as: {dataset_type}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV dataset: {str(e)}")
            raise
    
    def process_documents(self, df: pd.DataFrame, chunk_size: int = 1000, 
                         chunk_overlap: int = 200, content_column: str = "translation") -> List[Document]:
        """
        Process DataFrame into LangChain documents.
        
        Args:
            df: Input DataFrame
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            content_column: Name of the column to use as main content
            
        Returns:
            List of processed Document objects
        """
        try:
            # Validate content column exists
            if content_column not in df.columns:
                raise ValueError(f"Content column '{content_column}' not found. Available: {list(df.columns)}")
            
            # Create formatted content
            df["content"] = df.apply(
                lambda row: format_content(row.to_dict(), self.dataset_name, content_column), 
                axis=1
            )
            
            # Load documents using DataFrameLoader
            loader = DataFrameLoader(df, page_content_column="content")
            documents = list(loader.lazy_load())
            
            # Optional: Split documents if they're too long
            if chunk_size and chunk_size > 0:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                )
                documents = text_splitter.split_documents(documents)
            
            self.processed_data = df
            logger.info(f"Processed {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise
    
    def get_dataset_type(self) -> Optional[str]:
        """Get the current dataset type."""
        return self.dataset_name
    
    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get the processed DataFrame."""
        return self.processed_data
