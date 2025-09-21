"""
Document Processor Module.
Handles CSV loading, document creation, chunking, and embedding generation.
"""

import pandas as pd
import re
from typing import List, Optional, Tuple
from pathlib import Path
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from modules.config import Config


class DocumentProcessor:
    """Processes CSV files and creates document chunks for embedding."""
    
    def __init__(self):
        """Initialize document processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " "]
        )
    
    def load_csv(self, file_path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load CSV file into DataFrame.
        
        Args:
            file_path: Path to CSV file
            usecols: Optional list of columns to use
            
        Returns:
            DataFrame with loaded data
        """
        return pd.read_csv(file_path, usecols=usecols)
    
    def create_content_column(self, df: pd.DataFrame, content_column: str) -> pd.DataFrame:
        """
        Create formatted content column from dataframe.
        
        Args:
            df: Input dataframe
            content_column: Main content column name
            
        Returns:
            DataFrame with added content column
        """
        # Try to normalize chapter and verse columns (case-insensitive, synonyms)
        def find_col(possible_names):
            cols = {c.lower(): c for c in df.columns}
            for name in possible_names:
                if name in cols:
                    return cols[name]
            return None

        # Known aliases
        chapter_aliases = [
            'chapter', 'chapter/valli', 'chapter_valli', 'adhyaya', 'canto', 'book', 'section', 'valli'
        ]
        verse_aliases = [
            'verse', 'shloka', 'śloka', 'sloka', 'sutra', 'mantra'
        ]

        # Build lowercase map once
        lower_map = {c.lower(): c for c in df.columns}
        chapter_col = None
        for alias in chapter_aliases:
            if alias in lower_map:
                chapter_col = lower_map[alias]
                break
        verse_col = None
        for alias in verse_aliases:
            if alias in lower_map:
                verse_col = lower_map[alias]
                break

        # Create standardized 'chapter' and 'verse' columns if missing
        def parse_first_int(val):
            try:
                s = str(val)
                m = re.search(r"(\d+)", s)
                return int(m.group(1)) if m else None
            except Exception:
                return None

        if 'chapter' not in df.columns and chapter_col is not None:
            df['chapter'] = df[chapter_col].apply(parse_first_int)
        if 'verse' not in df.columns and verse_col is not None:
            df['verse'] = df[verse_col].apply(parse_first_int)

        # Build content string
        if 'chapter' in df.columns and 'verse' in df.columns:
            df['content'] = (
                "Chapter " + df['chapter'].astype(str) +
                " Verse " + df['verse'].astype(str) +
                " — " + df[content_column].astype(str).str.strip()
            )
        else:
            df['content'] = df[content_column].astype(str).str.strip()
        
        return df
    
    def create_documents(self, df: pd.DataFrame, filename: str) -> List[Document]:
        """
        Create Document objects from DataFrame.
        
        Args:
            df: DataFrame with content column
            filename: Source filename
            
        Returns:
            List of Document objects
        """
        loader = DataFrameLoader(
            df,
            page_content_column="content",
        )
        
        docs = list(loader.lazy_load())
        
        # Add filename and preserve metadata
        for i, doc in enumerate(docs):
            doc.metadata['filename'] = filename
            # Preserve all columns as metadata
            for col in df.columns:
                if col != 'content' and i < len(df):
                    doc.metadata[col] = df.iloc[i][col]
        
        return docs
    
    def group_documents(self, docs: List[Document], group_size: int = Config.GROUP_SIZE) -> List[Document]:
        """
        Group documents into larger chunks.
        
        Args:
            docs: List of documents to group
            group_size: Number of documents per group
            
        Returns:
            List of grouped documents
        """
        grouped_docs = []
        
        for i in range(0, len(docs), group_size):
            group = docs[i:i + group_size]
            content = "\n\n".join(d.page_content for d in group)
            
            # Aggregate metadata
            metadata = {
                "row_start": i,
                "row_end": i + len(group) - 1,
                "filename": group[0].metadata.get("filename")
            }
            
            # Collect chapters and verses if available
            chapters = []
            verses = []
            for d in group:
                if 'chapter' in d.metadata and d.metadata['chapter'] is not None:
                    try:
                        chapters.append(int(d.metadata['chapter']))
                    except (ValueError, TypeError):
                        pass
                if 'verse' in d.metadata and d.metadata['verse'] is not None:
                    try:
                        verses.append(int(d.metadata['verse']))
                    except (ValueError, TypeError):
                        pass
            
            if chapters:
                metadata['chapters'] = chapters
            if verses:
                metadata['verses'] = verses
            
            grouped_docs.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )
        
        return grouped_docs
    
    def split_documents(self, docs: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks if needed.
        
        Args:
            docs: List of documents to split
            
        Returns:
            List of split documents
        """
        final_docs = self.text_splitter.split_documents(docs)
        
        # Preserve filename metadata after splitting
        for doc in final_docs:
            if 'filename' not in doc.metadata and docs:
                doc.metadata['filename'] = docs[0].metadata.get('filename')
        
        return final_docs
    
    def process_csv_to_chunks(
        self, 
        file_path: str, 
        content_column: str,
        filename: Optional[str] = None
    ) -> Tuple[List[Document], pd.DataFrame]:
        """
        Complete pipeline to process CSV into document chunks.
        
        Args:
            file_path: Path to CSV file
            content_column: Column to use as main content
            filename: Optional filename override
            
        Returns:
            Tuple of (processed documents, original dataframe)
        """
        if filename is None:
            filename = Path(file_path).name
        
        # Load CSV
        df = self.load_csv(file_path)
        
        # Create content column
        df = self.create_content_column(df, content_column)
        
        # Create documents
        docs = self.create_documents(df, filename)
        
        # Group documents
        grouped_docs = self.group_documents(docs)
        
        # Split if needed
        final_docs = self.split_documents(grouped_docs)
        
        return final_docs, df
