"""
Query processing module for RAGveda application.

Handles query processing and similarity-based document retrieval.
"""

import numpy as np
from typing import List, Dict, Any
import logging

from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Handles query processing and document retrieval operations."""
    
    def __init__(self, embedding_handler, documents: List[Document]):
        """
        Initialize the query processor.
        
        Args:
            embedding_handler: EmbeddingHandler instance
            documents: List of processed documents
        """
        self.embedding_handler = embedding_handler
        self.documents = documents
    
    def query_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query documents using semantic similarity search.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of relevant documents with similarity scores
        """
        document_embeddings = self.embedding_handler.get_embeddings()
        
        if document_embeddings is None:
            raise ValueError("Document embeddings not generated. Generate embeddings first.")
        
        if not self.documents:
            raise ValueError("No documents available for querying.")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_handler.embed_query(query)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, document_embeddings)[0]
            
            # Get top-k most similar documents
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                doc = self.documents[idx]
                similarity_score = similarities[idx]
                
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(similarity_score),
                    "rank": len(results) + 1
                }
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} results for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            raise
    
    def generate_summary(self, query: str, results: List[Dict[str, Any]]) -> str:
        """
        Generate a comprehensive summary of the search results.
        
        Args:
            query: Original query string
            results: List of search results
            
        Returns:
            Generated summary text
        """
        if not results:
            return "No relevant results found for your query."
        
        # Get the most relevant results
        top_results = results[:3]
        
        # Extract key content and create a flowing summary
        summary_content = []
        
        for result in top_results:
            content = result['content']
            # Clean and extract meaningful content
            if '—' in content:
                # For structured content (Chapter X Verse Y — content)
                main_text = content.split('—', 1)[1].strip()
            else:
                main_text = content
            
            # Take first meaningful sentence or up to 150 characters
            if '.' in main_text:
                sentences = main_text.split('.')
                first_sentence = sentences[0].strip()
                if len(first_sentence) > 20:  # Ensure it's a meaningful sentence
                    summary_content.append(first_sentence)
            else:
                summary_content.append(main_text[:150].strip())
        
        # Create a cohesive summary
        if summary_content:
            summary_text = ". ".join(summary_content)
            if not summary_text.endswith('.'):
                summary_text += "."
            
            # Add context about additional results
            additional_info = ""
            if len(results) > 3:
                additional_info = f" This synthesis is based on the {len(results)} most relevant passages found in the spiritual texts."
                
            return f"In response to your query about '{query}': {summary_text}{additional_info}"
        else:
            return f"Found {len(results)} relevant passages related to '{query}'. The content discusses various aspects of this topic from the spiritual texts."
    
    def update_documents(self, documents: List[Document]):
        """Update the documents list."""
        self.documents = documents
