"""
Neo4j Manager Module.
Handles all Neo4j database operations including connection, vector store creation, and queries.
"""

from typing import List, Dict, Any, Optional
import time
from langchain_neo4j import Neo4jVector
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from modules.config import Config


class Neo4jManager:
    """Manages Neo4j database connections and operations."""
    
    def __init__(self, embedding_model: Optional[HuggingFaceEmbeddings] = None):
        """
        Initialize Neo4j manager.
        
        Args:
            embedding_model: Optional embedding model instance
        """
        self.embedding_model = embedding_model or self._create_embedding_model()
        self.vectorstore = None
        self.current_index = None
        
    def _create_embedding_model(self) -> HuggingFaceEmbeddings:
        """Create and return embedding model instance."""
        return HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': Config.EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def create_vector_store(self, documents: List[Document], index_name: str) -> Neo4jVector:
        """
        Create Neo4j vector store from documents.
        
        Args:
            documents: List of documents to store
            index_name: Name for the vector index
            
        Returns:
            Neo4jVector instance
        """
        self.vectorstore = Neo4jVector.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            url=Config.NEO4J_URL,
            username=Config.NEO4J_USERNAME,
            password=Config.NEO4J_PASSWORD,
            index_name=index_name,
            node_label=Config.NODE_LABEL,
            text_node_property=Config.TEXT_PROPERTY,
            embedding_node_property=Config.EMBEDDING_PROPERTY,
        )
        self.current_index = index_name
        return self.vectorstore
    
    def connect_to_existing_index(self, index_name: str) -> Neo4jVector:
        """
        Connect to an existing Neo4j vector index.
        
        Args:
            index_name: Name of existing index
            
        Returns:
            Neo4jVector instance
        """
        self.vectorstore = Neo4jVector.from_existing_index(
            embedding=self.embedding_model,
            url=Config.NEO4J_URL,
            username=Config.NEO4J_USERNAME,
            password=Config.NEO4J_PASSWORD,
            index_name=index_name,
            node_label=Config.NODE_LABEL,
            text_node_property=Config.TEXT_PROPERTY,
            embedding_node_property=Config.EMBEDDING_PROPERTY,
        )
        self.current_index = index_name
        return self.vectorstore
    
    def create_file_relationships(self, filename: str):
        """
        Create File node and relationships to chunks.
        
        Args:
            filename: Name of the file
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        # Create constraint for unique file names
        self.vectorstore.query(
            "CREATE CONSTRAINT file_name_unique IF NOT EXISTS FOR (f:File) REQUIRE f.name IS UNIQUE"
        )
        
        # Set filename for any chunks that might not have it
        self.vectorstore.query(f"""
            MATCH (c:Chunk) 
            WHERE c.filename IS NULL 
            SET c.filename = '{filename}'
        """)
        
        # Create File node and connect chunks
        self.vectorstore.query(f"""
            MATCH (c:Chunk) 
            WHERE c.filename IS NOT NULL
            MERGE (f:File {{name: c.filename}})
            MERGE (f)-[:HAS_CHUNK]->(c)
        """)
    
    def retrieve_with_filename_filter(
        self, 
        question: str, 
        filename: str, 
        top_k: int = Config.DEFAULT_TOP_K
    ) -> List[Document]:
        """
        Retrieve documents filtered by filename using Cypher query.
        
        Args:
            question: User query
            filename: Filename to filter by
            top_k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            return []
        
        # Get embedding for the query
        query_embedding = self.embedding_model.embed_query(question)
        
        # Cypher query to retrieve only chunks from current file
        cypher_query = """
        MATCH (c:Chunk {filename: $filename})
        WITH c, vector.similarity.cosine(c.embedding, $embedding) AS score
        ORDER BY score DESC
        LIMIT $k
        RETURN c.text AS text, c.filename AS filename, 
               c.chapter AS chapter, c.verse AS verse, 
               c.chapters AS chapters, c.verses AS verses,
               c.row_start AS row_start, c.row_end AS row_end,
               score
        """
        
        # Execute the query with a single reconnect-and-retry on connection errors
        try:
            results = self.vectorstore.query(
                cypher_query,
                params={
                    "filename": filename,
                    "embedding": query_embedding,
                    "k": top_k
                }
            )
        except Exception as e:
            # Attempt to reconnect to the current index and retry once
            try:
                if self.current_index:
                    self.connect_to_existing_index(self.current_index)
                    time.sleep(0.25)
                results = self.vectorstore.query(
                    cypher_query,
                    params={
                        "filename": filename,
                        "embedding": query_embedding,
                        "k": top_k
                    }
                )
            except Exception as e2:
                print(f"Neo4j query failed after reconnect: {e2}")
                return []
        
        # Convert results to Document objects
        docs = []
        for result in results:
            metadata = {
                "filename": result.get("filename"),
                "chapter": result.get("chapter"),
                "verse": result.get("verse"),
                "chapters": result.get("chapters"),
                "verses": result.get("verses"),
                "row_start": result.get("row_start"),
                "row_end": result.get("row_end"),
                "score": result.get("score")
            }
            # Remove None values from metadata
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            docs.append(Document(
                page_content=result.get("text", ""),
                metadata=metadata
            ))
        
        return docs
    
    def query_cypher(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute arbitrary Cypher query.
        
        Args:
            query: Cypher query string
            params: Query parameters
            
        Returns:
            Query results
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.query(query, params=params or {})
