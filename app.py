import os
from typing import List, Dict, Any
from datetime import datetime
import streamlit as st

from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import GraphQAChain
from langchain.prompts import PromptTemplate



def verify_neo4j_connection(url, username, password):
    """
    Verify Neo4j database connection with specialized Aura handling
    
    :param url: Neo4j database URL
    :param username: Neo4j username
    :param password: Neo4j password
    :return: Tuple of (success_status, message)
    """
    try:

        # Create driver without the encrypted parameter for neo4j+s
        driver = GraphDatabase.driver(
            url, 
            auth=(username, password),
            connection_timeout=10,  # Increased timeout
            max_connection_lifetime=3600,  # 1 hour max connection time
            connection_acquisition_timeout=5  # Increased acquisition timeout
        )
        
        # Verify actual connectivity with a simple query
        with driver.session(database="neo4j") as session:
            result = session.run("RETURN 1 AS test")
            result.consume()  # Explicitly consume the result
        
        driver.close()
        return True, f"Successfully connected to Neo4j database at {url}!"
    
    except Exception as e:
        # Provide more detailed error information
        import traceback
        return False, f"Failed to connect to Neo4j: {type(e).__name__} - {str(e)}\n{traceback.format_exc()}"
        
def _setup_qa_chain(self) -> GraphQAChain:
    """
    Setup the Graph QA Chain with custom prompts
    """
    # Custom prompt template for graph QA
    GRAPH_QA_TEMPLATE = """
    You are a knowledgeable assistant that answers questions based on the provided context from a knowledge graph.
    
    Context from the knowledge graph:
    {context}
    
    Question: {question}
    
    Please provide a detailed answer using the information from the knowledge graph.
    If you cannot find the exact information, explain what related information is available.
    
    Answer:"""

    prompt = PromptTemplate(
        template=GRAPH_QA_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    return GraphQAChain.from_llm(
        llm=self.llm,
        graph=self.graph,
        verbose=True,
        prompt=prompt
    )
    
def process_document(self, 
                    doc_path: str,
                    doc_type: str = "pdf") -> None:
    """
    Process document and create knowledge graph
    
    :param doc_path: Path to the document
    :param doc_type: Type of document (pdf, txt, etc.)
    """
    # Load and split document
    if doc_type.lower() == "pdf":
        loader = PyPDFLoader(doc_path)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = splitter.split_documents(documents)
        
        # Process each chunk and create graph nodes/relationships
        for i, doc in enumerate(docs):
            self._create_document_graph(doc, i)
    
def _create_document_graph(self, 
                            doc: Document, 
                            chunk_id: int) -> None:
    """
    Create graph structure from document chunk
    
    :param doc: Document chunk
    :param chunk_id: Unique identifier for the chunk
    """
    # Create Cypher query for document chunk
    cypher_query = """
    MERGE (c:Chunk {id: $chunk_id, content: $content})
    WITH c
    UNWIND $entities as entity
    MERGE (e:Entity {name: entity.name, type: entity.type})
    MERGE (c)-[:CONTAINS]->(e)
    """
    
    # Extract entities (This is a simplified version - you might want to use NER)
    # You can use spaCy or other NLP tools for better entity extraction
    entities = self._extract_entities(doc.page_content)
    
    # Execute Cypher query
    params = {
        "chunk_id": f"chunk_{chunk_id}",
        "content": doc.page_content,
        "entities": entities
    }
    
    self.graph.query(cypher_query, params=params)
    
def _extract_entities(self, text: str) -> List[Dict[str, str]]:
    """
    Extract entities from text (simplified version)
    In a production environment, use proper NER tools
    
    :param text: Text to extract entities from
    :return: List of extracted entities
    """
    # This is a placeholder - implement proper entity extraction
    # You might want to use spaCy or other NLP tools
    entities = []
    # Add example entity extraction logic here
    return entities
    
def query(self, question: str) -> str:
    """
    Query the knowledge graph
    
    :param question: User question
    :return: Answer based on graph context
    """
    try:
        # Get answer using the QA chain
        response = self.qa_chain.run(question)
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}"
    
def create_graph_visualization(self) -> str:
    """
    Generate Cypher query for graph visualization
    
    :return: Cypher query for visualization
    """
    return """
    MATCH (n)
    OPTIONAL MATCH (n)-[r]->(m)
    RETURN n, r, m LIMIT 100
    """


def main():
    st.set_page_config(page_title="RAGVeda", layout="wide")
    
    st.title("RAGVeda")
    
    # Improved credentials handling
    try:
        # Get credentials from Streamlit secrets or environment variables
        neo4j_url = st.secrets["NEO4J_URL"] 
        neo4j_user = st.secrets["NEO4J_USERNAME"] 
        neo4j_password = st.secrets["NEO4J_PASSWORD"] 
        gemini_api_key = st.secrets["GEMINI_API_KEY"] 

        # Validate that all required credentials are present
        if not all([neo4j_url, neo4j_user, neo4j_password, gemini_api_key]):
            st.error("Missing required credentials. Please check your Streamlit secrets or environment variables.")
            st.stop()

        # Verify Neo4j connection before proceeding
        with st.spinner("Verifying Neo4j connection..."):
            connection_success, message = verify_neo4j_connection(neo4j_url, neo4j_user, neo4j_password)
            
            if connection_success:
                st.success(message)
            else:
                st.error(message)
                st.stop()  # Stop execution if connection fails

        # Only proceed if connection is successful
        try:
            # Initialize graph_rag object
            graph = Neo4jGraph(
                url=neo4j_url,
                username=neo4j_user,
                password=neo4j_password
            )
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key)
            graph_rag = GraphRAG(graph=graph, llm=llm)  # Assuming GraphRAG is your custom class

            uploaded_files = st.file_uploader(
                "Upload Document (PDF)", 
                type=['pdf'],
                accept_multiple_files=True
            )
            
            for uploaded_file in uploaded_files:
                if uploaded_file:
                    with st.spinner("Processing document..."):
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        graph_rag.process_document(temp_path)
                        os.remove(temp_path)
                        
                    st.success("Document processed and graph created!")

            query = st.text_input("Ask a question about the document")
            if st.button("Get Answer") and query:
                with st.spinner("Searching knowledge graph..."):
                    response = graph_rag.query(query)
                    st.write("Answer:", response)

        except Exception as e:
            st.error(f"Error: {str(e)}")

    except Exception as e:
        st.error(f"Configuration Error: {str(e)}")
    


if __name__ == "__main__":
    main()