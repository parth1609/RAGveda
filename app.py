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



def verify_neo4j_connection(URI, USERNAME, PASSWORD):
   
    try:
        with GraphDatabase.driver(URI, auth=("USERNAME","PASSWORD")) as driver:
            driver.verify_connectivity()
            print("Connection established.")
    
    except Exception as e:
        # Provide more detailed error information
        import traceback
        return False, f"Failed to connect to Neo4j: {type(e).__name__} - {str(e)}\n{traceback.format_exc()}"
        


def main():
    st.set_page_config(page_title="RAGVeda", layout="wide")
    
    st.title("RAGVeda")
    
    # Improved credentials handling
    try:
        # Get credentials from Streamlit secrets or environment variables
        neo4j_url = st.secrets["NEO4J_URI"] 
        neo4j_user = st.secrets["NEO4J_USERNAME"] 
        neo4j_password = st.secrets["NEO4J_PASSWORD"] 
        groq_api_key = st.secrets["GROQ_API_KEY"] 

        Connection = verify_neo4j_connection(neo4j_url, neo4j_user, neo4j_password)
        if Connection:
            st.success("successful")
        else:
            st.error("not succesful")

    except Exception as e:
        st.error(f"Configuration Error: {str(e)}")
    


if __name__ == "__main__":
    main()