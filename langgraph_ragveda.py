"""
RAGVeda LangGraph Implementation.
Converts the RAGVeda application to use LangGraph for orchestration.
"""

from typing import Literal, List, Dict, Any, Optional
from typing_extensions import TypedDict, Annotated
import operator
import tempfile
from pathlib import Path
import pandas as pd

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

from modules.config import Config
from modules.neo4j_manager import Neo4jManager
from modules.document_processor import DocumentProcessor
from modules.retrieval import Retrieval
from modules.llm_chain import LLMChain
from modules.memory_manager import MemoryManager


# ==================== STATE DEFINITION ====================

class RAGVedaState(TypedDict):
    """Global state that flows through all nodes."""
    
    # User interaction
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    
    # File processing
    uploaded_file_path: str
    filename: str
    content_column: str
    
    # Document processing
    raw_dataframe: Optional[pd.DataFrame]
    processed_documents: List[Document]
    grouped_documents: List[Document]
    final_documents: List[Document]
    embeddings_created: bool
    index_name: str
    
    # Retrieval and context
    query_length: int
    needs_rewrite: bool
    rewritten_query: str
    retrieved_docs: List[Document]
    formatted_context: str
    
    # Memory management
    memory_enabled: bool
    memory_context: str
    conversation_summary: str
    
    # Response generation
    llm_response: Dict[str, Any]
    references: List[str]
    
    # Configuration and status
    api_config_valid: bool
    neo4j_connected: bool
    current_step: str
    error_message: Optional[str]
    top_k: int
    
    # Service instances (persistent across invocations)
    neo4j_manager: Optional[Neo4jManager]
    document_processor: Optional[DocumentProcessor]
    llm_chain: Optional[LLMChain]
    memory_manager: Optional[MemoryManager]


# ==================== CONFIGURATION NODES ====================

def check_api_config(state: RAGVedaState) -> Dict[str, Any]:
    """Check if API configuration is set."""
    config_errors = Config.validate()
    
    return {
        "api_config_valid": len(config_errors) == 0,
        "error_message": ", ".join(config_errors) if config_errors else None,
        "current_step": "config_checked"
    }


def initialize_services(state: RAGVedaState) -> Dict[str, Any]:
    """Initialize all services (Neo4j, LLM, Memory, etc.)."""
    try:
        # Initialize Neo4j Manager
        neo4j_manager = Neo4jManager()
        
        # Initialize Document Processor
        document_processor = DocumentProcessor()
        
        # Initialize LLM Chain
        llm_chain = LLMChain()
        
        # Initialize Memory Manager if enabled
        memory_manager = MemoryManager() if Config.MEMORY_ENABLED else None
        
        return {
            "neo4j_manager": neo4j_manager,
            "document_processor": document_processor,
            "llm_chain": llm_chain,
            "memory_manager": memory_manager,
            "memory_enabled": Config.MEMORY_ENABLED,
            "neo4j_connected": True,
            "current_step": "services_initialized",
            "error_message": None
        }
    except Exception as e:
        return {
            "neo4j_connected": False,
            "error_message": f"Service initialization failed: {str(e)}",
            "current_step": "initialization_failed"
        }


# ==================== DOCUMENT PROCESSING NODES ====================

def load_csv_file(state: RAGVedaState) -> Dict[str, Any]:
    """Load CSV file and prepare for processing."""
    processor = state.get("document_processor") or DocumentProcessor()
    
    try:
        df = processor.load_csv(state["uploaded_file_path"])
        
        return {
            "raw_dataframe": df,
            "current_step": "csv_loaded",
            "error_message": None
        }
    except Exception as e:
        return {
            "error_message": f"Failed to load CSV: {str(e)}",
            "current_step": "csv_load_failed"
        }


def parse_csv_content(state: RAGVedaState) -> Dict[str, Any]:
    """Parse CSV and create formatted content with metadata."""
    processor = state.get("document_processor") or DocumentProcessor()
    
    try:
        # Create formatted content column
        df = processor.create_content_column(
            state["raw_dataframe"], 
            state["content_column"]
        )
        
        # Create document objects
        docs = processor.create_documents(df, state["filename"])
        
        return {
            "processed_documents": docs,
            "current_step": "csv_parsed",
            "error_message": None
        }
    except Exception as e:
        return {
            "error_message": f"Failed to parse CSV: {str(e)}",
            "current_step": "csv_parse_failed"
        }


def group_documents(state: RAGVedaState) -> Dict[str, Any]:
    """Group documents into larger chunks."""
    processor = state.get("document_processor") or DocumentProcessor()
    
    try:
        grouped_docs = processor.group_documents(
            state["processed_documents"],
            Config.GROUP_SIZE
        )
        
        return {
            "grouped_documents": grouped_docs,
            "current_step": "documents_grouped",
            "error_message": None
        }
    except Exception as e:
        return {
            "error_message": f"Failed to group documents: {str(e)}",
            "current_step": "grouping_failed"
        }


def split_into_chunks(state: RAGVedaState) -> Dict[str, Any]:
    """Split documents into smaller chunks for embedding."""
    processor = state.get("document_processor") or DocumentProcessor()
    
    try:
        final_docs = processor.split_documents(state["grouped_documents"])
        
        return {
            "final_documents": final_docs,
            "current_step": "documents_chunked",
            "error_message": None
        }
    except Exception as e:
        return {
            "error_message": f"Failed to split documents: {str(e)}",
            "current_step": "splitting_failed"
        }


def generate_embeddings(state: RAGVedaState) -> Dict[str, Any]:
    """Generate embeddings and store in Neo4j."""
    neo4j_manager = state.get("neo4j_manager") or Neo4jManager()
    
    try:
        index_name = Config.get_index_name(state["filename"])
        
        neo4j_manager.create_vector_store(
            state["final_documents"],
            index_name
        )
        
        return {
            "embeddings_created": True,
            "index_name": index_name,
            "current_step": "embeddings_generated",
            "error_message": None
        }
    except Exception as e:
        return {
            "embeddings_created": False,
            "error_message": f"Failed to generate embeddings: {str(e)}",
            "current_step": "embedding_failed"
        }


def create_file_relationships(state: RAGVedaState) -> Dict[str, Any]:
    """Create file-chunk relationships in Neo4j."""
    neo4j_manager = state.get("neo4j_manager") or Neo4jManager()
    
    try:
        neo4j_manager.create_file_relationships(state["filename"])
        
        return {
            "current_step": "relationships_created",
            "error_message": None
        }
    except Exception as e:
        return {
            "error_message": f"Failed to create relationships: {str(e)}",
            "current_step": "relationship_failed"
        }


# ==================== QUERY PROCESSING NODES ====================

def check_query_complexity(state: RAGVedaState) -> Dict[str, Any]:
    """Check if query needs rewriting based on complexity."""
    query_length = len(state["user_query"].split())
    
    return {
        "query_length": query_length,
        "needs_rewrite": query_length > 5,
        "rewritten_query": state["user_query"],  # Default to original
        "current_step": "query_complexity_checked"
    }


def rewrite_query(state: RAGVedaState) -> Dict[str, Any]:
    """Rewrite complex queries for better retrieval."""
    llm_chain = state.get("llm_chain") or LLMChain()
    
    try:
        # Convert messages to chat history format
        chat_history = []
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                chat_history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                chat_history.append({"role": "assistant", "content": msg.content})
        
        rewrite_result = llm_chain.rewrite_query(
            state["user_query"],
            chat_history=chat_history,
            source_name=state.get("filename", "uploaded dataset"),
            fallback_on_error=True
        )
        
        return {
            "rewritten_query": rewrite_result["rewritten_query"],
            "current_step": "query_rewritten",
            "error_message": None
        }
    except Exception as e:
        # Fallback to original query on error
        return {
            "rewritten_query": state["user_query"],
            "current_step": "query_rewrite_failed",
            "error_message": f"Query rewrite failed: {str(e)}"
        }


def retrieve_documents(state: RAGVedaState) -> Dict[str, Any]:
    """Retrieve relevant documents from Neo4j."""
    neo4j_manager = state.get("neo4j_manager") or Neo4jManager()
    
    try:
        # Connect to existing index if needed
        if state.get("index_name") and not neo4j_manager.current_index:
            neo4j_manager.connect_to_existing_index(state["index_name"])
        
        docs = neo4j_manager.retrieve_with_filename_filter(
            state["rewritten_query"],
            state["filename"],
            state.get("top_k", Config.DEFAULT_TOP_K)
        )
        
        return {
            "retrieved_docs": docs,
            "current_step": "documents_retrieved",
            "error_message": None
        }
    except Exception as e:
        return {
            "retrieved_docs": [],
            "error_message": f"Retrieval failed: {str(e)}",
            "current_step": "retrieval_failed"
        }


def format_context(state: RAGVedaState) -> Dict[str, Any]:
    """Format retrieved documents into context string."""
    context = Retrieval.format_context(
        state["retrieved_docs"],
        Config.MAX_CONTEXT_CHARS
    )
    
    return {
        "formatted_context": context,
        "current_step": "context_formatted"
    }


def handle_no_results(state: RAGVedaState) -> Dict[str, Any]:
    """Handle case when no documents are retrieved."""
    return {
        "llm_response": {
            "text": "No relevant content found in the current dataset.",
            "refrence": []
        },
        "references": [],
        "current_step": "no_results"
    }


# ==================== MEMORY MANAGEMENT NODES ====================

def get_memory_context(state: RAGVedaState) -> Dict[str, Any]:
    """Get conversation memory context if available."""
    memory_manager = state.get("memory_manager")
    
    if memory_manager and memory_manager.is_available():
        memory_context = memory_manager.get_memory_context()
        return {
            "memory_context": memory_context,
            "current_step": "memory_retrieved"
        }
    
    return {
        "memory_context": "",
        "current_step": "memory_skipped"
    }


def save_to_memory(state: RAGVedaState) -> Dict[str, Any]:
    """Save conversation turn to memory."""
    memory_manager = state.get("memory_manager")
    
    if memory_manager and memory_manager.is_available():
        response_text = state["llm_response"].get("text", "")
        memory_manager.save_conversation_turn(
            state["user_query"],
            response_text
        )
    
    return {
        "current_step": "memory_saved"
    }


# ==================== RESPONSE GENERATION NODES ====================

def generate_llm_response(state: RAGVedaState) -> Dict[str, Any]:
    """Generate response using LLM with retrieved context."""
    llm_chain = state.get("llm_chain") or LLMChain()
    
    try:
        response = llm_chain.graph_qa_chain(
            state["user_query"],
            state.get("retrieved_docs", []),
            source_name=state.get("filename", "uploaded dataset"),
            memory_context=state.get("memory_context", ""),
            fallback_on_error=True
        )
        
        return {
            "llm_response": response,
            "current_step": "response_generated",
            "error_message": None
        }
    except Exception as e:
        return {
            "llm_response": {
                "text": f"Error generating response: {str(e)}",
                "refrence": []
            },
            "error_message": f"LLM error: {str(e)}",
            "current_step": "response_failed"
        }


def filter_references(state: RAGVedaState) -> Dict[str, Any]:
    """Filter references by similarity threshold."""
    threshold = Config.MIN_SIMILARITY_FOR_REFERENCES
    filtered_docs = []
    
    for doc in state.get("retrieved_docs", []):
        score = doc.metadata.get("score", 1.0) if hasattr(doc, 'metadata') else 1.0
        if score >= threshold:
            filtered_docs.append(doc)
    
    refs = Retrieval.topk_refs_from_docs(filtered_docs, len(filtered_docs))
    
    return {
        "references": refs,
        "current_step": "references_filtered"
    }


# ==================== CONDITIONAL EDGE FUNCTIONS ====================

def should_initialize_services(state: RAGVedaState) -> Literal["initialize_services", "process_ready"]:
    """Route based on API configuration status."""
    if not state.get("api_config_valid", False):
        return "initialize_services"
    if not state.get("neo4j_connected", False):
        return "initialize_services"
    return "process_ready"


def should_rewrite_query(state: RAGVedaState) -> Literal["rewrite_query", "retrieve_documents"]:
    """Route based on query complexity."""
    return "rewrite_query" if state.get("needs_rewrite", False) else "retrieve_documents"


def has_results(state: RAGVedaState) -> Literal["format_context", "handle_no_results"]:
    """Route based on retrieval results."""
    return "format_context" if state.get("retrieved_docs") else "handle_no_results"


def should_use_memory(state: RAGVedaState) -> Literal["generate_llm_response"]:
    """Proceed to generate LLM response after getting memory."""
    return "generate_llm_response"


# ==================== GRAPH BUILDERS ====================

def build_document_processing_graph() -> StateGraph:
    """Build the document processing subgraph."""
    builder = StateGraph(RAGVedaState)
    
    # Add nodes
    builder.add_node("load_csv_file", load_csv_file)
    builder.add_node("parse_csv_content", parse_csv_content)
    builder.add_node("group_documents", group_documents)
    builder.add_node("split_into_chunks", split_into_chunks)
    builder.add_node("generate_embeddings", generate_embeddings)
    builder.add_node("create_file_relationships", create_file_relationships)
    
    # Add edges
    builder.add_edge(START, "load_csv_file")
    builder.add_edge("load_csv_file", "parse_csv_content")
    builder.add_edge("parse_csv_content", "group_documents")
    builder.add_edge("group_documents", "split_into_chunks")
    builder.add_edge("split_into_chunks", "generate_embeddings")
    builder.add_edge("generate_embeddings", "create_file_relationships")
    builder.add_edge("create_file_relationships", END)
    
    return builder.compile()


def build_query_processing_graph() -> StateGraph:
    """Build the query processing and response generation subgraph."""
    builder = StateGraph(RAGVedaState)
    
    # Add nodes
    builder.add_node("check_query_complexity", check_query_complexity)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("retrieve_documents", retrieve_documents)
    builder.add_node("format_context", format_context)
    builder.add_node("handle_no_results", handle_no_results)
    builder.add_node("get_memory_context", get_memory_context)
    builder.add_node("generate_llm_response", generate_llm_response)
    builder.add_node("filter_references", filter_references)
    builder.add_node("save_to_memory", save_to_memory)
    
    # Add edges
    builder.add_edge(START, "check_query_complexity")
    
    # Conditional routing for query rewriting
    builder.add_conditional_edges(
        "check_query_complexity",
        should_rewrite_query,
        ["rewrite_query", "retrieve_documents"]
    )
    builder.add_edge("rewrite_query", "retrieve_documents")
    
    # Conditional routing based on retrieval results
    builder.add_conditional_edges(
        "retrieve_documents",
        has_results,
        ["format_context", "handle_no_results"]
    )
    
    # Memory path - simplified flow
    builder.add_edge("format_context", "get_memory_context")
    builder.add_edge("get_memory_context", "generate_llm_response")
    
    # Response generation
    builder.add_edge("generate_llm_response", "filter_references")
    builder.add_edge("filter_references", "save_to_memory")
    
    # No results path
    builder.add_edge("handle_no_results", END)
    
    # End after saving memory
    builder.add_edge("save_to_memory", END)
    
    return builder.compile()


def build_complete_ragveda_graph() -> StateGraph:
    """Build the complete RAGVeda LangGraph."""
    builder = StateGraph(RAGVedaState)
    
    # Configuration nodes
    builder.add_node("check_api_config", check_api_config)
    builder.add_node("initialize_services", initialize_services)
    builder.add_node("process_ready", lambda x: {"current_step": "ready"})
    
    # Document processing nodes
    builder.add_node("load_csv_file", load_csv_file)
    builder.add_node("parse_csv_content", parse_csv_content)
    builder.add_node("group_documents", group_documents)
    builder.add_node("split_into_chunks", split_into_chunks)
    builder.add_node("generate_embeddings", generate_embeddings)
    builder.add_node("create_file_relationships", create_file_relationships)
    
    # Query processing nodes
    builder.add_node("check_query_complexity", check_query_complexity)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("retrieve_documents", retrieve_documents)
    builder.add_node("format_context", format_context)
    builder.add_node("handle_no_results", handle_no_results)
    builder.add_node("get_memory_context", get_memory_context)
    builder.add_node("generate_llm_response", generate_llm_response)
    builder.add_node("filter_references", filter_references)
    builder.add_node("save_to_memory", save_to_memory)
    
    # Configuration flow
    builder.add_edge(START, "check_api_config")
    builder.add_conditional_edges(
        "check_api_config",
        should_initialize_services,
        ["initialize_services", "process_ready"]
    )
    builder.add_edge("initialize_services", "process_ready")
    
    # The rest of the flow depends on the operation type
    # This would need to be handled by the application logic
    
    return builder.compile()


# ==================== MAIN EXECUTION FUNCTIONS ====================

def process_document(file_path: str, filename: str, content_column: str) -> Dict[str, Any]:
    """Process a document through the document processing graph."""
    graph = build_document_processing_graph()
    
    initial_state = {
        "uploaded_file_path": file_path,
        "filename": filename,
        "content_column": content_column,
        "messages": []
    }
    
    result = graph.invoke(initial_state)
    return result


def process_query(query: str, filename: str, top_k: int = 5) -> Dict[str, Any]:
    """Process a query through the query processing graph."""
    graph = build_query_processing_graph()
    
    initial_state = {
        "user_query": query,
        "filename": filename,
        "top_k": top_k,
        "messages": []
    }
    
    result = graph.invoke(initial_state)
    return result


if __name__ == "__main__":
    # Example: visualize the complete graph
    graph = build_complete_ragveda_graph()
    
    # Generate Mermaid diagram
    print("Complete RAGVeda LangGraph Structure:")
    print(graph.get_graph().draw_mermaid())
