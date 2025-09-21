"""
LLM Chain Module.
Handles LLM operations, prompt templates, and QA chain execution.
"""

from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from modules.config import Config
from modules.retrieval import Retrieval


class LLMChain:
    """Manages LLM operations and QA chains."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM chain.
        
        Args:
            api_key: Optional API key override
        """
        self.api_key = api_key or Config.GROQ_API_KEY
        self.llm = None
        self.parser = JsonOutputParser()
        self.prompt_template = self._create_prompt_template()
        
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for QA.
        The prompt explicitly references the current dataset name via {source_name}.
        """
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. Use the provided context from {source_name} to answer "
             "the user's question accurately and concisely in 3–5 sentences. "
             "Do not include verse text in your answer. If the context is insufficient, say you are uncertain and ask for clarification. "
             "Follow the output format instructions exactly."),
            ("human",
             "Question:\n{question}\n\nContext:\n{context}\n\nOutput format:\n{format_instructions}")
        ])
    
    def get_llm(self) -> Optional[ChatGroq]:
        """
        Get or create LLM instance.
        
        Returns:
            ChatGroq instance or None if API key not available
        """
        if not self.api_key:
            return None
            
        if not self.llm:
            try:
                self.llm = ChatGroq(
                    model=Config.LLM_MODEL,
                    temperature=Config.LLM_TEMPERATURE,
                    groq_api_key=self.api_key
                )
            except Exception as e:
                print(f"Error initializing LLM: {e}")
                return None
        
        return self.llm
    
    def graph_qa_chain(
        self, 
        question: str, 
        docs: List[Document],
        source_name: Optional[str] = None,
        fallback_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Execute QA chain with retrieved documents.
        
        Args:
            question: User question
            docs: Retrieved documents
            source_name: Human-readable dataset name to condition the system prompt. If None, will
                attempt to infer from docs[0].metadata['filename'] or default to 'the uploaded dataset'.
            fallback_on_error: Whether to use fallback on LLM error
            
        Returns:
            Dictionary with 'text' and 'refrence' keys
        """
        # Extract context and top-k references (one per doc)
        context = Retrieval.format_context(docs)
        refs = Retrieval.topk_refs_from_docs(docs, k=len(docs))
        if source_name is None:
            try:
                source_name = (docs[0].metadata.get('filename') if docs and isinstance(docs[0].metadata, dict) else None) or 'the uploaded dataset'
            except Exception:
                source_name = 'the uploaded dataset'
        
        # Get LLM
        llm = self.get_llm()
        if llm is None:
            # Fallback: heuristic answer from top documents
            if fallback_on_error and docs:
                snippet = Retrieval.get_document_summary(docs, max_sentences=3)
                return {"text": snippet, "refrence": refs}
            else:
                return {
                    "text": "LLM service is not available. Please check your API configuration.",
                    "refrence": refs
                }
        
        # Format instructions for JSON output
        format_instructions = """Return ONLY a JSON object with this exact shape:
{
  "text": "<concise answer in 3–5 sentences>"
}
Do not include any other keys, no markdown, and no trailing text outside JSON.
"""
        
        try:
            # Execute chain
            result = (self.prompt_template | llm | self.parser).invoke({
                "question": question,
                "context": context,
                "format_instructions": format_instructions,
                "source_name": source_name
            })
            
            # Ensure we have the expected structure
            if not isinstance(result, dict):
                result = {"text": str(result)}
            elif "text" not in result:
                result = {"text": "Unable to parse response."}
            
            # Inject authoritative references (limited to top-k)
            result["refrence"] = refs
            
            return result
            
        except Exception as e:
            print(f"Error in QA chain: {e}")
            if fallback_on_error and docs:
                snippet = Retrieval.get_document_summary(docs, max_sentences=3)
                return {"text": snippet, "refrence": refs, "_error": str(e)}
            else:
                return {
                    "text": f"Error generating response: {str(e)}",
                    "refrence": refs
                }
    
    def create_custom_prompt(
        self, 
        system_prompt: str,
        human_template: str
    ) -> ChatPromptTemplate:
        """
        Create a custom prompt template.
        
        Args:
            system_prompt: System message
            human_template: Human message template
            
        Returns:
            ChatPromptTemplate instance
        """
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_template)
        ])
    
    def generate_response(
        self,
        prompt: ChatPromptTemplate,
        variables: Dict[str, Any],
        parse_json: bool = False
    ) -> Any:
        """
        Generate response using custom prompt.
        
        Args:
            prompt: Prompt template
            variables: Variables for the template
            parse_json: Whether to parse response as JSON
            
        Returns:
            Generated response
        """
        llm = self.get_llm()
        if not llm:
            return None
        
        try:
            if parse_json:
                chain = prompt | llm | self.parser
            else:
                chain = prompt | llm
            
            return chain.invoke(variables)
        except Exception as e:
            print(f"Error generating response: {e}")
            return None
