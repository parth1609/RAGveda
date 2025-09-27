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
        # Dedicated prompt for query rewriting prior to retrieval
        self.rewrite_prompt_template = self._create_rewrite_prompt_template()
        
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for QA.
        The prompt explicitly references the current dataset name via {source_name}.
        """
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. Use the provided context from {source_name} to answer "
             "the user's question accurately. "
             "Do not include verse text in your answer. If the context is insufficient, say you are uncertain. "
             "If conversation memory is provided, use it to maintain context. "
             "Follow the output format instructions exactly."),
            ("human",
             "Conversation Summary (if available):\n{memory_context}\n\n"
             "Question:\n{question}\n\nContext:\n{context}\n\nOutput format:\n{format_instructions}")
        ])
    
    def _create_rewrite_prompt_template(self) -> ChatPromptTemplate:
        """
        Create the prompt template for query rewriting.
        
        Purpose:
            Given a potentially vague or ambiguous user message (and optional short chat history),
            produce a precise, standalone retrieval query tailored to the current dataset.
        
        Returns:
            ChatPromptTemplate: A template that elicits a compact JSON response containing
            the rewritten query and brief notes.
        """
        return ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert assistant that rewrites vague user questions into clear, "
                "standalone search queries for a retrieval system over {source_name}. "
                "Rules:(1) Include exact numbers like chapter/verse "
                "if the user mentions them. (2) Prefer key nouns and important entities; include synonyms "
                "or alternate phrasings only if they are high value. (3) Do not hallucinate details or verses. "
                "(4) Avoid instructions; output just a search query, not a question. (5) Preserve the user's language."
            ),
            (
                "human",
                "Dataset: {source_name}\n"
                "Chat history (optional, short):\n{history}\n\n"
                "Original message:\n{question}\n\n"
                "Return ONLY JSON with this exact shape:\n"
                "{format_instructions}"
            ),
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
                    groq_api_key=self.api_key,
                    timeout=Config.LLM_TIMEOUT,
                    max_retries=Config.LLM_MAX_RETRIES
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
        memory_context: Optional[str] = None,
        fallback_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Execute QA chain with retrieved documents.
        
        Args:
            question: User question
            docs: Retrieved documents
            source_name: Human-readable dataset name to condition the system prompt. If None, will
                attempt to infer from docs[0].metadata['filename'] or default to 'the uploaded dataset'.
            memory_context: Optional conversation summary for context
            fallback_on_error: Whether to use fallback on LLM error
            
        Returns:
            Dictionary with 'text' and 'refrence' keys
        """
        # Compute context and candidate references
        context = Retrieval.format_context(docs)
        # Keep only documents above similarity threshold when computing references
        try:
            threshold = float(Config.MIN_SIMILARITY_FOR_REFERENCES)
        except Exception:
            threshold = 0.30
        filtered_docs = []
        for d in docs:
            score = None
            if isinstance(d.metadata, dict) and 'score' in d.metadata:
                try:
                    score = float(d.metadata.get('score'))
                except Exception:
                    score = None
            # If score is missing, keep the document; otherwise apply threshold
            if score is None or score >= threshold:
                filtered_docs.append(d)
        if filtered_docs:
            refs = Retrieval.topk_refs_from_docs(filtered_docs, k=len(filtered_docs))
        else:
            refs = []
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
  "text": "<concise answer in 1-2 sentences maximum>"
}
Do not include any other keys, no markdown, and no trailing text outside JSON.
"""
        
        try:
            # Execute chain
            result = (self.prompt_template | llm | self.parser).invoke({
                "question": question,
                "context": context,
                "memory_context": memory_context or "No previous conversation context.",
                "format_instructions": format_instructions,
                "source_name": source_name
            })
            
            # Ensure we have the expected structure
            if not isinstance(result, dict):
                result = {"text": str(result)}
            elif "text" not in result:
                result = {"text": "Unable to parse response."}
            
            # If the model signaled uncertainty, suppress references
            text_lower = result.get("text", "").lower()
            uncertainty_markers = [
                "insufficient", "uncertain", "not enough context", "i could not find",
                "cannot find", "does not contain", "no relevant", "not present", "no evidence"
            ]
            if any(m in text_lower for m in uncertainty_markers):
                result["refrence"] = []
            else:
                # Inject authoritative references (limited to confident top-k)
                if refs:
                    result["refrence"] = refs
                else:
                    # Fallback: if we do have retrieved docs but no refs due to thresholding,
                    # derive references from all docs so the Sources panel is populated.
                    result["refrence"] = Retrieval.topk_refs_from_docs(docs, k=len(docs)) if docs else []
            
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
    
    def rewrite_query(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        source_name: Optional[str] = None,
        fallback_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Rewrite a potentially vague user query into a precise retrieval query.
        
        Parameters:
            question (str): The raw user message.
            chat_history (Optional[List[Dict[str, Any]]]): Recent chat messages to provide context.
                Each item should have keys like 'role' and 'content'.
            source_name (Optional[str]): Human-readable dataset name; if None, defaults to
                'the uploaded dataset'. Used to bias the rewrite to the current scope.
            fallback_on_error (bool): If True, returns the original question when the LLM
                is unavailable or an error occurs.
        
        Returns:
            Dict[str, Any]: JSON with keys:
                - 'rewritten_query' (str): The improved query for retrieval.
                - 'did_rewrite' (bool): Whether a rewrite occurred.
                - 'notes' (str): Brief rationale or disambiguation steps.
        
        Side Effects:
            Calls the external Groq LLM API via LangChain if configured.
        
        Examples:
            >>> llm = LLMChain()
            >>> llm.rewrite_query("Tell me about karma", [], "Gita_questions.csv")
            {'rewritten_query': 'karma yoga duty action results detachment', 'did_rewrite': True, 'notes': 'focused on karma yoga'}
        """
        # Format short history to aid the model without overloading the prompt.
        def _format_history(items: Optional[List[Dict[str, Any]]], max_pairs: int = 3, max_len: int = 300) -> str:
            if not items:
                return ""
            # Keep the most recent interactions; include at most `max_pairs` user/assistant pairs.
            trimmed = items[-(max_pairs * 2):]
            lines: List[str] = []
            for m in trimmed:
                role = m.get('role', '')
                content = m.get('content', '')
                # If assistant content is a dict (our QA result), reduce to text
                if isinstance(content, dict):
                    content = content.get('text', str(content))
                content = str(content)
                if len(content) > max_len:
                    content = content[: max_len] + '…'
                lines.append(f"{role.capitalize()}: {content}")
            return "\n".join(lines)
        
        # Resolve source name
        if source_name is None:
            source_name = 'the uploaded dataset'
        
        llm = self.get_llm()
        if llm is None:
            return {
                "rewritten_query": question,
                "did_rewrite": False,
                "notes": "LLM unavailable; used original query"
            }
        
        # Expected JSON shape for the parser
        format_instructions = (
            '{\n'
            '  "rewritten_query": "<concise retrieval query>",\n'
            '  "did_rewrite": <true|false>,\n'
            '  "notes": "<very brief reasoning or disambiguation>"\n'
            '}'
        )
        variables = {
            "question": question,
            "history": _format_history(chat_history),
            "source_name": source_name or "the uploaded dataset",
            "format_instructions": format_instructions
        }
        try:
            result = (self.rewrite_prompt_template | llm | self.parser).invoke(variables)
            # Validate and normalize
            if not isinstance(result, dict):
                return {
                    "rewritten_query": question,
                    "did_rewrite": False,
                    "notes": "Unexpected parser output; fallback to original"
                }
            rewritten = str(result.get("rewritten_query") or "").strip()
            did_rewrite = bool(result.get("did_rewrite", bool(rewritten and rewritten != question)))
            notes = str(result.get("notes") or "").strip()
            if not rewritten:
                return {
                    "rewritten_query": question,
                    "did_rewrite": False,
                    "notes": "Empty rewrite; used original"
                }
            return {
                "rewritten_query": rewritten,
                "did_rewrite": did_rewrite,
                "notes": notes,
            }
        except Exception as e:
            print(f"Error in rewrite_query: {e}")
            if fallback_on_error:
                return {
                    "rewritten_query": question,
                    "did_rewrite": False,
                    "notes": f"Rewrite error: {e}; used original"
                }
            raise
    
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
