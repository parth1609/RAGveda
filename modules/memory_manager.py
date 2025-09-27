"""
Memory Manager Module.
Handles conversation memory using LangChain's ConversationSummaryMemory.
"""

from typing import Optional, Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from modules.config import Config


class MemoryManager:
    """Manages conversation memory with summarization."""
    
    def __init__(self, llm: Optional[ChatGroq] = None):
        """
        Initialize memory manager.
        
        Args:
            llm: Optional LLM instance for summarization
        """
        self.llm = llm or self._create_llm()
        self.memory = None
        self._reset_memory()
    
    def _create_llm(self) -> Optional[ChatGroq]:
        """Create LLM instance for memory summarization."""
        try:
            return ChatGroq(
                groq_api_key=Config.GROQ_API_KEY,
                model_name=Config.LLM_MODEL,
                temperature=Config.LLM_TEMPERATURE
            )
        except Exception as e:
            print(f"Failed to create LLM for memory: {e}")
            return None
    
    def _reset_memory(self):
        """Reset memory to empty state."""
        self.memory = {
            "turns": [],
            "summary": ""
        }
    
    def save_conversation_turn(self, human_input: str, ai_response: str):
        """
        Save a conversation turn to memory.
        
        Args:
            human_input: User's message
            ai_response: Assistant's response
        """
        if not self.memory:
            return
        
        try:
            # Add turn to memory
            self.memory["turns"].append({
                "human": human_input,
                "ai": ai_response
            })
            
            # If we have too many turns, create a summary
            if len(self.memory["turns"]) > Config.MEMORY_MAX_TURNS_BEFORE_SUMMARY:
                self._create_summary()
        except Exception as e:
            print(f"Failed to save conversation to memory: {e}")
    
    def get_memory_context(self) -> str:
        """
        Get the current memory context (summary).
        
        Returns:
            Memory context string
        """
        if not self.memory:
            return ""
        
        try:
            # Return summary if available, otherwise recent turns
            if self.memory["summary"]:
                return self.memory["summary"]
            
            # If no summary yet, return recent conversation turns
            if len(self.memory["turns"]) > 0:
                recent_turns = self.memory["turns"][-3:]  # Last 3 turns
                context_parts = []
                for turn in recent_turns:
                    context_parts.append(f"Human: {turn['human'][:100]}...")
                    context_parts.append(f"AI: {turn['ai'][:100]}...")
                return "\n".join(context_parts)
            
            return ""
        except Exception as e:
            print(f"Failed to load memory context: {e}")
            return ""
    
    def _create_summary(self):
        """Create a summary of the conversation turns."""
        if not self.llm or len(self.memory["turns"]) == 0:
            return
        
        try:
            # Format conversation turns for summarization
            conversation_text = []
            for turn in self.memory["turns"]:
                conversation_text.append(f"Human: {turn['human']}")
                conversation_text.append(f"AI: {turn['ai']}")
            
            full_conversation = "\n".join(conversation_text)
            
            # Create summary using LLM
            summary_prompt = f"""Summarize this conversation concisely in 2-3 sentences, focusing on key topics discussed:

{full_conversation}

Summary:"""
            
            summary_response = self.llm.invoke(summary_prompt)
            self.memory["summary"] = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
            
            # Keep only the most recent turns after summarizing
            self.memory["turns"] = self.memory["turns"][-2:]  # Keep last 2 turns
            
        except Exception as e:
            print(f"Failed to create memory summary: {e}")
    
    def clear_memory(self):
        """Clear all memory contents."""
        if self.memory:
            try:
                self.memory["turns"] = []
                self.memory["summary"] = ""
            except Exception as e:
                print(f"Failed to clear memory: {e}")
    
    def reset_session(self):
        """Reset memory for new session."""
        self._reset_memory()
    
    def is_available(self) -> bool:
        """Check if memory is available."""
        return self.memory is not None
    
    def get_buffer_string(self) -> str:
        """Get the raw memory buffer for debugging."""
        if not self.memory:
            return ""
        
        try:
            return getattr(self.memory, 'buffer', '')
        except:
            return ""
    
    def from_chat_history(self, chat_history: List[Dict[str, Any]]):
        """
        Initialize memory from existing chat history.
        
        Args:
            chat_history: List of chat messages with 'role' and 'content' keys
        """
        if not self.memory or not chat_history:
            return
        
        try:
            # Process chat history in pairs (user + assistant)
            for i in range(0, len(chat_history) - 1, 2):
                if i + 1 < len(chat_history):
                    user_msg = chat_history[i]
                    ai_msg = chat_history[i + 1]
                    
                    if (user_msg.get('role') == 'user' and 
                        ai_msg.get('role') == 'assistant'):
                        
                        user_content = user_msg.get('content', '')
                        ai_content = ai_msg.get('content', '')
                        
                        # Handle AI content that might be a dict (with text and references)
                        if isinstance(ai_content, dict):
                            ai_content = ai_content.get('text', str(ai_content))
                        
                        self.save_conversation_turn(str(user_content), str(ai_content))
        except Exception as e:
            print(f"Failed to initialize memory from chat history: {e}")
