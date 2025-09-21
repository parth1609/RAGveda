"""
Retrieval Module.
Handles document retrieval, context formatting, and reference extraction.
"""

import re
from typing import List, Dict, Any
from langchain_core.documents import Document
from modules.config import Config


class Retrieval:
    """Handles retrieval operations and document processing."""
    
    @staticmethod
    def extract_refs(docs: List[Document]) -> List[str]:
        """
        Extract unique chapter:verse references from Document metadata or content.
        Prioritizes metadata (chapter/verse or chapters/verses), falls back to regex.
        
        Args:
            docs: List of documents to extract references from
            
        Returns:
            List of unique chapter:verse references
        """
        refs = []
        
        for d in docs:
            # Prefer list-form metadata created during grouping
            ch_list = d.metadata.get("chapters")
            vs_list = d.metadata.get("verses")
            
            if (isinstance(ch_list, list) and isinstance(vs_list, list) and 
                len(ch_list) == len(vs_list) and len(ch_list) > 0):
                for ch, vs in zip(ch_list, vs_list):
                    try:
                        refs.append(f"Chapter {int(ch)} Verse {int(vs)}")
                    except Exception:
                        pass
                continue
            
            # Fallback: single chapter/verse in metadata
            ch = d.metadata.get("chapter")
            vs = d.metadata.get("verse")
            if ch is not None and vs is not None:
                try:
                    refs.append(f"Chapter {int(ch)} Verse {int(vs)}")
                    continue
                except Exception:
                    pass
            
            # Fallback: parse from the content using regex
            pattern = re.compile(r"Chapter\s+(\d+)\s+Verse\s+(\d+)", re.IGNORECASE)
            matches = pattern.findall(d.page_content)
            for ch, vs in matches:
                refs.append(f"Chapter {int(ch)} Verse {int(vs)}")
        
        # Deduplicate preserving order
        seen = set()
        unique_refs = []
        for r in refs:
            if r not in seen:
                seen.add(r)
                unique_refs.append(r)
        
        return unique_refs
    
    @staticmethod
    def topk_refs_from_docs(docs: List[Document], k: int) -> List[str]:
        """
        Return at most k references, selecting at most one reference per document in order.
        Prefers metadata (chapters/verses or chapter/verse), then regex from content.
        """
        results: List[str] = []
        seen = set()
        for d in docs:
            ref: str = ""
            # Prefer list-form metadata created during grouping
            ch_list = d.metadata.get("chapters") if isinstance(d.metadata, dict) else None
            vs_list = d.metadata.get("verses") if isinstance(d.metadata, dict) else None
            if (
                isinstance(ch_list, list) and isinstance(vs_list, list)
                and len(ch_list) > 0 and len(ch_list) == len(vs_list)
            ):
                try:
                    ref = f"Chapter {int(ch_list[0])} Verse {int(vs_list[0])}"
                except Exception:
                    ref = ""
            # Fallback: single chapter/verse
            if not ref and isinstance(d.metadata, dict):
                ch = d.metadata.get("chapter")
                vs = d.metadata.get("verse")
                if ch is not None and vs is not None:
                    try:
                        ref = f"Chapter {int(ch)} Verse {int(vs)}"
                    except Exception:
                        ref = ""
            # Fallback: regex from content
            if not ref:
                m = re.search(r"Chapter\s+(\d+)\s+Verse\s+(\d+)", d.page_content or "")
                if m:
                    try:
                        ref = f"Chapter {int(m.group(1))} Verse {int(m.group(2))}"
                    except Exception:
                        ref = ""
            if ref and ref not in seen:
                seen.add(ref)
                results.append(ref)
            if len(results) >= k:
                break
        return results
    
    @staticmethod
    def format_context(docs: List[Document], max_chars: int = Config.MAX_CONTEXT_CHARS) -> str:
        """
        Join doc page_contents into a single context string, capped in length to keep prompt manageable.
        
        Args:
            docs: List of documents to format
            max_chars: Maximum characters for context
            
        Returns:
            Formatted context string
        """
        buf = []
        total = 0
        
        for d in docs:
            chunk = d.page_content.strip()
            if not chunk:
                continue
            if total + len(chunk) > max_chars:
                break
            buf.append(chunk)
            total += len(chunk)
        
        return "\n\n".join(buf)
    
    @staticmethod
    def format_results_for_display(docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Format documents for display in UI.
        
        Args:
            docs: List of documents to format
            
        Returns:
            List of formatted results
        """
        results = []
        
        for i, doc in enumerate(docs, start=1):
            result = {
                'rank': i,
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': doc.metadata.get('score', 1.0 - (i * 0.1))
            }
            results.append(result)
        
        return results
    
    @staticmethod
    def get_document_summary(docs: List[Document], max_sentences: int = 3) -> str:
        """
        Get a summary from the top documents.
        
        Args:
            docs: List of documents
            max_sentences: Maximum sentences to include
            
        Returns:
            Summary string
        """
        if not docs:
            return "No context found."
        
        sentences = []
        for doc in docs:
            doc_sentences = doc.page_content.split('. ')
            sentences.extend(doc_sentences[:max_sentences])
            if len(sentences) >= max_sentences:
                break
        
        return '. '.join(sentences[:max_sentences]).strip()
    
    @staticmethod
    def filter_by_metadata(docs: List[Document], filters: Dict[str, Any]) -> List[Document]:
        """
        Filter documents by metadata criteria.
        
        Args:
            docs: List of documents to filter
            filters: Dictionary of metadata filters
            
        Returns:
            Filtered list of documents
        """
        filtered = []
        
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    match = False
                    break
            if match:
                filtered.append(doc)
        
        return filtered
