from typing import Dict, Any, List, Optional, Union
import re
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TokenizationUtils:
    """
    Centralized tokenization utility for consistent text preprocessing across all agents.
    Ensures reliable tokenization for experimental metrics and LLM performance.
    """
    
    def __init__(self, 
                 preserve_case: bool = False,
                 max_length: int = 4096,
                 min_length: int = 1):
        """
        Initialize tokenization utility.
        
        Args:
            preserve_case: Whether to preserve original capitalization
            max_length: Maximum text length to process
            min_length: Minimum text length to process
        """
        self.preserve_case = preserve_case
        self.max_length = max_length
        self.min_length = min_length
        
        # Common patterns for cleaning
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]')
        self.multiple_punctuation = re.compile(r'([.!?]){2,}')
        self.multiple_spaces = re.compile(r' {2,}')
        
        # JSON-specific patterns
        self.json_cleanup_patterns = [
            (r'```json\s*', ''),
            (r'```\s*$', ''),
            (r'^\s*```', ''),
            (r'```\s*', ''),
        ]
    
    def preprocess_llm_input(self, text: str) -> str:
        """
        Preprocess text before sending to LLM to ensure consistent tokenization.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Basic length validation
        if len(text) < self.min_length:
            return ""
        
        if len(text) > self.max_length:
            text = text[:self.max_length]
            logger.warning(f"Text truncated to {self.max_length} characters")
        
        # Step 1: Normalize whitespace
        text = self._normalize_whitespace(text)
        
        # Step 2: Fix common formatting issues
        text = self._fix_formatting_issues(text)
        
        # Step 3: Clean special characters (preserve essential punctuation)
        text = self._clean_special_characters(text)
        
        # Step 4: Ensure proper sentence structure
        text = self._fix_sentence_structure(text)
        
        # Step 5: Final validation
        text = self._validate_output(text)
        
        return text.strip()
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize user queries for consistent processing across agents.
        
        Args:
            query: Raw user query
            
        Returns:
            Normalized query
        """
        if not query or not isinstance(query, str):
            return ""
        
        # Step 1: Basic cleaning
        query = self._normalize_whitespace(query)
        
        # Step 2: Fix capitalization (unless preserve_case is True)
        if not self.preserve_case:
            query = self._normalize_capitalization(query)
        
        # Step 3: Remove extra punctuation
        query = self._clean_punctuation(query)
        
        # Step 4: Ensure proper question format
        query = self._ensure_question_format(query)
        
        # Step 5: Final validation
        query = self._validate_query(query)
        
        return query.strip()
    
    def postprocess_answer(self, answer: str, output_type: str = "text") -> str:
        """
        Clean and format LLM outputs to ensure consistent quality.
        
        Args:
            answer: Raw LLM output
            output_type: Type of output ("text", "json", "structured")
            
        Returns:
            Cleaned and formatted answer
        """
        if not answer or not isinstance(answer, str):
            return ""
        
        # Step 1: Basic cleaning
        answer = self._normalize_whitespace(answer)
        
        # Step 2: Type-specific processing
        if output_type == "json":
            answer = self._clean_json_output(answer)
        elif output_type == "structured":
            answer = self._clean_structured_output(answer)
        else:
            answer = self._clean_text_output(answer)
        
        # Step 3: Remove unwanted tokens and artifacts
        answer = self._remove_artifacts(answer)
        
        # Step 4: Final validation
        answer = self._validate_output(answer)
        
        return answer.strip()
    
    def clean_text_consistently(self, text: str, context: str = "general") -> str:
        """
        Apply consistent text cleaning across all agents.
        
        Args:
            text: Text to clean
            context: Context for cleaning ("query", "document", "answer", "general")
            
        Returns:
            Consistently cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Context-specific cleaning
        if context == "query":
            return self.normalize_query(text)
        elif context == "document":
            return self._clean_document_text(text)
        elif context == "answer":
            return self.postprocess_answer(text)
        else:
            return self.preprocess_llm_input(text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        # Replace multiple whitespace with single space
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Fix line breaks
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\r+', '\r', text)
        
        return text
    
    def _fix_formatting_issues(self, text: str) -> str:
        """Fix common formatting issues."""
        # Fix multiple punctuation
        text = self.multiple_punctuation.sub(r'\1', text)
        
        # Fix multiple spaces
        text = self.multiple_spaces.sub(' ', text)
        
        # Fix common typos in punctuation
        text = re.sub(r'\s+([.!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.!?])\s*([a-zA-Z])', r'\1 \2', text)  # Add space after punctuation
        
        return text
    
    def _clean_special_characters(self, text: str) -> str:
        """Clean special characters while preserving essential punctuation."""
        # Remove unwanted special characters
        text = self.special_chars_pattern.sub('', text)
        
        # Fix common encoding issues
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€', '"')
        text = text.replace('â€¢', '•')
        text = text.replace('â€"', '–')
        text = text.replace('â€"', '—')
        
        return text
    
    def _fix_sentence_structure(self, text: str) -> str:
        """Fix sentence structure issues."""
        # Ensure sentences start with capital letters
        sentences = re.split(r'([.!?])', text)
        result = []
        
        for i, part in enumerate(sentences):
            if part.strip():
                if i == 0 or (i > 0 and sentences[i-1] in '.!?'):
                    part = part[0].upper() + part[1:] if len(part) > 1 else part.upper()
                result.append(part)
        
        return ''.join(result)
    
    def _normalize_capitalization(self, text: str) -> str:
        """Normalize capitalization for queries."""
        # Keep first letter capitalized, rest lowercase for questions
        if text.endswith('?'):
            return text[0].upper() + text[1:].lower()
        else:
            return text[0].upper() + text[1:].lower()
    
    def _clean_punctuation(self, text: str) -> str:
        """Clean excessive punctuation."""
        # Remove multiple consecutive punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        text = re.sub(r'([,;:]){2,}', r'\1', text)
        
        # Remove punctuation at start/end (except question marks)
        text = re.sub(r'^[^\w\s]+', '', text)
        if not text.endswith('?'):
            text = re.sub(r'[^\w\s]+$', '', text)
        
        return text
    
    def _ensure_question_format(self, text: str) -> str:
        """Ensure proper question format."""
        # Add question mark if missing and it looks like a question
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'should', 'would']
        
        if not text.endswith('?'):
            text_lower = text.lower().strip()
            if any(text_lower.startswith(word) for word in question_words):
                text = text.rstrip('.!') + '?'
        
        return text
    
    def _clean_json_output(self, text: str) -> str:
        """Clean JSON output from LLM."""
        # Remove markdown code blocks
        for pattern, replacement in self.json_cleanup_patterns:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        
        # Remove any remaining markdown
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        # Try to extract JSON if wrapped in other text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group()
        
        return text
    
    def _clean_structured_output(self, text: str) -> str:
        """Clean structured output (lists, bullet points, etc.)."""
        # Fix bullet points
        text = re.sub(r'^\s*[-*•]\s*', '• ', text, flags=re.MULTILINE)
        
        # Fix numbered lists
        text = re.sub(r'^\s*(\d+)\.\s*', r'\1. ', text, flags=re.MULTILINE)
        
        # Remove excessive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def _clean_text_output(self, text: str) -> str:
        """Clean general text output."""
        # Remove common LLM artifacts
        artifacts = [
            r'^As an AI[^.]*\.',
            r'^I\'m an AI[^.]*\.',
            r'^I am an AI[^.]*\.',
            r'^Based on the information[^.]*\.',
            r'^According to[^.]*\.',
        ]
        
        for pattern in artifacts:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        return text
    
    def _clean_document_text(self, text: str) -> str:
        """Clean document text for retrieval."""
        # Remove excessive whitespace
        text = self._normalize_whitespace(text)
        
        # Remove unwanted characters but preserve essential punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]', '', text)
        
        # Fix common document formatting issues
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 line breaks
        
        return text
    
    def _remove_artifacts(self, text: str) -> str:
        """Remove common LLM artifacts and unwanted tokens."""
        # Remove common artifacts
        artifacts = [
            r'<\|.*?\|>',  # Special tokens
            r'\[.*?\]',    # Bracketed content
            r'\(.*?\)',    # Parenthesized content (be careful with this)
        ]
        
        for pattern in artifacts:
            text = re.sub(pattern, '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?]){3,}', r'\1\1', text)
        
        return text
    
    def _validate_output(self, text: str) -> str:
        """Final validation of processed text."""
        if not text or len(text) < self.min_length:
            return ""
        
        # Ensure text doesn't start/end with punctuation (except ?)
        text = text.strip()
        if text and not text.endswith('?'):
            text = text.rstrip('.,;:!')
        
        # Ensure proper capitalization
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        return text
    
    def _validate_query(self, query: str) -> str:
        """Validate processed query."""
        if not query or len(query) < self.min_length:
            return ""
        
        # Ensure query is properly formatted
        if not query.endswith('?') and not query.endswith('.'):
            query = query.rstrip('.,;:!') + '?'
        
        return query
    
    def validate_json_output(self, text: str) -> bool:
        """
        Validate if text is valid JSON.
        
        Args:
            text: Text to validate
            
        Returns:
            True if valid JSON, False otherwise
        """
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    
    def extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text that might contain other content.
        
        Args:
            text: Text that might contain JSON
            
        Returns:
            Extracted JSON as dictionary, or None if not found
        """
        try:
            # Try to find JSON object
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            
            # Try to find JSON array
            array_match = re.search(r'\[.*\]', text, re.DOTALL)
            if array_match:
                array_str = array_match.group()
                return json.loads(array_str)
            
            return None
        except (json.JSONDecodeError, ValueError):
            return None
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get statistics about processed text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with text statistics
        """
        if not text:
            return {
                "length": 0,
                "word_count": 0,
                "sentence_count": 0,
                "has_question_mark": False,
                "has_json": False
            }
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            "length": len(text),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "has_question_mark": '?' in text,
            "has_json": self.validate_json_output(text),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
        }


# Global instance for easy import
tokenization_utils = TokenizationUtils()

# Convenience functions for easy use
def preprocess_llm_input(text: str) -> str:
    """Preprocess text before sending to LLM."""
    return tokenization_utils.preprocess_llm_input(text)

def normalize_query(query: str) -> str:
    """Normalize user query."""
    return tokenization_utils.normalize_query(query)

def postprocess_answer(answer: str, output_type: str = "text") -> str:
    """Postprocess LLM output."""
    return tokenization_utils.postprocess_answer(answer, output_type)

def clean_text_consistently(text: str, context: str = "general") -> str:
    """Clean text consistently."""
    return tokenization_utils.clean_text_consistently(text, context)

