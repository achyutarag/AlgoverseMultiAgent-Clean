from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from .base_agent import BaseAgent, AgentResponse
import json
import logging

logger = logging.getLogger(__name__)

class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document with metadata."""
    text: str = Field(..., description="The text content of the chunk")
    document_id: str = Field(..., description="Unique identifier for the source document")
    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the chunk")
    relevance_score: Optional[float] = Field(None, description="Relevance score of this chunk to the query")

class ExtractorAgent(BaseAgent):
    """
    The Extractor Agent analyzes retrieved documents and extracts the most relevant
    passages that directly address the query using a local LLM.
    """
    
    def __init__(
        self, 
        model_config: Optional[Dict[str, Any]] = None,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        temperature: float = 0.1,
        max_tokens: int = 2048
    ):
        """
        Initialize the Extractor Agent.
        
        Args:
            model_config: Configuration for the LLM
            model_name: Name of the model to use
            temperature: Temperature for text generation (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
        """
        super().__init__("extractor_agent", model_config, model_name)
        self.temperature = max(0.0, min(1.0, temperature))
        self.max_tokens = max(100, min(4096, max_tokens))
        
        self.system_prompt = """You are an expert at extracting the most relevant information from documents.
        Your task is to analyze the retrieved documents and extract ONLY the specific passages that directly
        address the query. Be extremely selective - only include information that is directly relevant.
        
        For each relevant passage:
        1. Extract the exact text that answers the query
        2. Keep the original wording as much as possible
        3. Include only complete sentences or meaningful fragments
        4. Preserve any important context needed to understand the passage
        5. Assign a relevance score from 0.0 (irrelevant) to 1.0 (perfect match)
        
        Return a JSON object with this structure:
        {
            "query": "The original query",
            "extracted_passages": [
                {
                    "text": "The extracted passage",
                    "document_id": "ID of the source document",
                    "chunk_id": "ID of the specific chunk",
                    "relevance": 0.9,  // Score from 0.0 to 1.0
                    "reasoning": "Why this passage is relevant"
                }
            ]
        }
        """
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process the retrieved documents and extract relevant passages.
        
        Args:
            input_data: Dictionary containing:
                - 'query': The original query
                - 'documents': List of retrieved documents with their content and metadata
                - Optional 'history': Previous interactions for context
                - Optional 'max_documents': Maximum number of documents to process (default: 5)
                - Optional 'min_relevance': Minimum relevance score (0.0-1.0) to include passages
                
        Returns:
            AgentResponse containing the extracted passages and metadata
        """
        query = input_data.get('query', '').strip()
        documents = input_data.get('documents', [])
        history = input_data.get('history', [])
        max_documents = min(int(input_data.get('max_documents', 5)), 10)  # Cap at 10 documents
        min_relevance = max(0.0, min(1.0, float(input_data.get('min_relevance', 0.5))))
        
        if not query:
            return AgentResponse(
                content="Error: No query provided",
                metadata={"error": "No query provided"}
            )
            
        if not documents:
            return AgentResponse(
                content="No documents provided for extraction",
                metadata={"query": query, "num_extracted": 0}
            )
        
        try:
            # Limit number of documents to process
            documents = documents[:max_documents]
            
            # Prepare the prompt
            prompt = f"""{self.system_prompt}
            
            ### Query to answer:
            {query}
            
            ### Retrieved Documents:
            """
            
            # Add history if available
            if history:
                history_str = "\n".join(
                    f"{h.get('role', 'unknown').upper()}: {h.get('content', '')}"
                    for h in history[-3:]  # Last 3 history items
                )
                prompt += f"\n### Previous Context (most recent last):\n{history_str}\n"
            
            # Add documents to the prompt
            for i, doc in enumerate(documents):
                doc_id = doc.get('id', f'doc_{i+1}')
                metadata = json.dumps(doc.get('metadata', {}), ensure_ascii=False, indent=2)
                content = doc.get('page_content', '').strip()
                score = doc.get('score', 0.0)
                
                prompt += (
                    f"\n[Document {i+1}, ID: {doc_id}, Score: {score:.3f}]\n"
                    f"Metadata: {metadata}\n"
                    f"Content: {content[:2000]}"
                )
                if len(content) > 2000:
                    prompt += "... [truncated]"
            
            # Add extraction instructions
            prompt += """
            
            ### Instructions:
            Please extract ONLY the specific passages that directly address the query.
            Be extremely selective - only include information that is directly relevant.
            For each relevant passage, include:
            1. The exact text that answers the query
            2. The document_id and chunk_id it came from
            3. A relevance score from 0.0 to 1.0
            4. A brief explanation of why it's relevant
            
            Return your response as a valid JSON object with the structure shown above.
            """
            
            # Log the extraction request
            logger.info(f"Extracting relevant passages for query: {query[:100]}...")
            logger.debug(f"Processing {len(documents)} documents with min_relevance={min_relevance}")
            
            # Get the LLM response
            response = await self.generate_text(
                prompt=prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens
            )
            
            try:
                # Extract JSON from the response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No valid JSON found in response")
                
                # Validate the response structure
                if not all(key in result for key in ["query", "extracted_passages"]):
                    raise ValueError("Missing required fields in response")
                
                # Validate and filter extracted passages
                valid_passages = []
                for i, passage in enumerate(result["extracted_passages"]):
                    try:
                        if not all(k in passage for k in ["text", "document_id", "relevance"]):
                            logger.warning(f"Skipping passage {i}: Missing required fields")
                            continue
                            
                        # Ensure relevance is a float between 0 and 1
                        relevance = float(passage.get("relevance", 0.0))
                        if relevance < min_relevance:
                            continue
                            
                        valid_passages.append({
                            "text": passage["text"].strip(),
                            "document_id": str(passage["document_id"]),
                            "chunk_id": str(passage.get("chunk_id", "")),
                            "relevance": relevance,
                            "reasoning": str(passage.get("reasoning", "")).strip()
                        })
                        
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing passage {i}: {str(e)}")
                        continue
                
                # Sort passages by relevance (highest first)
                valid_passages.sort(key=lambda x: x["relevance"], reverse=True)
                
                # Prepare response data
                response_data = {
                    "query": query,
                    "extracted_passages": valid_passages
                }
                
                # Log extraction results
                logger.info(f"Extracted {len(valid_passages)} relevant passages "
                          f"(min_relevance={min_relevance})")
                
                # Update history
                self._update_history("user", f"Extract relevant information for: {query}")
                self._update_history(
                    "assistant",
                    f"Extracted {len(valid_passages)} relevant passages from {len(documents)} documents"
                )
                
                return AgentResponse(
                    content=json.dumps(response_data, ensure_ascii=False, indent=2),
                    metadata={
                        "query": query,
                        "num_extracted": len(valid_passages),
                        "num_documents_processed": len(documents),
                        "min_relevance": min_relevance,
                        "avg_relevance": (
                            sum(p["relevance"] for p in valid_passages) / len(valid_passages)
                            if valid_passages else 0.0
                        ),
                        "extraction_parameters": {
                            "max_documents": max_documents,
                            "min_relevance": min_relevance,
                            "model": self.model_name,
                            "temperature": self.temperature
                        }
                    }
                )
                
            except json.JSONDecodeError as e:
                error_msg = "Failed to parse LLM response as JSON"
                logger.error(f"{error_msg}: {e}")
                return AgentResponse(
                    content=f"Error: {error_msg}",
                    metadata={
                        "error": error_msg,
                        "llm_response": response,
                        "exception": str(e)
                    }
                )
                
            except Exception as e:
                error_msg = f"Error processing LLM response: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return AgentResponse(
                    content=error_msg,
                    metadata={
                        "error": "Response processing error",
                        "exception": str(e),
                        "llm_response": response
                    }
                )
                
        except Exception as e:
            error_msg = f"Error extracting information: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return AgentResponse(
                content=error_msg,
                metadata={
                    "error": "Extraction error",
                    "exception": str(e),
                    "query": query,
                    "num_documents": len(documents)
                }
            )
    
    @staticmethod
    def chunk_document(document: Dict[str, Any], chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Split a document into overlapping chunks.
        
        Args:
            document: Dictionary containing 'page_content' and 'metadata'
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of document chunks
        """
        if not document or 'page_content' not in document:
            return []
            
        text = document['page_content']
        metadata = document.get('metadata', {})
        doc_id = metadata.get('id', 'unknown')
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to end at a sentence boundary if possible
            if end < len(text):
                # Look for sentence-ending punctuation near the chunk boundary
                boundary = end
                for punct in ['. ', '\n', '? ', '! ']:
                    pos = text.rfind(punct, start, end)
                    if pos > start + chunk_size // 2:  # Only if it's in the second half of the chunk
                        boundary = pos + len(punct)
                        break
                end = boundary
            
            chunks.append({
                'page_content': text[start:end].strip(),
                'metadata': {
                    **metadata,
                    'chunk_id': f"{doc_id}_chunk_{len(chunks)+1}",
                    'chunk_start': start,
                    'chunk_end': end
                }
            })
            
            # Move start position, accounting for overlap
            start = end - overlap if end - overlap > start else end
            
            # Prevent infinite loop with very small chunks
            if start == end and start < len(text):
                start = end
                
        return chunks
