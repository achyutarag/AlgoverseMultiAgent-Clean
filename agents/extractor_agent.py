from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from .base_agent import BaseAgent, AgentResponse
from .tokenization_utils import tokenization_utils, TokenizationUtils
import json
import logging
import re

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
    Enhanced Extractor Agent that performs fine-grained selection and aggregation 
    of sentences/spans aligned with subqueries. This addresses context inefficiency 
    by filtering out noise and enabling effective evidence aggregation.
    """
    
    def __init__(
        self, 
        model_config: Optional[Dict[str, Any]] = None,
        model_name: str = "gemini-2.5-flash",  # LLM for extraction
        temperature: float = 0.3,
        max_tokens: int = 2048
    ):
        """
        Initialize the Enhanced Extractor Agent.
        
        Args:
            model_config: Configuration for the LLM
            model_name: Name of the model to use
            temperature: Temperature for text generation (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
        """
        super().__init__("extractor_agent", model_config, model_name)
        self.temperature = max(0.0, min(1.0, temperature))
        self.max_tokens = max(100, min(4096, max_tokens))
        
        self.system_prompt = """You are an expert at performing fine-grained selection and aggregation of information from retrieved documents. Your task is to:

1. **Fine-grained Selection**: Extract specific sentences or spans that are directly aligned with the current subquery, rather than appending entire chunks

2. **Noise Filtering**: Filter out redundant or irrelevant content that doesn't contribute to answering the subquery

3. **Evidence Aggregation**: Combine complementary information from multiple sources into a concise, task-specific evidence set

4. **Context Efficiency**: Address the "lost-in-the-middle" issue by focusing on the most relevant information

Guidelines for extraction:
- Extract sentences or spans that answer the subquery (directly or indirectly)
- If documents are provided, extract at least 1 passage (even with low relevance) explaining why it's relevant or not relevant
- Only return empty array if NO documents were provided
- Preserve important context needed to understand the extracted information
- Combine information from multiple sources when they complement each other
- Assign relevance scores based on how directly the content addresses the subquery
- Avoid redundant information - if multiple sources say the same thing, include the clearest version
- Maintain source attribution for each extracted piece

Return a JSON object with this structure:
{
    "query": "The original subquery",
    "extraction_reasoning": "Your reasoning for what to extract and why",
    "extracted_passages": [
        {
            "text": "The extracted sentence or span",
            "document_id": "ID of the source document",
            "chunk_id": "ID of the specific chunk",
            "relevance": 0.9,  // Score from 0.0 to 1.0
            "reasoning": "Why this passage is relevant",
            "source_context": "Brief context about where this came from"
        }
    ],
    "aggregated_evidence": "Combined summary of all relevant information",
    "extraction_summary": "Summary of what was extracted and why"
}

Examples of good extraction:

**Subquery**: "What are the environmental benefits of solar energy?"
**Document**: "Solar energy production has several environmental advantages. Unlike fossil fuels, solar panels don't emit greenhouse gases during operation. They also reduce air pollution and water usage compared to traditional power plants. However, manufacturing solar panels does require energy and materials."

**Extracted**: "Solar energy production has several environmental advantages. Unlike fossil fuels, solar panels don't emit greenhouse gases during operation. They also reduce air pollution and water usage compared to traditional power plants."
**Reasoning**: "Directly answers the subquery about environmental benefits, excluding the manufacturing caveat which doesn't address benefits."

**Subquery**: "How do Japan and South Korea differ in economic policy?"
**Multiple sources**: Extract specific policy differences rather than general descriptions of each country's economy.

**Complete JSON Example**:

For the subquery "What is Scott Derrickson's nationality?" with documents about Scott Derrickson, return:

{
    "query": "What is Scott Derrickson's nationality?",
    "extraction_reasoning": "I need to find information about Scott Derrickson's nationality. I'll extract any biographical information that might indicate his nationality or country of origin.",
    "extracted_passages": [
        {
            "text": "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
            "document_id": "hotpotqa_Scott_Derrickson",
            "chunk_id": "chunk_1",
            "relevance": 0.9,
            "reasoning": "This sentence directly states Scott Derrickson's nationality as American",
            "source_context": "From Scott Derrickson biographical article"
        },
        {
            "text": "He lives in Los Angeles, California.",
            "document_id": "hotpotqa_Scott_Derrickson", 
            "chunk_id": "chunk_1",
            "relevance": 0.6,
            "reasoning": "Living in California suggests American nationality",
            "source_context": "From Scott Derrickson biographical article"
        }
    ],
    "aggregated_evidence": "Scott Derrickson is an American director, screenwriter and producer, born July 16, 1966, living in Los Angeles, California.",
    "extraction_summary": "Successfully extracted nationality information showing Scott Derrickson is American, with supporting biographical details"
}

**IMPORTANT**: Always return valid JSON. Do not include any text before or after the JSON object."""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process the retrieved documents and perform fine-grained extraction and aggregation.
        
        Args:
            input_data: Dictionary containing:
                - 'query': The original subquery
                - 'documents': List of retrieved documents with their content and metadata
                - Optional 'history': Previous interactions for context
                - Optional 'max_documents': Maximum number of documents to process (default: 5)
                - Optional 'min_relevance': Minimum relevance score (0.0-1.0) to include passages
                - Optional 'context_needed': Types of context needed for this extraction
                
        Returns:
            AgentResponse containing the extracted passages and aggregated evidence
        """
        query = input_data.get('query', '').strip()
        subqueries = input_data.get('subqueries', [])  # NEW: Get subqueries if provided
        documents = input_data.get('documents', [])
        history = input_data.get('history', [])
        max_documents = min(int(input_data.get('max_documents', 8)), 15)
        min_relevance = max(0.0, min(1.0, float(input_data.get('min_relevance', 0.3))))
        context_needed = input_data.get('context_needed', ['factual'])
        
        # Normalize query for consistent processing
        query = tokenization_utils.normalize_query(query)
        
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
            model_context = getattr(self.llm.config, 'context_length', 8192) if hasattr(self.llm, 'config') else 8192
            system_overhead = 3000  # Conservative estimate for system prompt, metadata, instructions
            available_tokens = max(1000, model_context - system_overhead)  # Ensure minimum
            tokens_per_doc = available_tokens // max_documents if max_documents > 0 else available_tokens // 10
            max_chars_per_doc = int(tokens_per_doc / 0.25)  # ~0.25 tokens per character

            # Build the extraction query - prioritize subqueries if available
            if subqueries:
                extraction_query = "\n".join([f"- {sq}" for sq in subqueries])
                extraction_context = f"### Extract passages that answer these queries:\n{extraction_query}\n\n(Step context: {query})"
            else:
                extraction_query = query
                extraction_context = "Subquery to Extract For:"

            prompt = f"""{self.system_prompt}

            ### {extraction_context}
            {extraction_query}

            ### Context Types Needed:
            {', '.join(context_needed)
            }
            
            ### Retrieved Documents:
            """
            
            # Add history if available
            if history:
                history_str = "\n".join(
                    f"{h.get('role', 'unknown').upper()}: {h.get('content', '')}"
                    for h in history[-3:]  # Last 3 history items
                )
                prompt += f"\n### Previous Context (most recent last):\n{history_str}\n"
            
            # Add documents to the prompt with better formatting
            for i, doc in enumerate(documents):
                doc_id = doc.get('id', f'doc_{i+1}')
                metadata = json.dumps(doc.get('metadata', {}), ensure_ascii=False, indent=2)
                content = doc.get('page_content', '').strip()
                score = doc.get('score', 0.0)
                
                prompt += (
                    f"\n[Document {i+1}, ID: {doc_id}, Retrieval Score: {score:.3f}]\n"
                    f"Metadata: {metadata}\n"
                    f"Content: {content[:max_chars_per_doc]}"  # Use the calculated value"
                )
                if len(content) > max_chars_per_doc:
                    prompt += "... [truncated]"
            
            # Add extraction instructions
            prompt += f"""
            
            ### CRITICAL REQUIREMENTS (MUST FOLLOW - READ FIRST):   
            - ⚠️ Extract passages with relevance ≥ {min_relevance} (use this exact threshold, not 0.3)
            - ⚠️ Extract passages that have ANY semantic connection to the query (even if weak, relevance 0.2-0.4)"
            - ⚠️ If documents have NO semantic connection to the query, you may return empty array BUT must provide detailed extraction_reasoning explaining why no passages were relevant"
            - ⚠️ Do NOT extract completely unrelated passages just to populate the array - quality over quantity
            - ⚠️ If documents contain ANY information related to the subquery (even indirectly), you MUST extract at least 1-2 passages
            - ⚠️ Being "somewhat relevant" (relevance 0.2-0.4) is acceptable - extract these passages!
            - ⚠️ Extract specific sentences or spans that are relevant to the subquery (be inclusive, not restrictive)
            - ⚠️ Use relevance scores based on actual relevance - scores ≥ {min_relevance} are acceptable
            - ⚠️ Focus on context types: {', '.join(context_needed)}

            ### ❌ WRONG - DO NOT DO THIS:
            {{
                "extracted_passages": [
                    {{"text": "Irrelevant passage", "relevance": 0.1, "reasoning": "Just to fill array"}}  // ❌ DON'T do this
                ]
            }}


            ### ✅ CORRECT - ALWAYS DO THIS:
            {{
                "extracted_passages": [
                    {{
                        "text": "Exact sentence from document",
                        "document_id": "doc_1",
                        "chunk_id": "chunk_1",
                        "relevance": {min(min_relevance + 0.05, 0.35)},  // Score just above threshold (≥ {min_relevance}) - THIS IS VALID AND ACCEPTABLE
                        "reasoning": "This helps answer the query",
                        "source_context": "From document context"
                    }}
                ]
            }}


            ### ⚠️ IMPORTANT - When to use empty array:
            - ALWAYS extract at least 1 passage when documents are provided
            - If documents don't match the query, extract a passage explaining why (relevance 0.2-0.3)
            - Being "somewhat relevant" (relevance 0.2-0.4) is STILL RELEVANT - extract it!
            - Empty array is ONLY acceptable if NO documents were provided at all
                    
            ### Instructions:
            Please perform fine-grained extraction focusing on:
            1. Extract sentences or spans that are relevant to the subquery (be inclusive, not restrictive)
            2. Include information that helps answer the subquery, even if indirectly
            3. Combine complementary information from multiple sources
            Assign relevance scores based on actual relevance:
            - {min_relevance}-0.5: Somewhat relevant, indirectly related (ACCEPTABLE - extract these)
            - 0.5-0.7: Relevant, helps answer the query
            - 0.7-1.0: Highly relevant, directly answers the query
            - Below {min_relevance}: Do NOT extract (not relevant enough)
            5. Provide reasoning for each extraction decision
            6. Create an aggregated summary of all relevant evidence

            ### FINAL REMINDER BEFORE RETURNING JSON:
            1. Extract at least 1-2 passages if any document has relevance ≥ {min_relevance}
            2. If documents don't match the query, extract at least 1 passage explaining why (with low relevance 0.2-0.3)
            3. NEVER return completely empty array - always extract at least 1 passage when documents are provided
            4. Extract exact text from the documents provided above
            5. Do NOT summarize - extract specific passages
            
            
            Return your response as a valid JSON object with the structure shown above.
            """
            
            # Log the extraction request
            logger.info(f"Performing fine-grained extraction for subquery: {query[:100]}...")
            #logger.debug(f"Processing {len(documents)} documents with min_relevance={min_relevance}")
            
            # Get the LLM response
            response = await self.generate_text(
                prompt=prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens
            )
            # We will show the following for DEBUGGING:
            #1) The Exact raw response of the LLM
            #2) How postprocessing changed it
            #3) Whether the issue is in the LLM output or in processing

            # DEBUG: Log raw LLM response BEFORE any processing
            # logger.debug("=" * 80)
            # logger.debug("RAW LLM RESPONSE (BEFORE POSTPROCESSING):")
            # logger.debug("=" * 80)
            # logger.debug(f"Response type: {type(response)}")
            # logger.debug(f"Response length: {len(str(response))} characters")
            # logger.debug(f"Full response:\n{response}")
            # logger.debug("=" * 80)

            # Postprocess the LLM response
            response = tokenization_utils.postprocess_answer(response, output_type="json")

            # DEBUG: Log response AFTER postprocessing
            # logger.debug("=" * 80)
            # logger.debug("LLM RESPONSE AFTER POSTPROCESSING:")
            # logger.debug("=" * 80)
            # logger.debug(f"Response after postprocess:\n{response}")
            # logger.debug("=" * 80)

            #DEBUG: check for response length and preview

            # logger.debug(f"LLM response length: {len(response)} characters")
            # logger.debug(f"LLM response preview: {response[:200]}...")
            
            try:
                # Extract JSON from the response
                # Strip markdown formatting first
                clean_response = TokenizationUtils.strip_markdown_json(response)
                
                # Additional JSON repair for common issues
                if '"extracted_passages":' in clean_response and '"extracted_passages": [' not in clean_response:
                    # Fix incomplete extracted_passages field
                    clean_response = clean_response.replace('"extracted_passages": ', '"extracted_passages": []')
                
                # Try to repair JSON if it fails
                try:
                    result = json.loads(clean_response)
                except json.JSONDecodeError:
                    # Try to repair common JSON issues
                    repaired_response = TokenizationUtils.repair_json(clean_response)
                    result = json.loads(repaired_response)
                
                # Validate the response structure
                required_keys = ["query", "extracted_passages"]
                if not all(key in result for key in required_keys):
                    raise ValueError("Missing required fields in response")
                
                # DEBUG: Log what the LLM actually returned
                logger.debug(f"LLM returned {len(result.get('extracted_passages', []))} passages before validation")
                logger.debug(f"Raw extracted_passages: {result.get('extracted_passages', [])}")
                
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
                            #Added logging in the validationloop (after line 277)
                            logger.debug(f"Passage {i} filtered: relevance {relevance} < min_relevance {min_relevance}")
                            continue
                            
                        # Clean and validate the extracted text
                        text = passage["text"].strip()
                        if len(text) < 10:  # Skip very short extractions
                            #DEBUG: check for text length filtering
                            logger.debug(f"Passage {i} filtered: text too short ({len(text)} chars < 10)")
                            continue
                            
                        valid_passages.append({
                            "text": text,
                            "document_id": str(passage["document_id"]),
                            "chunk_id": str(passage.get("chunk_id", "")),
                            "relevance": relevance,
                            "reasoning": str(passage.get("reasoning", "")).strip(),
                            "source_context": str(passage.get("source_context", "")).strip()
                        })
                        
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing passage {i}: {str(e)}")
                        continue
                
                # Sort passages by relevance (highest first)
                valid_passages.sort(key=lambda x: x["relevance"], reverse=True)
                
                # Fallback: If no passages were extracted, create one from aggregated evidence
                if not valid_passages and result.get("aggregated_evidence"):
                    logger.warning("No passages extracted, creating fallback passage from aggregated evidence")
                    aggregated_evidence = result.get("aggregated_evidence", "")
                    if aggregated_evidence:
                        valid_passages.append({
                            "text": aggregated_evidence,
                            "document_id": "fallback",
                            "chunk_id": "fallback",
                            "relevance": 0.5,
                            "reasoning": "Fallback passage created from aggregated evidence",
                            "source_context": "Generated from aggregated evidence"
                        })
                
                # Remove duplicate or very similar passages
                deduplicated_passages = self._deduplicate_passages(valid_passages)
                
                # Prepare response data
                response_data = {
                    "query": query,
                    "extraction_reasoning": result.get("extraction_reasoning", "Fine-grained extraction performed"),
                    "extracted_passages": deduplicated_passages,
                    "aggregated_evidence": result.get("aggregated_evidence", ""),
                    "extraction_summary": result.get("extraction_summary", f"Extracted {len(deduplicated_passages)} relevant passages")
                }
                
                # Log extraction results
                logger.info(f"Extracted {len(deduplicated_passages)} relevant passages "
                          f"(min_relevance={min_relevance})")
                
                # Update history
                self._update_history("user", f"Extract relevant information for: {query}")
                self._update_history(
                    "assistant",
                    f"Extracted {len(deduplicated_passages)} relevant passages from {len(documents)} documents"
                )
                
                return AgentResponse(
                    content=json.dumps(response_data, ensure_ascii=False, indent=2),
                    metadata={
                        "query": query,
                        "num_extracted": len(deduplicated_passages),
                        "num_documents_processed": len(documents),
                        "min_relevance": min_relevance,
                        "avg_relevance": (
                            sum(p["relevance"] for p in deduplicated_passages) / len(deduplicated_passages)
                            if deduplicated_passages else 0.0
                        ),
                        "context_needed": context_needed,
                        "extraction_parameters": {
                            "max_documents": max_documents,
                            "min_relevance": min_relevance,
                            "model": self.model_name,
                            "temperature": self.temperature
                        },
                        "extracted_passages": deduplicated_passages  # Include in metadata for easy access
                    }
                )
                
            except json.JSONDecodeError as e:
                error_msg = "Failed to parse LLM response as JSON"
                logger.error(f"{error_msg}: {e}")
                
                # Fallback: simple extraction based on keyword matching
                fallback_passages = self._fallback_extraction(query, documents, min_relevance)
                
                return AgentResponse(
                    content=json.dumps({
                        "query": query,
                        "extraction_reasoning": "Fallback extraction due to parsing error",
                        "extracted_passages": fallback_passages,
                        "aggregated_evidence": "Fallback extraction performed",
                        "extraction_summary": f"Fallback: extracted {len(fallback_passages)} passages"
                    }),
                    metadata={
                        "error": error_msg,
                        "llm_response": response,
                        "exception": str(e),
                        "fallback": True,
                        "num_extracted": len(fallback_passages)
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
    
    def _deduplicate_passages(self, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or very similar passages."""
        if not passages:
            return passages
        
        deduplicated = []
        for passage in passages:
            # Check if this passage is too similar to any already included passage
            is_duplicate = False
            for existing in deduplicated:
                if self._passages_similar(passage["text"], existing["text"]):
                    # Keep the one with higher relevance
                    if passage["relevance"] > existing["relevance"]:
                        deduplicated.remove(existing)
                        deduplicated.append(passage)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(passage)
        
        return deduplicated
    
    def _passages_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two passages are similar based on word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity >= threshold
    
    def _fallback_extraction(self, query: str, documents: List[Dict[str, Any]], min_relevance: float) -> List[Dict[str, Any]]:
        """Fallback extraction using simple keyword matching."""
        query_words = set(query.lower().split())
        passages = []
        
        for i, doc in enumerate(documents):
            content = doc.get('page_content', '')
            doc_id = doc.get('id', f'doc_{i+1}')
            
            # Split into sentences
            sentences = re.split(r'[.!?]+', content)
            
            for j, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) < 20:  # Skip very short sentences
                    continue
                
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                
                if overlap > 0:
                    relevance = min(0.9, overlap / len(query_words))
                    if relevance >= min_relevance:
                        passages.append({
                            "text": sentence,
                            "document_id": doc_id,
                            "chunk_id": f"{doc_id}_sentence_{j+1}",
                            "relevance": relevance,
                            "reasoning": f"Keyword overlap: {overlap} words",
                            "source_context": f"Document {i+1}"
                        })
        
        # Sort by relevance and limit to top 5
        passages.sort(key=lambda x: x["relevance"], reverse=True)
        return passages[:5]
    
    @staticmethod
    def chunk_document(document: Dict[str, Any], chunk_size: int = 1200, overlap: int = 50) -> List[Dict[str, Any]]:
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
