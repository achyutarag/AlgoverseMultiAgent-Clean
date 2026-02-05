from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from .base_agent import BaseAgent, AgentResponse
from .tokenization_utils import tokenization_utils, TokenizationUtils
import json
import logging

logger = logging.getLogger(__name__)

class Evidence(BaseModel):
    """Supporting evidence for an answer."""
    text: str = Field(..., description="The text of the evidence")
    source: str = Field(..., description="Source document ID for this evidence")
    relevance: float = Field(0.0, description="Relevance score of this evidence")

class Answer(BaseModel):
    """Structured answer with supporting evidence."""
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The generated answer")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    sources: List[str] = Field(default_factory=list, description="List of source document IDs")
    supporting_evidence: List[Evidence] = Field(
        default_factory=list,
        description="List of supporting evidence with text and source"
    )
    reasoning: str = Field("", description="Reasoning process for the answer")
    evidence_term: Optional[str] = Field(None, description="Term actually used in evidence (e.g., 'partner' for 'spouse')")

class QAAgent(BaseAgent):
    """
    QA Agent that synthesizes answers using in-context learning with 
    step-specific context. It produces responses for each step which are passed 
    to the next iteration, enabling grounded reasoning throughout the trajectory.
    
    Performs basic answer synthesis with no entropy-aware or diffusion-aware features.
    """
    
    def __init__(
        self, 
        model_config: Optional[Dict[str, Any]] = None,
        model_name: str = "gemini-2.5-flash",  # LLM for answer synthesis
        temperature: float = 0.0,
        max_tokens: int = 1024
    ):
        """
        Initialize the Enhanced QA Agent.
        
        Args:
            model_config: Configuration for the LLM
            model_name: Name of the model to use
            temperature: Temperature for text generation (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
        """
        # deterministic defaults; enforce top_p=1.0
        if model_config is None:
            model_config = {
                "model_name": model_name,
                "model_type": "google_gemini",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": max_tokens,
            }
        else:
            model_config = {**model_config, "temperature": 0.0, "top_p": 1.0}

        super().__init__("qa_agent", model_config, model_name)
        self.temperature = 0.0
        self.max_tokens = max(100, min(4096, max_tokens))
        self.conversation_history: List[Dict[str, str]] = []
        
        self.system_prompt = """You are an extractive QA system. Use only the provided evidence.

Rules:
- Extract the entity that fulfills the asked role according to the evidence, 
  even if the evidence wording differs from the question (e.g., "partner" for 
  "spouse", "founded" for "founder").
- Copy the entity name exactly as written in the evidence.
- Record the evidence term used (e.g., "partner", "wife", "husband") if it 
  differs from the question role (e.g., "spouse").
- Only return "unknown" if the evidence genuinely does not contain information 
  that could answer the question.
- No explanations. Return JSON only.

Return JSON:
{
  "answer": "Entity Name",
  "confidence": 0.9,
  "sources": ["doc_id"],
  "supporting_evidence": [
    {"text": "...", "source": "doc_id", "relevance": 0.9}
  ],
  "evidence_term": "partner"  // Optional: the term actually used in evidence
}"""

    def _extract_answer_from_evidence(
        self,
        question: str,
        passage_text: str,
        all_passages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Extract answer from evidence when QA said "unknown" but evidence exists.
        
        Uses simple pattern matching to extract entities/locations from passages.
        This is a fallback when the LLM hesitates despite having valid evidence.
        """
        import re
        
        question_lower = question.lower()
        location_keywords = ["headquarter", "headquarters", "hq", "based", "located", "location", "where", "city", "country", "province", "state"]
        is_location_q = any(kw in question_lower for kw in location_keywords)
        
        if is_location_q:
            pattern1 = r'\b(?:in|from|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?:,\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*))?'
            matches = re.findall(pattern1, passage_text)
            if matches:
                for match in matches:
                    if match[1]:
                        return f"{match[0]}, {match[1]}"
                    elif match[0]:
                        return match[0]
            pattern2 = r'(?:operated|run|controlled|managed)\s+(?:from|in)\s+[^,]+(?:,\s*)?in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
            matches = re.findall(pattern2, passage_text)
            if matches:
                return matches[0]
        
        capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', passage_text)
        exclude_words = {"The", "A", "An", "This", "That", "Yes", "No", "Unknown", "Question", "Answer", "Evidence", "Source", "Document"}
        question_words = set(re.findall(r'\b([A-Z][a-z]+)\b', question))
        exclude_words.update(question_words)
        
        for entity in capitalized:
            entity_words = entity.split()
            if not all(word in exclude_words for word in entity_words):
                if len(entity_words) >= 2 or len(entity) > 6:
                    return entity
        
        for entity in capitalized:
            if len(entity) > 4 and entity not in exclude_words:
                return entity
        
        return None

    def _infer_slot(self, question: str, step_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Infer semantic slot from question/step context.
        Returns slot label like "spouse", "performer", "founder", "parent", etc.
        """
        if not question:
            return "default"
        
        q_lower = question.lower()
        
        # Slot inference patterns
        if any(kw in q_lower for kw in ["spouse", "wife", "husband", "married", "partner"]):
            return "spouse"
        elif any(kw in q_lower for kw in ["performer", "artist", "musician", "singer", "actor"]):
            return "performer"
        elif any(kw in q_lower for kw in ["founder", "founded", "created", "established"]):
            return "founder"
        elif any(kw in q_lower for kw in ["parent", "father", "mother", "dad", "mom"]):
            return "parent"
        elif any(kw in q_lower for kw in ["child", "son", "daughter", "offspring"]):
            return "child"
        elif any(kw in q_lower for kw in ["author", "writer", "wrote", "written by"]):
            return "author"
        elif any(kw in q_lower for kw in ["owner", "owns", "owned by"]):
            return "owner"
        elif any(kw in q_lower for kw in ["headquarters", "hq", "located", "based"]):
            return "location"
        elif any(kw in q_lower for kw in ["company", "organization", "corporation"]):
            return "company"
        else:
            # Fallback: use step description if available
            if step_context:
                step_desc = step_context.get("description", "").lower()
                if "spouse" in step_desc:
                    return "spouse"
                elif "performer" in step_desc or "artist" in step_desc:
                    return "performer"
            
            return "default"

    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Generate a step-specific answer using basic in-context learning.
        
        Performs simple answer synthesis from provided evidence with no entropy-aware
        or diffusion-aware features.
        
        Args:
            input_data: Dictionary containing:
                - 'question': The subquery to answer
                - 'context': List of extracted passages with their sources and relevance
                - Optional 'history': Previous interactions for context
                - Optional 'step_context': Information about the current step
                - Optional 'overall_query': The main question being answered
                - Optional 'previous_answers': Answers from previous steps
                - Optional 'max_history_items': Max history items to include (default: 4)
                - Optional 'min_confidence': Minimum confidence threshold (0.0-1.0)
                
        Returns:
            AgentResponse containing:
                - Synthesized answer
                - Confidence score
                - Supporting evidence and sources
        """
        question = input_data.get('question', '').strip()
        context = input_data.get('context', [])
        history = input_data.get('history', [])
        step_context = input_data.get('step_context', {})
        overall_query = input_data.get('overall_query', '')
        previous_answers = input_data.get('previous_answers', {})
        max_history = int(input_data.get('max_history_items', 4))
        min_confidence = max(0.0, min(1.0, float(input_data.get('min_confidence', 0.0))))
        
        # Normalize question for consistent processing
        question = tokenization_utils.normalize_query(question)
        
        if not question:
            return AgentResponse(
                content=json.dumps({
                    "question": "",
                    "answer": "Error: No question provided",
                    "confidence": 0.0,
                    "reasoning": "No question provided",
                    "sources": [],
                    "supporting_evidence": []
                }),
                metadata={
                    "error": "No question provided",
                    "question": "",
                    "has_context": bool(context),
                    "num_sources": 0
                }
            )
            
        if not context:
            return AgentResponse(
                content=json.dumps({
                    "question": question,
                    "answer": "No context provided to answer this question.",
                    "confidence": 0.0,
                    "reasoning": "No evidence available",
                    "sources": [],
                    "supporting_evidence": []
                }),
                metadata={
                    "question": question,
                    "has_context": False,
                    "num_sources": 0,
                    "confidence": 0.0
                }
            )
        
        try:
            # Build a concise extractive prompt
            prompt_lines = [self.system_prompt]
            prompt_lines.append(f"Question: {tokenization_utils.preprocess_llm_input(question)}")
            if overall_query:
                prompt_lines.append(f"Overall question (context): {tokenization_utils.preprocess_llm_input(overall_query)}")
            prompt_lines.append("Evidence:")
            for i, doc in enumerate(context):
                doc_id = doc.get('document_id', f'doc_{i+1}')
                text = doc.get('text', '').strip()
                relevance = doc.get('relevance', 0.0)
                prompt_lines.append(f"- [source: {doc_id}, relevance: {relevance:.2f}] {text}")
            


            prompt_lines.append("Return only the JSON object.")
            prompt = "\n".join(prompt_lines)

            # Log the QA request
            logger.info(f"Generating step-specific answer for subquery: {question[:100]}...")
            logger.debug(f"Using {len(context)} context items, min_confidence={min_confidence}")
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            # Get the LLM response with token tracking
            response_text, token_usage = await self.generate_text_with_usage(
                prompt=prompt,
                temperature=self.temperature
            )
            
            logger.debug(f"LLM response length: {len(response_text)} characters")
            logger.debug(f"LLM response preview: {response_text[:200]}...")
            
            if not response_text or not response_text.strip():
                raise ValueError("Empty response from LLM")
            
            # Postprocess the LLM response
            response = tokenization_utils.postprocess_answer(response_text, output_type="json")
            
            try:
                # Extract JSON from the response
                # Strip markdown formatting first
                clean_response = TokenizationUtils.strip_markdown_json(response)
                
                # Additional JSON repair for common issues
                if '"sources":' in clean_response and '"sources": [' not in clean_response:
                    # Fix incomplete sources field
                    clean_response = clean_response.replace('"sources": ', '"sources": []')
                
                if '"supporting_evidence":' in clean_response and '"supporting_evidence": [' not in clean_response:
                    # Fix incomplete supporting_evidence field
                    clean_response = clean_response.replace('"supporting_evidence": ', '"supporting_evidence": []')
                
                # Try to repair JSON if it fails
                try:
                    result = json.loads(clean_response)
                except json.JSONDecodeError:
                    # Try to repair common JSON issues
                    repaired_response = TokenizationUtils.repair_json(clean_response)
                    result = json.loads(repaired_response)
                
                # Validate the response structure
                required_keys = ["answer"]
                if not all(key in result for key in required_keys):
                    raise ValueError("Missing required fields in response")
                
                # Parse the response
                supporting_evidence = [
                    Evidence(
                        text=str(e.get("text", "")), 
                        source=str(e.get("source", "")),
                        relevance=float(e.get("relevance", 0.0))
                    )
                    for e in result.get("supporting_evidence", [])
                    if e.get("text") and e.get("source")
                ]
                
                answer = Answer(
                    question=result.get("question", question),
                    answer=result.get("answer", ""),
                    confidence=min(1.0, max(0.0, float(result.get("confidence", 0.0)))),
                    reasoning=result.get("reasoning", ""),
                    sources=list(set(result.get("sources", []))),  # Remove duplicates
                    supporting_evidence=supporting_evidence,
                    evidence_term=result.get("evidence_term")  # Capture from LLM if provided
                )
                
                # Filter out evidence without sources
                answer.supporting_evidence = [
                    e for e in answer.supporting_evidence 
                    if e.source and e.source != "unknown"
                ]
                
                # Update sources list based on actual evidence
                answer.sources = list(set(e.source for e in answer.supporting_evidence))
                
                # ✅ FIX 1: Disallow "unknown" when valid extracted evidence exists
                # This enforces the invariant: "if evidence exists, use it"
                # Prevents semantic hesitation from causing information loss
                if answer.answer.lower().strip() == "unknown":
                    # Two-tier relevance threshold
                    STRONG_EVIDENCE_THRESHOLD = 0.30
                    WEAK_EVIDENCE_THRESHOLD = 0.20
                    
                    # Separate strong and weak evidence
                    strong_evidence = [
                        c for c in context 
                        if isinstance(c, dict) 
                        and isinstance(c.get("relevance"), (int, float))
                        and float(c.get("relevance", 0.0)) >= STRONG_EVIDENCE_THRESHOLD
                        and c.get("text", "").strip()
                    ]
                    
                    weak_evidence = [
                        c for c in context 
                        if isinstance(c, dict) 
                        and isinstance(c.get("relevance"), (int, float))
                        and float(c.get("relevance", 0.0)) >= WEAK_EVIDENCE_THRESHOLD
                        and float(c.get("relevance", 0.0)) < STRONG_EVIDENCE_THRESHOLD
                        and c.get("text", "").strip()
                    ]
                    
                    # Prefer strong evidence, fall back to weak
                    valid_evidence = strong_evidence if strong_evidence else weak_evidence
                    evidence_tier = "strong" if strong_evidence else "weak"
                    
                    if valid_evidence:
                        # Force extraction from evidence instead of "unknown"
                        best_passage = max(valid_evidence, key=lambda x: float(x.get("relevance", 0.0)))
                        best_text = best_passage.get("text", "").strip()
                        best_relevance = float(best_passage.get("relevance", 0.0))
                        
                        extracted_answer = self._extract_answer_from_evidence(
                            question=question,
                            passage_text=best_text,
                            all_passages=valid_evidence
                        )
                        
                        # ✅ Guard: Only override if extraction returns valid, non-generic answer
                        invalid_answers = {"unknown", "n/a", "na", "none", "", "the", "a", "an"}
                        extracted_lower = extracted_answer.lower().strip() if extracted_answer else ""
                        is_valid_extraction = (
                            extracted_answer 
                            and extracted_lower not in invalid_answers
                            and len(extracted_lower) > 2  # Must be substantial
                        )
                        
                        # For weak evidence, require clean extraction success
                        if evidence_tier == "weak" and not is_valid_extraction:
                            logger.debug(
                                f"[QA] Weak evidence (relevance: {best_relevance:.3f}) but extraction failed or invalid, "
                                f"keeping original 'unknown' answer"
                            )
                        elif is_valid_extraction:
                            logger.info(
                                f"[QA] Blocked 'unknown' answer: {len(valid_evidence)} {evidence_tier} passages exist "
                                f"(best relevance: {best_relevance:.3f}), extracted '{extracted_answer}' from evidence"
                            )
                            answer.answer = extracted_answer
                            # Update confidence based on evidence quality and tier
                            confidence_boost = best_relevance if evidence_tier == "strong" else min(0.6, best_relevance)
                            answer.confidence = max(answer.confidence, confidence_boost)
                            answer.reasoning = (
                                f"Extracted from {evidence_tier} evidence (relevance: {best_relevance:.3f}). "
                                f"Original reasoning: {answer.reasoning}"
                            )
                            # Update supporting evidence to include the best passage
                            if not any(e.text == best_text for e in answer.supporting_evidence):
                                answer.supporting_evidence.append(
                                    Evidence(
                                        text=best_text,
                                        source=str(best_passage.get("document_id", "unknown")),
                                        relevance=best_relevance
                                    )
                                )
                
                # ✅ REMOVED: Dangerous rewriting of "unknown" to extracted entity
                # This breaks separation of concerns: QA compresses, Manager tracks, Gate commits
                # Evidence extraction now happens via slot_candidates (Fix 1)
                
                # If confidence is below threshold, update the answer
                if answer.confidence < min_confidence:
                    answer.answer = (
                        f"I'm not very confident about this answer (confidence: {answer.confidence:.2f}). "
                        f"Here's my best attempt based on the available information:\n\n{answer.answer}"
                    )
                
                # Update history
                self.conversation_history.append({"role": "user", "content": f"Q: {question}"})
                self.conversation_history.append({"role": "assistant", "content": f"A: {answer.answer[:200]}..."})
                
                # Prepare metadata
                answer_dict = answer.model_dump()
                
                metadata = {
                    "question": question,
                    "confidence": answer.confidence,
                    "reasoning": answer.reasoning,
                    "num_sources": len(answer.sources),
                    "num_evidence": len(answer.supporting_evidence),
                    "sources": answer.sources,
                    "has_context": True,
                    "min_confidence": min_confidence,
                    "step_context": step_context,
                    "overall_query": overall_query,
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "qa_parameters": {
                        "max_history": max_history,
                        "min_confidence": min_confidence,
                        "temperature": self.temperature
                    },
                    "token_usage": token_usage
                }
                
                # ✅ INFER SLOT AND EMIT SLOT-LABELED CANDIDATES
                # Emit candidates from evidence even if answer = "unknown"
                slot = self._infer_slot(question, step_context)
                
                # Build evidence candidates (tentative hypotheses from evidence)
                evidence_candidates = []
                import re
                for ev in answer.supporting_evidence:
                    # Extract entity-like spans from evidence text
                    matches = re.findall(
                        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
                        ev.text
                    )
                    for m in matches:
                        # Filter: minimum length and exclude common words
                        if len(m) > 2 and m not in ["The", "A", "An", "This", "That", "Yes", "No"]:
                            evidence_candidates.append({
                                "answer": m,
                                "slot": slot,
                                "confidence": max(answer.confidence, 0.4),  # Tentative confidence
                                "evidence_count": 1,
                                "sources": [ev.source],
                                "status": "tentative"  # Tentative, not committed
                            })
                
                # Build final slot_candidates list
                slot_candidates = []
                
                # If answer is not "unknown", emit the extracted candidate
                if answer.answer.lower().strip() != "unknown":
                    # ✅ FIX #1: Extract evidence_term DESCRIPTIVELY (don't normalize)
                    evidence_term = None
                    
                    # Try to get from LLM response first (if it provided evidence_term)
                    if answer.evidence_term:
                        evidence_term = answer.evidence_term
                    else:
                        # Fallback: extract from supporting_evidence descriptively
                        for ev in answer.supporting_evidence:
                            ev_text_lower = ev.text.lower()
                            # ✅ FIX #1: Keep descriptive, don't normalize
                            if "partner" in ev_text_lower:
                                evidence_term = "partner"
                            elif "wife" in ev_text_lower:
                                evidence_term = "wife"
                            elif "husband" in ev_text_lower:
                                evidence_term = "husband"
                            elif "married" in ev_text_lower or "marriage" in ev_text_lower:
                                evidence_term = "married"
                            elif "spouse" in ev_text_lower:
                                evidence_term = "spouse"
                            # Add more role-specific terms as needed
                            elif "founded" in ev_text_lower or "founder" in ev_text_lower:
                                evidence_term = "founded"
                            elif "performer" in ev_text_lower or "performed" in ev_text_lower:
                                evidence_term = "performer"
                            elif "headquartered" in ev_text_lower or "headquarters" in ev_text_lower:
                                evidence_term = "headquarters"
                            if evidence_term:
                                break  # Use first match found
                    
                    slot_candidates.append({
                        "answer": answer.answer,
                        "slot": slot,
                        "confidence": answer.confidence,
                        "evidence_count": len(answer.supporting_evidence),
                        "sources": answer.sources,
                        "status": "extracted",  # ✅ FIX #2: Extracted, not committed
                        "evidence_term": evidence_term  # What evidence actually said
                    })
                
                # Always add evidence candidates (tentative hypotheses)
                slot_candidates.extend(evidence_candidates)
                
                metadata["slot_candidates"] = slot_candidates
                
                # ✅ FIX 4: Explicitly mark abstention in metadata
                metadata["abstained"] = (answer.answer.lower().strip() == "unknown")
                
                return AgentResponse(
                    content=json.dumps(answer_dict),
                    metadata=metadata
                )
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing LLM response: {str(e)}")
                
                # Fallback response when parsing fails
                fallback_evidence = [
                    Evidence(
                        text=c.get("text", ""),
                        source=c.get("document_id", "unknown"),
                        relevance=c.get("relevance", 0.0)
                    )
                    for c in context[:3]  # Include first 3 context items as evidence
                ]
                
                fallback_answer = Answer(
                    question=question,
                    answer=(
                        "I'm having trouble understanding the context. "
                        "Here's what I can say based on the information: "
                        f"{response[:500]}"
                    ),
                    confidence=0.3,
                    reasoning="Fallback reasoning due to parsing error",
                    sources=list(set(c.get("document_id", "unknown") for c in context)),
                    supporting_evidence=fallback_evidence
                )
                
                return AgentResponse(
                    content=fallback_answer.model_dump_json(),
                    metadata={
                        "question": question,
                        "confidence": 0.3,
                        "reasoning": "Fallback reasoning",
                        "num_sources": len(fallback_answer.sources),
                        "num_evidence": len(fallback_answer.supporting_evidence),
                        "sources": fallback_answer.sources,
                        "error": f"Error parsing response: {str(e)}",
                        "fallback": True
                    }
                )
                
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return AgentResponse(
                content=json.dumps({
                    "question": question,
                    "answer": f"Error: {error_msg}",
                    "confidence": 0.0,
                    "reasoning": "Error occurred during processing",
                    "sources": [],
                    "supporting_evidence": []
                }),
                metadata={
                    "error": error_msg,
                    "question": question,
                    "confidence": 0.0,
                    "reasoning": "Error occurred",
                    "num_sources": 0,
                    "num_evidence": 0,
                    "has_context": bool(context)
                }
            )
    
    
    async def generate_followup_questions(
        self, 
        question: str, 
        answer: str, 
        num_questions: int = 3,
        context: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Generate relevant follow-up questions based on the original Q&A and context.
        
        Args:
            question: The original question
            answer: The answer provided
            num_questions: Number of follow-up questions to generate (1-10)
            context: Optional list of context documents used for the answer
            
        Returns:
            List of follow-up questions (up to num_questions)
        """
        # Validate input
        num_questions = max(1, min(10, int(num_questions)))
        
        # Prepare the prompt
        prompt = f"""Generate {num_questions} relevant follow-up questions based on the Q&A below.
        
        ### Original Question:
        {question}
        
        ### Answer:
        {answer}
        """
        
        # Add context if available
        if context:
            prompt += "\n### Context Used for Answer:\n"
            for i, doc in enumerate(context[:3]):  # Include first 3 context items
                doc_id = doc.get('document_id', f'doc_{i+1}')
                text = doc.get('text', '').strip()
                prompt += f"\n[Context {i+1}, ID: {doc_id}]\n{text[:500]}"
                if len(text) > 500:
                    prompt += "... [truncated]"
        
        # Add instructions
        prompt += f"""
        
        ### Instructions:
        1. Generate exactly {num_questions} follow-up questions that:
           - Explore related aspects not fully covered
           - Seek clarification on complex points
           - Ask for examples or applications
           - Question assumptions or limitations
        
        2. Make sure questions are:
           - Clear and specific
           - Directly related to the original Q&A
           - Not answerable with just 'yes' or 'no'
        
        3. Format: One question per line, no numbering or bullet points.
        """
        
        try:
            # Log the request
            logger.info(f"Generating {num_questions} follow-up questions for: {question[:100]}...")
            
            # Get the LLM response
            response = await self.generate_text(
                prompt=prompt,
                temperature=min(0.7, self.temperature + 0.1)  # Slightly more creative
            )
            
            # Parse the response
            questions = [
                q.strip() 
                for q in response.split('\n') 
                if q.strip() and len(q.strip()) > 5  # Filter out very short lines
            ]
            
            # Clean up questions
            cleaned_questions = []
            for q in questions:
                # Remove any numbering or bullets
                q = q.lstrip('0123456789.-*• ').strip()
                # Ensure it's a question
                if not q.endswith('?'):
                    q = f"{q}?"
                if q not in cleaned_questions:  # Avoid duplicates
                    cleaned_questions.append(q)
            
            # Log the results
            logger.debug(f"Generated {len(cleaned_questions)} follow-up questions")
            
            return cleaned_questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {str(e)}")
            
            # Fallback questions
            fallback_questions = [
                "Can you provide more details about this topic?",
                "What are some related aspects I should know about?",
                "Are there any limitations or exceptions to this answer?",
                "How does this compare to similar concepts?",
                "What are the practical applications of this?",
                "What are the key factors to consider here?",
                "How would this work in a different context?",
                "What are the potential challenges or risks?",
                "What are the next steps I should take?",
                "Where can I find more information about this?"
            ]
            
            return fallback_questions[:num_questions]
