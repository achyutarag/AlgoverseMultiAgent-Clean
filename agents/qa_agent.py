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

class QAAgent(BaseAgent):
    """
    Enhanced QA Agent that synthesizes answers using in-context learning with 
    step-specific context. It produces responses for each step which are passed 
    to the next iteration, enabling grounded reasoning throughout the trajectory.
    """
    
    def __init__(
        self, 
        model_config: Optional[Dict[str, Any]] = None,
        model_name: str = "gemini-2.5-flash",  # LLM for answer synthesis
        temperature: float = 0.3,
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
        super().__init__("qa_agent", model_config, model_name)
        self.temperature = max(0.0, min(1.0, temperature))
        self.max_tokens = max(100, min(4096, max_tokens))
        self.conversation_history: List[Dict[str, str]] = []
        
        self.system_prompt = """You are an expert at answering questions using provided evidence. 

Your task is to:
1. Answer the question using ONLY the provided evidence
2. Be concise and accurate
3. Provide confidence scores based on evidence quality
4. Return valid JSON

Return a JSON object with this structure:
{
    "answer": "Your answer to the question",
    "confidence": 0.95,
    "reasoning": "Why you gave this answer",
    "sources": ["source1", "source2"],
    "supporting_evidence": [
        {
            "text": "Relevant text from evidence",
            "source": "source_id",
            "relevance": 0.9
        }
    ]
}

**IMPORTANT**: Always return valid JSON. Do not include any text before or after the JSON object."""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Generate a step-specific answer using in-context learning with provided evidence.
        
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
            AgentResponse containing the synthesized answer and metadata
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
            # Prepare the enhanced prompt with step-specific context (preprocess for LLM)
            prompt = f"""{self.system_prompt}
            
### Subquery to Answer:
{tokenization_utils.preprocess_llm_input(question)}

### Step Context:
{json.dumps(step_context, indent=2) if step_context else "No specific step context"}
"""
            
            # Add overall query context prominently at the top (if available)
            if overall_query:
                prompt += f"\n\n### ORIGINAL QUESTION (USE TO GUIDE ANSWER FORMAT):\n{overall_query}\n"
            
            # Add previous answers if available
            if previous_answers:
                prompt += "\n\n### Previous Step Answers:"
                for step_id, answer in previous_answers.items():
                    answer_preview = str(answer)[:200] + "..." if len(str(answer)) > 200 else str(answer)
                    prompt += f"\n- Step {step_id}: {answer_preview}"
            
            # Add conversation history if available
            if history:
                history_str = "\n".join(
                    f"{h.get('role', 'user').upper()}: {h.get('content', '')}"
                    for h in history[-max_history:]
                )
                prompt += f"\n### Previous Conversation (most recent last):\n{history_str}\n\n"
            
            # Add context documents with enhanced formatting
            prompt += "\n### Extracted Evidence:"
            for i, doc in enumerate(context):
                doc_id = doc.get('document_id', f'doc_{i+1}')
                text = doc.get('text', '').strip()
                relevance = doc.get('relevance', 0.0)
                reasoning = doc.get('reasoning', '')
                
                prompt += (
                    f"\n[Evidence {i+1}, Source: {doc_id}, Relevance: {relevance:.2f}]\n"
                    f"Text: {text}\n"
                    f"Reasoning: {reasoning}\n"
                )
            
            # Add instructions for synthesis
            prompt += f"""

### Instructions:
Please synthesize a CONCISE, DIRECT answer to the subquery using the evidence above.
Your response MUST be a valid JSON object with this exact structure:
{{
    "question": "The original subquery",
    "answer": "Your DIRECT answer - be concise and specific (typically 1-5 words for factual questions, just 'Yes' or 'No' for yes/no questions)",
    "confidence": 0.0-1.0,
    "reasoning": "Your step-by-step reasoning process",
    "sources": ["doc1_id", "doc2_id"],
    "supporting_evidence": [
        {{
            "text": "Relevant passage from context",
            "source": "source_document_id",
            "relevance": 0.9
        }}
    ]
}}

### CRITICAL: Answer Format Rules (READ FIRST):

**1. Entity Names (e.g., "What is the name of...", "Who created...", "formed by who?")**
   - Rule: Extract ONLY the entity name - nothing else
   - ❌ WRONG: "[group name] formed by [entity]" (includes extra context)
   - ✅ CORRECT: "[entity name]" (just the entity)
   - Example: Question "formed by who?" → Answer: "YG Entertainment" (not "winner formed by YG Entertainment")

**2. Numerical Questions (e.g., "how many people?", "how many cars?", "what capacity?")**
   - Rule: Extract ONLY the number and unit (if specified) - nothing else
   - ❌ WRONG: "[venue name] [number] people" (includes venue name)
   - ❌ WRONG: "[number] people" when ground truth is "[number] seated" (wrong unit)
   - ✅ CORRECT: "[number]" or "[number] [unit]" (just the number and correct unit from evidence)
   - Example: Question "can serve how many guests?" → Answer: "400 guests" (not "white house 400 people")
   - If evidence specifies a unit (e.g., "seated", "people", "cars"), use that exact unit

**3. Location Questions (e.g., "in what [city]?", "located in what [city]?")**
   - Rule: Return format "[neighborhood], [city]" when question asks for location within a city
   - ❌ WRONG: "[neighborhood]" (missing city)
   - ✅ CORRECT: "[neighborhood], [city]" (complete location)
   - Example: Question "in what [city]?" → Answer: "[neighborhood], [city]" (not just "[neighborhood]")
   - Example: Question "located in what city?" → Answer: "Downtown District, Chicago" (not just "Downtown District")
   - If question asks for a neighborhood within a specific city, include both neighborhood and city

**4. Specific Positions/Titles (e.g., "What position did X hold?", "What was X's role?")**
   - Rule: Extract ONLY ONE position - the most prominent/relevant one if multiple exist
   - ❌ WRONG: "[position1] and [position2] and [position3]" (listing multiple)
   - ❌ WRONG: "[position] of [country/organization]" (adding irrelevant context)
   - ✅ CORRECT: "[position name]" (ONE position - extract the most prominent/relevant one)
   - Rule: If the question asks for "a position" or "the position" (singular) but evidence shows multiple positions:
     * Extract ONLY ONE position based on the evidence:
       - Choose the position that appears most prominently in the evidence (mentioned first, emphasized, most detailed, or held for the longest time period)
       - Choose the position most directly relevant to the question's context (e.g., if question mentions a specific time period, choose the position held during that time)
       - If positions are equally prominent, choose the most significant/highest-ranking one
     * DO NOT list multiple positions - extract ONLY the position name itself (no "of [country]", "of [organization]", etc.)

**5. Yes/No Questions (e.g., "Are X and Y the same?", "Did X do Y?")**
   - Rule: Answer with ONLY "Yes" or "No" - nothing else
   - Even if the question mentions multiple entities with "and" or "both", answer with just "Yes" or "No"

**6. Nationalities/Attributes (e.g., "What nationality was X?")**
   - Rule: Extract ONLY the attribute asked for
   - ❌ WRONG: "[person] was [nationality]" (includes person name)
   - ✅ CORRECT: "[nationality]" (just the attribute)

**7. Time Period Questions (e.g., "during what years?", "served during what timeframe?")**
   - Rule: Extract the time period with connecting words if present in evidence
   - ❌ WRONG: "1450-1494" (missing connector)
   - ✅ CORRECT: "1450 until 1494" or "1450-1494" (preserve format from evidence using connector)

### Guidelines:
1. **BE CONCISE**: Answer the subquery directly - typically 1-5 words, rarely more than 1 sentence
2. **BE SPECIFIC**: Extract ONLY the exact answer requested - nothing else
3. **DO NOT** include descriptions, explanations, or multiple facts
4. **DO NOT** list multiple positions/entities - extract ONLY the one asked for
5. **DO NOT** provide reasoning in the answer field - put it in "reasoning" field
6. **DO NOT** include venue names, organization names, or other context unless the question specifically asks for it
7. If the question asks for one thing, provide ONLY that thing
8. Rate your confidence honestly based on evidence quality

**HANDLING AMBIGUOUS QUESTIONS**: If the question asks for one thing but evidence contains multiple valid answers:
   - Choose the answer that is most prominently featured in the evidence:
     * Most emphasized (repeatedly mentioned, highlighted)
     * Most detailed (has more description, context, or information)
     * Longest time period (held for the longest duration)
     * Mentioned first or most prominently
   - Choose the one most directly relevant to the question's context and intent
   - Base your decision on the evidence provided, not on assumptions about what the "correct" answer might be
   - Extract ONE answer, not multiple
"""
            
            # Log the QA request
            logger.info(f"Generating step-specific answer for subquery: {question[:100]}...")
            logger.debug(f"Using {len(context)} context items, min_confidence={min_confidence}")
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            # Get the LLM response
            response = await self.generate_text(
                prompt=prompt,
                temperature=self.temperature
            )
            
            logger.debug(f"LLM response length: {len(response)} characters")
            logger.debug(f"LLM response preview: {response[:200]}...")
            
            if not response or not response.strip():
                raise ValueError("Empty response from LLM")
            
            # Postprocess the LLM response
            response = tokenization_utils.postprocess_answer(response, output_type="json")
            
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
                    supporting_evidence=supporting_evidence
                )
                
                # Filter out evidence without sources
                answer.supporting_evidence = [
                    e for e in answer.supporting_evidence 
                    if e.source and e.source != "unknown"
                ]
                
                # Update sources list based on actual evidence
                answer.sources = list(set(e.source for e in answer.supporting_evidence))
                
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
                    }
                }
                
                return AgentResponse(
                    content=answer.model_dump_json(),
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
