from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from .base_agent import BaseAgent, AgentResponse
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
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
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
        
        self.system_prompt = """You are an expert at synthesizing answers using in-context learning with step-specific context. Your task is to:

1. **Step-Specific Synthesis**: Generate answers that are tailored to the current step's objective and context

2. **Evidence Integration**: Synthesize information from multiple extracted passages into coherent, well-structured answers

3. **Grounded Reasoning**: Base your answers strictly on the provided evidence while maintaining logical coherence

4. **Context Awareness**: Consider the broader context of the overall query and previous step results

5. **Confidence Assessment**: Provide honest confidence scores based on the quality and completeness of available evidence

Guidelines for answer generation:
- Synthesize information from multiple sources when they complement each other
- Address the specific subquery directly and comprehensively
- Maintain logical flow and coherence in your reasoning
- Include relevant details that support your main answer
- Note any limitations or uncertainties in the available evidence
- Provide clear reasoning for your conclusions
- Rate confidence honestly based on evidence quality and completeness

Format your response as a valid JSON object with this structure:
{
    "question": "The original subquery",
    "answer": "Your comprehensive answer based on the evidence",
    "confidence": 0.0-1.0,  // Your confidence in the answer
    "reasoning": "Your step-by-step reasoning process",
    "sources": ["doc1_id", "doc2_id"],  // IDs of sources used
    "supporting_evidence": [
        {
            "text": "Relevant passage from context",
            "source": "source_document_id",
            "relevance": 0.9  // How relevant this evidence is
        }
    ]
}

Examples of good answer synthesis:

**Subquery**: "What are the environmental benefits of solar energy?"
**Evidence**: Multiple passages about solar energy benefits
**Answer**: "Solar energy provides several key environmental benefits: 1) Zero greenhouse gas emissions during operation, unlike fossil fuels; 2) Reduced air pollution from power generation; 3) Lower water usage compared to traditional power plants; 4) Minimal environmental impact during operation. However, manufacturing does require energy and materials."
**Reasoning**: "Combined information from multiple sources to provide comprehensive overview of environmental benefits while noting limitations."

**Subquery**: "How do Japan and South Korea differ in monetary policy?"
**Evidence**: Specific policy details from both countries
**Answer**: "Japan and South Korea differ significantly in their monetary policy approaches: Japan maintains ultra-low interest rates (near zero) with quantitative easing, while South Korea uses more conventional monetary policy with higher interest rates. Japan focuses on combating deflation, while South Korea targets inflation control."
**Reasoning**: "Compared specific policy elements from both countries to highlight key differences."""
    
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
            # Prepare the enhanced prompt with step-specific context
            prompt = f"""{self.system_prompt}
            
            ### Subquery to Answer:
            {question}
            
            ### Step Context:
            {json.dumps(step_context, indent=2) if step_context else "No specific step context"}
            """
            
            # Add overall query context if available
            if overall_query:
                prompt += f"\n### Overall Query Context:\n{overall_query}"
            
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
            prompt += """
            
            ### Instructions:
            Please synthesize a comprehensive answer to the subquery using the evidence above.
            Your response MUST be a valid JSON object with this exact structure:
            {
                "question": "The original subquery",
                "answer": "Your comprehensive answer based on the evidence",
                "confidence": 0.0-1.0,  // Your confidence in the answer
                "reasoning": "Your step-by-step reasoning process",
                "sources": ["doc1_id", "doc2_id"],  // IDs of sources used
                "supporting_evidence": [
                    {
                        "text": "Relevant passage from context",
                        "source": "source_document_id",
                        "relevance": 0.9  // How relevant this evidence is
                    }
                ]
            }
            
            Guidelines:
            1. Synthesize information from multiple sources when they complement each other
            2. Address the subquery directly and comprehensively
            3. Maintain logical flow and coherence in your reasoning
            4. Include relevant details that support your main answer
            5. Note any limitations or uncertainties in the available evidence
            6. Rate your confidence honestly based on evidence quality and completeness
            7. Provide clear reasoning for your conclusions
            """
            
            # Log the QA request
            logger.info(f"Generating step-specific answer for subquery: {question[:100]}...")
            logger.debug(f"Using {len(context)} context items, min_confidence={min_confidence}")
            
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
                    content=answer.json(ensure_ascii=False, indent=2),
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
                    content=fallback_answer.json(ensure_ascii=False, indent=2),
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
    
    async def generate_text(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The prompt to generate text from
            temperature: Temperature for text generation (0.0-1.0)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        # This is a placeholder implementation that should be overridden by a real LLM call
        # In a real implementation, this would call the actual LLM API
        return ""
    
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
                temperature=min(0.7, self.temperature + 0.1),  # Slightly more creative
                max_new_tokens=512
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
