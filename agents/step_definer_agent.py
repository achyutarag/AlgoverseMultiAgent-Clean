from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from .base_agent import BaseAgent, AgentResponse
from .tokenization_utils import tokenization_utils
import json
import logging

logger = logging.getLogger(__name__)

class SubQuery(BaseModel):
    """A sub-query for a specific step."""
    id: str = Field(..., description="Unique identifier for the sub-query")
    query: str = Field(..., description="The specific question to answer")
    purpose: str = Field(..., description="What this sub-query aims to accomplish")
    priority: int = Field(1, description="Priority level (1=highest)")
    context_needed: List[str] = Field(default_factory=list, description="Types of context needed for this sub-query")

class StepDefinerAgent(BaseAgent):
    """
    The Step Definer Agent makes abstract steps executable by generating detailed 
    subqueries tailored for retrieval. It conditions on the original query, plan, 
    current step, and accumulated history to bridge high-level intent and low-level execution.
    """
    
    def __init__(
        self, 
        model_config: Optional[Dict[str, Any]] = None,
        model_name: str = "llama-2-13b-chat-hf",  # LLM for step definition
        max_subqueries: int = 3
    ):
        """
        Initialize the Step Definer Agent.
        
        Args:
            model_config: Configuration for the LLM
            model_name: Name of the model to use (if model_config not provided)
            max_subqueries: Maximum number of sub-queries to generate per step
        """
        super().__init__("step_definer_agent", model_config, model_name)
        self.max_subqueries = max_subqueries
        
        self.system_prompt = """You are an expert at converting abstract reasoning steps into specific, executable sub-queries for retrieval-augmented generation. Your task is to:

1. **Context Grounding**: Condition on the original query q, the overall plan P, the current step si, and accumulated history Hi-1 = {(s1, a1), ..., (si-1, ai-1)}

2. **Subquery Generation**: Generate detailed subqueries tailored for retrieval that bridge high-level intent and low-level execution

3. **Precision Focus**: Create subqueries that enable precise and relevant document retrieval by being specific about what information is needed

Guidelines for subquery generation:
- Make subqueries specific and actionable for retrieval
- Consider what context from previous steps might be relevant
- Ensure subqueries are focused enough to retrieve precise information
- Include necessary context and constraints in the subquery
- Prioritize subqueries based on their importance to the step objective
- Consider what types of documents or information sources would be most relevant

Return your response as a JSON object with this structure:
{
    "step_id": "ID of the current step",
    "step_description": "Description of the current step",
    "reasoning": "Your reasoning for these specific sub-queries",
    "context_analysis": "Analysis of relevant context from previous steps",
    "sub_queries": [
        {
            "id": "subquery_1",
            "query": "Specific, focused question for retrieval",
            "purpose": "What this sub-query aims to accomplish",
            "priority": 1,
            "context_needed": ["factual", "statistical", "comparative"]
        }
    ]
}

Examples of good subquery generation:

**Step**: "Research environmental benefits of renewable energy"
- Subquery 1: "What are the specific environmental benefits of solar energy production?"
- Subquery 2: "What environmental advantages does wind energy have over fossil fuels?"
- Subquery 3: "How does hydroelectric power impact local ecosystems?"

**Step**: "Compare economic policies between countries"
- Subquery 1: "What are Japan's current monetary policy rates and targets?"
- Subquery 2: "What fiscal policies has South Korea implemented in the last 5 years?"
- Subquery 3: "How do Japan and South Korea differ in their trade policy approaches?"
"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process the current step and generate detailed subqueries conditioned on context and history.
        
        Args:
            input_data: Dictionary containing:
                - 'step': Dict with 'id', 'description', 'objective', 'dependencies', 'critical'
                - 'plan': The overall plan with all steps
                - 'history': List of previous interactions and their results
                - 'context': Additional context for the task
                - 'previous_answers': Dict of {step_id: answer} for completed steps
                - 'max_subqueries': Override default max subqueries
                
        Returns:
            AgentResponse containing the generated sub-queries and metadata
        """
        step = input_data.get('step', {})
        plan = input_data.get('plan', {})
        history = input_data.get('history', [])
        context = input_data.get('context', {})
        previous_answers = input_data.get('previous_answers', {})
        max_subqueries = min(input_data.get('max_subqueries', self.max_subqueries), 5)  # Cap at 5
        
        if not step or 'id' not in step:
            return AgentResponse(
                content="Error: Invalid or missing step information",
                metadata={"error": "Invalid step data"}
            )
        
        try:
            # Prepare the enhanced prompt with full context (preprocess for LLM)
            prompt = f"""{self.system_prompt}
            
            ### Main Question:
            {tokenization_utils.preprocess_llm_input(plan.get('main_question', 'Not specified'))}
            
            ### Disambiguated Query:
            {tokenization_utils.preprocess_llm_input(plan.get('disambiguated_query', plan.get('main_question', 'Not specified')))}
            
            ### Query Type:
            {plan.get('query_type', 'unknown')}
            
            ### Current Step:
            ID: {step.get('id', 'unknown')}
            Description: {tokenization_utils.preprocess_llm_input(step.get('description', 'No description'))}
            Objective: {tokenization_utils.preprocess_llm_input(step.get('objective', 'No objective'))}
            Dependencies: {', '.join(step.get('dependencies', [])) or 'None'}
            Critical: {step.get('critical', False)}
            Expected Output: {tokenization_utils.preprocess_llm_input(step.get('expected_output', 'Not specified'))}
            """
            
            # Add context if available
            if context:
                prompt += f"\n### Additional Context:\n{json.dumps(context, indent=2)}"
            
            # Add information about previous steps and their answers
            if previous_answers:
                prompt += "\n\n### Previous Steps and Answers:"
                for step_id, answer in previous_answers.items():
                    # Truncate long answers for context
                    answer_preview = str(answer)[:300] + "..." if len(str(answer)) > 300 else str(answer)
                    prompt += f"\n- Step {step_id}: {answer_preview}"
            
            # Add relevant conversation history
            if history:
                prompt += "\n\n### Conversation History:"
                for i, h in enumerate(history[-5:]):  # Last 5 messages for more context
                    role = h.get('role', 'unknown').upper()
                    content = h.get('content', '')[:400]  # Truncate long content
                    prompt += f"\n{role}: {content}"
            
            # Add the actual instruction
            prompt += f"""
            
            ### Instructions:
            Please generate specific sub-queries that would help accomplish the current step.
            Consider what information is needed and how it contributes to the step's objective.
            Focus on creating subqueries that are:
            1. Specific enough for precise retrieval
            2. Grounded in the context from previous steps
            3. Aligned with the step's objective
            4. Prioritized by importance
            
            Generate up to {max_subqueries} subqueries, prioritizing the most important ones.
            """
            
            # Generate the response using the LLM
            response = await self.generate_text(
                prompt,
                temperature=0.3,
                max_new_tokens=1536
            )
            
            # Postprocess the LLM response
            response = tokenization_utils.postprocess_answer(response, output_type="json")
            
            # Try to extract JSON from the response
            try:
                # Find JSON part in the response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No valid JSON found in response")
                
                # Validate the response structure
                required_keys = ["step_id", "step_description", "reasoning", "sub_queries"]
                if not all(key in result for key in required_keys):
                    raise ValueError(f"Missing required keys in response. Expected: {required_keys}")
                
                # Validate sub_queries structure
                if not isinstance(result["sub_queries"], list):
                    raise ValueError("sub_queries must be a list")
                
                # Limit subqueries to max_subqueries
                result["sub_queries"] = result["sub_queries"][:max_subqueries]
                
                # Validate and enhance each subquery
                validated_subqueries = []
                for i, sq in enumerate(result["sub_queries"]):
                    if not all(k in sq for k in ["id", "query", "purpose"]):
                        logger.warning(f"Sub-query {i} is missing required fields, skipping")
                        continue
                    
                    # Add priority if not present
                    if "priority" not in sq:
                        sq["priority"] = i + 1
                    
                    # Add context_needed if not present
                    if "context_needed" not in sq:
                        sq["context_needed"] = ["factual"]  # Default context type
                    
                    validated_subqueries.append(sq)
                
                result["sub_queries"] = validated_subqueries
                
                # Add context_analysis if not present
                if "context_analysis" not in result:
                    result["context_analysis"] = "Analysis of relevant context from previous steps"
                
                # Update history
                self._update_history("user", f"Define sub-queries for step: {step.get('id')}")
                self._update_history("assistant", response)
                
                # Parse sub-queries into SubQuery objects
                sub_queries = [
                    SubQuery(
                        id=sq.get("id", f"subq_{i+1}"),
                        query=sq["query"],
                        purpose=sq["purpose"],
                        priority=int(sq.get("priority", i + 1)),
                        context_needed=sq.get("context_needed", ["factual"])
                    )
                    for i, sq in enumerate(result["sub_queries"])
                ]
                
                # Sort sub-queries by priority (ascending, so 1 comes first)
                sub_queries.sort(key=lambda x: x.priority)
                
                return AgentResponse(
                    content=json.dumps({
                        "step_id": result["step_id"],
                        "step_description": result["step_description"],
                        "reasoning": result["reasoning"],
                        "context_analysis": result["context_analysis"],
                        "sub_queries": [sq.dict() for sq in sub_queries]
                    }),
                    metadata={
                        "step_id": result["step_id"],
                        "step_description": result["step_description"],
                        "reasoning": result["reasoning"],
                        "context_analysis": result["context_analysis"],
                        "sub_queries": [sq.dict() for sq in sub_queries],
                        "num_subqueries": len(sub_queries),
                        "step_definer_parameters": {
                            "max_subqueries": max_subqueries,
                            "model": self.model_name,
                            "temperature": 0.3
                        }
                    }
                )
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                
                # Fallback: create a simple subquery
                fallback_subquery = {
                    "step_id": step.get("id", "unknown"),
                    "step_description": step.get("description", "No description"),
                    "reasoning": "Unable to parse detailed subqueries, using fallback approach",
                    "context_analysis": "Limited context analysis due to parsing error",
                    "sub_queries": [
                        {
                            "id": "subquery_1",
                            "query": step.get("description", "Research relevant information"),
                            "purpose": f"Accomplish step objective: {step.get('objective', 'Unknown objective')}",
                            "priority": 1,
                            "context_needed": ["factual"]
                        }
                    ]
                }
                
                return AgentResponse(
                    content=json.dumps(fallback_subquery),
                    metadata={
                        "step_id": step.get("id", "unknown"),
                        "step_description": step.get("description", "No description"),
                        "reasoning": "Fallback subquery due to parsing error",
                        "context_analysis": "Limited analysis",
                        "sub_queries": fallback_subquery["sub_queries"],
                        "error": f"Failed to parse response: {str(e)}",
                        "fallback": True
                    }
                )
                
            except Exception as e:
                logger.error(f"Error processing LLM response: {e}")
                return AgentResponse(
                    content=f"Error processing LLM response: {str(e)}",
                    metadata={
                        "error": "Response processing error",
                        "exception": str(e),
                        "llm_response": response
                    }
                )
                
        except Exception as e:
            logger.error(f"Error in StepDefinerAgent.process: {e}", exc_info=True)
            return AgentResponse(
                content=f"Error generating sub-queries: {str(e)}",
                metadata={
                    "error": "Processing error",
                    "exception": str(e)
                }
            )
