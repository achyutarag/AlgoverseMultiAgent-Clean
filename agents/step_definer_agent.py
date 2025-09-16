from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from .base_agent import BaseAgent, AgentResponse
import json
import logging

logger = logging.getLogger(__name__)

class SubQuery(BaseModel):
    """A sub-query for a specific step."""
    id: str = Field(..., description="Unique identifier for the sub-query")
    query: str = Field(..., description="The specific question to answer")
    purpose: str = Field(..., description="What this sub-query aims to accomplish")
    priority: int = Field(1, description="Priority level (1=highest)")

class StepDefinerAgent(BaseAgent):
    """
    The Step Definer Agent breaks down high-level steps into specific, actionable
    sub-queries based on the current context and conversation history.
    """
    
    def __init__(
        self, 
        model_config: Optional[Dict[str, Any]] = None,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    ):
        """
        Initialize the Step Definer Agent.
        
        Args:
            model_config: Configuration for the LLM
            model_name: Name of the model to use (if model_config not provided)
        """
        super().__init__("step_definer_agent", model_config, model_name)
        
        self.system_prompt = """You are an expert at breaking down complex tasks into specific, 
        answerable sub-queries. Your task is to analyze the current step in the context of the 
        overall plan and generate precise sub-queries that will help accomplish the step's objective.
        
        For each sub-query, consider:
        1. What specific information is needed
        2. How it contributes to the step's objective
        3. What context from previous steps might be relevant
        4. How to phrase it to get the most precise answer
        
        Return your response as a JSON object with this structure:
        {
            "step_id": "ID of the current step",
            "step_description": "Description of the current step",
            "reasoning": "Your step-by-step reasoning for these sub-queries",
            "sub_queries": [
                {
                    "id": "subquery_1",
                    "query": "Specific, focused question",
                    "purpose": "What this sub-query aims to accomplish",
                    "priority": 1  // 1=highest priority
                },
                ...
            ]
        }
        """
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process the current step and generate specific, actionable sub-queries.
        
        Args:
            input_data: Dictionary containing:
                - 'step': Dict with 'id', 'description', 'objective', 'dependencies'
                - 'plan': The overall plan with all steps
                - 'history': List of previous interactions and their results
                - 'context': Additional context for the task
                - 'previous_answers': Dict of {step_id: answer} for completed steps
                
        Returns:
            AgentResponse containing the generated sub-queries and metadata
        """
        step = input_data.get('step', {})
        plan = input_data.get('plan', {})
        history = input_data.get('history', [])
        context = input_data.get('context', '')
        previous_answers = input_data.get('previous_answers', {})
        
        if not step or 'id' not in step:
            return AgentResponse(
                content="Error: Invalid or missing step information",
                metadata={"error": "Invalid step data"}
            )
        
        try:
            # Prepare the prompt
            prompt = f"""{self.system_prompt}
            
            ### Main Question:
            {plan.get('main_question', 'Not specified')}
            
            ### Current Step:
            ID: {step.get('id', 'unknown')}
            Description: {step.get('description', 'No description')}
            Objective: {step.get('objective', 'No objective')}
            Dependencies: {', '.join(step.get('dependencies', [])) or 'None'}
            """
            
            # Add context if available
            if context:
                prompt += f"\n### Additional Context:\n{context}"
            
            # Add information about previous steps and their answers
            if previous_answers:
                prompt += "\n\n### Previous Steps and Answers:"
                for step_id, answer in previous_answers.items():
                    prompt += f"\n- Step {step_id}: {answer[:200]}..."
            
            # Add relevant conversation history
            if history:
                prompt += "\n\n### Conversation History:"
                for i, h in enumerate(history[-3:]):  # Last 3 messages
                    role = h.get('role', 'unknown').upper()
                    content = h.get('content', '')[:300]  # Truncate long content
                    prompt += f"\n{role}: {content}"
            
            # Add the actual instruction
            prompt += """
            
            Please generate specific sub-queries that would help accomplish the current step.
            Consider what information is needed and how it contributes to the step's objective.
            """
            
            # Generate the response using the LLM
            response = await self.generate_text(
                prompt,
                temperature=0.3,
                max_new_tokens=1024
            )
            
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
                
                for i, sq in enumerate(result["sub_queries"]):
                    if not all(k in sq for k in ["id", "query", "purpose"]):
                        raise ValueError(f"Sub-query {i} is missing required fields. Need 'id', 'query', and 'purpose'")
                
                # Update history
                self._update_history("user", f"Define sub-queries for step: {step.get('id')}")
                self._update_history("assistant", response)
                
                # Parse sub-queries into SubQuery objects
                sub_queries = [
                    SubQuery(
                        id=sq.get("id", f"subq_{i+1}"),
                        query=sq["query"],
                        purpose=sq["purpose"],
                        priority=int(sq.get("priority", 1))
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
                        "sub_queries": [sq.dict() for sq in sub_queries]
                    }),
                    metadata={
                        "step_id": result["step_id"],
                        "step_description": result["step_description"],
                        "reasoning": result["reasoning"],
                        "sub_queries": [sq.dict() for sq in sub_queries]
                    }
                )
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                return AgentResponse(
                    content="Error: Failed to parse LLM response as JSON",
                    metadata={
                        "error": "JSON parse error",
                        "llm_response": response,
                        "exception": str(e)
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
