from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from .base_agent import BaseAgent, AgentResponse
from .tokenization_utils import tokenization_utils, TokenizationUtils
import json
import logging

logger = logging.getLogger(__name__)

class PlanStep(BaseModel):
    """A single step in the plan."""
    id: str = Field(..., description="Unique identifier for the step")
    description: str = Field(..., description="Description of the step")
    objective: str = Field(..., description="Objective of this step")
    dependencies: List[str] = Field(default_factory=list, description="List of step IDs this step depends on")
    critical: bool = Field(False, description="Whether this step is critical for the overall plan")

class PlannerAgent(BaseAgent):
    """
    The Planner Agent performs query disambiguation and task decomposition.
    It analyzes input queries to identify ambiguities and creates structured 
    reasoning plans with chain-of-thought prompting.
    """
    
    def __init__(
        self, 
        model_config: Optional[Dict[str, Any]] = None,
        model_name: str = "gemini-2.5-flash",  # LLM for complex planning
        max_steps: int = 5
    ):
        """
        Initialize the Planner Agent.
        
        Args:
            model_config: Configuration for the LLM
            model_name: Name of the model to use (if model_config not provided)
            max_steps: Maximum number of steps in the plan
        """
        super().__init__("planner_agent", model_config, model_name)
        self.max_steps = max_steps
        
        self.system_prompt = """You are a query planner. Create a step-by-step plan to answer the user's question.

CRITICAL: You MUST return a complete JSON object with ALL required fields, especially the "steps" array.

Return your response as a JSON object with this EXACT structure:
{
    "main_question": "The original question",
    "disambiguated_query": "Clarified version of the query if needed",
    "reasoning": "Your chain-of-thought reasoning for the plan",
    "query_type": "simple|multi-hop|comparative|analytical",
    "steps": [
        {
            "id": "step_1",
            "description": "Clear description of what this step does",
            "objective": "Specific objective this step aims to accomplish",
            "dependencies": [],
            "critical": true,
            "expected_output": "What kind of information this step should produce"
        }
    ]
}

IMPORTANT RULES:
1. Always include at least 2-3 steps in the "steps" array
2. Each step must have "id", "description", "objective", "dependencies", "critical", and "expected_output"
3. Make steps simple and actionable
4. The JSON must be complete and valid

Example for "Who created X?":
{
    "main_question": "Who created X?",
    "disambiguated_query": "Identify the creator of X",
    "reasoning": "Simple factual lookup requiring identification of creator",
    "query_type": "simple",
    "steps": [
        {
            "id": "step_1",
            "description": "Search for information about X and its creator",
            "objective": "Find who created X",
            "dependencies": [],
            "critical": true,
            "expected_output": "Name of creator"
        },
        {
            "id": "step_2",
            "description": "Verify the creator information",
            "objective": "Confirm accuracy",
            "dependencies": ["step_1"],
            "critical": true,
            "expected_output": "Verified creator name"
        }
    ]
}"""
        
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process the input query and create a structured reasoning plan with disambiguation.
        
        Args:
            input_data: Dictionary containing:
                - 'query': The main question to plan for
                - Optional 'context': Additional context for planning
                - Optional 'max_steps': Override default max steps
                
        Returns:
            AgentResponse containing the structured plan with disambiguation
        """
        query = input_data.get('query', '')
        context = input_data.get('context', {})
        max_steps = min(input_data.get('max_steps', self.max_steps), 10)  # Cap at 10 steps
        
        # Normalize query for consistent processing
        query = tokenization_utils.normalize_query(query)
        
        if not query:
            return AgentResponse(
                content="Error: No query provided",
                metadata={"error": "No query provided"}
            )
        
        try:
            # First, get basic plan info
            basic_prompt = f"""Analyze this question: {query}

Return a JSON object with ONLY these fields:
{{
    "main_question": "the original question",
    "disambiguated_query": "clarified version",
    "reasoning": "brief reasoning about how to answer",
    "query_type": "simple"
}}

Return ONLY the JSON, nothing else."""
            
            # Generate basic plan info
            basic_response = await self.generate_text(
                basic_prompt,
                temperature=0.2,
                max_new_tokens=512
            )
            
            basic_response = tokenization_utils.postprocess_answer(basic_response, output_type="json")
            
            try:
                clean_basic = TokenizationUtils.repair_json(basic_response)
                result = json.loads(clean_basic)
            except Exception as e:
                logger.error(f"Failed to parse basic plan: {e}")
                # Create fallback
                result = {
                    "main_question": query,
                    "disambiguated_query": query,
                    "reasoning": "Direct factual lookup",
                    "query_type": "simple"
                }
            
            # Now generate steps one by one
            steps = []
            num_steps = min(3, max_steps)  # Generate 3 steps
            
            for i in range(num_steps):
                step_num = i + 1
                previous_steps = "\n".join([f"Step {j+1}: {s['description']}" for j, s in enumerate(steps)])
                
                step_prompt = f"""Question: {query}

Previous steps:
{previous_steps if previous_steps else "None yet"}

Generate step {step_num} to answer this question. Return a JSON object:
{{
    "id": "step_{step_num}",
    "description": "what to do in this step",
    "objective": "goal of this step",
    "dependencies": {json.dumps([f"step_{j+1}" for j in range(i)]) if i > 0 else "[]"},
    "critical": true,
    "expected_output": "what this step should produce"
}}

Return ONLY the JSON."""
                
                step_response = await self.generate_text(
                    step_prompt,
                    temperature=0.2,
                    max_new_tokens=512
                )
                
                step_response = tokenization_utils.postprocess_answer(step_response, output_type="json")
                
                try:
                    clean_step = TokenizationUtils.repair_json(step_response)
                    step = json.loads(clean_step)
                    
                    # Validate step has required fields
                    if all(k in step for k in ["id", "description", "objective"]):
                        steps.append(step)
                        logger.info(f"Generated step {step_num}: {step['description'][:50]}")
                    else:
                        logger.warning(f"Step {step_num} missing required fields, skipping")
                except Exception as e:
                    logger.warning(f"Failed to parse step {step_num}: {e}")
                    continue
            
            # Add steps to result
            result["steps"] = steps
            logger.info(f"Generated plan with {len(steps)} steps")
            
            # Validate the response structure
            required_keys = ["main_question", "reasoning", "steps"]
            if not all(key in result for key in required_keys):
                raise ValueError(f"Missing required keys in response. Expected: {required_keys}")
            
            # Validate steps structure
            if not isinstance(result["steps"], list):
                raise ValueError("Steps must be a list")
            
            # Enhance each step (they're already validated in the loop above)
            for i, step in enumerate(result["steps"]):
                # Ensure dependencies is a list
                if "dependencies" not in step:
                    step["dependencies"] = []
                elif not isinstance(step["dependencies"], list):
                    step["dependencies"] = [step["dependencies"]]
                
                # Add critical flag if not present
                if "critical" not in step:
                    step["critical"] = True
                
                # Add expected output if not present
                if "expected_output" not in step:
                    step["expected_output"] = f"Information needed for: {step['objective']}"
            
            # Add disambiguated query if not present
            if "disambiguated_query" not in result:
                result["disambiguated_query"] = result["main_question"]
            
            # Add query type if not present
            if "query_type" not in result:
                query_lower = query.lower()
                if any(word in query_lower for word in ["compare", "difference", "vs", "versus"]):
                    result["query_type"] = "comparative"
                elif any(word in query_lower for word in ["how", "why", "what", "when", "where"]):
                    result["query_type"] = "multi-hop"
                elif any(word in query_lower for word in ["analyze", "analysis", "evaluate"]):
                    result["query_type"] = "analytical"
                else:
                    result["query_type"] = "simple"
            
            # Update history
            self._update_history("user", f"Plan for: {query}")
            self._update_history("assistant", json.dumps(result, indent=2))
            
            return AgentResponse(
                content=json.dumps(result, indent=2),
                metadata={
                    "main_question": result["main_question"],
                    "disambiguated_query": result["disambiguated_query"],
                    "query_type": result["query_type"],
                    "num_steps": len(result["steps"]),
                    "reasoning": result["reasoning"],
                    "steps": result["steps"],
                    "planning_parameters": {
                        "max_steps": max_steps,
                        "model": self.model_name,
                        "temperature": 0.2
                    }
                }
            )
                
        except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing LLM response: {str(e)}")
                logger.debug(f"Response content: {response}")
                
                # Fallback: create a simple plan
                fallback_plan = {
                    "main_question": query,
                    "disambiguated_query": query,
                    "reasoning": "Unable to parse detailed plan, using fallback approach",
                    "query_type": "simple",
                    "steps": [
                        {
                            "id": "step_1",
                            "description": f"Research information about: {query}",
                            "objective": f"Gather relevant information to answer: {query}",
                            "dependencies": [],
                            "critical": True,
                            "expected_output": "Relevant facts and information"
                        }
                    ]
                }
                
                return AgentResponse(
                    content=json.dumps(fallback_plan, indent=2),
                    metadata={
                        "main_question": query,
                        "disambiguated_query": query,
                        "query_type": "simple",
                        "num_steps": 1,
                        "reasoning": "Fallback plan due to parsing error",
                        "steps": fallback_plan["steps"],
                        "error": f"Failed to parse detailed plan: {str(e)}",
                        "fallback": True
                    }
                )
                
        except Exception as e:
            logger.error(f"Error in PlannerAgent: {str(e)}", exc_info=True)
            return AgentResponse(
                content=f"Error processing query: {str(e)}",
                metadata={
                    "error": str(e),
                    "query": query,
                    "context": context
                }
            )
