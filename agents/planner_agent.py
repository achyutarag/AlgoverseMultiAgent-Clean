from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from .base_agent import BaseAgent, AgentResponse
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
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
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
        
        self.system_prompt = """You are an expert planner specialized in query disambiguation and task decomposition for retrieval-augmented generation. Your task is to:

1. **Query Disambiguation**: Identify potential ambiguities or underspecified elements in the query and reformulate them into clearer sub-questions if necessary.

2. **Task Decomposition**: For complex or multi-hop questions, create a structured plan P = {s1, s2, ..., sn} where each si denotes a reasoning subtask.

3. **Chain-of-Thought Reasoning**: Use step-by-step reasoning to ensure interpretable decomposition that supports grounded reasoning in downstream modules.

Guidelines for planning:
- Break down complex queries into manageable, sequential steps
- Identify dependencies between steps
- Consider what specific information is needed for each step
- Ensure each step can be executed independently when dependencies are met
- Mark critical steps that are essential for answering the main question
- Use clear, actionable descriptions for each step

Return your response as a JSON object with this structure:
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

Examples of good step decomposition:

**Multi-hop question**: "What are the environmental impacts of renewable energy adoption in Germany?"
- Step 1: Identify types of renewable energy used in Germany
- Step 2: Research environmental benefits of each renewable energy type
- Step 3: Investigate environmental challenges/concerns for each type
- Step 4: Analyze Germany-specific environmental policies and regulations
- Step 5: Synthesize findings into comprehensive environmental impact assessment

**Comparative question**: "How do the economic policies of Japan and South Korea differ?"
- Step 1: Research Japan's key economic policies
- Step 2: Research South Korea's key economic policies  
- Step 3: Compare monetary policies between the two countries
- Step 4: Compare fiscal policies between the two countries
- Step 5: Analyze differences in trade and industrial policies
- Step 6: Synthesize comprehensive comparison"""
        
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
        
        if not query:
            return AgentResponse(
                content="Error: No query provided",
                metadata={"error": "No query provided"}
            )
        
        try:
            # Prepare the enhanced prompt with context
            prompt = f"""{self.system_prompt}
            
            ### Query to Analyze:
            {query}
            
            ### Additional Context:
            {json.dumps(context, indent=2) if context else "No additional context provided"}
            
            ### Instructions:
            Please analyze this query and create a structured plan following the guidelines above.
            Focus on:
            1. Identifying any ambiguities in the query
            2. Determining the query type (simple, multi-hop, comparative, analytical)
            3. Creating a logical sequence of steps that will lead to a comprehensive answer
            4. Ensuring each step is actionable and has clear objectives
            
            Remember to limit the plan to {max_steps} steps maximum."""
            
            # Generate the response using the LLM
            response = await self.generate_text(
                prompt,
                temperature=0.2,  # Lower temperature for more consistent planning
                max_new_tokens=2048
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
                required_keys = ["main_question", "reasoning", "steps"]
                if not all(key in result for key in required_keys):
                    raise ValueError(f"Missing required keys in response. Expected: {required_keys}")
                
                # Validate steps structure
                if not isinstance(result["steps"], list):
                    raise ValueError("Steps must be a list")
                
                # Limit steps to max_steps
                result["steps"] = result["steps"][:max_steps]
                
                # Validate and enhance each step
                validated_steps = []
                for i, step in enumerate(result["steps"]):
                    if not all(k in step for k in ["id", "description", "objective"]):
                        logger.warning(f"Step {i} is missing required fields, skipping")
                        continue
                    
                    # Ensure dependencies is a list
                    if "dependencies" not in step:
                        step["dependencies"] = []
                    elif not isinstance(step["dependencies"], list):
                        step["dependencies"] = [step["dependencies"]]
                    
                    # Add critical flag if not present
                    if "critical" not in step:
                        step["critical"] = i == 0 or "synthesize" in step["description"].lower()
                    
                    # Add expected output if not present
                    if "expected_output" not in step:
                        step["expected_output"] = f"Information needed for: {step['objective']}"
                    
                    validated_steps.append(step)
                
                result["steps"] = validated_steps
                
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
