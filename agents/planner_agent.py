from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
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

class PlannerAgent(BaseAgent):
    """
    The Planner Agent is responsible for creating a high-level plan to answer
    complex questions by breaking them down into manageable steps.
    """
    
    def __init__(
        self, 
        model_config: Optional[Dict[str, Any]] = None,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    ):
        """
        Initialize the Planner Agent.
        
        Args:
            model_config: Configuration for the LLM
            model_name: Name of the model to use (if model_config not provided)
        """
        super().__init__("planner_agent", model_config, model_name)
        
        self.system_prompt = """You are an expert planner specialized in breaking down complex questions 
        into logical, sequential steps. Your task is to analyze the main question and create a clear, 
        step-by-step plan to answer it comprehensively.
        
        For each step, consider:
        1. What specific information is needed
        2. What order the steps should be performed in
        3. Any dependencies between steps
        4. How to verify the correctness of each step
        
        Return your response as a JSON object with this structure:
        {
            "main_question": "The original question",
            "reasoning": "Your step-by-step reasoning for the plan",
            "steps": [
                {
                    "id": "step_1",
                    "description": "First step description",
                    "objective": "What this step aims to accomplish",
                    "dependencies": []  // Any steps that must be completed first
                },
                ...
            ]
        }
        """
        
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process the input query and create a step-by-step plan.
        
        Args:
            input_data: Dictionary containing:
                - 'query': The main question to plan for
                - Optional 'context': Additional context for planning
                
        Returns:
            AgentResponse containing the structured plan
        """
        query = input_data.get('query', '')
        context = input_data.get('context', '')
        
        if not query:
            return AgentResponse(
                content="Error: No query provided",
                metadata={"error": "No query provided"}
            )
        
        try:
            # Prepare the prompt
            prompt = f"""{self.system_prompt}
            
            Main Question: {query}"""
            
            if context:
                prompt += f"\n\nAdditional Context:\n{context}"
            
            # Get the LLM response
            llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)
            messages = [SystemMessage(content=self.system_prompt)]
            
            if context:
                messages.append(HumanMessage(
                    content=f"Context for planning: {context}\n\nMain question: {query}"
                ))
            else:
                messages.append(HumanMessage(content=f"Main question: {query}"))
            
            response = llm(messages)
            
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
                
                for i, step in enumerate(result["steps"]):
                    if not all(k in step for k in ["id", "description", "objective"]):
                        raise ValueError(f"Step {i} is missing required fields")
                    
                    # Ensure dependencies is a list
                    if "dependencies" not in step:
                        step["dependencies"] = []
                    elif not isinstance(step["dependencies"], list):
                        step["dependencies"] = [step["dependencies"]]
                
                # Update history
                self._update_history("user", f"Plan for: {query}")
                self._update_history("assistant", json.dumps(result, indent=2))
                
                return AgentResponse(
                    content=json.dumps(result, indent=2),
                    metadata={
                        "main_question": result["main_question"],
                        "num_steps": len(result["steps"]),
                        "reasoning": result["reasoning"],
                        "steps": result["steps"]
                    }
                )
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing LLM response: {str(e)}")
                logger.debug(f"Response content: {response}")
                
                return AgentResponse(
                    content=f"Error: Failed to generate valid plan. {str(e)}",
                    metadata={
                        "error": str(e),
                        "response": response,
                        "query": query
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
