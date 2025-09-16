from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from .llm_wrapper import LLMConfig, get_llm

class AgentResponse(BaseModel):
    """Standard response format for all agents"""
    content: str = Field(..., description="The main content of the response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the response")

class BaseAgent(ABC):
    """Base class for all agents in the pipeline"""
    
    def __init__(
        self, 
        name: str, 
        model_config: Optional[Union[Dict[str, Any], LLMConfig]] = None,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf"  # Default model
    ):
        self.name = name
        self.history: List[Dict[str, str]] = []
        
        # Set up LLM
        if model_config is None:
            model_config = {
                "model_name": model_name,
                "model_type": "huggingface",  # Default to huggingface
                "temperature": 0.7,
                "max_new_tokens": 1024,
                "use_quantization": True,
                "load_in_4bit": True
            }
        
        self.llm = get_llm(model_config)
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> 'AgentResponse':
        """
        Process the input and return a response.
        
        Args:
            input_data: Dictionary containing the input data for the agent
            
        Returns:
            AgentResponse containing the agent's response and metadata
        """
        pass
    
    def _update_history(self, role: str, content: str):
        """
        Update the conversation history.
        
        Args:
            role: Either 'user' or 'assistant'
            content: The message content
        """
        self.history.append({"role": role, "content": content})
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return self.history.copy()
    
    def clear_history(self):
        """
        Clear the conversation history.
        
        Note:
            This will remove all previous messages from the agent's memory.
        """
        self.history = []
        
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the agent's LLM.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        response = await self.llm.generate(prompt, **kwargs)
        return response.text
