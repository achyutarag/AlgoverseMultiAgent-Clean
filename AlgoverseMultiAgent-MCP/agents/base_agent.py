from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import random
import numpy as np
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
        model_name: str = "gemini-2.5-flash"  # Default model
    ):
        self.name = name
        self.model_name = model_name  # Store model name
        self.history: List[Dict[str, str]] = []
        
        # Set up LLM
        if model_config is None:
            model_config = {
                "model_name": model_name,
                "model_type": "google_gemini",  # Default to google_gemini
                "temperature": 0.0,  # deterministic default
                "top_p": 1.0,
                "seed": 1234,
                "max_new_tokens": 1024,
                "use_quantization": True,
                "load_in_4bit": True
            }
        elif isinstance(model_config, dict) and "model_name" in model_config:
            self.model_name = model_config["model_name"]
        
        # Seed PRNGs for deterministic runs if seed provided
        seed_val = model_config.get("seed") if isinstance(model_config, dict) else getattr(model_config, "seed", None)
        if seed_val is not None:
            random.seed(seed_val)
            np.random.seed(seed_val)
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
        try:
            response = await self.llm.generate(prompt, **kwargs)
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                raise Exception(f"API quota exceeded. Please wait before retrying. Original error: {error_msg}")
            elif "404" in error_msg and "not found" in error_msg.lower():
                raise Exception(f"Model not found or not supported. Please check model configuration. Original error: {error_msg}")
            else:
                raise Exception(f"Error in text generation: {error_msg}")
    
    async def generate_text_with_usage(self, prompt: str, **kwargs) -> tuple[str, Dict[str, int]]:
        """
        Generate text using the agent's LLM and return token usage.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (generated_text, token_usage_dict) where token_usage_dict contains:
            - prompt_tokens: Number of tokens in the input prompt
            - generated_tokens: Number of tokens in the generated text
            - total_tokens: Total tokens used
        """
        try:
            response = await self.llm.generate(prompt, **kwargs)
            return response.text, response.usage
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                raise Exception(f"API quota exceeded. Please wait before retrying. Original error: {error_msg}")
            elif "404" in error_msg and "not found" in error_msg.lower():
                raise Exception(f"Model not found or not supported. Please check model configuration. Original error: {error_msg}")
            else:
                raise Exception(f"Error in text generation: {error_msg}")