from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import torch
import google.generativeai as genai
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import logging

logger = logging.getLogger(__name__)

class LLMConfig(BaseModel):
    """Configuration for LLM models."""
    model_name: str = "gemini-2.5-flash"
    model_type: str = "google_gemini"  # google_gemini, huggingface, etc.
    device: str = "auto"  # auto, cuda, cpu, mps
    temperature: float = 0.7
    max_new_tokens: int = 1024
    context_length: int = 8192
    use_quantization: bool = True
    load_in_4bit: bool = True
    use_flash_attention: bool = False
    trust_remote_code: bool = True
    

class LLMResponse(BaseModel):
    """Standardized response from LLM."""
    text: str
    model_name: str
    model_type: str
    usage: Dict[str, int] = Field(default_factory=dict)
    finish_reason: str = "stop"
    logprobs: Optional[List[float]] = None

class BaseLLMWrapper(ABC):
    """Base class for LLM wrappers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from a prompt."""
        pass
    
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in the text."""
        if not self.tokenizer:
            return len(text.split())  # Fallback to word count
        return len(self.tokenizer.encode(text, add_special_tokens=False))

class GoogleGeminiLLM(BaseLLMWrapper):
    """Wrapper for Google Gemini models."""
    
    def _load_model(self):
        """Load the Google Gemini model."""
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv()

            # Try API key first, then fall back to service account
            api_key = os.getenv("GOOGLE_API_KEY")
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            
            if api_key:
                # Configure Gemini with API key
                genai.configure(api_key=api_key)
                logger.info("Using Google API Key for authentication")
            elif credentials_path:
                # Set up service account authentication
                if not os.path.exists(credentials_path):
                    raise ValueError(f"Credentials file not found: {credentials_path}")
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
                genai.configure()
                logger.info("Using Service Account for authentication")
            else:
                raise ValueError(
                    "Neither GOOGLE_API_KEY nor GOOGLE_APPLICATION_CREDENTIALS set in .env file. "
                    "Please set one of these environment variables."
                )
            
            # Load Gemini model
            self.model = genai.GenerativeModel(self.config.model_name)
            
            logger.info(f"Loaded Google Gemini model: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading Google model: {str(e)}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from a prompt using Google Gemini."""
        try:
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            # Extract the generated text
            generated_text = response.text if response.text else ""
            
            return LLMResponse(
                text=generated_text,
                model_name=self.config.model_name,
                model_type="google_gemini",
                usage={
                    "prompt_tokens": len(prompt.split()),
                    "generated_tokens": len(generated_text.split()),
                    "total_tokens": len(prompt.split()) + len(generated_text.split())
                }
            )
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            raise

class HuggingFaceLLM(BaseLLMWrapper):
    """Wrapper for Hugging Face models."""
    
    def _load_model(self):
        """Load the Hugging Face model."""
        try:
            # Configure device
            device = "cuda" if torch.cuda.is_available() and self.config.device != "cpu" else "cpu"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Configure quantization if needed
            quantization_config = None
            if self.config.use_quantization and self.config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=self.config.trust_remote_code
            )
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            logger.info(f"Loaded Hugging Face model: {self.config.model_name} on {device}")
            
        except Exception as e:
            logger.error(f"Error loading Hugging Face model: {str(e)}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from a prompt using Hugging Face model."""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if torch.cuda.is_available() and self.config.device != "cpu":
                inputs = inputs.to("cuda")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                    temperature=kwargs.get("temperature", self.config.temperature),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the response
            response_text = generated_text[len(prompt):].strip()
            
            return LLMResponse(
                text=response_text,
                model_name=self.config.model_name,
                model_type="huggingface",
                usage={
                    "prompt_tokens": len(inputs[0]),
                    "generated_tokens": len(outputs[0]) - len(inputs[0]),
                    "total_tokens": len(outputs[0])
                }
            )
            
        except Exception as e:
            logger.error(f"Error in Hugging Face text generation: {str(e)}")
            raise

def get_llm(config: Union[Dict[str, Any], LLMConfig]) -> BaseLLMWrapper:
    """Factory function to get the appropriate LLM wrapper."""
    if not isinstance(config, LLMConfig):
        config = LLMConfig(**config)
    
    if config.model_type == "google_gemini":
        return GoogleGeminiLLM(config)
    elif config.model_type == "huggingface":
        return HuggingFaceLLM(config)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Example with Google Gemini model
    gemini_config = {
        "model_name": "gemini-2.5-flash",
        "model_type": "google_gemini",
        "device": "auto",
        "temperature": 0.7,
        "max_new_tokens": 512
    }
    
    # Example with Hugging Face model
    hf_config = {
        "model_name": "microsoft/DialoGPT-medium",
        "model_type": "huggingface",
        "device": "auto",
        "temperature": 0.7,
        "max_new_tokens": 512,
        "use_quantization": True,
        "load_in_4bit": True
    }
    
    
    async def test_llm():
        # Test with Google Gemini
        print("Testing Google Gemini LLM...")
        gemini_llm = get_llm(gemini_config)
        response = await gemini_llm.generate("Explain quantum computing in simple terms.")
        print(f"Gemini Response: {response.text[:200]}...")
        
        # Test with Hugging Face (uncomment to test)
        # print("\nTesting Hugging Face LLM...")
        # hf_llm = get_llm(hf_config)
        # response = await hf_llm.generate("Explain quantum computing in simple terms.")
        # print(f"HF Response: {response.text[:200]}...")
        
    
    asyncio.run(test_llm())
