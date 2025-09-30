from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList
)
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import logging

logger = logging.getLogger(__name__)

class LLMConfig(BaseModel):
    """Configuration for LLM models."""
    model_name: str = "google/gemini-pro"
    model_type: str = "huggingface"  # huggingface, llama_cpp, etc.
    device: str = "auto"  # auto, cuda, cpu, mps
    temperature: float = 0.7
    max_new_tokens: int = 1024
    context_length: int = 4096
    use_quantization: bool = True
    load_in_4bit: bool = True
    use_flash_attention: bool = False
    trust_remote_code: bool = True
    
    # LLaMA.cpp specific
    model_path: Optional[str] = None
    n_gpu_layers: int = -1  # -1 for all layers
    n_ctx: int = 2048
    n_batch: int = 512

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

class HuggingFaceLLM(BaseLLMWrapper):
    """Wrapper for Hugging Face models."""
    
    def _load_model(self):
        """Load the Hugging Face model and tokenizer."""
        try:
            quantization_config = None
            if self.config.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=self.config.load_in_4bit,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            
            # Set device map
            device_map = self.config.device
            if device_map == "auto":
                device_map = "auto"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map=device_map,
                quantization_config=quantization_config,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                use_flash_attention_2=self.config.use_flash_attention
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Loaded model {self.config.model_name} on {self.model.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from a prompt."""
        try:
            # Prepare inputs
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, 
                                 max_length=self.config.context_length - self.config.max_new_tokens)
            
            # Move to the same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                    temperature=kwargs.get("temperature", self.config.temperature),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **{k: v for k, v in kwargs.items() if k not in ["max_new_tokens", "temperature"]}
                )
            
            # Decode the output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the output
            if kwargs.get("remove_prompt", True):
                generated_text = generated_text[len(prompt):].strip()
            
            return LLMResponse(
                text=generated_text,
                model_name=self.config.model_name,
                model_type="huggingface",
                usage={
                    "prompt_tokens": inputs["input_ids"].shape[1],
                    "generated_tokens": outputs.shape[1] - inputs["input_ids"].shape[1],
                    "total_tokens": outputs.shape[1]
                }
            )
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            raise

class LlamaCppLLM(BaseLLMWrapper):
    """Wrapper for LLaMA.cpp models."""
    
    def _load_model(self):
        """Load the LLaMA.cpp model."""
        try:
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            self.llm = LlamaCpp(
                model_path=self.config.model_path or self.config.model_name,
                n_ctx=self.config.n_ctx,
                n_batch=self.config.n_batch,
                n_gpu_layers=self.config.n_gpu_layers,
                callback_manager=callback_manager,
                verbose=True,
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
                n_threads=4  # Adjust based on your CPU
            )
            
            logger.info(f"Loaded LLaMA.cpp model: {self.config.model_path or self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading LLaMA.cpp model: {str(e)}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from a prompt using LLaMA.cpp."""
        try:
            response = self.llm(
                prompt,
                max_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                **{k: v for k, v in kwargs.items() if k not in ["max_new_tokens", "temperature"]}
            )
            
            # For LLaMA.cpp, we don't have token counts in the response
            return LLMResponse(
                text=response,
                model_name=self.config.model_name,
                model_type="llama_cpp",
                usage={
                    "prompt_tokens": self.get_token_count(prompt),
                    "generated_tokens": self.get_token_count(response) - self.get_token_count(prompt),
                    "total_tokens": self.get_token_count(response)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in LLaMA.cpp text generation: {str(e)}")
            raise

def get_llm(config: Union[Dict[str, Any], LLMConfig]) -> BaseLLMWrapper:
    """Factory function to get the appropriate LLM wrapper."""
    if not isinstance(config, LLMConfig):
        config = LLMConfig(**config)
    
    if config.model_type == "huggingface":
        return HuggingFaceLLM(config)
    elif config.model_type == "llama_cpp":
        return LlamaCppLLM(config)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Example with Hugging Face model
    hf_config = {
        "model_name": "google/gemini-pro",
        "model_type": "huggingface",
        "device": "auto",
        "temperature": 0.7,
        "max_new_tokens": 512,
        "use_quantization": True,
        "load_in_4bit": True
    }
    
    # Example with LLaMA.cpp model
    llama_cpp_config = {
        "model_path": "models/llama-2-7b-chat.Q4_K_M.gguf",  # Path to your GGUF model
        "model_type": "llama_cpp",
        "n_ctx": 2048,
        "n_batch": 512,
        "n_gpu_layers": 35,  # Adjust based on your GPU
        "temperature": 0.7,
        "max_new_tokens": 512
    }
    
    async def test_llm():
        # Test with Hugging Face
        print("Testing Hugging Face LLM...")
        hf_llm = get_llm(hf_config)
        response = await hf_llm.generate("Explain quantum computing in simple terms.")
        print(f"Response: {response.text[:200]}...")
        
        # Test with LLaMA.cpp (uncomment to test)
        # print("\nTesting LLaMA.cpp LLM...")
        # llama_llm = get_llm(llama_cpp_config)
        # response = await llama_llm.generate("Explain quantum computing in simple terms.")
        # print(f"Response: {response.text[:200]}...")
    
    asyncio.run(test_llm())
