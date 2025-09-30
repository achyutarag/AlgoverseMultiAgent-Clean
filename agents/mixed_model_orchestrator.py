from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import json
import logging
import asyncio
from datetime import datetime

# Import existing agents
from .planner_agent import PlannerAgent
from .step_definer_agent import StepDefinerAgent
from .retriever_agent import RetrieverAgent
from .extractor_agent import ExtractorAgent
from .qa_agent import QAAgent
from .state_manager import StateManager
from .final_assembler import FinalAssembler
from .orchestrator import MARAGOrchestrator, PipelineResult

logger = logging.getLogger(__name__)

class MixedModelOrchestrator(MARAGOrchestrator):
    """
    MA-RAG Orchestrator with optimized SLM/LLM model configuration.
    
    Uses:
    - SLMs for retrieval and extraction (fast, efficient)
    - LLMs for planning, step definition, and QA (complex reasoning)
    """
    
    def __init__(
        self,
        # SLM models for retrieval and extraction
        retrieval_model: str = "all-MiniLM-L6-v2",
        extraction_model: str = "distilbert-base-uncased",
        
        # LLM models for reasoning tasks
        planning_model: str = "llama-2-13b-chat-hf",
        step_definition_model: str = "llama-2-13b-chat-hf", 
        qa_model: str = "llama-2-13b-chat-hf",
        
        # Model configurations
        slm_config: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        
        # Pipeline configuration
        max_concurrent_steps: int = 1,
        timeout_seconds: int = 300,
        
        # Agent-specific parameters
        max_steps: int = 5,
        max_subqueries: int = 3,
        top_k: int = 5,
        min_similarity: float = 0.6
    ):
        """
        Initialize the Mixed Model Orchestrator with optimal SLM/LLM configuration.
        
        Args:
            retrieval_model: SLM model for document retrieval
            extraction_model: SLM model for text extraction
            planning_model: LLM model for query planning
            step_definition_model: LLM model for step definition
            qa_model: LLM model for answer synthesis
            slm_config: Configuration for SLM models
            llm_config: Configuration for LLM models
            max_concurrent_steps: Maximum concurrent steps
            timeout_seconds: Timeout for pipeline execution
            max_steps: Maximum steps in plan
            max_subqueries: Maximum subqueries per step
            top_k: Number of documents to retrieve
            min_similarity: Minimum similarity threshold
        """
        
        # Default configurations
        if slm_config is None:
            slm_config = {
                "use_cuda": False,  # SLMs can run on CPU
                "load_in_4bit": False,
                "temperature": 0.1,
                "max_new_tokens": 512
            }
        
        if llm_config is None:
            llm_config = {
                "use_cuda": True,  # LLMs need GPU
                "load_in_4bit": True,  # Memory optimization
                "temperature": 0.3,
                "max_new_tokens": 1024
            }
        
        # Initialize agents with optimal models
        logger.info("Initializing Mixed Model MA-RAG Orchestrator...")
        logger.info(f"SLM Models: Retrieval={retrieval_model}, Extraction={extraction_model}")
        logger.info(f"LLM Models: Planning={planning_model}, StepDef={step_definition_model}, QA={qa_model}")
        
        # SLM Agents (fast, efficient)
        retriever_agent = RetrieverAgent(
            model_name=retrieval_model,
            model_config=slm_config,
            top_k=top_k,
            min_similarity=min_similarity
        )
        
        extractor_agent = ExtractorAgent(
            model_name=extraction_model,
            model_config=slm_config
        )
        
        # LLM Agents (complex reasoning)
        planner_agent = PlannerAgent(
            model_name=planning_model,
            model_config=llm_config,
            max_steps=max_steps
        )
        
        step_definer_agent = StepDefinerAgent(
            model_name=step_definition_model,
            model_config=llm_config,
            max_subqueries=max_subqueries
        )
        
        qa_agent = QAAgent(
            model_name=qa_model,
            model_config=llm_config
        )
        
        # Initialize supporting components
        state_manager = StateManager()
        final_assembler = FinalAssembler()
        
        # Initialize parent orchestrator
        super().__init__(
            planner_agent=planner_agent,
            step_definer_agent=step_definer_agent,
            retriever_agent=retriever_agent,
            extractor_agent=extractor_agent,
            qa_agent=qa_agent,
            state_manager=state_manager,
            final_assembler=final_assembler,
            max_concurrent_steps=max_concurrent_steps,
            timeout_seconds=timeout_seconds
        )
        
        # Store model information
        self.model_configuration = {
            "slm_models": {
                "retrieval": retrieval_model,
                "extraction": extraction_model
            },
            "llm_models": {
                "planning": planning_model,
                "step_definition": step_definition_model,
                "qa": qa_model
            },
            "configurations": {
                "slm_config": slm_config,
                "llm_config": llm_config
            }
        }
        
        logger.info("Mixed Model Orchestrator initialized successfully")
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model configuration.
        
        Returns:
            Dictionary with model configuration details
        """
        return {
            "model_configuration": self.model_configuration,
            "agent_types": {
                "planner": "LLM (complex reasoning)",
                "step_definer": "LLM (context grounding)",
                "retriever": "SLM (fast retrieval)",
                "extractor": "SLM (efficient extraction)",
                "qa": "LLM (answer synthesis)"
            },
            "performance_characteristics": {
                "slm_agents": "Fast, memory-efficient, good for pattern matching",
                "llm_agents": "Slower, memory-intensive, good for complex reasoning"
            }
        }
    
    async def benchmark_agent_performance(self) -> Dict[str, Any]:
        """
        Benchmark the performance of different agent types.
        
        Returns:
            Performance metrics for SLM vs LLM agents
        """
        logger.info("Running agent performance benchmark...")
        
        # Test SLM agents (retrieval, extraction)
        slm_start = datetime.now()
        try:
            # Test retrieval agent
            retrieval_test = await self.retriever.process({
                "query": "test query",
                "k": 3
            })
            slm_retrieval_time = (datetime.now() - slm_start).total_seconds()
        except Exception as e:
            slm_retrieval_time = -1
            logger.warning(f"Retrieval test failed: {str(e)}")
        
        # Test LLM agents (planning, QA)
        llm_start = datetime.now()
        try:
            # Test planner agent
            planning_test = await self.planner.process({
                "query": "What are the benefits of renewable energy?"
            })
            llm_planning_time = (datetime.now() - llm_start).total_seconds()
        except Exception as e:
            llm_planning_time = -1
            logger.warning(f"Planning test failed: {str(e)}")
        
        return {
            "slm_performance": {
                "retrieval_time": slm_retrieval_time,
                "status": "fast" if slm_retrieval_time > 0 else "error"
            },
            "llm_performance": {
                "planning_time": llm_planning_time,
                "status": "complex" if llm_planning_time > 0 else "error"
            },
            "recommendation": "SLMs for retrieval/extraction, LLMs for reasoning tasks"
        }


# Convenience functions for easy usage
async def create_optimized_marag_pipeline(
    retrieval_model: str = "all-MiniLM-L6-v2",
    extraction_model: str = "distilbert-base-uncased",
    planning_model: str = "llama-2-13b-chat-hf",
    step_definition_model: str = "llama-2-13b-chat-hf",
    qa_model: str = "llama-2-13b-chat-hf",
    **kwargs
) -> MixedModelOrchestrator:
    """
    Create an optimized MA-RAG pipeline with mixed SLM/LLM models.
    
    Args:
        retrieval_model: SLM for document retrieval
        extraction_model: SLM for text extraction  
        planning_model: LLM for query planning
        step_definition_model: LLM for step definition
        qa_model: LLM for answer synthesis
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured MixedModelOrchestrator
    """
    return MixedModelOrchestrator(
        retrieval_model=retrieval_model,
        extraction_model=extraction_model,
        planning_model=planning_model,
        step_definition_model=step_definition_model,
        qa_model=qa_model,
        **kwargs
    )

async def run_optimized_marag_pipeline(
    query: str,
    retrieval_model: str = "all-MiniLM-L6-v2",
    extraction_model: str = "distilbert-base-uncased", 
    planning_model: str = "llama-2-13b-chat-hf",
    step_definition_model: str = "llama-2-13b-chat-hf",
    qa_model: str = "llama-2-13b-chat-hf",
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> PipelineResult:
    """
    Run the complete MA-RAG pipeline with optimized model configuration.
    
    Args:
        query: The user's question
        retrieval_model: SLM for document retrieval
        extraction_model: SLM for text extraction
        planning_model: LLM for query planning
        step_definition_model: LLM for step definition
        qa_model: LLM for answer synthesis
        context: Optional additional context
        **kwargs: Additional configuration parameters
        
    Returns:
        PipelineResult with complete execution results
    """
    orchestrator = await create_optimized_marag_pipeline(
        retrieval_model=retrieval_model,
        extraction_model=extraction_model,
        planning_model=planning_model,
        step_definition_model=step_definition_model,
        qa_model=qa_model,
        **kwargs
    )
    
    return await orchestrator.execute_pipeline(query, context)


