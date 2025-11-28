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
from .tokenization_utils import tokenization_utils

logger = logging.getLogger(__name__)

class PipelineResult(BaseModel):
    """Result of the complete MA-RAG pipeline execution."""
    main_query: str = Field(..., description="The original user query")
    disambiguated_query: str = Field(..., description="Disambiguated version of the query")
    query_type: str = Field(..., description="Type of query (simple, multi-hop, etc.)")
    execution_time: float = Field(..., description="Total execution time in seconds")
    steps_completed: int = Field(..., description="Number of steps completed")
    final_answer: str = Field(..., description="The final synthesized answer")
    confidence: float = Field(..., description="Overall confidence score")
    reasoning_trajectory: List[Dict[str, Any]] = Field(..., description="Step-by-step reasoning")
    sources: List[str] = Field(..., description="All source documents used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class MARAGOrchestrator:
    """
    Main orchestrator that manages the dynamic, modular agent invocation
    based on the structure of the reasoning plan. Implements the MA-RAG pipeline
    with seamless integration to existing agents.
    """
    
    def __init__(
        self,
        planner_agent: Optional[PlannerAgent] = None,
        step_definer_agent: Optional[StepDefinerAgent] = None,
        retriever_agent: Optional[RetrieverAgent] = None,
        extractor_agent: Optional[ExtractorAgent] = None,
        qa_agent: Optional[QAAgent] = None,
        state_manager: Optional[StateManager] = None,
        final_assembler: Optional[FinalAssembler] = None,
        max_concurrent_steps: int = 1,
        timeout_seconds: int = 300
    ):
        """
        Initialize the MA-RAG Orchestrator.
        
        Args:
            planner_agent: Planner agent instance
            step_definer_agent: Step definer agent instance
            retriever_agent: Retriever agent instance
            extractor_agent: Extractor agent instance
            qa_agent: QA agent instance
            state_manager: State manager instance
            final_assembler: Final assembler instance
            max_concurrent_steps: Maximum concurrent steps (default: 1 for sequential)
            timeout_seconds: Timeout for pipeline execution
        """
        # Initialize agents (create defaults if not provided)
        self.planner = planner_agent or PlannerAgent()
        self.step_definer = step_definer_agent or StepDefinerAgent()
        self.retriever = retriever_agent or RetrieverAgent()
        self.extractor = extractor_agent or ExtractorAgent()
        self.qa = qa_agent or QAAgent()
        
        # Initialize supporting components
        self.state_manager = state_manager or StateManager()
        self.final_assembler = final_assembler or FinalAssembler()
        
        # Pipeline configuration
        self.max_concurrent_steps = max_concurrent_steps
        self.timeout_seconds = timeout_seconds
        
        # Execution tracking
        self.current_execution_id: Optional[str] = None
        self.start_time: Optional[datetime] = None
        
        logger.info("MA-RAG Orchestrator initialized with all components")
    
    async def execute_pipeline(self, query: str, context: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """
        Execute the complete MA-RAG pipeline following the paper's methodology.
        
        Args:
            query: The user's question to answer
            context: Optional additional context
            
        Returns:
            PipelineResult containing the complete execution results
        """
        # Initialize execution tracking
        self.current_execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now()
        
        logger.info(f"Starting MA-RAG pipeline execution: {self.current_execution_id}")
        logger.info(f"Query: {query[:100]}...")
        
        try:
            # Step 1: Initialize state
            await self.state_manager.initialize_execution(
                execution_id=self.current_execution_id,
                main_query=query,
                context=context or {}
            )
            
            # Step 2: Planner Agent (once at beginning)
            plan_result = await self._execute_planner(query, context)
            if not plan_result:
                raise Exception("Failed to generate plan")
            
            # Step 3: Execute each step in the plan
            step_results = await self._execute_plan_steps(plan_result)
            
            # Step 4: Assemble final answer
            final_result = await self._assemble_final_answer(plan_result, step_results)
            
            # Step 5: Create pipeline result
            execution_time = (datetime.now() - self.start_time).total_seconds()
            
            pipeline_result = PipelineResult(
                main_query=query,
                disambiguated_query=plan_result.get("disambiguated_query", query),
                query_type=plan_result.get("query_type", "unknown"),
                execution_time=execution_time,
                steps_completed=len(step_results),
                final_answer=final_result["final_answer"],
                confidence=final_result.get("confidence", 0.0),
                reasoning_trajectory=step_results,
                sources=final_result.get("sources", []),
                metadata={
                    "execution_id": self.current_execution_id,
                    "plan": plan_result,
                    "state_snapshots": await self.state_manager.get_execution_snapshots(),
                    "performance_metrics": {
                        "total_time": execution_time,
                        "avg_step_time": execution_time / len(step_results) if step_results else 0,
                        "steps_per_second": len(step_results) / execution_time if execution_time > 0 else 0
                    }
                }
            )
            
            logger.info(f"Pipeline execution completed successfully in {execution_time:.2f}s")
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise Exception(f"MA-RAG pipeline execution failed: {str(e)}")
        
        finally:
            # Cleanup
            await self.state_manager.cleanup_execution(self.current_execution_id)
            self.current_execution_id = None
            self.start_time = None
    
    async def _execute_planner(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute the Planner Agent to generate the reasoning plan.
        
        Args:
            query: The user query
            context: Optional context
            
        Returns:
            Plan result from planner
        """
        logger.info("Executing Planner Agent...")
        
        try:
            planner_input = {
                "query": query,
                "context": context or {},
                "max_steps": 5  # Default max steps
            }
            
            planner_response = await self.planner.process(planner_input)
            
            if planner_response.metadata.get("error"):
                raise Exception(f"Planner error: {planner_response.metadata.get('error')}")
            
            # Parse the plan from response
            plan_data = json.loads(planner_response.content)
            
            # Update state with plan
            await self.state_manager.update_plan(plan_data)
            
            logger.info(f"Plan generated with {len(plan_data.get('steps', []))} steps")
            return plan_data
            
        except Exception as e:
            logger.error(f"Planner execution failed: {str(e)}")
            raise Exception(f"Failed to generate plan: {str(e)}")
    
    async def _execute_plan_steps(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute all steps in the plan following MA-RAG methodology.
        
        Args:
            plan: The generated plan with steps
            
        Returns:
            List of step results
        """
        steps = plan.get("steps", [])
        if not steps:
            logger.warning("No steps in plan")
            return []
        
        logger.info(f"Executing {len(steps)} plan steps...")
        
        # Resolve step dependencies
        ordered_steps = await self.state_manager.resolve_step_dependencies(steps)
        
        step_results = []
        
        for i, step in enumerate(ordered_steps):
            try:
                logger.info(f"Executing step {i+1}/{len(ordered_steps)}: {step.get('id', 'unknown')}")
                
                # Execute single step following MA-RAG sequence
                step_result = await self._execute_single_step(step, plan)
                
                # Update state with step result
                await self.state_manager.add_step_result(step["id"], step_result)
                
                step_results.append({
                    "step_id": step["id"],
                    "step_description": step.get("description", ""),
                    "result": step_result,
                    "execution_order": i + 1,
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"Step {step['id']} completed successfully")
                
            except Exception as e:
                logger.error(f"Step {step.get('id', 'unknown')} failed: {str(e)}")
                
                # Add error result to maintain trajectory
                step_results.append({
                    "step_id": step["id"],
                    "step_description": step.get("description", ""),
                    "result": {"error": str(e), "success": False},
                    "execution_order": i + 1,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Continue with next step (don't fail entire pipeline)
                continue
        
        logger.info(f"Completed {len(step_results)} steps")
        return step_results
    
    async def _execute_single_step(self, step: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step following MA-RAG sequence:
        1. Step Definer → subqueries
        2. Retrieval Tool → documents
        3. Extractor Agent → evidence
        4. QA Agent → answer
        
        Args:
            step: The current step to execute
            plan: The overall plan
            
        Returns:
            Step execution result
        """
        try:
            # Get accumulated history for context
            history = await self.state_manager.get_accumulated_history()
            previous_answers = await self.state_manager.get_previous_answers()
            
            # 1. Step Definer Agent
            logger.debug(f"Step {step['id']}: Executing Step Definer...")
            step_definer_input = {
                "step": step,
                "plan": plan,
                "history": history,
                "previous_answers": previous_answers
            }
            
            step_definer_response = await self.step_definer.process(step_definer_input)
            if step_definer_response.metadata.get("error"):
                raise Exception(f"Step definer failed: {step_definer_response.metadata.get('error')}")
            
            subqueries_data = json.loads(step_definer_response.content)
            subqueries = subqueries_data.get("sub_queries", [])
            
            if not subqueries:
                raise Exception("No subqueries generated")
            
            # 2. Retrieval Tool (for each subquery)
            logger.debug(f"Step {step['id']}: Executing retrieval for {len(subqueries)} subqueries...")
            all_retrieved_docs = []
            
            for subquery in subqueries:
                retrieval_input = {
                    "query": subquery["query"],
                    "k": 5,  # Default retrieval count
                    "min_similarity": 0.6
                }
                
                retrieval_response = await self.retriever.process(retrieval_input)
                if retrieval_response.metadata.get("error"):
                    logger.warning(f"Retrieval failed for subquery: {subquery['query']}")
                    continue
                
                retrieved_docs = json.loads(retrieval_response.content)
                all_retrieved_docs.extend(retrieved_docs.get("documents", []))
            
            if not all_retrieved_docs:
                raise Exception("No documents retrieved")
            
            # 3. Extractor Agent
            logger.debug(f"Step {step['id']}: Executing extraction on {len(all_retrieved_docs)} documents...")
            extractor_input = {
                "query": step["description"],
                "documents": all_retrieved_docs,
                "history": history
            }
            
            extractor_response = await self.extractor.process(extractor_input)
            if extractor_response.metadata.get("error"):
                raise Exception(f"Extractor failed: {extractor_response.metadata.get('error')}")
            
            extracted_data = json.loads(extractor_response.content)
            extracted_passages = extracted_data.get("extracted_passages", [])
            
            if not extracted_passages:
                raise Exception("No passages extracted")
            
            # 4. QA Agent
            logger.debug(f"Step {step['id']}: Executing QA synthesis...")
            qa_input = {
                "question": step["description"],
                "context": extracted_passages,
                "step_context": step,
                "overall_query": plan.get("main_question", ""),
                "previous_answers": previous_answers
            }
            
            qa_response = await self.qa.process(qa_input)
            if qa_response.metadata.get("error"):
                raise Exception(f"QA agent failed: {qa_response.metadata.get('error')}")
            
            qa_result = json.loads(qa_response.content)
            
            # Return comprehensive step result
            return {
                "step_id": step["id"],
                "step_description": step.get("description", ""),
                "subqueries": subqueries,
                "retrieved_documents": all_retrieved_docs,
                "extracted_passages": extracted_passages,
                "qa_result": qa_result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Step execution failed: {str(e)}")
            return {
                "step_id": step["id"],
                "step_description": step.get("description", ""),
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _assemble_final_answer(self, plan: Dict[str, Any], step_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assemble the final answer from all step results.
        
        Args:
            plan: The original plan
            step_results: Results from all executed steps
            
        Returns:
            Final assembled answer
        """
        logger.info("Assembling final answer...")
        
        try:
            assembler_input = {
                "main_query": plan.get("main_question", ""),
                "disambiguated_query": plan.get("disambiguated_query", ""),
                "query_type": plan.get("query_type", "unknown"),
                "step_results": step_results,
                "plan": plan
            }
            
            final_result = await self.final_assembler.assemble_final_answer(assembler_input)
            
            logger.info("Final answer assembled successfully")
            return final_result
            
        except Exception as e:
            logger.error(f"Final assembly failed: {str(e)}")
            # Return fallback answer
            return {
                "final_answer": f"Error assembling final answer: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "error": str(e)
            }
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline execution status.
        
        Returns:
            Status information
        """
        return {
            "execution_id": self.current_execution_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "elapsed_time": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "state": await self.state_manager.get_current_state() if self.state_manager else {},
            "agents_initialized": {
                "planner": self.planner is not None,
                "step_definer": self.step_definer is not None,
                "retriever": self.retriever is not None,
                "extractor": self.extractor is not None,
                "qa": self.qa is not None
            }
        }
    
    async def cancel_execution(self) -> bool:
        """
        Cancel current pipeline execution.
        
        Returns:
            True if cancelled successfully
        """
        if not self.current_execution_id:
            return False
        
        try:
            logger.info(f"Cancelling execution: {self.current_execution_id}")
            await self.state_manager.cleanup_execution(self.current_execution_id)
            self.current_execution_id = None
            self.start_time = None
            return True
        except Exception as e:
            logger.error(f"Failed to cancel execution: {str(e)}")
            return False


# Convenience function for easy usage
async def run_marag_pipeline(
    query: str,
    retriever_agent: Optional[RetrieverAgent] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> PipelineResult:
    """
    Convenience function to run the complete MA-RAG pipeline.
    
    Args:
        query: The user's question
        retriever_agent: Optional retriever agent (must be initialized with documents)
        context: Optional additional context
        **kwargs: Additional arguments for orchestrator
        
    Returns:
        PipelineResult with complete execution results
    """
    orchestrator = MARAGOrchestrator(retriever_agent=retriever_agent, **kwargs)
    return await orchestrator.execute_pipeline(query, context)

