import asyncio
import json
import logging
import datetime
import torch
from typing import Dict, Any, List, Optional, Union

from pydantic import BaseModel, Field

from agents.planner_agent import PlannerAgent
from agents.step_definer_agent import StepDefinerAgent
from agents.retriever_agent import RetrieverAgent
from agents.extractor_agent import ExtractorAgent, DocumentChunk
from agents.qa_agent import QAAgent
from agents.base_agent import AgentResponse

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_CONFIG = {
    "model": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "temperature": 0.2,
        "max_tokens": 1024,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
    },
    "embedding": {
        "model": "sentence-transformers/all-mpnet-base-v2",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 32
    },
    "retrieval": {
        "top_k": 5,
        "min_similarity": 0.6,
        "max_documents": 10
    },
    "extraction": {
        "max_documents": 3,
        "min_relevance": 0.5,
        "max_tokens": 1000
    },
    "qa": {
        "min_confidence": 0.5,
        "max_followup_questions": 3
    },
    "max_steps": 5,
    "max_subqueries": 3
}

class QueryResult(BaseModel):
    """Structured result of a query through the pipeline."""
    main_question: str = Field(..., description="The original question")
    final_answer: str = Field(..., description="The final answer to the question")
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="List of steps taken to arrive at the answer")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="List of sources used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the query")

class RAGPipeline:
    """
    A multi-agent RAG pipeline that coordinates between different specialized agents
    to answer complex questions.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        **overrides
    ) -> None:
        """Initialize the RAG pipeline with configuration.
        
        Args:
            config: Full configuration dictionary (will be merged with defaults)
            **overrides: Individual configuration overrides
        """
        import torch
        from copy import deepcopy
        
        # Merge default config with provided config and overrides
        self.config = deepcopy(DEFAULT_CONFIG)
        if config:
            self._update_config(self.config, config)
        if overrides:
            self._update_config(self.config, overrides)
        
        # Set up device
        self.device = self.config["model"].get("device", "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        # Initialize components with proper configuration
        model_kwargs = {
            k: v for k, v in self.config["model"].items()
            if k not in ["name", "device"]
        }
        
        logger.info(f"Initializing RAG Pipeline with config: {self._get_config_summary()}")
        
        # Initialize agents with shared configuration
        agent_kwargs = {
            "model_name": self.config["model"]["name"],
            "device": self.device,
            **model_kwargs
        }
        
        self.planner = PlannerAgent(
            **agent_kwargs,
            max_steps=self.config.get("max_steps", 5)
        )
        
        self.step_definer = StepDefinerAgent(
            **agent_kwargs,
            max_subqueries=self.config.get("max_subqueries", 3)
        )
        
        self.retriever = RetrieverAgent(
            model_name=self.config["embedding"]["model"],
            device=self.config["embedding"].get("device", self.device),
            top_k=self.config["retrieval"].get("top_k"),
            min_similarity=self.config["retrieval"].get("min_similarity"),
            batch_size=self.config["embedding"].get("batch_size")
        )
        
        self.extractor = ExtractorAgent(
            **agent_kwargs,
            max_documents=self.config["extraction"].get("max_documents"),
            min_relevance=self.config["extraction"].get("min_relevance"),
            max_tokens=self.config["extraction"].get("max_tokens")
        )
        
        self.qa_agent = QAAgent(
            **agent_kwargs,
            min_confidence=self.config["qa"].get("min_confidence"),
            max_followup_questions=self.config["qa"].get("max_followup_questions")
        )
        
        # Initialize conversation history and document sources
        self.conversation_history: List[Dict[str, str]] = []
        self.document_sources: Dict[str, Dict[str, Any]] = {}
        
        logger.info("RAG Pipeline initialized successfully")
    
    def _update_config(self, base: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Recursively update a config dictionary with updates."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_config(base[key], value)
            else:
                base[key] = value
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration."""
        return {
            "model": {
                "name": self.config["model"]["name"],
                "device": self.device,
                "temperature": self.config["model"]["temperature"],
                "max_tokens": self.config["model"]["max_tokens"]
            },
            "embedding": {
                "model": self.config["embedding"]["model"],
                "device": self.config["embedding"].get("device")
            },
            "retrieval": {
                "top_k": self.config["retrieval"].get("top_k"),
                "min_similarity": self.config["retrieval"].get("min_similarity")
            }
        }
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the retriever's vector store."""
        self.retriever.add_documents(documents)
    
    def _update_history(self, role: str, content: str):
        """Update the conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    async def _process_step(
        self,
        step: Dict[str, Any],
        plan: Dict[str, Any],
        context: Dict[str, Any],
        max_retries: int = 2,
        initial_retry_delay: float = 1.0
    ) -> Dict[str, Any]:
        """
        Process a single step in the plan with retry logic.
        
        Args:
            step: The step to process
            plan: The overall plan
            context: Additional context
            max_retries: Maximum number of retry attempts
            initial_retry_delay: Initial delay between retries in seconds (will be doubled each retry)
            
        Returns:
            Dictionary with step results
        """
        step_id = step.get("id", f"step_{len(self.conversation_history)}")
        step_description = step.get("description", "")
        
        # Track retries
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                # Define sub-queries for this step
                step_result = await self.step_definer.process({
                    "step": step,
                    "plan": plan,
                    "history": self.conversation_history[-10:],  # Limit history length
                    "context": context
                })
                
                if step_result.metadata.get("error"):
                    raise ValueError(f"Step definer error: {step_result.metadata['error']}")
                
                sub_queries = step_result.metadata.get("sub_queries", [])
                if not sub_queries:
                    logger.warning(f"No sub-queries generated for step {step_id}")
                    sub_queries = [{"id": "default", "query": step_description}]
                
                step_results = []
                
                # Process each sub-query in parallel
                sub_query_tasks = []
                for sub_query in sub_queries:
                    sub_query_tasks.append(self._process_subquery(
                        sub_query, step_id, context
                    ))
                
                # Gather all sub-query results
                step_results = await asyncio.gather(*sub_query_tasks, return_exceptions=True)
                
                # Process results
                valid_results = []
                for i, result in enumerate(step_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error in sub-query {i}: {str(result)}")
                        step_results[i] = {
                            "status": "error",
                            "error": str(result)
                        }
                    valid_results.append(step_results[i])
                
                # Check if we have any successful results
                if not any(r.get("status") == "completed" for r in valid_results):
                    raise ValueError("All sub-queries failed to complete successfully")
                
                return {
                    "step_id": step_id,
                    "description": step_description,
                    "status": "completed",
                    "sub_queries": valid_results,
                    "metadata": {
                        "retry_count": retry_count,
                        "num_subqueries": len(valid_results),
                        "successful_subqueries": sum(1 for r in valid_results if r.get("status") == "completed")
                    }
                }
                
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count <= max_retries:
                    delay = initial_retry_delay * (2 ** (retry_count - 1))
                    logger.warning(
                        f"Error in step {step_id} (attempt {retry_count}/{max_retries + 1}): {str(e)}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Failed to process step {step_id} after {max_retries + 1} attempts")
                    return {
                        "step_id": step_id,
                        "description": step_description,
                        "status": "error",
                        "error": f"Failed after {max_retries + 1} attempts: {str(last_error)}",
                        "metadata": {
                            "retry_count": retry_count,
                            "last_error": str(last_error)
                        }
                    }
    
    async def _process_subquery(
        self,
        sub_query: Dict[str, Any],
        step_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single sub-query within a step.
        
        Args:
            sub_query: The sub-query to process
            step_id: ID of the parent step
            context: Additional context
            
        Returns:
            Dictionary with sub-query results
        """
        import uuid
        from typing import List, Dict, Any, Optional
        
        query_id = sub_query.get("id", str(uuid.uuid4())[:8])
        query_text = sub_query.get("query", "")
        
        try:
            # Retrieve relevant documents with similarity threshold
            retrieval_result = await self.retriever.process({
                "query": query_text,
                "k": 5,  # Number of documents to retrieve
                "min_similarity": 0.6  # Minimum similarity score
            })
            
            if retrieval_result.metadata.get("error"):
                raise ValueError(f"Retrieval error: {retrieval_result.metadata['error']}")
            
            documents = retrieval_result.metadata.get("documents", [])
            if not documents:
                return {
                    "sub_query_id": query_id,
                    "query": query_text,
                    "status": "no_results",
                    "answer": "No relevant documents found.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Extract relevant passages from documents
            extract_result = await self.extractor.process({
                "query": query_text,
                "documents": documents,
                "history": self.conversation_history[-5:],  # Include recent history
                "max_documents": 3,  # Limit number of documents to process
                "min_relevance": 0.5  # Minimum relevance score
            })
            
            if extract_result.metadata.get("error"):
                raise ValueError(f"Extraction error: {extract_result.metadata['error']}")
            
            extracted_passages = extract_result.metadata.get("extracted_passages", [])
            if not extracted_passages:
                return {
                    "sub_query_id": query_id,
                    "query": query_text,
                    "status": "no_relevant_content",
                    "answer": "No relevant content found in the documents.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Generate answer using QA agent
            qa_result = await self.qa_agent.process({
                "question": query_text,
                "context": extracted_passages,
                "history": self.conversation_history[-5:],  # Include recent history
                "min_confidence": 0.5  # Minimum confidence threshold
            })
            
            # Parse the answer from the response
            try:
                answer_data = json.loads(qa_result.content)
                answer = answer_data.get("answer", "")
                confidence = answer_data.get("confidence", 0.0)
                sources = answer_data.get("sources", [])
            except (json.JSONDecodeError, AttributeError):
                answer = qa_result.content if hasattr(qa_result, 'content') else str(qa_result)
                confidence = 0.5  # Default confidence for unparsable responses
                sources = []
            
            # Update conversation history with this Q&A
            self._update_history("user", f"[Step {step_id}] {query_text}")
            self._update_history("assistant", answer)
            
            # Track document sources
            for doc in documents:
                doc_id = doc.get("id", str(uuid.uuid4()))
                if doc_id not in self.document_sources:
                    self.document_sources[doc_id] = {
                        "id": doc_id,
                        "metadata": doc.get("metadata", {}),
                        "content_preview": doc.get("content", "")[:200] + "...",
                        "times_used": 0
                    }
                self.document_sources[doc_id]["times_used"] += 1
            
            return {
                "sub_query_id": query_id,
                "query": query_text,
                "status": "completed",
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "metadata": {
                    "num_documents": len(documents),
                    "num_extracted_passages": len(extracted_passages),
                    "avg_relevance": extract_result.metadata.get("avg_relevance", 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing sub-query {query_id}: {str(e)}", exc_info=True)
            return {
                "sub_query_id": query_id,
                "query": query_text,
                "status": "error",
                "error": str(e),
                "sources": [],
                "confidence": 0.0
            }
    
    async def query(
        self, 
        question: str, 
        context: Optional[Dict[str, Any]] = None,
        max_steps: int = 5,
        timeout: Optional[float] = 60.0
    ) -> QueryResult:
        """
        Process a query through the entire RAG pipeline with enhanced error handling and timeouts.
        
        Args:
            question: The question to answer
            context: Optional additional context for the query
            max_steps: Maximum number of steps to process
            timeout: Maximum time in seconds to spend on the query
            
        Returns:
            QueryResult containing the answer and metadata
            
        Raises:
            asyncio.TimeoutError: If the query takes longer than the specified timeout
            Exception: If there's an error processing the query
        """
        start_time = datetime.datetime.utcnow()
        
        if context is None:
            context = {}
        
        try:
            # Start with the planner
            logger.info(f"Starting query processing for: {question}")
            
            # Set up timeout
            if timeout is not None:
                # Create a task with timeout
                task = asyncio.create_task(self._execute_query_plan(question, context, max_steps))
                done, pending = await asyncio.wait(
                    [task], 
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                if pending:
                    # Timeout occurred
                    task.cancel()
                    raise asyncio.TimeoutError(f"Query timed out after {timeout} seconds")
                
                # Get the result from the completed task
                result = task.result()
            else:
                # No timeout
                result = await self._execute_query_plan(question, context, max_steps)
            
            # Log completion
            duration = (datetime.datetime.utcnow() - start_time).total_seconds()
            logger.info(
                f"Query completed in {duration:.2f}s. "
                f"Steps: {len(result.steps)}, Sources: {len(result.sources)}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return QueryResult(
                main_question=question,
                final_answer=f"An error occurred while processing your query: {str(e)}",
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_seconds": (datetime.datetime.utcnow() - start_time).total_seconds()
                }
            )
    
    async def _execute_query_plan(
        self,
        question: str,
        context: Dict[str, Any],
        max_steps: int
    ) -> QueryResult:
        """
        Execute the query plan without timeout handling.
        
        Args:
            question: The question to answer
            context: Additional context for the query
            max_steps: Maximum number of steps to process
            
        Returns:
            QueryResult with the answer and metadata
        """
        # Start with the planner
        logger.debug("Generating execution plan...")
        plan_result = await self.planner.process({
            "query": question,
            "context": context,
            "max_steps": max_steps
        })
        
        if plan_result.metadata.get("error"):
            error_msg = f"Error in planning: {plan_result.metadata['error']}"
            logger.error(error_msg)
            return QueryResult(
                main_question=question,
                final_answer=error_msg,
                metadata={"error": plan_result.metadata["error"], "step": "planning"}
            )
        
        plan = {
            "main_question": plan_result.metadata.get("main_question", question),
            "reasoning": plan_result.metadata.get("reasoning", ""),
            "steps": plan_result.metadata.get("steps", [])[:max_steps]  # Limit steps
        }
        
        logger.info(f"Plan generated with {len(plan['steps'])} steps")
        
        # Process each step in the plan
        steps_results = []
        for i, step in enumerate(plan["steps"]):
            step_id = step.get("id", f"step_{i+1}")
            logger.info(f"Processing step {i+1}/{len(plan['steps'])}: {step_id}")
            
            try:
                step_result = await self._process_step(step, plan, context)
                steps_results.append(step_result)
                
                if step_result.get("status") == "error":
                    logger.warning(
                        f"Step {step_id} failed: {step_result.get('error')}"
                    )
                    
                    # If it's a critical step, we might want to abort
                    if step.get("critical", False):
                        logger.error("Critical step failed, aborting query")
                        break
                        
            except Exception as e:
                logger.error(f"Unexpected error in step {step_id}: {str(e)}", exc_info=True)
                steps_results.append({
                    "step_id": step_id,
                    "status": "error",
                    "error": f"Unexpected error: {str(e)}",
                    "exception_type": type(e).__name__
                })
        
        # Generate final answer by summarizing the steps
        logger.debug("Generating final answer...")
        final_answer = await self._generate_final_answer(question, plan, steps_results)
        
        # Extract all sources used
        logger.debug("Extracting sources...")
        sources = self._extract_sources(steps_results)
        
        # Calculate statistics
        successful_steps = sum(1 for s in steps_results if s.get("status") == "completed")
        total_subqueries = sum(len(s.get("sub_queries", [])) for s in steps_results)
        successful_subqueries = sum(
            sum(1 for sq in s.get("sub_queries", []) if sq.get("status") == "completed")
            for s in steps_results
        )
        
        return QueryResult(
            main_question=question,
            final_answer=final_answer,
            steps=steps_results,
            sources=sources,
            metadata={
                "plan": plan,
                "num_steps": len(steps_results),
                "successful_steps": successful_steps,
                "total_subqueries": total_subqueries,
                "successful_subqueries": successful_subqueries,
                "num_sources": len(sources),
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "pipeline_version": "1.0.0"
            }
        )
    
    async def _generate_final_answer(
        self,
        question: str,
        plan: Dict[str, Any],
        steps_results: List[Dict[str, Any]],
        max_context_length: int = 4000
    ) -> str:
        """
        Generate a coherent final answer by synthesizing information from all steps.
        
        Args:
            question: The original question
            plan: The execution plan
            steps_results: Results from each processing step
            max_context_length: Maximum context length to use for the final answer
            
        Returns:
            A well-structured final answer
        """
        # Collect all answers and their metadata
        answers = []
        total_confidence = 0.0
        num_answers = 0
        all_sources = set()
        
        # First pass: collect all answers and calculate average confidence
        for step in steps_results:
            if step.get("status") != "completed":
                continue
                
            for sq in step.get("sub_queries", []):
                if sq.get("status") == "completed" and sq.get("answer"):
                    confidence = float(sq.get("confidence", 0.5))
                    answers.append({
                        "answer": sq["answer"],
                        "confidence": confidence,
                        "sources": sq.get("sources", []),
                        "step_id": step.get("step_id", ""),
                        "query": sq.get("query", "")
                    })
                    total_confidence += confidence
                    num_answers += 1
                    all_sources.update(sq.get("sources", []))
        
        if not answers:
            return "I couldn't find enough information to answer your question."
        
        avg_confidence = total_confidence / max(num_answers, 1)
        
        # Sort answers by confidence (highest first)
        answers.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Prepare context for the final answer generation
        context_parts = []
        current_length = 0
        
        # Add the most confident answers first, up to context limit
        for ans in answers:
            answer_text = f"Answer (confidence: {ans['confidence']:.2f}): {ans['answer']}"
            if current_length + len(answer_text) > max_context_length:
                break
            context_parts.append(answer_text)
            current_length += len(answer_text)
        
        if not context_parts:
            # If we couldn't fit any answers, take the beginning of the most confident one
            context_parts.append(answers[0]["answer"][:max_context_length])
        
        context = "\n\n".join(context_parts)
        
        # Generate final answer using QA agent
        final_prompt = f"""
        Original question: {question}
        
        Here are the key findings from my research:
        {context}
        
        Please provide a comprehensive, well-structured answer to the original question.
        - Start with a direct answer if possible
        - Include relevant details and examples
        - Be concise but thorough
        - If there are multiple perspectives, present them clearly
        - End with a brief summary
        - If you're not confident about something, say so
        """
        
        qa_result = await self.qa_agent.process({
            "question": final_prompt,
            "context": [{"text": context, "document_id": "synthesis"}],
            "min_confidence": 0.3  # Be more lenient for the final synthesis
        })
        
        try:
            # Try to parse as JSON first
            result_data = json.loads(qa_result.content)
            final_answer = result_data.get("answer", str(qa_result.content))
        except (json.JSONDecodeError, AttributeError):
            final_answer = str(qa_result.content)
        
        # Add confidence indicator if available
        if avg_confidence < 0.5:
            final_answer = (
                "⚠️ Note: I'm not very confident about this answer. "
                "Please verify this information if it's critical.\n\n"
                + final_answer
            )
        
        return final_answer
    
    def _extract_sources(self, steps_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract, deduplicate, and enhance source information from all steps.
        
        Args:
            steps_results: Results from processing steps
            
        Returns:
            List of unique sources with metadata
        """
        sources = {}
        
        for step in steps_results:
            if step.get("status") != "completed":
                continue
                
            step_id = step.get("step_id", "unknown")
            
            for sq in step.get("sub_queries", []):
                if sq.get("status") != "completed":
                    continue
                    
                for source_id in sq.get("sources", []):
                    # Get or create source entry
                    if source_id not in sources:
                        # Try to get from document_sources first
                        doc_info = self.document_sources.get(source_id, {})
                        sources[source_id] = {
                            "id": source_id,
                            "title": doc_info.get("metadata", {}).get("title", ""),
                            "url": doc_info.get("metadata", {}).get("url", ""),
                            "content_preview": doc_info.get("content_preview", ""),
                            "times_used": 0,
                            "used_in_steps": set(),
                            "used_in_queries": set(),
                            "first_used_at": None,
                            "last_used_at": None
                        }
                    
                    # Update source usage information
                    sources[source_id]["times_used"] += 1
                    sources[source_id]["used_in_steps"].add(step_id)
                    sources[source_id]["used_in_queries"].add(sq.get("query", ""))
                    
                    # Update timestamps
                    current_time = datetime.datetime.utcnow().isoformat()
                    if not sources[source_id]["first_used_at"]:
                        sources[source_id]["first_used_at"] = current_time
                    sources[source_id]["last_used_at"] = current_time
        
        # Convert sets to lists for JSON serialization and add additional metadata
        result = []
        for source_id, source_info in sources.items():
            doc_info = self.document_sources.get(source_id, {})
            
            # Calculate relevance score based on usage
            relevance_score = min(1.0, 0.3 + (0.7 * (source_info["times_used"] / 5)))
            
            result.append({
                "id": source_id,
                "title": source_info.get("title") or doc_info.get("metadata", {}).get("title", ""),
                "url": source_info.get("url") or doc_info.get("metadata", {}).get("url", ""),
                "content_preview": source_info.get("content_preview") or doc_info.get("content_preview", ""),
                "times_used": source_info["times_used"],
                "relevance_score": round(relevance_score, 2),
                "used_in_steps": sorted(list(source_info["used_in_steps"])),
                "used_in_queries": list(source_info["used_in_queries"]),
                "first_used_at": source_info["first_used_at"],
                "last_used_at": source_info["last_used_at"],
                "metadata": doc_info.get("metadata", {})
            })
        
        # Sort by relevance and usage
        result.sort(key=lambda x: (-x["relevance_score"], -x["times_used"]))
        
        return result

async def example_usage():
    """
    Example usage of the RAG pipeline with sample documents and queries.
    
    This demonstrates how to:
    1. Initialize the RAG pipeline with configuration
    2. Add documents to the knowledge base
    3. Query the pipeline with a question
    4. Process and display the results
    """
    import json
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Sample documents (in a real application, these would be your knowledge base)
    documents = [
        {
            "page_content": "The capital of France is Paris. Paris is known as the 'City of Light' and is famous for the Eiffel Tower.",
            "metadata": {
                "source": "geography_facts", 
                "id": "doc1",
                "title": "Basic Geography Facts",
                "url": "https://example.com/geography/facts",
                "author": "Geo Research Team",
                "date_published": "2023-01-15"
            }
        },
        {
            "page_content": "The Eiffel Tower is located in Paris, France. It was completed in 1889 and is one of the most visited monuments in the world.",
            "metadata": {
                "source": "landmark_info", 
                "id": "doc2",
                "title": "Famous Landmarks",
                "url": "https://example.com/landmarks/eiffel"
            }
        },
        {
            "page_content": "French cuisine is known for its bread, cheese, and wine. Popular dishes include croissants, baguettes, and coq au vin.",
            "metadata": {
                "source": "culinary_info", 
                "id": "doc3",
                "title": "French Cuisine Guide",
                "url": "https://example.com/cuisine/french"
            }
        },
        {
            "page_content": "The Louvre Museum in Paris houses the Mona Lisa and is the world's largest art museum.",
            "metadata": {
                "source": "museum_info",
                "id": "doc4",
                "title": "Art and Museums",
                "url": "https://example.com/art/louvre"
            }
        }
    ]
    
    try:
        # Initialize the pipeline with configuration
        pipeline = RAGPipeline(
            config={
                "model": {
                    "name": "meta-llama/Llama-2-7b-chat-hf",
                    "temperature": 0.3,
                    "max_tokens": 1024
                },
                "retrieval": {
                    "top_k": 3,
                    "min_similarity": 0.6
                }
            }
        )
        
        # Add documents to the knowledge base
        pipeline.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to the knowledge base")
        
        # Example questions to ask
        questions = [
            "What is the capital of France and what is it known for?",
            "Tell me about famous landmarks in Paris.",
            "What is French cuisine known for?"
        ]
        
        for question in questions:
            print("\n" + "="*80)
            print(f"Question: {question}")
            print("-" * 80)
            
            # Process the query
            start_time = datetime.datetime.now()
            result = await pipeline.query(question, timeout=30.0)
            duration = (datetime.datetime.now() - start_time).total_seconds()
            
            # Display results
            print(f"\nAnswer (took {duration:.2f}s):")
            print(result.final_answer)
            
            # Show sources if available
            if result.sources:
                print("\nSources:")
                for i, source in enumerate(result.sources[:3], 1):  # Show top 3 sources
                    title = source.get('title', 'Untitled')
                    url = source.get('url', 'No URL')
                    print(f"{i}. {title} - {url}")
            
            # Show metadata if available
            if result.metadata:
                print("\nMetadata:")
                print(f"- Steps: {result.metadata.get('num_steps', 0)} total, "
                      f"{result.metadata.get('successful_steps', 0)} successful")
                print(f"- Subqueries: {result.metadata.get('total_subqueries', 0)} total, "
                      f"{result.metadata.get('successful_subqueries', 0)} successful")
                print(f"- Sources used: {len(result.sources)}")
    
    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import asyncio
    import logging
    
    # Configure logging for the main script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_pipeline.log')
        ]
    )
    
    # Run the example
    try:
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.exception("Unhandled exception in main")
