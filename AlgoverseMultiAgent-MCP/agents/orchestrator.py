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
from .tokenization_utils import TokenizationUtils
from .tokenization_utils import tokenization_utils
from .mcp_reasoning_state import mcp_state_manager
from .mcp_reasoning_state import MCPReasoningStateManager

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
    token_usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "generated_tokens": 0, "total_tokens": 0}, description="Total token usage across all agents")

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
        mcp_state_manager: Optional[MCPReasoningStateManager] = None,
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
        self.token_usage: Dict[str, int] = {"prompt_tokens": 0, "generated_tokens": 0, "total_tokens": 0}
        
        logger.info("MA-RAG Orchestrator initialized with all components")
    
    def _extract_and_aggregate_token_usage(self, agent_response: Any) -> None:
        """
        Extract token usage from agent response and aggregate it.
        
        Args:
            agent_response: AgentResponse object that may contain token usage in metadata
        """
        if hasattr(agent_response, 'metadata') and agent_response.metadata:
            token_usage = agent_response.metadata.get("token_usage", {})
            if token_usage:
                self.token_usage["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
                self.token_usage["generated_tokens"] += token_usage.get("generated_tokens", 0)
                self.token_usage["total_tokens"] += token_usage.get("total_tokens", 0)
    
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
        # Reset token usage for this execution
        self.token_usage = {"prompt_tokens": 0, "generated_tokens": 0, "total_tokens": 0}

        
        logger.info(f"Starting MA-RAG pipeline execution: {self.current_execution_id}")
        logger.info(f"Query: {query[:100]}...")

        
        # Normalize question for consistent processing
        question = tokenization_utils.normalize_query(query)
        
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
            
            # NEW: Step 2.5: Initialize MCP Reasoning State
            mcp_state = mcp_state_manager.create_state(
                execution_id=self.current_execution_id,
                main_question=query,
                disambiguated_query=plan_result.get("disambiguated_query", query),
                reasoning_intent=plan_result.get("reasoning", "Answer the user's question"),
                query_type=plan_result.get("query_type", "unknown"),
                original_plan=plan_result
            )
            logger.info("MCP reasoning state initialized")


            
            
            
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
                token_usage=self.token_usage.copy(),
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
            self._extract_and_aggregate_token_usage(planner_response)
            
            if planner_response.metadata.get("error"):
                raise Exception(f"Planner error: {planner_response.metadata.get('error')}")
            
            # Parse the plan from response
            clean_plan = TokenizationUtils.strip_markdown_json(planner_response.content)
            plan_data = json.loads(clean_plan)
            
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
        Includes early stopping if answer is found.
        """
        steps = plan.get("steps", [])
        if not steps:
            logger.warning("No steps in plan")
            return []
        
        logger.info(f"Executing {len(steps)} plan steps...")
        
        # Resolve step dependencies
        ordered_steps = await self.state_manager.resolve_step_dependencies(steps)
        
        step_results = []
        
        # Get main query from plan for diffusion-aware retrieval
        main_query = plan.get("query") or plan.get("disambiguated_query") or ""
        
        for i, step in enumerate(ordered_steps):
            try:
                logger.info(f"Executing step {i+1}/{len(ordered_steps)}: {step.get('id', 'unknown')}")
                
                # Execute single step following MA-RAG sequence
                # Pass hop number (i+1) and main query for diffusion-aware retrieval
                step_result = await self._execute_single_step(step, plan, hop=i+1, plan_goal=main_query)
                
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
    
    async def _execute_single_step(
        self, 
        step: Dict[str, Any], 
        plan: Dict[str, Any],
        hop: int = 1,
        plan_goal: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a single step following MA-RAG sequence:
        1. Step Definer → subqueries
        2. Diffusion-Aware Retrieval → documents (with stabilization, entropy tracking, anchors)
        3. Extractor Agent → evidence
        4. QA Agent → answer
        
        Args:
            step: The current step to execute
            plan: The overall plan
            hop: Current hop number (for entropy tracking and reasoning flow)
            plan_goal: Main query/goal (for plan alignment regulator)
            
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
            self._extract_and_aggregate_token_usage(step_definer_response)
            if step_definer_response.metadata.get("error"):
                raise Exception(f"Step definer failed: {step_definer_response.metadata.get('error')}")
            
            clean_subqueries = TokenizationUtils.strip_markdown_json(step_definer_response.content)
            subqueries_data = json.loads(clean_subqueries)
            subqueries = subqueries_data.get("sub_queries", [])
            
            # Check if this step has a direct answer (no retrieval needed)
            if subqueries and len(subqueries) == 1 and subqueries[0].get("direct_answer"):
                direct_answer = subqueries[0].get("direct_answer", "")
                
                # ✅ FIRST PRINCIPLES FIX: Validate direct_answer using diffusion-aware components
                # This prevents accepting incomplete answers (e.g., municipality when state is needed)
                should_skip_retrieval = True
                validation_reason = ""
                
                if plan_goal and self.state_manager.regulator_manager:
                    try:
                        # Update flow state to get reasoning context
                        flow_snapshot = self.state_manager._update_flow_state(
                            hop=hop,
                            previous_answers=previous_answers,
                            plan_goal=plan_goal
                        )
                        
                        # Find PlanRegulator to check alignment
                        plan_reg = None
                        for regulator in self.state_manager.regulator_manager.regulators:
                            if hasattr(regulator, 'name') and 'plan' in regulator.name.lower():
                                plan_reg = regulator
                                break
                        
                        if plan_reg and flow_snapshot:
                            # Check if direct_answer aligns with plan goal
                            # Use the step description as the "proposed query" to check alignment
                            step_description = step.get("description", "")
                            
                            # Convert flow_snapshot to dict if needed
                            if hasattr(flow_snapshot, 'model_dump'):
                                reasoning_state = flow_snapshot.model_dump()
                            elif hasattr(flow_snapshot, 'dict'):
                                reasoning_state = flow_snapshot.dict()
                            else:
                                reasoning_state = {}
                            
                            constraint = plan_reg.apply_constraint(
                                proposed_query=step_description,  # Check if step aligns with goal
                                reasoning_state=reasoning_state,
                                previous_answers=previous_answers,
                                plan_goal=plan_goal
                            )
                            
                            alignment = constraint.parameters.get("alignment", 1.0)
                            
                            # Check for hierarchical level mismatch
                            # If plan asks for "administrative territorial entity" (state/province)
                            # but answer is a municipality, that's a level mismatch
                            plan_lower = plan_goal.lower()
                            answer_lower = direct_answer.lower()
                            step_lower = step_description.lower()
                            
                            asks_for_state_level = any(term in plan_lower for term in [
                                "administrative territorial entity", "state", "province", 
                                "administrative entity", "territorial entity"
                            ])
                            answer_is_municipality = "municipality" in answer_lower
                            
                            # If plan asks for state-level but answer is municipality, that's wrong
                            if asks_for_state_level and answer_is_municipality:
                                should_skip_retrieval = False
                                validation_reason = (
                                    f"Hierarchical level mismatch: plan asks for state/province level "
                                    f"but direct_answer is municipality '{direct_answer}'. "
                                    f"Need to retrieve state information."
                                )
                            # If alignment is very low, the answer might be incomplete
                            elif alignment < 0.3:
                                should_skip_retrieval = False
                                validation_reason = (
                                    f"Low plan alignment ({alignment:.2f}) for direct_answer "
                                    f"'{direct_answer}'. Proceeding with retrieval to verify."
                                )
                            else:
                                validation_reason = (
                                    f"Direct answer '{direct_answer}' validated: "
                                    f"alignment={alignment:.2f}, level check passed"
                                )
                    except Exception as e:
                        logger.warning(
                            f"Failed to validate direct_answer with diffusion-aware components: {e}. "
                            f"Proceeding with retrieval to be safe."
                        )
                        should_skip_retrieval = False
                        validation_reason = f"Validation error: {e}"
                
                # Only skip retrieval if validation passed
                if should_skip_retrieval:
                    logger.info(
                        f"Step {step['id']} has validated direct answer from previous steps, "
                        f"skipping retrieval. {validation_reason}"
                    )
                    
                    # Structure qa_result to match what final assembler expects
                    qa_result = {
                        "answer": direct_answer,
                        "confidence": 1.0,
                        "sources": ["previous_step"],
                        "supporting_evidence": []
                    }
                    
                    # Update MCP state after step completes
                    step_result_dict = {
                        "step_id": step["id"],
                        "answer": direct_answer,
                        "sources": ["previous_step"],
                        "confidence": 1.0
                    }
                    
                    mcp_state_manager.update_state(
                        step_id=step["id"],
                        step_result=step_result_dict,
                        execution_id=self.current_execution_id
                    )
                    
                    # Skip retrieval and extraction, go directly to QA with the direct answer
                    step_result = {
                        "step_id": step['id'],
                        "step_description": step.get("description", ""),
                        "subqueries": subqueries,
                        "retrieved_documents": [],  # No documents retrieved
                        "extracted_passages": [{
                            "text": direct_answer,
                            "document_id": "previous_step",
                            "chunk_id": "direct_answer",
                            "relevance": 1.0,
                            "reasoning": "Answer extracted directly from previous step results",
                            "source_context": "Previous step answer"
                        }],
                        "qa_result": qa_result,  # Match structure expected by final assembler
                        "success": True,
                        "direct_answer": True,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Update state and return
                    await self.state_manager.add_step_result(step['id'], step_result)
                    return step_result
                else:
                    # Validation failed - proceed with retrieval
                    logger.warning(
                        f"Step {step['id']}: Direct answer validation failed. {validation_reason} "
                        f"Proceeding with diffusion-aware retrieval."
                    )
                    # Clear direct_answer flag so we proceed with normal retrieval flow
                    subqueries[0].pop("direct_answer", None)
            
            if not subqueries:
                # For simple steps, use the step description as a single query
                logger.warning(f"No subqueries generated for step {step['id']}, using step description as query")
                subqueries = [{
                    "id": "default_sq",
                    "query": step.get('description', step.get('objective', 'search for relevant information')),
                    "purpose": step.get('objective', 'accomplish step goal'),
                    "priority": 1
                }]
            
            # 2. Diffusion-Aware Retrieval (for each subquery)
            # ✅ NEW: Uses stabilize_and_retrieve() which includes:
            #   - Query stabilization via regulators
            #   - Entropy tracking and diffusion awareness
            #   - Reasoning flow updates
            #   - Early termination checks
            #   - Anchor corrections
            logger.debug(f"Step {step['id']}: Executing diffusion-aware retrieval for {len(subqueries)} subqueries...")
            all_retrieved_docs = []
            seen_doc_ids = set()  # Track seen document IDs for deduplication
            early_terminated = False
            
            for subquery in subqueries:
                try:
                    # Use diffusion-aware retrieval with stabilization
                    # Get main query from plan if plan_goal not provided
                    main_query = plan_goal or plan.get("query") or plan.get("disambiguated_query") or ""
                    retrieval_result = await self.state_manager.stabilize_and_retrieve(
                        proposed_query=subquery["query"],
                        hop=hop,
                        previous_answers=previous_answers,
                        plan_goal=main_query,
                        retriever_agent=self.retriever
                    )
                    
                    # Check for early termination
                    if retrieval_result.get("direct_answer"):
                        logger.info(f"Step {step['id']}: Early termination triggered - entropy low, confidence high")
                        early_terminated = True
                        # Use the direct answer as the result
                        direct_answer = retrieval_result.get("answer", "")
                        confidence = retrieval_result.get("confidence", 0.9)
                        
                        # Structure result for early termination
                        step_result = {
                            "step_id": step['id'],
                            "step_description": step.get("description", ""),
                            "subqueries": subqueries,
                            "retrieved_documents": [],
                            "extracted_passages": [{
                                "text": direct_answer,
                                "document_id": "early_termination",
                                "chunk_id": "direct_answer",
                                "relevance": confidence,
                                "reasoning": retrieval_result.get("reasoning", "Early termination based on entropy and confidence"),
                                "source_context": "Early termination"
                            }],
                            "qa_result": {
                                "answer": direct_answer,
                                "confidence": confidence,
                                "sources": ["early_termination"],
                                "supporting_evidence": []
                            },
                            "success": True,
                            "early_termination": True,
                            "stabilized_query": retrieval_result.get("stabilized_query", subquery["query"]),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Update state and return early termination result
                        await self.state_manager.add_step_result(step['id'], step_result)
                        return step_result
                    
                    # Normal retrieval - get documents from stabilized retrieval
                    docs_from_retrieval = retrieval_result.get("documents", [])
                    stabilized_query = retrieval_result.get("stabilized_query", subquery["query"])
                    
                    logger.debug(
                        f"Step {step['id']}: Stabilized query '{subquery['query']}' → '{stabilized_query}'. "
                        f"Retrieved {len(docs_from_retrieval)} documents"
                    )
                    
                    # Deduplicate by document ID and add to collection
                    for doc in docs_from_retrieval:
                        doc_id = doc.get("id") or doc.get("metadata", {}).get("id")
                        if doc_id and doc_id not in seen_doc_ids:
                            seen_doc_ids.add(doc_id)
                            all_retrieved_docs.append(doc)
                        elif not doc_id:
                            # If no ID, add anyway (will be deduplicated by content if needed)
                            all_retrieved_docs.append(doc)
                    
                    # Log regulator constraints if available
                    constraints = retrieval_result.get("constraints", [])
                    if constraints:
                        constraint_names = []
                        for c in constraints:
                            if isinstance(c, dict):
                                constraint_names.append(c.get('regulator_name', 'unknown'))
                            elif hasattr(c, 'regulator_name'):
                                constraint_names.append(c.regulator_name)
                        logger.debug(
                            f"Step {step['id']}: Applied {len(constraints)} regulator constraints "
                            f"({', '.join(constraint_names)})"
                        )
                    
                except Exception as e:
                    logger.warning(f"Diffusion-aware retrieval failed for subquery '{subquery['query']}': {e}")
                    # Fallback to direct retrieval if stabilize_and_retrieve fails
                    try:
                        retrieval_input = {
                            "query": subquery["query"],
                            "k": getattr(self.retriever, 'top_k', 10),
                            "min_similarity": getattr(self.retriever, 'min_similarity', 0.3),
                            "filter": {}
                        }
                        retrieval_response = await self.retriever.process(retrieval_input)
                        self._extract_and_aggregate_token_usage(retrieval_response)
                        
                        if retrieval_response.metadata.get("error"):
                            logger.warning(f"Fallback retrieval also failed for subquery: {subquery['query']}")
                            continue
                        
                        docs_from_metadata = retrieval_response.metadata.get("documents", [])
                        for doc in docs_from_metadata:
                            doc_id = doc.get("id") or doc.get("metadata", {}).get("id")
                            if doc_id and doc_id not in seen_doc_ids:
                                seen_doc_ids.add(doc_id)
                                all_retrieved_docs.append(doc)
                    except Exception as fallback_error:
                        logger.error(f"Fallback retrieval failed: {fallback_error}")
                        continue
            
            if not all_retrieved_docs:
                raise Exception("No documents retrieved")
            
            # Sort documents by relevance score (highest first) if scores available
            all_retrieved_docs.sort(
                key=lambda x: x.get("score", 0.0) if isinstance(x.get("score"), (int, float)) else 0.0,
                reverse=True
            )
            
            # Limit documents before passing to extractor (reduce load)
            max_docs_for_extractor = min(len(all_retrieved_docs), 15)  # Cap at 15 documents
            limited_docs = all_retrieved_docs[:max_docs_for_extractor]
            
            logger.info(f"Step {step['id']}: Collected {len(all_retrieved_docs)} unique documents, "
                       f"limiting to {len(limited_docs)} for extractor")
            
            # 3. Extractor Agent
            logger.debug(f"Step {step['id']}: Executing extraction on {len(limited_docs)} documents...")
            
            # Extract subquery texts for the extractor
            subquery_texts = [sq["query"] for sq in subqueries]
            
            extractor_input = {
                "query": step["description"],  # Keep for context
                "subqueries": subquery_texts,  # CRITICAL: Pass the actual subqueries used for retrieval
                "documents": limited_docs,  # Use limited, sorted, deduplicated docs
                "history": history,
                "max_documents": max_docs_for_extractor,  # Pass the limit explicitly
                "min_relevance": 0.2  # Lower threshold for extraction
            }

            # DEBUG: Log extractor input details
            logger.debug(f"[EXTRACTOR DEBUG] Step {step['id']}: Extractor Input Details:")
            logger.debug(f"  - Query: {step['description'][:200]}...")  # Truncate if too long
            logger.debug(f"  - Subqueries: {subquery_texts}")  # ADD THIS LINE to verify subqueries
            logger.debug(f"  - Number of documents: {len(all_retrieved_docs)}")
            logger.debug(f"  - Max documents to process: {extractor_input['max_documents']}")
            logger.debug(f"  - Min relevance: {extractor_input['min_relevance']}")
            logger.debug(f"  - History length: {len(history)}")
            
            # Show sample document structure (first 2-3 documents)
            logger.debug(f"[EXTRACTOR DEBUG] Sample documents (first {min(3, len(all_retrieved_docs))}):")
            for i, doc in enumerate(all_retrieved_docs[:3]):
                doc_id = doc.get('id', 'NO_ID')
                page_content_preview = doc.get('page_content', 'NO_CONTENT')[:200]  # First 200 chars
                score = doc.get('score', 'NO_SCORE')
                logger.debug(f"  Document {i+1}:")
                logger.debug(f"    - ID: {doc_id}")
                logger.debug(f"    - Score: {score}")
                logger.debug(f"    - Content preview: {page_content_preview}...")
                logger.debug(f"    - Content length: {len(doc.get('page_content', ''))}")
                logger.debug(f"    - Metadata keys: {list(doc.get('metadata', {}).keys())}")
            
            # Check if documents have content
            docs_with_content = sum(1 for doc in all_retrieved_docs if doc.get('page_content', '').strip())
            docs_empty = len(all_retrieved_docs) - docs_with_content
            logger.debug(f"[EXTRACTOR DEBUG] Document content status:")
            logger.debug(f"  - Documents with content: {docs_with_content}")
            logger.debug(f"  - Empty documents: {docs_empty}")
            
            extractor_response = await self.extractor.process(extractor_input)
            self._extract_and_aggregate_token_usage(extractor_response)
            
            if extractor_response.metadata.get("error"):
                raise Exception(f"Extractor failed: {extractor_response.metadata.get('error')}")
            
            clean_extraction = TokenizationUtils.strip_markdown_json(extractor_response.content)
            extracted_data = json.loads(clean_extraction)
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
            self._extract_and_aggregate_token_usage(qa_response)
            if qa_response.metadata.get("error"):
                raise Exception(f"QA agent failed: {qa_response.metadata.get('error')}")
            
            clean_qa = TokenizationUtils.strip_markdown_json(qa_response.content)
            qa_result = json.loads(clean_qa)

            #Update MCP state after step completes
            step_result_dict = {
                "step_id": step["id"],
                "answer": qa_result.get("answer", ""),
                "sources": qa_result.get("sources", []),
                "confidence": qa_result.get("confidence", 0.0)
            }

            mcp_state_manager.update_state(
                step_id=step["id"],
                step_result=step_result_dict,
                execution_id=self.current_execution_id
            )
            
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

