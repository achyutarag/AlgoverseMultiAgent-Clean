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
from .validation.convergence_gate import ConvergenceValidityGate
from .state_manager.protected_answer_manager import ProtectedAnswerManager
from .tokenization_utils import TokenizationUtils
from .tokenization_utils import tokenization_utils
from .mcp_reasoning_state import mcp_state_manager
from .mcp_reasoning_state import MCPReasoningStateManager

logger = logging.getLogger(__name__)

# Support-aware planning / retrieval defaults
ENABLE_SUPPORT_AWARE_PLANNER = True
SUPPORT_BELIEF_FLOOR = 1
SUPPORT_EVIDENCE_TERM_FLOOR = 2
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
        self.convergence_gate = ConvergenceValidityGate()
        self.protected_answer_manager = ProtectedAnswerManager()
        
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

    def _compute_support_signals(self, flow_snapshot: Optional[Any], previous_answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute lightweight epistemic support signals for planner/step-definer biasing.
        """
        belief_count = 0
        evidence_terms_count = 0

        if flow_snapshot:
            fs_dict = flow_snapshot.dict() if hasattr(flow_snapshot, "dict") else (
                flow_snapshot.model_dump() if hasattr(flow_snapshot, "model_dump") else {}
            )
            belief_count = len(fs_dict.get("beliefs", {}) or {})
            evidence_terms_count = len(fs_dict.get("evidence_terms", []) or [])
        else:
            # Fallback to previous_answers as a proxy if no snapshot yet
            if previous_answers:
                belief_count = len(previous_answers)

        epistemic_support_low = (
            belief_count < SUPPORT_BELIEF_FLOOR or evidence_terms_count < SUPPORT_EVIDENCE_TERM_FLOOR
        )

        return {
            "belief_count": belief_count,
            "evidence_terms_count": evidence_terms_count,
            "epistemic_support_low": epistemic_support_low,
        }
    
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
        
        # Reset protected answer manager for new query
        self.protected_answer_manager.clear()

        
        logger.info(f"Starting MA-RAG pipeline execution: {self.current_execution_id}")
        logger.info(f"Query: {query[:100]}...")

        
        # Normalize question for consistent processing
        question = tokenization_utils.normalize_query(query)
        
        # Infer target slot ONCE per query (enforces question semantics)
        target_slot = self.qa._infer_slot(question, step_context=None)
        self.protected_answer_manager.set_target_slot(target_slot)
        logger.debug(f"[Orchestrator] Inferred target_slot='{target_slot}' from query: {question[:50]}...")
        
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
                # Pass hop number (i+1), step context, and main query for diffusion-aware retrieval
                step_result = await self._execute_single_step(
                    step, 
                    plan, 
                    hop=i+1, 
                    plan_goal=main_query,
                    current_step_index=i,
                    total_steps=len(ordered_steps)
                )
                
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
        plan_goal: Optional[str] = None,
        current_step_index: Optional[int] = None,
        total_steps: Optional[int] = None
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
            current_step_index: Current step index (0-based) in the plan
            total_steps: Total number of steps in the plan
            
        Returns:
            Step execution result
        """
        try:
            # Get accumulated history for context
            history = await self.state_manager.get_accumulated_history()
            previous_answers = await self.state_manager.get_previous_answers()
            flow_snapshot = await self.state_manager.get_current_flow_state()
            support_signals = self._compute_support_signals(flow_snapshot, previous_answers)
            epistemic_low = (
                support_signals.get("belief_count", 0) < SUPPORT_BELIEF_FLOOR
                or support_signals.get("evidence_terms_count", 0) < SUPPORT_EVIDENCE_TERM_FLOOR
            )
            # Hard gate: if epistemic support is low, force gather_context (block answer mode)
            if ENABLE_SUPPORT_AWARE_PLANNER and epistemic_low:
                planner_mode = "gather_context"
            else:
                planner_mode = "answer"

            # 1. Step Definer Agent
            logger.debug(f"Step {step['id']}: Executing Step Definer...")
            step_definer_input = {
                "step": step,
                "plan": plan,
                "history": history,
                "previous_answers": previous_answers,
                "support_signals": support_signals,
                "planner_mode": planner_mode,
            }
            
            step_definer_response = await self.step_definer.process(step_definer_input)
            self._extract_and_aggregate_token_usage(step_definer_response)
            if step_definer_response.metadata.get("error"):
                raise Exception(f"Step definer failed: {step_definer_response.metadata.get('error')}")
            
            clean_subqueries = TokenizationUtils.strip_markdown_json(step_definer_response.content)
            subqueries_data = json.loads(clean_subqueries)
            subqueries = subqueries_data.get("sub_queries", [])
            # Defensive clamp: if planner_mode demanded gather_context, ensure no answer-intent slips through
            if planner_mode == "gather_context":
                forced = False
                for sq in subqueries:
                    intent = sq.get("intent") or sq.get("purpose") or ""
                    if isinstance(intent, str) and intent.lower() in {"answer", "direct_answer", "respond"}:
                        sq["intent"] = "gather_context"
                        sq["forced_by_support_gate"] = True
                        forced = True
                if forced:
                    logger.debug(f"Support gate enforced gather_context on subqueries for step {step.get('id')}")
            
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
                            plan_goal=plan_goal,
                            protected_answer_manager=self.protected_answer_manager
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
                            
                            # Check for hierarchical level mismatch using GranularityRegulator
                            # (generalized, not hardcoded to location terms)
                            from .regulators.granularity_regulator import GranularityRegulator
                            granularity_reg = GranularityRegulator()
                            required_domain, required_level = granularity_reg._infer_required_level(plan_goal)
                            answer_domain, answer_level, _ = granularity_reg.classify_entity_level(direct_answer)
                            
                            # Check if answer violates required level
                            is_level_violation = granularity_reg.is_level_violation(
                                required_domain, required_level,
                                answer_domain, answer_level
                            )
                            
                            # If answer violates required hierarchical level, that's wrong
                            if is_level_violation:
                                should_skip_retrieval = False
                                validation_reason = (
                                    f"Hierarchical level mismatch: plan requires {required_domain}/{required_level} "
                                    f"but direct_answer is {answer_domain}/{answer_level} '{direct_answer}'. "
                                    f"Need to retrieve correct-level information."
                                )
                            # If alignment is very low, the answer might be incomplete
                            elif alignment < 0.3:
                                should_skip_retrieval = False
                                validation_reason = (
                                    f"Low plan alignment ({alignment:.2f}) for direct_answer "
                                    f"'{direct_answer}'. Proceeding with retrieval to verify."
                                )
                            else:
                                # ✅ FIX 4: Boundary Condition Check (HARD FILTER)
                                # Check if direct_answer satisfies the query's boundary condition
                                # This is a CONSTRAINT, not a score - answers that violate are rejected
                                query_to_check = step_description or plan_goal or ""
                                if query_to_check and hasattr(self.state_manager, '_check_answer_satisfies_boundary_condition'):
                                    try:
                                        boundary_check = self.state_manager._check_answer_satisfies_boundary_condition(
                                            direct_answer,
                                            query_to_check
                                        )
                                        
                                        if not boundary_check.get("satisfies", True):
                                            should_skip_retrieval = False
                                            validation_reason = (
                                                f"Boundary condition violation: direct_answer '{direct_answer}' "
                                                f"is in input space (not solution space). "
                                                f"{boundary_check.get('reason', 'answer violates query boundary condition')}. "
                                                f"Proceeding with retrieval to find valid solution."
                                            )
                                            logger.warning(
                                                f"Step {step['id']}: Direct answer '{direct_answer}' rejected due to "
                                                f"boundary condition violation: {boundary_check.get('reason', 'unknown')}"
                                            )
                                        else:
                                            validation_reason = (
                                                f"Direct answer '{direct_answer}' validated: "
                                                f"alignment={alignment:.2f}, level check passed, "
                                                f"boundary condition satisfied"
                                            )
                                    except Exception as boundary_error:
                                        logger.warning(
                                            f"Boundary condition check failed for direct_answer: {boundary_error}. "
                                            f"Proceeding with retrieval to be safe."
                                        )
                                        should_skip_retrieval = False
                                        validation_reason = f"Boundary condition check error: {boundary_error}"
                                else:
                                    # If boundary check not available, still validate other checks passed
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

                    # ✅ Explicitly declare retrieval skipped (postcondition satisfaction)
                    # This hop is allowed to proceed without stabilize_and_retrieve().
                    step_result["debug_metadata"] = step_result.get("debug_metadata", {})
                    step_result["debug_metadata"]["orchestrator"] = {
                        "retrieval_attempted": False,
                        "retrieval_skipped": True,
                        "retrieval_skipped_reason": "direct_answer_used",
                        "forced_retrieval": False,
                        "forced_retrieval_reason": None,
                        "forced_retrieval_query": None,
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
            last_retrieval_result = None  # Store last retrieval_result for QA agent
            stabilized_query = None  # ✅ FIX: Initialize before loop to prevent NameError
            # ✅ HOP VALIDITY POSTCONDITION (control-only)
            # A hop is valid only if:
            # - stabilize_and_retrieve() was executed (retrieval_attempted == True), OR
            # - retrieval was explicitly skipped and declared (retrieval_skipped == True with a reason).
            # If neither is true, we must NOT silently continue to extractor/QA.
            retrieval_attempted = False
            retrieval_skipped = False
            retrieval_skipped_reason = None
            forced_retrieval = False
            forced_retrieval_reason = None
            forced_retrieval_query = None
            
            for subquery in subqueries:
                try:
                    retrieval_attempted = True
                    # Use diffusion-aware retrieval with stabilization
                    # Get main query from plan if plan_goal not provided
                    main_query = plan_goal or plan.get("query") or plan.get("disambiguated_query") or None
                    
                    # Build kwargs to avoid argument conflicts
                    retrieval_kwargs = {
                        "proposed_query": subquery["query"],
                        "hop": hop,
                        "previous_answers": previous_answers,
                        "retriever_agent": self.retriever,
                        "current_step_index": current_step_index,
                        "total_steps": total_steps
                    }
                    # Only pass plan_goal if it has a value (not None or empty string)
                    if main_query:
                        retrieval_kwargs["plan_goal"] = main_query
                    
                    retrieval_result = await self.state_manager.stabilize_and_retrieve(**retrieval_kwargs)
                    
                    # ✅ REMOVED: Early termination check before retrieval
                    # Early termination now happens AFTER QA produces the current step's answer
                    # This ensures the current step always executes and finds its own answer
                    
                    # Normal retrieval - get documents from stabilized retrieval
                    docs_from_retrieval = retrieval_result.get("documents", [])
                    stabilized_query = retrieval_result.get("stabilized_query", subquery["query"])
                    last_retrieval_result = retrieval_result  # Store for QA agent
                    
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
                    retrieval_attempted = True
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

            # ✅ POSTCONDITION ENFORCEMENT: If retrieval was neither attempted nor explicitly skipped, force recovery.
            # (This prevents "phantom hops" that emit QA results without running retrieval/regulators.)
            if not retrieval_attempted and not retrieval_skipped:
                forced_retrieval = True
                forced_retrieval_reason = "hop_validity_violation_no_retrieval_or_skip_declared"

                # Fallback preference: last stabilized query (from StateManager) -> plan goal -> step description/objective.
                last_stabilized = getattr(self.state_manager, "_last_stabilized_query", None)
                forced_retrieval_query = (
                    last_stabilized
                    or plan_goal
                    or step.get("objective")
                    or step.get("description")
                    or "search for relevant information"
                )

                main_query = plan_goal or plan.get("query") or plan.get("disambiguated_query") or None
                retrieval_kwargs = {
                    "proposed_query": str(forced_retrieval_query),
                    "hop": hop,
                    "previous_answers": previous_answers,
                    "retriever_agent": self.retriever,
                    "current_step_index": current_step_index,
                    "total_steps": total_steps
                }
                if main_query:
                    retrieval_kwargs["plan_goal"] = main_query

                retrieval_attempted = True
                retrieval_result = await self.state_manager.stabilize_and_retrieve(**retrieval_kwargs)
                docs_from_retrieval = retrieval_result.get("documents", [])
                stabilized_query = retrieval_result.get("stabilized_query", str(forced_retrieval_query))
                last_retrieval_result = retrieval_result

                for doc in docs_from_retrieval:
                    doc_id = doc.get("id") or doc.get("metadata", {}).get("id")
                    if doc_id and doc_id not in seen_doc_ids:
                        seen_doc_ids.add(doc_id)
                        all_retrieved_docs.append(doc)
                    elif not doc_id:
                        all_retrieved_docs.append(doc)

            # Hard guardrail: if still neither attempted nor explicitly skipped, abort hop.
            if not retrieval_attempted and not retrieval_skipped:
                raise Exception(
                    "Hop contract violation: neither retrieval_attempted nor retrieval_skipped "
                    "(recovery failed). Aborting hop."
                )
            
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

            # ✅ CONCISE: Only essential extractor info
            logger.debug(f"Extractor: {len(all_retrieved_docs)} docs → {extractor_input['max_documents']} for processing")
            
            extractor_response = await self.extractor.process(extractor_input)
            self._extract_and_aggregate_token_usage(extractor_response)
            
            if extractor_response.metadata.get("error"):
                raise Exception(f"Extractor failed: {extractor_response.metadata.get('error')}")
            
            clean_extraction = TokenizationUtils.strip_markdown_json(extractor_response.content)
            extracted_data = json.loads(clean_extraction)
            extracted_passages = extracted_data.get("extracted_passages", [])
            
            if not extracted_passages:
                raise Exception("No passages extracted")

            # Attach extractor metadata (including evidence_terms) to the retrieval result/flow snapshot
            extractor_metadata = extractor_response.metadata or {}
            if last_retrieval_result is not None:
                last_retrieval_result["extracted_passages_metadata"] = extractor_metadata
                try:
                    fs = last_retrieval_result.get("flow_snapshot") or {}
                    if hasattr(fs, "get"):
                        fs_evidence = fs.get("evidence_terms") or []
                        ext_terms = extractor_metadata.get("evidence_terms") or []
                        if ext_terms and not fs_evidence:
                            fs["evidence_terms"] = ext_terms
                        last_retrieval_result["flow_snapshot"] = fs
                except Exception:
                    pass
            

            # 4. QA Agent - Enhanced with diffusion-aware stabilized belief field
            # ====================================================================
            # DIFFUSION MODEL: Multi-hop reasoning is a diffusion process where
            # beliefs P(x,t) spread through document space. The QA agent performs
            # entropy-aware compression, collapsing probability mass into anchors
            # for the next hop. This is the compression step: P(x,t+1) = compress(P(x,t), anchors)
            # ====================================================================
            logger.debug(f"Step {step['id']}: Executing QA synthesis with diffusion-aware compression...")
            
            # Get flow_snapshot and constraints from retrieval_result (stabilized belief field)
            # Use last_retrieval_result from the loop, or fallback to state_manager
            flow_snapshot = None
            constraints = []
            # ✅ FIX: Safe fallback - use stabilized_query if available, else use first subquery, else step description
            stabilized_query_for_qa = stabilized_query if stabilized_query else (subqueries[0]["query"] if subqueries else step.get("description", ""))
            
            # Try to get from last_retrieval_result (from stabilize_and_retrieve)
            if last_retrieval_result:
                flow_snapshot = last_retrieval_result.get("flow_snapshot")
                constraints = last_retrieval_result.get("constraints", [])
                stabilized_query_for_qa = last_retrieval_result.get("stabilized_query", stabilized_query_for_qa)
            
            # If not available, try to get from state_manager (for entropy tracking)
            if not flow_snapshot and hasattr(self.state_manager, 'reasoning_flow'):
                try:
                    # Get current flow state from state_manager
                    current_state = await self.state_manager.get_current_flow_state()
                    if current_state:
                        flow_snapshot = current_state
                except Exception as e:
                    logger.debug(f"Could not get flow_snapshot from state_manager: {e}")
            
            qa_input = {
                "question": step["description"],
                "context": extracted_passages,
                "step_context": step,
                "overall_query": plan.get("main_question", ""),
                "previous_answers": previous_answers,
                # ✅ DIFFUSION-AWARE INPUTS: Stabilized belief field
                "flow_snapshot": flow_snapshot,  # Contains H(t), D(t), anchors, beliefs (may be None)
                "regulator_constraints": constraints,  # Boundary conditions from regulators
                "stabilized_query": stabilized_query_for_qa,  # Stabilized query used for retrieval
                "hop": hop  # Current hop number (time step in diffusion)
            }
            
            qa_response = await self.qa.process(qa_input)
            self._extract_and_aggregate_token_usage(qa_response)
            if qa_response.metadata.get("error"):
                raise Exception(f"QA agent failed: {qa_response.metadata.get('error')}")
            
            clean_qa = TokenizationUtils.strip_markdown_json(qa_response.content)
            qa_result = json.loads(clean_qa)

            # Required granularity (computed once, used for both manager and gate)
            required_domain = None
            required_level = None
            try:
                from .regulators.granularity_regulator import GranularityRegulator
                required_domain, required_level, _ = GranularityRegulator.infer_required(
                    plan_goal, stabilized_query_for_qa
                )
            except Exception:
                pass

            # Extract slot candidates from QA metadata
            slot_candidates = qa_response.metadata.get("slot_candidates", [])
            slot = None
            if slot_candidates:
                self.protected_answer_manager.propose_candidates(
                    slot_candidates,
                    required_domain=required_domain,
                    required_level=required_level
                )
                # Extract slot from first candidate for gate
                slot = slot_candidates[0].get("slot") if slot_candidates else None
                # Add slot_candidates to qa_result for assembler use
                qa_result["slot_candidates"] = slot_candidates
                qa_result["slot"] = slot

            # Pass manager (read-only) to gate
            protected_answers = self.protected_answer_manager.get_protected_answers()  # read-only

            # Convergence / validity gate (single control point)
            gate_decision = self.convergence_gate.evaluate(
                qa_result=qa_result,
                protected_answers=protected_answers,
                required_domain=required_domain,
                required_level=required_level,
                slot=slot,
            )

            # Add logging to see gate decisions
            answer_preview = qa_result.get("answer", "")[:30] if isinstance(qa_result, dict) else ""
            logger.info(
                f"[Gate] Step {step.get('id', 'unknown')}: "
                f"decision={gate_decision.get('decision')}, "
                f"reason={gate_decision.get('reason')}, "
                f"answer='{answer_preview}...', "
                f"slot={slot}"
            )

            decision = gate_decision.get("decision")
            if decision == "reject":
                step_result = {
                    "step_id": step["id"],
                    "step_description": step.get("description", ""),
                    "subqueries": subqueries,
                    "retrieved_documents": all_retrieved_docs,
                    "extracted_passages": extracted_passages,
                    "qa_result": qa_result,
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                    "gate_decision": gate_decision,
                }
                return step_result

            if decision == "needs_more_evidence":
                step_result = {
                    "step_id": step["id"],
                    "step_description": step.get("description", ""),
                    "subqueries": subqueries,
                    "retrieved_documents": all_retrieved_docs,
                    "extracted_passages": extracted_passages,
                    "qa_result": qa_result,
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                    "gate_decision": gate_decision,
                    "needs_more_evidence": True,
                }
                return step_result

            # Protected anchor commit (normalized, step-scoped) after accept
            qa_answer_raw = (qa_result.get("answer") or "").strip()
            qa_conf = qa_result.get("confidence", 0.0)
            qa_support = qa_result.get("supporting_evidence") or []
            import re
            norm = lambda s: re.sub(r"\s+", " ", s.strip().lower()) if s else ""
            if (
                qa_answer_raw
                and qa_answer_raw.lower() not in {"unknown", "none", "n/a"}
                and qa_conf >= 0.7
                and qa_support
                and last_retrieval_result is not None
            ):
                fs = last_retrieval_result.get("flow_snapshot") or {}
                protected = set(fs.get("protected_anchors") or [])
                protected.add(norm(qa_answer_raw))
                fs["protected_anchors"] = list(protected)
                fs["protected_anchors_step"] = step.get("id")
                last_retrieval_result["flow_snapshot"] = fs
                # Also store on state_manager for flow_update access
                try:
                    if hasattr(self.state_manager, "__dict__"):
                        self.state_manager._protected_anchors = {
                            "anchors": list(protected),
                            "step_id": step.get("id"),
                        }
                except Exception:
                    pass

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
            
            # Post-extraction support refresh: surface evidence terms/anchors for next planner decision
            try:
                if hasattr(self.state_manager, "_extract_evidence_terms"):
                    refreshed_terms = self.state_manager._extract_evidence_terms({
                        step["id"]: {"answer": qa_result.get("answer", ""), "evidence": extracted_passages}
                    })
                    if isinstance(refreshed_terms, list):
                        # attach to debug for visibility
                        qa_result["refreshed_evidence_terms"] = refreshed_terms
            except Exception as e:
                logger.debug(f"Post-extraction support refresh skipped: {e}")

            # Return comprehensive step result
            step_result = {
                "step_id": step["id"],
                "step_description": step.get("description", ""),
                "subqueries": subqueries,
                "retrieved_documents": all_retrieved_docs,
                "extracted_passages": extracted_passages,
                "qa_result": qa_result,
                "success": True,
                "timestamp": datetime.now().isoformat()
                
            }

            # ✅ INVARIANT METADATA: Make retrieval behavior observable per hop
            step_result["debug_metadata"] = {
                "retrieval_attempted": retrieval_attempted,
                "retrieval_skipped": retrieval_skipped,
                "retrieval_skipped_reason": retrieval_skipped_reason,
                "forced_retrieval": forced_retrieval,
                "forced_retrieval_reason": forced_retrieval_reason,
                "forced_retrieval_query": forced_retrieval_query,
                "support_signals": support_signals,
                "planner_mode": planner_mode,
                "retrieval_support_signals": (last_retrieval_result or {}).get("support_signals"),
            }

            return step_result
            
        except Exception as e:
            logger.error(f"Step execution failed: {str(e)}")
            step_result = {
                "step_id": step["id"],
                "step_description": step.get("description", ""),
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
            # ✅ Always surface hop validity state even on errors (never allow debug_metadata == {})
            step_result["debug_metadata"] = step_result.get("debug_metadata", {})
            step_result["debug_metadata"]["orchestrator"] = {
                "retrieval_attempted": locals().get("retrieval_attempted", False),
                "retrieval_skipped": locals().get("retrieval_skipped", False),
                "retrieval_skipped_reason": locals().get("retrieval_skipped_reason", None),
                "forced_retrieval": locals().get("forced_retrieval", False),
                "forced_retrieval_reason": locals().get("forced_retrieval_reason", None),
                "forced_retrieval_query": locals().get("forced_retrieval_query", None),
                "invalid_hop": (not locals().get("retrieval_attempted", False) and not locals().get("retrieval_skipped", False)),
            }
            return step_result
    
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
            # Include execution_state (protected_answers is now managed by ProtectedAnswerManager)
            execution_state = None
            try:
                execution_state = self.state_manager.executions.get(self.state_manager.current_execution_id)
            except Exception:
                execution_state = None

            # Get protected answers from manager only (no fallback to legacy state)
            protected_answers = self.protected_answer_manager.get_protected_answers()

            assembler_input = {
                "main_query": plan.get("main_question", ""),
                "disambiguated_query": plan.get("disambiguated_query", ""),
                "query_type": plan.get("query_type", "unknown"),
                "step_results": step_results,
                "plan": plan,
                "execution_state": execution_state,
                "protected_answers": protected_answers,  # From manager only
                "protected_answer_manager": self.protected_answer_manager,  # Pass manager for slot-aware ranking
                # Legacy: protected_anchors kept for compatibility if used elsewhere
                "protected_anchors": getattr(self.state_manager, "_protected_anchors", {})
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

