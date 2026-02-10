from typing import Dict, Any, List, Optional, Union, Set
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
from .granularity_posterior_module import GranularityPosteriorModule

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
        
        # ✅ LAYER 2: Track all candidates across steps (for analysis only, not prior building)
        self._all_candidates: List[Dict[str, Any]] = []
        self._granularity_prior: Optional[Dict[str, float]] = None  # Renamed from _granularity_posterior
        # ✅ FIX #1, #4, #5, #8: Store granularity once from original question
        self._original_query: Optional[str] = None
        self._required_domain: Optional[str] = None
        self._required_level: Optional[str] = None
        
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
    
    def _compute_adaptive_extractor_limit(
        self,
        all_retrieved_docs: List[Dict[str, Any]],
        flow_snapshot: Optional[Any] = None,
        hop: int = 1
    ) -> int:
        """
        Minimal adaptive document limit based on retrieval quality.
        
        Purpose:
        - Expand coverage when retrieval confidence is low
        - Stay predictable and debuggable
        
        Returns:
            Limit between 15 (base) and 30 (expanded)
        """
        BASE_LIMIT = 15
        EXPANDED_LIMIT = 30

        if not all_retrieved_docs:
            return BASE_LIMIT

        # ------------------------------------------------------------
        # Signal 1: Similarity quality (primary signal)
        # ------------------------------------------------------------
        scores = [
            doc.get("score", 0.0)
            for doc in all_retrieved_docs[:BASE_LIMIT]
            if isinstance(doc.get("score"), (int, float))
        ]

        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        weak_retrieval = (
            max_score < 0.4 or
            avg_score < 0.35
        )

        # ------------------------------------------------------------
        # ✅ EXPERIMENT 3: Signal 2: Confidence (uncertainty hint) - removed entropy
        # ------------------------------------------------------------
        confidence = 0.5
        if flow_snapshot:
            try:
                # Try multiple extraction paths
                if hasattr(flow_snapshot, "confidence"):
                    confidence = float(flow_snapshot.confidence)
                elif isinstance(flow_snapshot, dict):
                    confidence = float(flow_snapshot.get("confidence", 0.5))
                elif hasattr(flow_snapshot, "__dict__"):
                    confidence = float(getattr(flow_snapshot, "confidence", 0.5))
                # Clamp to valid range
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError, AttributeError):
                confidence = 0.5

        high_uncertainty = confidence < 0.5  # Low confidence = high uncertainty

        # ------------------------------------------------------------
        # Signal 3: Multi-hop modifier
        # ------------------------------------------------------------
        multihop_uncertain = hop > 1 and max_score < 0.5

        # ------------------------------------------------------------
        # Expansion decision (simple OR logic)
        # ------------------------------------------------------------
        if weak_retrieval or high_uncertainty or multihop_uncertain:
            logger.info(
                f"📈 Adaptive expansion: {BASE_LIMIT} → {EXPANDED_LIMIT} "
                f"(max_score={max_score:.3f}, avg_score={avg_score:.3f}, "
                f"confidence={confidence:.3f}, hop={hop})"
            )
            return min(EXPANDED_LIMIT, len(all_retrieved_docs))

        return BASE_LIMIT
    
    def _coverage_exhausted(
        self,
        slot_id: Optional[str],
        current_step_index: Optional[int],
        total_steps: Optional[int],
        seen_doc_ids: Set[str],
        all_retrieved_docs: List[Dict[str, Any]],
        hop: int,
        retrieval_result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if search coverage is exhausted for a given slot.
        
        Coverage answers: "Have we looked enough?" (not "Is the answer bad?")
        
        Coverage is exhausted if:
        1. We've reached max steps in the plan
        2. We've searched enough unique sources for this slot type
        3. Retrieval is no longer finding new documents (semantic space explored)
        4. Similarity degradation + novelty exhaustion (weak signal, only if coverage already broad)
        
        Args:
            slot_id: Semantic slot being searched (e.g., "spouse", "founder")
            current_step_index: Current step index (0-based)
            total_steps: Total steps in plan
            seen_doc_ids: Set of document IDs already retrieved
            all_retrieved_docs: List of all retrieved documents
            hop: Current hop number
            retrieval_result: Last retrieval result (for similarity scores)
            
        Returns:
            True if coverage is exhausted, False if more searching is useful
        """
        # ✅ FIX #1: Slot-aware source limits
        SOURCE_LIMITS = {
            "spouse": 3,      # Usually converges quickly
            "founder": 5,     # May need more sources
            "performer": 4,
            "location": 5,
            "headquarters": 5,
            "company": 5,
            "author": 4,
            "parent": 3,
            "child": 3,
            None: 6  # fallback for unknown slots
        }
        slot_limit = SOURCE_LIMITS.get(slot_id, SOURCE_LIMITS[None])
        
        # 1) Step budget exhausted
        # Don't mark last step as exhausted - it might be a synthesis step
        # Only mark as exhausted if we've gone past the plan
        if total_steps and current_step_index is not None:
            if current_step_index >= total_steps:  # Past last step, not on it
                logger.debug(
                    f"[Coverage] Exhausted: past last step ({current_step_index + 1} > {total_steps}) "
                    f"for slot '{slot_id}'"
                )
                return True
        
        # 2) Slot-aware source limit reached
        unique_sources = len(seen_doc_ids)
        if unique_sources >= slot_limit:
            logger.debug(
                f"[Coverage] Exhausted: searched {unique_sources} unique sources "
                f"(limit={slot_limit} for slot '{slot_id}')"
            )
            return True
        
        # 3) Retrieval no longer finding new documents (semantic space explored)
        # This is the PRIMARY coverage signal
        if retrieval_result:
            docs_from_last_retrieval = retrieval_result.get("documents", [])
            if docs_from_last_retrieval:
                new_docs_count = sum(
                    1 for doc in docs_from_last_retrieval
                    if (doc.get("id") or doc.get("metadata", {}).get("id")) not in seen_doc_ids
                )
                total_docs_retrieved = len(docs_from_last_retrieval)
                if total_docs_retrieved > 0:
                    new_docs_ratio = new_docs_count / total_docs_retrieved
                    if new_docs_ratio < 0.2:  # Less than 20% new documents
                        logger.debug(
                            f"[Coverage] Exhausted: semantic space explored - only {new_docs_ratio:.1%} "
                            f"new documents ({new_docs_count}/{total_docs_retrieved}) for slot '{slot_id}'"
                        )
                        return True
        
        # 4) ✅ FIX #2: Similarity degradation ONLY if novelty already exhausted
        # This is a weak signal - only applies if we've already searched broadly
        # and are no longer finding new documents
        if retrieval_result and all_retrieved_docs:
            docs_from_last = retrieval_result.get("documents", [])
            if docs_from_last:
                # Check novelty first (already computed above)
                new_docs_count = sum(
                    1 for doc in docs_from_last
                    if (doc.get("id") or doc.get("metadata", {}).get("id")) not in seen_doc_ids
                )
                total_docs = len(docs_from_last)
                new_docs_ratio = new_docs_count / total_docs if total_docs > 0 else 0.0
                
                # Get similarity scores
                last_similarities = []
                for doc in docs_from_last:
                    score = doc.get("score") or doc.get("similarity", 0.0)
                    if isinstance(score, (int, float)):
                        last_similarities.append(score)
                
                if last_similarities:
                    max_similarity = max(last_similarities)
                    
                    # ✅ FIX #2: Require BOTH novelty exhaustion AND low similarity
                    # This prevents premature exhaustion during diffusion (weak → strong)
                    if (
                        max_similarity < 0.3
                        and unique_sources >= slot_limit  # Already searched broadly
                        and new_docs_ratio < 0.2  # AND no longer finding new docs
                    ):
                        logger.debug(
                            f"[Coverage] Exhausted: low similarity ({max_similarity:.3f}) "
                            f"+ novelty exhausted ({new_docs_ratio:.1%}) "
                            f"after {unique_sources} sources for slot '{slot_id}'"
                        )
                        return True
        
        # 5) Hop budget (prevent infinite loops)
        MAX_HOPS_PER_SLOT = 3  # Configurable
        if hop > MAX_HOPS_PER_SLOT:
            logger.debug(
                f"[Coverage] Exhausted: exceeded max hops ({hop} > {MAX_HOPS_PER_SLOT}) "
                f"for slot '{slot_id}'"
            )
            return True
        
        # Coverage still available
        logger.debug(
            f"[Coverage] Remaining: {unique_sources}/{slot_limit} sources, "
            f"hop={hop}, slot='{slot_id}'"
        )
        return False

    def _is_answer_type_valid(self, answer: str, target_type: Optional[str]) -> bool:
        """
        Lightweight type validation gate to prevent category drift
        (e.g., awarding 'Young Artist Award' for PERSON).
        """
        if not target_type or not answer:
            return True
        answer_l = answer.strip().lower()
        target = target_type.strip().upper()

        # Generic category words that are not entities
        generic_roles = {"actress", "actor", "singer", "performer", "musician", "politician", "artist"}
        non_person_keywords = {
            "award", "prize", "festival", "band", "company", "city", "state", "province",
            "river", "album", "film", "movie", "song", "book", "school", "university",
            "magazine", "newspaper", "organization", "foundation", "committee", "station"
        }

        if target == "PERSON":
            if answer_l in generic_roles:
                return False
            if any(k in answer_l for k in non_person_keywords):
                return False
        return True
    
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
        
        # ✅ LAYER 2: Reset candidate tracking for new pipeline execution
        self._all_candidates = []
        self._granularity_prior = None  # Renamed from _granularity_posterior
        # ✅ FIX #1, #4, #5, #8: Reset granularity state
        self._original_query = None
        self._required_domain = None
        self._required_level = None
        
        # Reset protected answer manager for new query
        self.protected_answer_manager.clear()

        
        logger.info(f"Starting MA-RAG pipeline execution: {self.current_execution_id}")
        logger.info(f"Query: {query[:100]}...")

        
        # ✅ FIX #1: Infer slot from ORIGINAL query (before normalization)
        # This preserves keywords that normalization might strip (e.g., "spouse", "founder")
        target_slot = self.qa._infer_slot(query, step_context=None)  # Use original query
        
        # Normalize question for other processing (but slot already inferred)
        question = tokenization_utils.normalize_query(query)
        
        # Validate that we got a meaningful slot (not "default")
        if target_slot == "default":
            logger.warning(
                f"[Orchestrator] ⚠️ Slot inference returned 'default' for query: '{query[:60]}...'. "
                f"Normalized query: '{question[:60]}...'. This may indicate missing keywords."
            )
            # Try again with normalized query as fallback
            target_slot_fallback = self.qa._infer_slot(question, step_context=None)
            if target_slot_fallback != "default":
                logger.info(f"[Orchestrator] Fallback inference succeeded: '{target_slot_fallback}'")
                target_slot = target_slot_fallback
        
        self.protected_answer_manager.set_target_slot(target_slot)
        logger.info(f"[Orchestrator] ✅ Inferred target_slot='{target_slot}' from query: '{query[:60]}...'")
        
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

            # ✅ FIX #1, #4, #5, #8: Infer granularity once from original question
            self._original_query = query  # Store original query
            try:
                from .regulators.granularity_regulator import GranularityRegulator
                gran_reg = GranularityRegulator()
                # Infer from original question (plan_goal as fallback context, but query is primary)
                plan_goal = plan_result.get("disambiguated_query") or plan_result.get("query") or query
                self._required_domain, self._required_level, _ = GranularityRegulator.infer_required(
                    plan_goal=plan_goal,  # Use plan goal as context
                    query=query  # But primary source is original question
                )
                
                # Build granularity prior once from original question
                self._granularity_prior = GranularityRegulator.build_granularity_posterior(
                    candidates=[],  # Empty - prior is from question only
                    query=query,  # Original question
                    gran_regulator=gran_reg
                )
                
                logger.info(
                    f"[Orchestrator] ✅ Granularity inferred once: domain='{self._required_domain}', "
                    f"level='{self._required_level}' from original query"
                )
            except Exception as e:
                logger.warning(f"[Orchestrator] Failed to infer granularity: {e}")
                self._required_domain = None
                self._required_level = None
                self._granularity_prior = None


            
            
            
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
            target_type = None
            for sq in subqueries:
                if sq.get("target_type"):
                    target_type = sq.get("target_type")
                    break
            
            # Fallback: infer target_type if missing
            if not target_type:
                step_desc = (step.get("description") or "").lower()
                if any(term in step_desc for term in ["who", "spouse", "person", "performer", "actor", "actress"]):
                    target_type = "PERSON"
                elif any(term in step_desc for term in ["where", "location", "country", "city", "state", "province"]):
                    target_type = "LOC"
                elif any(term in step_desc for term in ["organization", "company", "institution", "agency"]):
                    target_type = "ORG"
                elif any(term in step_desc for term in ["when", "year", "date"]):
                    target_type = "DATE"
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
                    # Entity-anchored hop retrieval (if provided)
                    entity_name = subquery.get("entity_name")
                    if not entity_name and previous_answers:
                        # Pull the most recent answer as a fallback anchor
                        last_answer = list(previous_answers.values())[-1]
                        if isinstance(last_answer, dict):
                            entity_name = last_answer.get("qa_result", {}).get("answer")
                        elif isinstance(last_answer, str):
                            entity_name = last_answer
                        if entity_name:
                            subquery["entity_name"] = entity_name
                    if entity_name:
                        query_text = subquery.get("query", "")
                        if entity_name.lower() not in query_text.lower():
                            subquery["query"] = f"{entity_name} {query_text}".strip()
                            subquery["entity_anchor_added"] = True
                    retrieval_attempted = True
                    # Use diffusion-aware retrieval with stabilization
                    # Get main query from plan if plan_goal not provided
                    main_query = plan_goal or plan.get("query") or plan.get("disambiguated_query") or None
                    
                    # Extract breadcrumb_scope from subquery (from StepDefinerAgent Search Schema)
                    breadcrumb_scope = subquery.get("breadcrumb_scope")
                    
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
                    # Pass breadcrumb_scope for Bayesian re-ranking
                    if breadcrumb_scope:
                        retrieval_kwargs["breadcrumb_scope"] = breadcrumb_scope
                    
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
            
            # ✅ GRANULARITY POSTERIOR MODULE: Filter documents by hierarchical level alignment
            # Extract prior granularity from GranularityRegulator constraints
            prior_domain = None
            prior_level = None
            if last_retrieval_result:
                constraints = last_retrieval_result.get("constraints", [])
                for constraint in constraints:
                    # Handle both dict and object formats
                    constraint_dict = constraint.dict() if hasattr(constraint, 'dict') else (
                        constraint if isinstance(constraint, dict) else {}
                    )
                    if constraint_dict.get('regulator_name') == 'Granularity':
                        params = constraint_dict.get('parameters', {})
                        prior_domain = params.get('required_domain')
                        prior_level = params.get('required_level')
                        break
            
            # Apply Granularity Posterior Module
            posterior_module = GranularityPosteriorModule(
                filter_threshold=0.3,  # Configurable threshold
                weight_by_posterior=True
            )
            
            # Get stabilized query from last retrieval result
            query_for_posterior = None
            if last_retrieval_result:
                query_for_posterior = last_retrieval_result.get("stabilized_query")
            
            posterior_result = posterior_module.filter_documents(
                documents=all_retrieved_docs,
                prior_domain=prior_domain,
                prior_level=prior_level,
                query=query_for_posterior
            )
            
            # Update documents with filtered results
            all_retrieved_docs = posterior_result['filtered_documents']
            logger.debug(
                f"Step {step['id']}: Granularity Posterior filtered {posterior_result['filtered_count']} documents, "
                f"kept {len(all_retrieved_docs)} (prior: {prior_domain}/{prior_level})"
            )
            
            # ✅ ADAPTIVE COVERAGE: Expand limit based on retrieval quality
            flow_snapshot = last_retrieval_result.get("flow_snapshot") if last_retrieval_result else None
            max_docs_for_extractor = self._compute_adaptive_extractor_limit(
                all_retrieved_docs=all_retrieved_docs,
                flow_snapshot=flow_snapshot,
                hop=hop
            )
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

            # ✅ FIX #1, #4, #5, #8: Use pre-computed granularity (no per-step inference)
            # Granularity was inferred once at pipeline start from original question
            required_domain = self._required_domain
            required_level = self._required_level
            granularity_prior = self._granularity_prior  # Renamed from _granularity_posterior

            # Extract slot candidates from QA metadata
            slot_candidates = qa_response.metadata.get("slot_candidates", [])
            slot = None
            if slot_candidates:
                # Track candidates (for debugging/analysis, not for prior building)
                self._all_candidates.extend(slot_candidates)
                
                # Use pre-computed granularity prior (built once from original question)
                self.protected_answer_manager.propose_candidates(
                    slot_candidates,
                    required_domain=required_domain,
                    required_level=required_level,
                    granularity_posterior=granularity_prior  # Will be renamed in Group 4
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
                granularity_posterior=granularity_prior,  # Will be renamed in Group 4
            )

            # Add logging to see gate decisions
            answer_preview = qa_result.get("answer", "")[:30] if isinstance(qa_result, dict) else ""

            # Type-validation gate (prevents category drift)
            if target_type and not self._is_answer_type_valid(qa_result.get("answer", ""), target_type):
                gate_decision = {
                    "decision": "needs_more_evidence",
                    "reason": "type_mismatch"
                }
                logger.info(
                    f"[Gate] Step {step.get('id', 'unknown')}: "
                    f"decision=needs_more_evidence, reason=type_mismatch, "
                    f"answer='{answer_preview}...', target_type={target_type}"
                )
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
                # ✅ Rule 3: Check if more searching is useful (coverage check)
                coverage_exhausted = self._coverage_exhausted(
                    slot_id=slot,
                    current_step_index=current_step_index,
                    total_steps=total_steps,
                    seen_doc_ids=seen_doc_ids,
                    all_retrieved_docs=all_retrieved_docs,
                    hop=hop,
                    retrieval_result=last_retrieval_result
                )
                
                if coverage_exhausted:
                    # Abstain gracefully - don't overwrite protected answers
                    logger.info(
                        f"[Orchestrator] Coverage exhausted for slot '{slot}' in step {step.get('id')}, "
                        f"abstaining (do NOT overwrite protected answers)"
                    )
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
                        "abstained": True,  # ✅ Explicit abstention
                        "reason": "coverage_exhausted",
                        "coverage_info": {
                            "unique_sources": len(seen_doc_ids),
                            "hop": hop,
                            "current_step": current_step_index,
                            "total_steps": total_steps,
                            "slot": slot
                        }
                    }
                    return step_result
                else:
                    # Coverage remaining - continue searching
                    logger.info(
                        f"[Orchestrator] Coverage remaining for slot '{slot}' in step {step.get('id')}, "
                        f"continuing search"
                    )
                    # For now, return step_result but mark that more evidence is needed
                    # Future: could trigger re-retrieval with different parameters
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
                        "coverage_remaining": True,  # ✅ Coverage still available
                        "coverage_info": {
                            "unique_sources": len(seen_doc_ids),
                            "hop": hop,
                            "current_step": current_step_index,
                            "total_steps": total_steps,
                            "slot": slot
                        }
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

