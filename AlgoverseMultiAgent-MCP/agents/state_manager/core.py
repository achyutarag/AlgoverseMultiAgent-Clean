# agents/state_manager/core.py
from typing import Dict, Any, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
import json
import logging
from datetime import datetime
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ExecutionState(BaseModel):
    """State of a single pipeline execution."""
    execution_id: str = Field(..., description="Unique execution identifier")
    main_query: str = Field(..., description="Original user query")
    disambiguated_query: str = Field("", description="Disambiguated version of query")
    query_type: str = Field("unknown", description="Type of query")
    plan: Optional[Dict[str, Any]] = Field(None, description="Generated plan")
    completed_steps: List[str] = Field(default_factory=list, description="IDs of completed steps")
    step_results: Dict[str, Any] = Field(default_factory=dict, description="Results from each step")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Accumulated history Hi")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    start_time: Optional[datetime] = Field(None, description="Execution start time")
    last_update: Optional[datetime] = Field(None, description="Last state update time")
    
    # ✅ GLOBAL ANCHOR STATE (No Silent Forgetting Invariant)
    # Validated anchors persist across hops unless explicitly revoked.
    validated_anchors: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Globally validated anchors that must persist across hops unless "
            "explicitly revoked with a recorded reason and evidence."
        ),
    )
    revoked_anchors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "History of explicitly revoked anchors: "
            "[{'anchor': str, 'hop': int, 'reason': str, 'evidence': Optional[Any]}]."
        ),
    )

    # Protected high-confidence answers per slot (monotonic, evidence-backed)
    protected_answers: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-slot protected answers with confidence ≥ threshold and evidence."
    )

    class Config:
        extra = "allow"

class StateManager:
    """
    Enhanced State Manager with diffusion-aware retrieval control.
    
    Now acts as the central controller that:
    - Persists state across hops
    - Shapes what each agent sees
    - Stabilizes the overall pipeline
    - Enforces continuity and direction
    - Prevents drift between steps
    - Performs entropy-aware retrieval
    """
    
    def __init__(self, max_history_size: int = 100):
        """
        Initialize the State Manager with diffusion-aware components.
        
        Args:
            max_history_size: Maximum number of history items to keep
        """
        self.max_history_size = max_history_size
        self.executions: Dict[str, ExecutionState] = {}
        self.current_execution_id: Optional[str] = None
        
        # Initialize diffusion-aware components
        try:
            # ✅ EXPERIMENT 3: Removed EntropyTracker - using simple heuristics instead
            # from ..entropy_tracker import EntropyTracker
            from ..reasoning_flow import ReasoningFlowIndex
            from ..regulators.regulator_manager import RegulatorManager
            from ..regulators.granularity_regulator import GranularityRegulator
            # ✅ EXPERIMENT 1b: Adding back EntityRegulator
            from ..regulators.entity_regulator import EntityRegulator
            # ✅ EXPERIMENT 1b: Removed EvidenceRegulator, RelationRegulator, ConfidenceRegulator - no benefit
            # ✅ EXPERIMENT 2: PlanRegulator confirmed as CORE component - restores stability/trajectory control
            from ..regulators.plan_regulator import PlanRegulator
            
            # ✅ EXPERIMENT 3: Removed entropy tracker - using simple heuristics instead
            self.entropy_tracker = None
            
            # Initialize reasoning flow (without entropy tracker integration)
            self.reasoning_flow = ReasoningFlowIndex(entropy_tracker=None)
            
            # Initialize regulators
            # ✅ GranularityRegulator FIRST - Initial Condition (u(x,0))
            # This sets the correct hierarchical level before retrieval begins,
            # preventing hierarchical leakage and level-mismatch cascades
            # ✅ FINAL: Optimal 3-regulator configuration (Granularity + Entity + Plan)
            # PlanRegulator confirmed as CORE - increases stability/trajectory control
            # (fewer hops + fewer anchors + fewer catastrophic step drift cases)
            # EvidenceRegulator, RelationRegulator, ConfidenceRegulator removed - no benefit
            regulators = [
                GranularityRegulator(weight=1.0),  # FIRST - initial condition
                EntityRegulator(weight=0.9),        # ✅ CRITICAL - restores full MCP performance
                PlanRegulator(weight=0.9)          # ✅ CORE - stability/trajectory control
            ]
            self.regulator_manager = RegulatorManager(regulators)
            
            logger.info("State Manager initialized with diffusion-aware components")
        except ImportError as e:
            logger.warning(f"Could not import diffusion-aware components: {e}")
            self.entropy_tracker = None
            self.reasoning_flow = None
            self.regulator_manager = None
        
        # Import and bind methods from other modules
        from .flow_update import (
            _update_flow_state,
            _extract_beliefs,
            _extract_entity_anchors,
            _extract_evidence_terms,
            _detect_relation_direction,
            _calculate_plan_alignment,
            _calculate_confidence,
            _should_create_anchor,
            _calculate_entropy_aware_strength
        )
        from .retrieval import (
            stabilize_and_retrieve,
            _entropy_aware_retrieve,
            _check_termination_validity,
            _check_answer_satisfies_boundary_condition,
            _extract_subject_entities,
            _calculate_query_complexity
        )
        from .metrics import (
            get_execution_summary,
            get_execution_snapshots,
            get_step_dependencies_status
        )
        
        # Bind methods to this instance
        self._update_flow_state = _update_flow_state.__get__(self, StateManager)
        self._extract_beliefs = _extract_beliefs.__get__(self, StateManager)
        self._extract_entity_anchors = _extract_entity_anchors.__get__(self, StateManager)
        self._extract_evidence_terms = _extract_evidence_terms.__get__(self, StateManager)
        self._detect_relation_direction = _detect_relation_direction.__get__(self, StateManager)
        self._calculate_plan_alignment = _calculate_plan_alignment.__get__(self, StateManager)
        self._calculate_confidence = _calculate_confidence.__get__(self, StateManager)
        self._should_create_anchor = _should_create_anchor.__get__(self, StateManager)
        self._calculate_entropy_aware_strength = _calculate_entropy_aware_strength.__get__(self, StateManager)
        
        # ✅ FIX: Properly bind stabilize_and_retrieve to avoid argument conflicts
        # Create a bound method wrapper that ensures correct argument passing
        async def bound_stabilize_and_retrieve(
            proposed_query,
            hop,
            previous_answers,
            plan_goal=None,
            retriever_agent=None,
            current_step_index=None,
            total_steps=None,
            breadcrumb_scope=None
        ):
            return await stabilize_and_retrieve(
                self,
                proposed_query,
                hop,
                previous_answers,
                plan_goal,
                retriever_agent,
                current_step_index,
                total_steps,
                breadcrumb_scope
            )
        self.stabilize_and_retrieve = bound_stabilize_and_retrieve
        
        self._entropy_aware_retrieve = _entropy_aware_retrieve.__get__(self, StateManager)
        self._check_termination_validity = _check_termination_validity.__get__(self, StateManager)
        self._check_answer_satisfies_boundary_condition = _check_answer_satisfies_boundary_condition.__get__(self, StateManager)
        self._extract_subject_entities = _extract_subject_entities.__get__(self, StateManager)
        self._calculate_query_complexity = _calculate_query_complexity.__get__(self, StateManager)
        self.get_execution_summary = get_execution_summary.__get__(self, StateManager)
        self.get_execution_snapshots = get_execution_snapshots.__get__(self, StateManager)
        self.get_step_dependencies_status = get_step_dependencies_status.__get__(self, StateManager)
    
    async def get_current_flow_state(self):
        """
        Get current flow snapshot from reasoning flow.
        
        Returns:
            FlowSnapshot if available, None otherwise
        """
        if self.reasoning_flow:
            try:
                return self.reasoning_flow.get_flow_snapshot()
            except Exception as e:
                logger.debug(f"Could not get flow snapshot: {e}")
                return None
        return None
    
    async def initialize_execution(
        self, 
        execution_id: str, 
        main_query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize a new pipeline execution.
        
        Args:
            execution_id: Unique identifier for this execution
            main_query: The original user query
            context: Optional additional context
        """
        logger.info(f"Initializing execution state: {execution_id}")
        
        self.current_execution_id = execution_id
        
        # Create new execution state
        execution_state = ExecutionState(
            execution_id=execution_id,
            main_query=main_query,
            context=context or {},
            start_time=datetime.now(),
            last_update=datetime.now()
        )
        
        self.executions[execution_id] = execution_state
        
        # Reset reasoning flow and entropy tracker for new execution
        if self.reasoning_flow:
            self.reasoning_flow.flow_states.clear()
            self.reasoning_flow.bucket_anchors.clear()
        
        # ✅ EXPERIMENT 3: Removed entropy tracker - no history to clear
        # if self.entropy_tracker:
        #     self.entropy_tracker.entropy_history.clear()
        
        logger.info(f"Execution state initialized for query: {main_query[:100]}...")
    
    async def update_plan(self, plan: Dict[str, Any]) -> None:
        """Update the execution state with the generated plan."""
        if not self.current_execution_id:
            raise Exception("No active execution to update")
        
        execution_state = self.executions[self.current_execution_id]
        execution_state.plan = plan
        execution_state.disambiguated_query = plan.get("disambiguated_query", execution_state.main_query)
        execution_state.query_type = plan.get("query_type", "unknown")
        execution_state.last_update = datetime.now()
        
        logger.info(f"Plan updated with {len(plan.get('steps', []))} steps")
    
    async def resolve_step_dependencies(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve step dependencies and return executable order."""
        if not steps:
            return []
        
        logger.info(f"Resolving dependencies for {len(steps)} steps")
        
        # Build dependency graph
        step_map = {step["id"]: step for step in steps}
        dependencies = {step["id"]: set(step.get("dependencies", [])) for step in steps}
        in_degree = {step["id"]: len(step.get("dependencies", [])) for step in steps}
        
        # Topological sort using Kahn's algorithm
        queue = deque([step_id for step_id, degree in in_degree.items() if degree == 0])
        ordered_steps = []
        
        while queue:
            current_step_id = queue.popleft()
            ordered_steps.append(step_map[current_step_id])
            
            # Update in-degrees of dependent steps
            for step_id, deps in dependencies.items():
                if current_step_id in deps:
                    in_degree[step_id] -= 1
                    if in_degree[step_id] == 0:
                        queue.append(step_id)
        
        # Check for circular dependencies
        if len(ordered_steps) != len(steps):
            remaining_steps = [step for step in steps if step not in ordered_steps]
            logger.warning(f"Circular dependencies detected in steps: {[s['id'] for s in remaining_steps]}")
            ordered_steps.extend(remaining_steps)
        
        logger.info(f"Step execution order: {[s['id'] for s in ordered_steps]}")
        return ordered_steps
    
    async def add_step_result(self, step_id: str, result: Dict[str, Any]) -> None:
        """Add step result to the execution state and update history."""
        if not self.current_execution_id:
            raise Exception("No active execution to update")
        
        execution_state = self.executions[self.current_execution_id]
        
        # Add to completed steps
        if step_id not in execution_state.completed_steps:
            execution_state.completed_steps.append(step_id)
        
        # Store step result
        execution_state.step_results[step_id] = result
        
        # Update history Hi = {(s1, a1), ..., (si, ai)}
        history_entry = {
            "step_id": step_id,
            "step_description": result.get("step_description", ""),
            "answer": result.get("qa_result", {}).get("answer", ""),
            "confidence": result.get("qa_result", {}).get("confidence", 0.0),
            "sources": result.get("qa_result", {}).get("sources", []),
            "timestamp": datetime.now().isoformat()
        }
        
        execution_state.history.append(history_entry)
        
        # Trim history if too large
        if len(execution_state.history) > self.max_history_size:
            execution_state.history = execution_state.history[-self.max_history_size:]
        
        execution_state.last_update = datetime.now()
        
        logger.info(f"Step result added: {step_id}")
    
    async def get_accumulated_history(self) -> List[Dict[str, Any]]:
        """Get the accumulated history Hi-1 = {(s1, a1), ..., (si-1, ai-1)}."""
        if not self.current_execution_id:
            return []
        
        execution_state = self.executions[self.current_execution_id]
        return execution_state.history.copy()
    
    async def get_previous_answers(self) -> Dict[str, Any]:
        """Get previous answers in a format suitable for agent context."""
        if not self.current_execution_id:
            return {}
        
        execution_state = self.executions[self.current_execution_id]
        
        previous_answers = {}
        for step_id, result in execution_state.step_results.items():
            qa_result = result.get("qa_result", {})
            if qa_result:
                previous_answers[step_id] = {
                    "answer": qa_result.get("answer", ""),
                    "confidence": qa_result.get("confidence", 0.0),
                    "sources": qa_result.get("sources", []),
                    # ✅ FIX: Include diffusion_metadata so posterior anchors can be consumed
                    "diffusion_metadata": qa_result.get("diffusion_metadata", {})
                }
        
        return previous_answers
    
    async def get_current_state(self) -> Dict[str, Any]:
        """Get the current execution state."""
        if not self.current_execution_id:
            return {"status": "no_active_execution"}
        
        execution_state = self.executions[self.current_execution_id]
        
        return {
            "execution_id": execution_state.execution_id,
            "main_query": execution_state.main_query,
            "disambiguated_query": execution_state.disambiguated_query,
            "query_type": execution_state.query_type,
            "plan_available": execution_state.plan is not None,
            "steps_completed": len(execution_state.completed_steps),
            "total_steps": len(execution_state.plan.get("steps", [])) if execution_state.plan else 0,
            "history_size": len(execution_state.history),
            "start_time": execution_state.start_time.isoformat() if execution_state.start_time else None,
            "last_update": execution_state.last_update.isoformat() if execution_state.last_update else None
        }
    
    async def cleanup_execution(self, execution_id: Optional[str] = None) -> bool:
        """Clean up execution state."""
        target_id = execution_id or self.current_execution_id
        if not target_id:
            return False
        
        try:
            if target_id in self.executions:
                del self.executions[target_id]
                logger.info(f"Cleaned up execution: {target_id}")
            
            if target_id == self.current_execution_id:
                self.current_execution_id = None
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup execution {target_id}: {str(e)}")
            return False
    
    async def clear_all_executions(self) -> int:
        """Clear all execution states."""
        count = len(self.executions)
        self.executions.clear()
        self.current_execution_id = None
        
        logger.info(f"Cleared {count} executions")
        return count
    
    async def get_all_executions(self) -> List[Dict[str, Any]]:
        """Get information about all executions."""
        summaries = []
        for execution_id in self.executions:
            summary = await self.get_execution_summary(execution_id)
            summaries.append(summary)
        
        return summaries

