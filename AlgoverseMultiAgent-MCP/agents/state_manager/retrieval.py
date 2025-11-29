# agents/state_manager/retrieval.py
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

async def stabilize_and_retrieve(
    self,
    proposed_query: str,
    hop: int,
    previous_answers: Dict[str, Any],
    plan_goal: Optional[str] = None,
    retriever_agent=None
) -> Dict[str, Any]:
    """
    Stabilize reasoning flow then perform entropy-aware retrieval.
    
    This is the central method that:
    1. Updates reasoning flow state
    2. Applies regulators to stabilize query
    3. Checks if early termination is valid
    4. Performs entropy-aware retrieval
    
    Args:
        proposed_query: Raw query from Step Definer
        hop: Current hop number
        previous_answers: Answers from previous steps
        plan_goal: Overall plan goal/question
        retriever_agent: Retriever agent instance
        
    Returns:
        Dictionary with documents or early termination answer
    """
    if not self.reasoning_flow or not self.regulator_manager:
        # Fallback to direct retrieval if components not available
        logger.warning("Diffusion-aware components not available, using direct retrieval")
        if retriever_agent:
            result = await retriever_agent.process({"query": proposed_query})
            return {
                "documents": result.metadata.get("documents", []),
                "stabilized_query": proposed_query,
                "constraints": []
            }
        return {"error": "retriever_agent required"}
    
    # 1. Update reasoning flow state
    flow_snapshot = self._update_flow_state(hop, previous_answers, plan_goal)
    
    # 2. Apply regulators to stabilize query
    stabilized_query, constraints = self.regulator_manager.apply_all(
        proposed_query=proposed_query,
        reasoning_state=flow_snapshot.dict() if flow_snapshot else {},
        previous_answers=previous_answers,
        plan_goal=plan_goal
    )
    
    # 3. Check if early termination is valid
    if flow_snapshot and self._check_termination_validity(flow_snapshot, plan_goal):
        logger.info(f"Early termination at hop {hop}: entropy low, confidence high")
        last_answer = list(previous_answers.values())[-1] if previous_answers else {}
        return {
            "direct_answer": True,
            "answer": last_answer.get("answer", "") if isinstance(last_answer, dict) else str(last_answer),
            "confidence": flow_snapshot.confidence,
            "reasoning": "Entropy low, confidence high - early termination valid"
        }
    
    # 4. Entropy-aware retrieval
    if not retriever_agent:
        raise ValueError("retriever_agent required for entropy-aware retrieval")
    
    return await self._entropy_aware_retrieve(
        stabilized_query=stabilized_query,
        flow_snapshot=flow_snapshot,
        constraints=constraints,
        retriever_agent=retriever_agent
    )

async def _entropy_aware_retrieve(
    self,
    stabilized_query: str,
    flow_snapshot: Optional[Any],
    constraints: List,
    retriever_agent
) -> Dict[str, Any]:
    """
    Perform entropy-aware retrieval with regulator constraints.
    
    Args:
        stabilized_query: Query after regulator stabilization
        flow_snapshot: Unified flow snapshot
        constraints: List of regulator constraints
        retriever_agent: Retriever agent instance
        
    Returns:
        Dictionary with retrieved documents and metadata
    """
    # Build retrieval input with constraints
    retrieval_input = {
        "query": stabilized_query,
        "regulator_constraints": [c.dict() if hasattr(c, 'dict') else c for c in constraints],
        "flow_snapshot": flow_snapshot.dict() if flow_snapshot and hasattr(flow_snapshot, 'dict') else {},
        "entropy_penalty": flow_snapshot.entropy if flow_snapshot else 0.0,
        "diffusion_penalty": flow_snapshot.diffusion_coefficient if flow_snapshot else 0.0
    }
    
    # Call retriever with constraints
    result = await retriever_agent.process(retrieval_input)
    
    logger.debug(
        f"Entropy-aware retrieval: query='{stabilized_query}', "
        f"H(t)={flow_snapshot.entropy:.3f if flow_snapshot else 0.0}, "
        f"D(t)={flow_snapshot.diffusion_coefficient:.3f if flow_snapshot else 0.0}, "
        f"documents={len(result.metadata.get('documents', []))}"
    )
    
    return {
        "documents": result.metadata.get("documents", []),
        "stabilized_query": stabilized_query,
        "constraints": constraints,
        "flow_snapshot": flow_snapshot.dict() if flow_snapshot and hasattr(flow_snapshot, 'dict') else None
    }

def _check_termination_validity(
    self,
    flow_snapshot: Any,
    plan_goal: Optional[str] = None
) -> bool:
    """
    Check if reasoning can terminate early.
    
    Args:
        flow_snapshot: Current flow snapshot
        plan_goal: Plan goal to check against
        
    Returns:
        True if can terminate early, False otherwise
    """
    if not flow_snapshot or not self.entropy_tracker:
        return False
    
    # Use entropy tracker's termination check
    entropy_state = self.entropy_tracker.get_current_state()
    if not entropy_state:
        return False
    
    return self.entropy_tracker.should_terminate_early(
        current_state=entropy_state,
        plan_goal=plan_goal
    )

