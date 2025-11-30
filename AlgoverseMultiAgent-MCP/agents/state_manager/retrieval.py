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
    retriever_agent=None,
    current_step_index: Optional[int] = None,
    total_steps: Optional[int] = None
) -> Dict[str, Any]:
    """
    Stabilize reasoning flow then perform entropy-aware retrieval.
    
    This is the central method that:
    1. Updates reasoning flow state
    2. Applies regulators to stabilize query
    3. Checks if early termination is valid (convergence conditions)
    4. Performs entropy-aware retrieval
    
    ✅ FIRST PRINCIPLES: Early termination is a CONVERGENCE condition, not just entropy.
    Requires: low entropy + evidence coverage + plan completion + hierarchical level + answer stability.
    
    Args:
        proposed_query: Raw query from Step Definer
        hop: Current hop number
        previous_answers: Answers from previous steps
        plan_goal: Overall plan goal/question
        retriever_agent: Retriever agent instance
        current_step_index: Current step index (0-based) in the plan
        total_steps: Total number of steps in the plan
        
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
    # ====================================================================
    # DIFFUSION PROCESS: Initial Condition (u(x,0))
    # ====================================================================
    # GranularityRegulator is applied FIRST as the initial condition,
    # setting the correct hierarchical level before retrieval begins.
    # This prevents hierarchical leakage and level-mismatch cascades.
    # Other regulators then apply iteratively to guide convergence.
    # ====================================================================
    stabilized_query, constraints = self.regulator_manager.apply_all(
        proposed_query=proposed_query,
        reasoning_state=flow_snapshot.dict() if flow_snapshot else {},
        previous_answers=previous_answers,
        plan_goal=plan_goal
    )
    
    # 3. Check if early termination is valid (CONVERGENCE condition, not just entropy)
    # ✅ FIRST PRINCIPLES FIX: Early termination requires convergence, not just low entropy
    termination_info = self._check_termination_validity(
        flow_snapshot=flow_snapshot,
        plan_goal=plan_goal,
        hop=hop,
        previous_answers=previous_answers,
        constraints=constraints,
        current_step_index=current_step_index,
        total_steps=total_steps,
        current_query=proposed_query  # ✅ DIFFUSION-AWARE: Pass current query to verify answer satisfies boundary condition
    )
    
    if termination_info.get("can_terminate", False):
        last_answer = list(previous_answers.values())[-1] if previous_answers else {}
        answer_text = last_answer.get("answer", "") if isinstance(last_answer, dict) else str(last_answer)
        
        # ✅ FIX: Don't terminate early if answer is "unknown" or empty
        # "unknown" means we haven't found the answer yet, not that we're confident
        answer_lower = answer_text.lower().strip()
        if answer_lower in ["unknown", "none", "n/a", ""]:
            logger.debug(
                f"Skipping early termination: last answer is '{answer_text}' "
                f"(entropy={flow_snapshot.entropy:.3f}, confidence={flow_snapshot.confidence:.3f})"
            )
        else:
            logger.info(
                f"Early termination at hop {hop}: {termination_info.get('reason', 'convergence conditions met')}"
            )
            return {
                "direct_answer": True,
                "answer": answer_text,
                "confidence": flow_snapshot.confidence,
                "reasoning": termination_info.get("reason", "Convergence conditions met - safe to finalize"),
                "stabilized_query": stabilized_query  # Include stabilized query even in early termination
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
    # ✅ FIX: Explicitly pass k and min_similarity from retriever defaults
    # ✅ FIRST PRINCIPLES: Pass hierarchical context for parent entity extraction
    flow_snapshot_dict = flow_snapshot.dict() if flow_snapshot and hasattr(flow_snapshot, 'dict') else {}
    
    # Extract hierarchical level requirement from GranularityRegulator constraint
    required_domain = None
    required_level = None
    for constraint in constraints:
        constraint_dict = constraint.dict() if hasattr(constraint, 'dict') else (constraint if isinstance(constraint, dict) else {})
        if constraint_dict.get('regulator_name') == 'Granularity':
            params = constraint_dict.get('parameters', {})
            required_domain = params.get('required_domain')
            required_level = params.get('required_level')
            break
    
    # Add hierarchical context to flow snapshot
    if required_domain and required_level:
        flow_snapshot_dict['required_domain'] = required_domain
        flow_snapshot_dict['required_level'] = required_level
    
    retrieval_input = {
        "query": stabilized_query,
        "k": getattr(retriever_agent, 'top_k', 15),  # Use retriever's top_k, default to 15 for scattered docs
        "min_similarity": getattr(retriever_agent, 'min_similarity', 0.2),  # Use retriever's min_similarity
        "regulator_constraints": [c.dict() if hasattr(c, 'dict') else c for c in constraints],
        "flow_snapshot": flow_snapshot_dict,
        "entropy_penalty": flow_snapshot.entropy if flow_snapshot else 0.0,
        "diffusion_penalty": flow_snapshot.diffusion_coefficient if flow_snapshot else 0.0
    }
    
    # Call retriever with constraints
    result = await retriever_agent.process(retrieval_input)
    
    logger.debug(
        f"Entropy-aware retrieval: query='{stabilized_query}', "
        f"k={retrieval_input.get('k')}, min_similarity={retrieval_input.get('min_similarity')}, "
        f"H(t)={(flow_snapshot.entropy if flow_snapshot else 0.0):.3f}, "
        f"D(t)={(flow_snapshot.diffusion_coefficient if flow_snapshot else 0.0):.3f}, "
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
    plan_goal: Optional[str] = None,
    hop: int = 1,
    previous_answers: Optional[Dict[str, Any]] = None,
    constraints: Optional[List] = None,
    current_step_index: Optional[int] = None,
    total_steps: Optional[int] = None,
    current_query: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check if reasoning can terminate early based on CONVERGENCE conditions.
    
    ✅ FIRST PRINCIPLES: Stopping is a convergence condition, not just an entropy condition.
    
    Convergence requires ALL of:
    1. Low entropy + evidence coverage high + plan steps completed + 
       granularity target satisfied + retrieval no longer surfacing new information
    2. Answer stable across cycles (last 2-3 answers are consistent)
    3. No new retrieval improves the evidence score (implicitly checked via answer stability)
    4. Required hierarchical level is met
    5. ✅ DIFFUSION-AWARE: Answer satisfies the query's boundary condition (not from input space)
    
    Args:
        flow_snapshot: Current flow snapshot
        plan_goal: Plan goal to check against
        hop: Current hop number
        previous_answers: Answers from previous steps
        constraints: Regulator constraints (for hierarchical level check)
        current_step_index: Current step index (0-based) in the plan
        total_steps: Total number of steps in the plan
        current_query: Current query being processed (boundary condition to verify against)
        
    Returns:
        Dict with "can_terminate" (bool) and "reason" (str) explaining why/why not
    """
    if not flow_snapshot or not self.entropy_tracker:
        return {"can_terminate": False, "reason": "Missing flow snapshot or entropy tracker"}
    
    entropy_state = self.entropy_tracker.get_current_state()
    if not entropy_state:
        return {"can_terminate": False, "reason": "No entropy state available"}
    
    # ====================================================================
    # CONVERGENCE CONDITION 1: Plan steps must be completed (or on last step)
    # ====================================================================
    # ✅ CRITICAL: Don't terminate early if there are more steps to execute
    if current_step_index is not None and total_steps is not None:
        # Only allow termination if we're on the last step (or all steps are effectively done)
        if current_step_index < total_steps - 1:
            return {
                "can_terminate": False,
                "reason": f"Not on last step: step {current_step_index + 1}/{total_steps} - must complete all steps"
            }
    
    # ====================================================================
    # CONVERGENCE CONDITION 2: Low entropy + high confidence + low drift
    # ====================================================================
    # Must have low entropy (high certainty)
    if entropy_state.entropy >= 0.5:
        return {"can_terminate": False, "reason": f"Entropy too high: {entropy_state.entropy:.3f} >= 0.5"}
    
    # Must have high confidence
    if entropy_state.confidence < 0.8:
        return {"can_terminate": False, "reason": f"Confidence too low: {entropy_state.confidence:.3f} < 0.8"}
    
    # Must have low drift (stable beliefs)
    if entropy_state.drift_from_previous >= 0.3:
        return {"can_terminate": False, "reason": f"Drift too high: {entropy_state.drift_from_previous:.3f} >= 0.3"}
    
    # ====================================================================
    # CONVERGENCE CONDITION 3: Answer stable across cycles
    # ====================================================================
    if previous_answers and len(previous_answers) >= 2:
        # Check if last 2-3 answers are consistent
        answer_values = []
        for answer_data in list(previous_answers.values())[-3:]:
            if isinstance(answer_data, dict):
                answer_text = answer_data.get("answer", "").lower().strip()
            else:
                answer_text = str(answer_data).lower().strip()
            if answer_text and answer_text not in ["unknown", "none", "n/a", ""]:
                answer_values.append(answer_text)
        
        # Answers must be consistent (all same or very similar)
        if len(answer_values) >= 2:
            unique_answers = set(answer_values)
            if len(unique_answers) > 1:
                return {
                    "can_terminate": False,
                    "reason": f"Answer not stable across cycles - answers differ: {list(unique_answers)[:3]}"
                }
    
    # ====================================================================
    # CONVERGENCE CONDITION 4: Required hierarchical level is met
    # ====================================================================
    if constraints:
        try:
            from agents.regulators.granularity_regulator import GranularityRegulator
        except ImportError:
            from ..regulators.granularity_regulator import GranularityRegulator
        granularity_reg = GranularityRegulator()
        
        # Extract required hierarchical level from GranularityRegulator constraint
        required_domain = None
        required_level = None
        for constraint in constraints:
            constraint_dict = constraint.dict() if hasattr(constraint, 'dict') else (constraint if isinstance(constraint, dict) else {})
            if constraint_dict.get('regulator_name') == 'Granularity':
                params = constraint_dict.get('parameters', {})
                required_domain = params.get('required_domain')
                required_level = params.get('required_level')
                break
        
        # If hierarchical level is required, check if last answer meets it
        if required_domain and required_level and previous_answers:
            last_answer_data = list(previous_answers.values())[-1]
            last_answer = last_answer_data.get("answer", "") if isinstance(last_answer_data, dict) else str(last_answer_data)
            
            if last_answer and last_answer.lower() not in ["unknown", "none", "n/a", ""]:
                # Check if answer is at required hierarchical level
                answer_domain, answer_level, _ = granularity_reg.classify_entity_level(last_answer)
                if answer_domain == required_domain and answer_level:
                    answer_level_num = granularity_reg.get_level_number(answer_domain, answer_level)
                    required_level_num = granularity_reg.get_level_number(required_domain, required_level)
                    
                    # Answer must be at required level (not higher or lower)
                    if answer_level_num != required_level_num:
                        return {
                            "can_terminate": False,
                            "reason": f"Hierarchical level not met: answer is {answer_level} (level {answer_level_num}), "
                                     f"required is {required_level} (level {required_level_num})"
                        }
    
    # ====================================================================
    # CONVERGENCE CONDITION 5: Evidence coverage check
    # ====================================================================
    # Check if we have sufficient evidence (not just low entropy)
    if previous_answers:
        # Count non-"unknown" answers as evidence
        valid_answers = sum(
            1 for answer_data in previous_answers.values()
            if isinstance(answer_data, dict) and answer_data.get("answer", "").lower().strip() not in ["unknown", "none", "n/a", ""]
            or (not isinstance(answer_data, dict) and str(answer_data).lower().strip() not in ["unknown", "none", "n/a", ""])
        )
        
        # Need at least some valid answers (evidence coverage)
        if valid_answers == 0:
            return {"can_terminate": False, "reason": "No valid answers found - insufficient evidence coverage"}
    
    # ====================================================================
    # CONVERGENCE CONDITION 6: Answer satisfies query boundary condition (DIFFUSION-AWARE)
    # ====================================================================
    # ✅ DIFFUSION PERSPECTIVE: The query defines a boundary condition u(x, t=0) = f(x).
    # The answer must be in the solution space that satisfies this boundary condition,
    # not in the input space (entities mentioned in the query).
    # 
    # In diffusion terms: if the query asks "What is X's Y?", then:
    # - X is in the input/initial condition space
    # - Y is the property/relationship being queried
    # - The answer must be in the solution space (a value for Y), not X itself
    #
    # This prevents convergence on answers that are semantically the input entity
    # rather than a valid solution to the query's boundary condition.
    if current_query and previous_answers:
        last_answer_data = list(previous_answers.values())[-1]
        last_answer = last_answer_data.get("answer", "") if isinstance(last_answer_data, dict) else str(last_answer_data)
        
        if last_answer and last_answer.lower() not in ["unknown", "none", "n/a", ""]:
            # Check if answer satisfies the query's boundary condition
            relevance_result = self._check_answer_satisfies_boundary_condition(last_answer, current_query)
            if not relevance_result["satisfies"]:
                return {
                    "can_terminate": False,
                    "reason": (
                        f"Answer '{last_answer}' does not satisfy query boundary condition: "
                        f"{relevance_result.get('reason', 'answer is in input space, not solution space')}"
                    )
                }
    
    # All convergence conditions met
    return {
        "can_terminate": True,
        "reason": (
            f"Convergence conditions met: entropy={entropy_state.entropy:.3f} (low), "
            f"confidence={entropy_state.confidence:.3f} (high), "
            f"drift={entropy_state.drift_from_previous:.3f} (low), "
            f"answer stable, hierarchical level satisfied, plan steps completed, "
            f"answer satisfies query boundary condition"
        )
    }

def _check_answer_satisfies_boundary_condition(self, answer: str, query: str) -> Dict[str, Any]:
    """
    Check if an answer satisfies the query's boundary condition from a diffusion perspective.
    
    ✅ DIFFUSION PERSPECTIVE: In a diffusion process, the boundary condition u(x, 0) = f(x)
    defines what we're solving for. The answer must be in the solution space that satisfies
    this boundary condition, not in the input/initial condition space.
    
    Semantic Structure Analysis:
    - Query defines: subject (input entity) + predicate (what we're asking) + expected answer type
    - Answer must be semantically distinct from the subject/input entities
    - Answer must be in the solution space (the type of thing being asked for)
    
    This is generalizable because it analyzes semantic structure, not specific patterns.
    
    Args:
        answer: The answer to check
        query: The current query (boundary condition)
        
    Returns:
        Dict with "satisfies" (bool) and "reason" (str) explaining why/why not
    """
    if not answer or not query:
        return {"satisfies": True, "reason": "Cannot verify - missing answer or query"}
    
    answer_lower = answer.lower().strip()
    query_lower = query.lower().strip()
    
    # Extract semantic structure: subject entities (input space) and query type
    import re
    
    # Extract all potential subject entities from the query
    # These are entities in the "input space" - the answer should NOT be one of these
    subject_entities = self._extract_subject_entities(query_lower)
    
    # Check if answer matches any subject entity (input space violation)
    for subject_entity in subject_entities:
        # Normalize for comparison (handle variations)
        subject_normalized = subject_entity.lower().strip()
        answer_normalized = answer_lower.strip()
        
        # Exact match or answer is contained in subject (e.g., "Steve" matches "Steve Hillage")
        if answer_normalized == subject_normalized:
            return {
                "satisfies": False,
                "reason": (
                    f"Answer '{answer}' is in input space (matches subject entity '{subject_entity}') "
                    f"rather than solution space. Query asks for information about '{subject_entity}', "
                    f"not '{subject_entity}' itself."
                )
            }
        
        # Check if answer is a substring of subject (e.g., "Steve" in "Steve Hillage")
        # This catches cases where answer is just part of the subject entity
        if len(answer_normalized) > 2 and answer_normalized in subject_normalized:
            # But allow if answer is longer/more specific (e.g., "Steve Hillage" is valid answer to "Who is Steve?")
            if len(answer_normalized) < len(subject_normalized) * 0.7:  # Answer is significantly shorter
                return {
                    "satisfies": False,
                    "reason": (
                        f"Answer '{answer}' appears to be part of subject entity '{subject_entity}' "
                        f"(input space), not a solution to the query."
                    )
                }
    
    # Check for relationship/attribute queries where answer should be semantically distinct
    # Pattern: "X's Y" or "Y of X" where X is subject, Y is what we're asking for
    possessive_pattern = r"(\w+(?:\s+\w+)*)'s\s+(\w+)"
    match = re.search(possessive_pattern, query_lower)
    if match:
        subject_entity = match.group(1).strip()
        attribute_or_relation = match.group(2).strip()
        
        # If answer matches subject, it's wrong (input space, not solution space)
        if answer_lower == subject_entity.lower():
            return {
                "satisfies": False,
                "reason": (
                    f"Answer '{answer}' is the subject entity '{subject_entity}' (input space). "
                    f"Query asks for '{attribute_or_relation}' of '{subject_entity}', "
                    f"so answer must be in solution space (a value for '{attribute_or_relation}'), "
                    f"not the subject itself."
                )
            }
    
    # Check for "who is X's Y?" type queries
    who_pattern = r"who\s+is\s+(\w+(?:\s+\w+)*)'s\s+(\w+)"
    match = re.search(who_pattern, query_lower)
    if match:
        subject_entity = match.group(1).strip()
        relation = match.group(2).strip()
        
        if answer_lower == subject_entity.lower():
            return {
                "satisfies": False,
                "reason": (
                    f"Answer '{answer}' is the subject '{subject_entity}' (input space). "
                    f"Query asks 'who is {subject_entity}'s {relation}?', "
                    f"so answer must be a different entity (the {relation}), not the subject."
                )
            }
    
    # If none of the violations detected, assume answer satisfies boundary condition
    return {
        "satisfies": True,
        "reason": "Answer is semantically distinct from query subject entities and appears to be in solution space"
    }

def _extract_subject_entities(self, query: str) -> List[str]:
    """
    Extract subject entities (input space) from a query using semantic analysis.
    
    This is generalizable because it uses linguistic patterns to identify
    entities that are part of the query's input/initial condition, not the solution.
    
    Args:
        query: The query to analyze
        
    Returns:
        List of subject entity strings (normalized)
    """
    import re
    entities = []
    
    # Pattern 1: Possessive constructions "X's Y" - X is subject
    possessive_pattern = r"(\w+(?:\s+\w+)*)'s\s+\w+"
    matches = re.findall(possessive_pattern, query)
    entities.extend([m.strip() for m in matches])
    
    # Pattern 2: "Y of X" constructions - X is subject
    of_pattern = r"\w+\s+of\s+(\w+(?:\s+\w+)*)"
    matches = re.findall(of_pattern, query)
    entities.extend([m.strip() for m in matches])
    
    # Pattern 3: "who/what/which X" - X might be subject if followed by verb
    # But we're more conservative here - only extract if it's clearly a subject
    
    # Pattern 4: Named entities (capitalized sequences) that appear early in query
    # These are often subject entities
    capitalized_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
    matches = re.findall(capitalized_pattern, query)
    # Only take first few as they're likely subjects
    entities.extend([m.strip() for m in matches[:2]])
    
    # Remove duplicates and normalize
    unique_entities = []
    seen = set()
    for entity in entities:
        entity_lower = entity.lower().strip()
        if entity_lower and entity_lower not in seen and len(entity_lower) > 2:
            unique_entities.append(entity)
            seen.add(entity_lower)
    
    return unique_entities

