# agents/state_manager/flow_update.py
from typing import Dict, Any, List, Optional
import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

def _update_flow_state(
    self,
    hop: int,
    previous_answers: Dict[str, Any],
    plan_goal: Optional[str] = None
) -> Optional[Any]:
    """
    Update reasoning flow state for current hop.
    
    Args:
        hop: Current hop number
        previous_answers: Answers from previous steps
        plan_goal: Overall plan goal
        
    Returns:
        FlowSnapshot or None
    """
    if not self.reasoning_flow:
        return None
    
    # Extract beliefs from previous answers
    beliefs = _extract_beliefs(self, previous_answers)
    
    # FEEDBACK LOOP: Apply anchor corrections (potential wells)
    # This closes the feedback loop - anchors correct beliefs
    if self.reasoning_flow:
        beliefs = self.reasoning_flow.apply_anchor_correction(beliefs)
    
    # Extract entity anchors
    entity_anchors = _extract_entity_anchors(self, previous_answers)
    
    # Extract evidence terms
    evidence_terms = _extract_evidence_terms(self, previous_answers)
    
    # Detect relation direction
    relation_direction = _detect_relation_direction(self, previous_answers, plan_goal)
    
    # Calculate plan alignment
    plan_alignment = _calculate_plan_alignment(self, previous_answers, plan_goal)
    
    # Get confidence from previous answers
    confidence = _calculate_confidence(self, previous_answers)
    
    # Add state to reasoning flow (this integrates with entropy tracker)
    self.reasoning_flow.add_state(
        hop=hop,
        beliefs=beliefs,  # Already corrected by anchors
        entity_anchors=entity_anchors,
        relation_direction=relation_direction,
        evidence_terms=evidence_terms,
        plan_alignment=plan_alignment,
        confidence=confidence
    )
    
    # Add bucket anchors for key entities
    for entity, anchor_data in entity_anchors.items():
        self.reasoning_flow.add_bucket_anchor(
            anchor_type="entity",
            value=entity,
            hop=hop,
            strength=anchor_data.get("strength", 0.8),
            context=anchor_data.get("context", {})
        )
    
    # Get flow snapshot (unified state for regulators)
    return self.reasoning_flow.get_flow_snapshot()

def _extract_beliefs(self, previous_answers: Dict[str, Any]) -> Dict[str, float]:
    """Extract belief distribution from previous answers."""
    beliefs = {}
    
    for step_id, answer_data in previous_answers.items():
        if isinstance(answer_data, dict):
            answer = answer_data.get("answer", "")
            confidence = answer_data.get("confidence", 0.5)
        else:
            answer = str(answer_data)
            confidence = 0.5
        
        # Treat each unique answer as a belief, weighted by confidence
        if answer:
            beliefs[answer] = beliefs.get(answer, 0.0) + confidence
    
    # Normalize to probability distribution
    total = sum(beliefs.values())
    if total > 0:
        beliefs = {k: v / total for k, v in beliefs.items()}
    
    return beliefs

def _extract_entity_anchors(self, previous_answers: Dict[str, Any]) -> Dict[str, Any]:
    """Extract entity anchors from previous answers."""
    entity_anchors = {}
    
    for step_id, answer_data in previous_answers.items():
        if isinstance(answer_data, dict):
            answer = answer_data.get("answer", "")
            confidence = answer_data.get("confidence", 0.5)
        else:
            answer = str(answer_data)
            confidence = 0.5
        
        # Simple entity extraction: capitalized words, proper nouns
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer)
        
        for entity in words:
            # Filter out common words
            if entity.lower() not in ["the", "and", "or", "for", "with", "from", "this", "that"]:
                if entity not in entity_anchors:
                    entity_anchors[entity] = {
                        "strength": confidence,
                        "hop": step_id,
                        "context": {"source": step_id, "answer": answer}
                    }
                else:
                    # Strengthen if seen multiple times
                    entity_anchors[entity]["strength"] = min(1.0, entity_anchors[entity]["strength"] + 0.1)
    
    return entity_anchors

def _extract_evidence_terms(self, previous_answers: Dict[str, Any]) -> List[str]:
    """Extract evidence terms from previous answers."""
    all_terms = []
    
    for step_id, answer_data in previous_answers.items():
        if isinstance(answer_data, dict):
            answer = answer_data.get("answer", "")
        else:
            answer = str(answer_data)
        
        # Tokenize and filter
        tokens = re.findall(r'\b\w+\b', answer.lower())
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were"}
        filtered = [t for t in tokens if t not in stop_words and len(t) > 2]
        all_terms.extend(filtered)
    
    # Return top terms by frequency
    term_counts = Counter(all_terms)
    top_terms = [term for term, count in term_counts.most_common(10)]
    
    return top_terms

def _detect_relation_direction(
    self,
    previous_answers: Dict[str, Any],
    plan_goal: Optional[str] = None
) -> Optional[str]:
    """Detect relational direction from context."""
    # Check plan goal for relation keywords
    if plan_goal:
        plan_lower = plan_goal.lower()
        if any(kw in plan_lower for kw in ["contains", "includes", "has", "owns"]):
            return "contains"
        elif any(kw in plan_lower for kw in ["in", "within", "located in", "part of"]):
            return "contained_in"
        elif any(kw in plan_lower for kw in ["hierarchical", "administrative", "territorial"]):
            return "hierarchical"
    
    # Check previous answers
    for answer_data in previous_answers.values():
        if isinstance(answer_data, dict):
            answer = answer_data.get("answer", "")
        else:
            answer = str(answer_data)
        
        answer_lower = answer.lower()
        if any(kw in answer_lower for kw in ["contains", "includes"]):
            return "contains"
        elif any(kw in answer_lower for kw in ["in", "within", "located"]):
            return "contained_in"
    
    return None

def _calculate_plan_alignment(
    self,
    previous_answers: Dict[str, Any],
    plan_goal: Optional[str]
) -> float:
    """Calculate alignment with plan goal."""
    if not plan_goal:
        return 0.5
    
    # Simple alignment: check if answers contain plan keywords
    plan_keywords = set(re.findall(r'\b\w+\b', plan_goal.lower()))
    plan_keywords = {kw for kw in plan_keywords if len(kw) > 3}  # Filter short words
    
    if not plan_keywords:
        return 0.5
    
    matching_count = 0
    total_answers = len(previous_answers)
    
    for answer_data in previous_answers.values():
        if isinstance(answer_data, dict):
            answer = answer_data.get("answer", "")
        else:
            answer = str(answer_data)
        
        answer_lower = answer.lower()
        matches = sum(1 for kw in plan_keywords if kw in answer_lower)
        if matches > 0:
            matching_count += 1
    
    # Alignment score: proportion of answers that match plan keywords
    alignment = matching_count / total_answers if total_answers > 0 else 0.5
    
    return alignment

def _calculate_confidence(self, previous_answers: Dict[str, Any]) -> float:
    """Calculate overall confidence from previous answers."""
    if not previous_answers:
        return 0.5
    
    confidences = []
    for answer_data in previous_answers.values():
        if isinstance(answer_data, dict):
            confidence = answer_data.get("confidence", 0.5)
        else:
            confidence = 0.5
        confidences.append(confidence)
    
    return sum(confidences) / len(confidences) if confidences else 0.5

