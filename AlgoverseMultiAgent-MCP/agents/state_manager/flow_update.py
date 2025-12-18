# agents/state_manager/flow_update.py
from typing import Dict, Any, List, Optional
import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

# Import for quality checks
try:
    from ..regulators.evidence_regulator import EvidenceRegulator
except ImportError:
    EvidenceRegulator = None

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

    # ✅ Get current execution state for global anchor bookkeeping
    execution_state = None
    current_execution_id = getattr(self, "current_execution_id", None)
    executions = getattr(self, "executions", None)
    if current_execution_id and isinstance(executions, dict):
        execution_state = executions.get(current_execution_id)

    # ✅ Ensure anchor containers exist on execution_state (if available)
    if execution_state is not None:
        if not hasattr(execution_state, "validated_anchors") or execution_state.validated_anchors is None:
            execution_state.validated_anchors = {}
        if not hasattr(execution_state, "revoked_anchors") or execution_state.revoked_anchors is None:
            execution_state.revoked_anchors = []

        # ✅ PERSISTENCE RULE: carry forward all validated anchors unchanged by default
        # This enforces: validated_anchors[t+1] ⊇ validated_anchors[t] \ revoked_anchors[t+1]
        for anchor_name in execution_state.validated_anchors.keys():
            logger.debug(
                f"[Anchors] Carrying forward validated anchor '{anchor_name}' unchanged into hop {hop}"
            )
    
    # Extract beliefs from previous answers
    beliefs = _extract_beliefs(self, previous_answers)
    
    # FEEDBACK LOOP: Apply anchor corrections (potential wells)
    # This closes the feedback loop - anchors correct beliefs
    if self.reasoning_flow:
        beliefs = self.reasoning_flow.apply_anchor_correction(beliefs)
    
    # Extract entity anchors
    entity_anchors = _extract_entity_anchors(self, previous_answers)

    # ✅ REVOCATION RULE (explicit only)
    # Look for explicit revocation signals coming from QA/regulators via diffusion_metadata.
    # We DO NOT infer contradictions ourselves here (no new heuristics).
    if execution_state is not None:
        for step_id, answer_data in previous_answers.items():
            if not isinstance(answer_data, dict):
                continue

            diffusion_metadata = answer_data.get("diffusion_metadata", {}) or {}
            explicit_revocations = diffusion_metadata.get("revoked_anchors", [])

            # Allow either a list[str] or list[dict] with {anchor, reason, evidence} etc.
            for rev in explicit_revocations:
                if isinstance(rev, str):
                    anchor_name = rev
                    reason = "explicit_revocation"
                    evidence = answer_data  # full answer context as evidence
                elif isinstance(rev, dict):
                    anchor_name = rev.get("anchor")
                    reason = rev.get("reason", "explicit_revocation")
                    evidence = rev.get("evidence", answer_data)
                else:
                    continue

                if not anchor_name:
                    continue

                if anchor_name in execution_state.validated_anchors:
                    # Record revocation event
                    rev_record = {
                        "anchor": anchor_name,
                        "hop": hop,
                        "reason": reason,
                        "evidence": evidence,
                    }
                    execution_state.revoked_anchors.append(rev_record)

                    # Remove from validated_anchors (explicit only)
                    del execution_state.validated_anchors[anchor_name]

                    logger.info(
                        f"[Anchors] Explicitly revoked validated anchor '{anchor_name}' "
                        f"at hop {hop}: reason='{reason}'"
                    )
    
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
    
    # ✅ FIRST PRINCIPLES FIX: Quality-gated anchor creation
    # Only add anchors for high-quality evidence that passes quality checks
    # This aligns anchor creation with EvidenceRegulator's quality standards
    current_state = self.reasoning_flow.get_current_state()
    entropy = current_state.entropy if current_state else 0.0
    confidence = current_state.confidence if current_state else 0.5
    
    for entity, anchor_data in entity_anchors.items():
        # ✅ FIX: Use anchor's own confidence from anchor_data, not flow state aggregate
        # The anchor's confidence (from QA's posterior) is stored in anchor_data["strength"]
        anchor_confidence = anchor_data.get("strength", confidence)  # Use anchor's confidence, fallback to flow state
        
        # Quality check: Only create anchors for high-quality evidence
        if self._should_create_anchor(
            entity=entity,
            anchor_data=anchor_data,
            entropy=entropy,
            confidence=anchor_confidence,  # ✅ FIX: Use anchor's own confidence, not flow state
            plan_goal=plan_goal
        ):
            # Entropy-aware strength: High entropy = weaker, low entropy = stronger
            base_strength = anchor_data.get("strength", 0.8)
            entropy_adjusted_strength = self._calculate_entropy_aware_strength(
                base_strength=base_strength,
                entropy=entropy,
                confidence=anchor_confidence  # ✅ FIX: Also use anchor confidence here for consistency
            )
            
            self.reasoning_flow.add_bucket_anchor(
                anchor_type="entity",
                value=entity,
                hop=hop,
                strength=entropy_adjusted_strength,
                context=anchor_data.get("context", {})
            )
            logger.debug(
                f"✅ Quality-gated anchor created: {entity} (strength={entropy_adjusted_strength:.3f}, "
                f"entropy={entropy:.3f}, anchor_confidence={anchor_confidence:.3f})"  # ✅ FIX: Log anchor confidence
            )

            # ✅ PROMOTION RULE: promote created anchors to validated_anchors
            if execution_state is not None:
                if entity not in execution_state.validated_anchors:
                    execution_state.validated_anchors[entity] = {
                        "entity": entity,
                        "strength": entropy_adjusted_strength,
                        "validated_hop": hop,
                        "context": anchor_data.get("context", {}),
                    }
                    logger.info(
                        f"[Anchors] Promoted validated anchor '{entity}' at hop {hop} "
                        f"(strength={entropy_adjusted_strength:.3f})"
                    )
                else:
                    # Already validated: keep it; we just log that it remains validated.
                    logger.debug(
                        f"[Anchors] Anchor '{entity}' already validated; "
                        f"keeping existing validated state at hop {hop}"
                    )
        else:
            logger.debug(
                f"⏭️ Skipped anchor creation for '{entity}': failed quality checks "
                f"(entropy={entropy:.3f}, anchor_confidence={anchor_confidence:.3f}, "  # ✅ FIX: Log anchor confidence
                f"flow_state_confidence={confidence:.3f})"  # Also log flow state for debugging
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



# In state_manager/flow_update.py, modify _extract_entity_anchors():

def _extract_entity_anchors(self, previous_answers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract entity anchors from previous answers.
    
    ✅ POSTERIOR INGESTION FIX: Consumes QA's structured new_anchors from diffusion_metadata
    first (Bayesian posterior), then falls back to raw text extraction only if QA provided
    no structured anchors OR if all structured anchors were filtered out.
    """
    entity_anchors = {}
    
    # Initialize EvidenceRegulator for quality checks
    evidence_reg = EvidenceRegulator() if EvidenceRegulator else None
    
    for step_id, answer_data in previous_answers.items():
        if isinstance(answer_data, dict):
            answer = answer_data.get("answer", "")
            confidence = answer_data.get("confidence", 0.5)
            query = answer_data.get("query", "")
            # ✅ FIX: Get diffusion_metadata containing QA's posterior anchors
            diffusion_metadata = answer_data.get("diffusion_metadata", {})
        else:
            answer = str(answer_data)
            confidence = 0.5
            query = ""
            diffusion_metadata = {}
        
        # ✅ POSTERIOR INGESTION: Consume QA's structured new_anchors first
        new_anchors = diffusion_metadata.get("new_anchors", [])
        anchors_added_from_posterior = False  # ✅ FIX: Track if we actually added any anchors
        
        if new_anchors and isinstance(new_anchors, list):
            # QA provided structured posterior anchors - consume these directly
            for anchor_obj in new_anchors:
                if isinstance(anchor_obj, dict):
                    entity = anchor_obj.get("entity", "")
                    anchor_confidence = anchor_obj.get("confidence", confidence)
                    anchor_hop = anchor_obj.get("hop", step_id)
                    anchor_type = anchor_obj.get("type", "extracted_answer")
                    anchor_source = anchor_obj.get("source", "qa_compression")
                    
                    if entity and entity.lower() not in ["the", "a", "an", "this", "that", "yes", "no", "unknown"]:
                        # ✅ Filter invalid evidence terms (same quality check as raw extraction)
                        if evidence_reg and evidence_reg._is_invalid_evidence_term(entity):
                            logger.debug(
                                f"Entity anchor extraction: Skipping invalid evidence term '{entity}' "
                                f"from QA posterior anchors (step {step_id})"
                            )
                            continue
                        
                        # Use QA's posterior anchor directly
                        if entity not in entity_anchors:
                            entity_anchors[entity] = {
                                "strength": anchor_confidence,
                                "hop": anchor_hop,
                                "context": {
                                    "source": step_id,
                                    "answer": answer,
                                    "query": query,
                                    "anchor_type": anchor_type,
                                    "anchor_source": anchor_source,
                                    "from_posterior": True  # Mark as posterior anchor
                                }
                            }
                            anchors_added_from_posterior = True  # ✅ FIX: Mark that we added an anchor
                            logger.debug(
                                f"✅ Consumed posterior anchor '{entity}' from QA diffusion_metadata "
                                f"(confidence={anchor_confidence:.3f}, hop={anchor_hop})"
                            )
                        else:
                            # Strengthen if confidence is high (same logic as raw extraction)
                            if anchor_confidence > 0.7:
                                entity_anchors[entity]["strength"] = min(
                                    1.0, entity_anchors[entity]["strength"] + 0.1
                                )
                                anchors_added_from_posterior = True  # ✅ FIX: Also mark on strengthening
                                logger.debug(
                                    f"✅ Strengthened posterior anchor '{entity}' "
                                    f"(confidence={anchor_confidence:.3f})"
                                )
            
            # ✅ FIX: Only skip raw text extraction if we actually added anchors from posterior
            # If all posterior anchors were filtered out, fall through to fallback
            if anchors_added_from_posterior:
                logger.debug(
                    f"Posterior anchors consumed for step {step_id}, skipping raw text extraction"
                )
                continue
            else:
                logger.debug(
                    f"QA provided new_anchors for step {step_id} but all were filtered out, "
                    f"falling back to raw text extraction"
                )
        
        # ✅ FALLBACK: Raw text extraction when:
        # 1. QA provided no structured anchors, OR
        # 2. QA provided structured anchors but all were filtered out
        if evidence_reg and evidence_reg._is_invalid_evidence_term(answer):
            logger.debug(
                f"Entity anchor extraction: Skipping invalid evidence term '{answer}' from step {step_id}"
            )
            continue
        
        # Simple entity extraction: capitalized words, proper nouns
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer)
        
        for entity in words:
            # Filter out common words
            if entity.lower() not in ["the", "and", "or", "for", "with", "from", "this", "that"]:
                if entity not in entity_anchors:
                    entity_anchors[entity] = {
                        "strength": confidence,
                        "hop": step_id,
                        "context": {
                            "source": step_id,
                            "answer": answer,
                            "query": query,
                            "from_posterior": False  # Mark as raw extraction fallback
                        }
                    }
                    logger.debug(
                        f"⚠️ Fallback: Extracted anchor '{entity}' from raw text "
                        f"(QA provided no valid structured anchors for step {step_id})"
                    )
                else:
                    # ✅ FIX: Only strengthen if confidence is high (quality check)
                    # Don't strengthen on repetition alone - repetition of wrong answer is still wrong
                    if confidence > 0.7:
                        entity_anchors[entity]["strength"] = min(
                            1.0, entity_anchors[entity]["strength"] + 0.1
                        )
    
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

def _should_create_anchor(
    self,
    entity: str,
    anchor_data: Dict[str, Any],
    entropy: float,
    confidence: float,
    plan_goal: Optional[str] = None
) -> bool:
    """
    ✅ FIRST PRINCIPLES: Quality check before creating anchor.
    
    Only create anchors for answers that:
    1. Pass EvidenceRegulator's invalid term filter
    2. Satisfy query boundary condition (not in input space)
    3. Have high confidence (>= 0.7) AND low entropy (<= 0.3)
    4. Are semantically relevant
    
    Args:
        entity: Entity to anchor
        anchor_data: Anchor metadata including answer and query
        entropy: Current entropy (uncertainty)
        confidence: Current confidence
        plan_goal: Overall plan goal
        
    Returns:
        True if anchor should be created, False otherwise
    """
    # Check 1: Invalid evidence term filter (same as EvidenceRegulator)
    if EvidenceRegulator:
        evidence_reg = EvidenceRegulator()
        if evidence_reg._is_invalid_evidence_term(entity):
            logger.debug(f"Anchor quality check: '{entity}' is invalid evidence term")
            return False
    
    # Check 2: Confidence and entropy thresholds
    # High confidence (>= 0.7) AND low entropy (<= 0.3) = certain, high-quality evidence
    if confidence < 0.7:
        logger.debug(f"Anchor quality check: '{entity}' confidence too low ({confidence:.3f} < 0.7)")
        return False
    
    if entropy > 0.3:
        logger.debug(f"Anchor quality check: '{entity}' entropy too high ({entropy:.3f} > 0.3)")
        return False
    
    # Check 3: Boundary condition (answer satisfies query, not in input space)
    answer = anchor_data.get("context", {}).get("answer", "")
    query = anchor_data.get("context", {}).get("query", "")
    
    if answer and query:
        # Use boundary condition check if available (from retrieval.py)
        try:
            if hasattr(self, '_check_answer_satisfies_boundary_condition'):
                boundary_check = self._check_answer_satisfies_boundary_condition(answer, query)
                if not boundary_check.get("satisfies", True):
                    logger.debug(
                        f"Anchor quality check: '{entity}' answer '{answer}' does not satisfy "
                        f"query boundary condition: {boundary_check.get('reason', 'unknown')}"
                    )
                    return False
        except Exception as e:
            # If boundary check not available or fails, skip it (graceful fallback)
            logger.debug(f"Boundary condition check skipped for '{entity}': {e}")
    
    # All quality checks passed
    return True

def _calculate_entropy_aware_strength(
    self,
    base_strength: float,
    entropy: float,
    confidence: float
) -> float:
    """
    ✅ FIRST PRINCIPLES: Calculate entropy-aware anchor strength.
    
    High entropy (uncertainty) → weaker anchors
    Low entropy (certainty) → stronger anchors
    
    Formula: strength = base_strength * (1 - entropy) * confidence
    
    Args:
        base_strength: Base anchor strength (0.0 to 1.0)
        entropy: Current entropy (0.0 to 1.0, higher = more uncertain)
        confidence: Current confidence (0.0 to 1.0)
        
    Returns:
        Entropy-adjusted anchor strength (0.0 to 1.0)
    """
    # Entropy penalty: (1 - entropy) means high entropy reduces strength
    # Confidence multiplier: Higher confidence increases strength
    entropy_factor = 1.0 - entropy  # 0.0 (high entropy) to 1.0 (low entropy)
    confidence_factor = confidence  # 0.0 to 1.0
    
    # Combined adjustment: base_strength * entropy_factor * confidence_factor
    adjusted_strength = base_strength * entropy_factor * confidence_factor
    
    # Clamp to valid range
    adjusted_strength = max(0.0, min(1.0, adjusted_strength))
    
    logger.debug(
        f"Entropy-aware strength: base={base_strength:.3f}, entropy={entropy:.3f}, "
        f"confidence={confidence:.3f} → adjusted={adjusted_strength:.3f}"
    )
    
    return adjusted_strength

