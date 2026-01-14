from typing import Any, Dict, Optional
import re
import logging

logger = logging.getLogger(__name__)


def _norm_answer(ans: str) -> str:
    """Normalize answers for comparison (lowercase, strip punctuation/whitespace)."""
    if not ans:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", str(ans).lower())
    return " ".join(cleaned.split())


def _is_valid_span(ans: str) -> bool:
    """Layer-1 validity gate to filter junk/placeholder answers before scoring."""
    if not ans:
        return False
    norm = _norm_answer(ans)
    if not norm:
        return False
    # ✅ FIX 2: Remove "unknown" from hard rejection list
    # Let it pass through with soft penalties instead (allows comparative evaluation)
    confirmations = {"yes", "no", "none", "n/a", "na", ""}  # Removed "unknown"
    if norm in confirmations:
        return False
    articles = {"the", "a", "an"}
    tokens = norm.split()
    if all(tok in articles for tok in tokens):
        return False
    if len(tokens) < 2 and len(norm) < 6:
        return False
    return True


class ConvergenceValidityGate:
    """
    Single convergence/validity gate that decides accept | needs_more_evidence | reject.
    Keeps local agents simple and centralizes global correctness checks.
    """

    def evaluate(
        self,
        qa_result: Dict[str, Any],
        protected_answers: Optional[Dict[str, Any]] = None,
        required_domain: Optional[str] = None,
        required_level: Optional[str] = None,
        slot: Optional[str] = None,
        granularity_posterior: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        protected_answers = protected_answers or {}
        answer_raw = (qa_result.get("answer") or "").strip()
        confidence = float(qa_result.get("confidence", 0.0) or 0.0)
        sources = qa_result.get("sources") or []
        supporting = qa_result.get("supporting_evidence") or []
        diffusion_metadata = qa_result.get("diffusion_metadata", {}) or {}
        entropy = diffusion_metadata.get("entropy")

        evidence_count = len(supporting) if supporting else len(sources)
        slot_id = slot or qa_result.get("slot")

        # 1) Span validity
        if not _is_valid_span(answer_raw):
            return {
                "decision": "reject",
                "reason": "invalid_span",
                "action": "fallback",
            }

        norm_ans = _norm_answer(answer_raw)

        # ✅ FIX 2: Apply soft penalty for "unknown" instead of hard rejection
        # Allow "unknown" to pass through but with heavy penalty for comparative evaluation
        if norm_ans == "unknown":
            # Heavy penalty but not rejection - let final assembler compare against other candidates
            confidence = confidence * 0.1  # Reduce confidence by 90%
            logger.debug(
                f"[Gate] 'unknown' answer detected: applying soft penalty "
                f"(confidence: {qa_result.get('confidence', 0.0):.3f} → {confidence:.3f})"
            )

        # 2) Protected conflict/match (slot optional)
        if slot_id:
            pa = protected_answers.get(slot_id)
            if pa:
                pa_norm = pa.get("normalized") or _norm_answer(pa.get("answer", ""))
                if pa_norm and pa_norm != norm_ans:
                    return {
                        "decision": "reject",
                        "reason": "protected_conflict",
                        "action": "fallback",
                    }
                if pa_norm and pa_norm == norm_ans:
                    protected_match = True
                else:
                    protected_match = False
            else:
                protected_match = False
        else:
            protected_match = False

        # 3) Evidence sufficiency (coverage)
        if evidence_count < 1:
            return {
                "decision": "needs_more_evidence",
                "reason": "insufficient_evidence",
                "action": "continue",
            }

        # ✅ NEW: Semantic equivalence check (if evidence_term provided)
        evidence_term = qa_result.get("evidence_term")
        semantic_penalty = 0.0
        
        if slot_id and evidence_term:
            # Gate decides semantic equivalence (QA is descriptive, Gate is normative)
            if slot_id == "spouse":
                if evidence_term == "spouse":
                    # Perfect match
                    semantic_penalty = 0.0
                elif evidence_term in ["wife", "husband"]:
                    # Strong equivalence
                    semantic_penalty = 0.0
                elif evidence_term == "partner":
                    # Possible equivalence, but less certain
                    semantic_penalty = 0.1  # Slight confidence reduction
                elif evidence_term == "married":
                    # Indirect, needs more context
                    semantic_penalty = 0.2
                else:
                    # Unknown term, be cautious
                    semantic_penalty = 0.3
            elif slot_id == "founder":
                if evidence_term in ["founder", "founded", "established", "created"]:
                    semantic_penalty = 0.0
                else:
                    semantic_penalty = 0.2
            elif slot_id == "performer":
                if evidence_term in ["performer", "performed", "artist", "musician", "singer", "actor"]:
                    semantic_penalty = 0.0
                else:
                    semantic_penalty = 0.2
            # Add more slot-specific mappings as needed
            
            # Adjust confidence based on semantic gap
            adjusted_confidence = confidence - semantic_penalty
            if adjusted_confidence < 0.5:  # Threshold for acceptance
                return {
                    "decision": "needs_more_evidence",
                    "reason": "semantic_gap",
                    "action": "continue",
                    "details": f"Evidence term '{evidence_term}' vs slot '{slot_id}' (penalty: {semantic_penalty:.2f})"
                }
            # Use adjusted confidence for remaining checks
            confidence = adjusted_confidence

        # 4) Confidence threshold
        if confidence < 0.5:
            return {
                "decision": "needs_more_evidence",
                "reason": "low_confidence",
                "action": "continue",
            }

        # 5) Entropy check (soft signal, never forces reject alone)
        if entropy is not None:
            try:
                entropy_val = float(entropy)
                if entropy_val > 0.5:
                    # High entropy suggests uncertainty, downgrade to needs_more_evidence
                    return {
                        "decision": "needs_more_evidence",
                        "reason": "high_entropy",
                        "action": "continue",
                    }
            except Exception:
                pass

        # ✅ LAYER 2: Use posterior-based penalty if available, else fallback to point estimate
        granularity_penalty = 0.0
        if granularity_posterior:
            try:
                from ..regulators.granularity_regulator import GranularityRegulator
                gran_reg = GranularityRegulator()
                dom, lvl, _ = gran_reg.classify_entity_level(answer_raw)
                if lvl:
                    # Use posterior probability: lower probability = stronger penalty
                    level_prob = granularity_posterior.get(lvl, 0.0)
                    
                    # ✅ IMPROVEMENT 5: Dynamic penalty factor based on evidence quality
                    # More evidence + wrong level = stronger signal that it's wrong
                    base_penalty_factor = 0.2 if evidence_count >= 2 else 0.1
                    
                    # Adjust factor based on posterior confidence
                    # If posterior is very confident (high prob for other levels), increase penalty
                    max_other_prob = max(
                        (prob for level, prob in granularity_posterior.items() if level != lvl),
                        default=0.0
                    )
                    # If max_other_prob is high (e.g., 0.8), we're confident in a different level
                    confidence_boost = max_other_prob * 0.1  # Up to 0.1 additional penalty
                    penalty_factor = base_penalty_factor + confidence_boost
                    
                    granularity_penalty = (1.0 - level_prob) * penalty_factor
                    # ✅ FIX 2: Multiplicative penalty (preserves relative ordering, Bayesian-consistent)
                    # Convert absolute penalty to multiplicative factor
                    penalty_factor_mult = min(granularity_penalty, 0.5)  # Cap at 50% reduction
                    confidence = max(0.0, confidence * (1.0 - penalty_factor_mult))
                    logger.debug(
                        f"[Gate] Granularity penalty (posterior, multiplicative) applied: "
                        f"penalty_factor={penalty_factor_mult:.3f} (level={lvl}, posterior={level_prob:.3f}, evidence={evidence_count}, "
                        f"base_factor={penalty_factor:.3f}), adjusted confidence: {confidence:.3f}"
                    )
                else:
                    # ✅ FIX #3: Penalize unclassified candidates independently of posterior
                    # Unclassified = uncertainty, should be penalized based on whether level is required
                    if required_level:
                        # If a specific level is required, unclassified is a problem
                        penalty_factor_mult = 0.15  # Fixed penalty when level is required
                    else:
                        # If no level requirement, less penalty for unclassified
                        penalty_factor_mult = 0.05  # Lower penalty when no requirement
                    
                    # ✅ FIX 2: Multiplicative penalty
                    confidence = max(0.0, confidence * (1.0 - min(penalty_factor_mult, 0.5)))
                    logger.debug(
                        f"[Gate] Granularity penalty (unclassified, multiplicative) applied: penalty_factor={penalty_factor_mult:.3f} "
                        f"(required_level={required_level}), adjusted confidence: {confidence:.3f}"
                    )
            except Exception as e:
                logger.warning(f"Granularity check failed: {e}")
        elif required_domain or required_level:
            # Fallback to old point-estimate method
            try:
                from ..regulators.granularity_regulator import GranularityRegulator
                gran_reg = GranularityRegulator()
                dom, lvl, _ = gran_reg.classify_entity_level(answer_raw)
                if gran_reg.is_level_violation(required_domain, required_level, dom, lvl):
                    if evidence_count >= 2:
                        penalty_factor_mult = 0.2
                    else:
                        penalty_factor_mult = 0.1
                    
                    # ✅ FIX 2: Multiplicative penalty
                    confidence = max(0.0, confidence * (1.0 - penalty_factor_mult))
                    logger.debug(
                        f"[Gate] Granularity penalty (point estimate, multiplicative) applied: penalty_factor={penalty_factor_mult:.2f} "
                        f"(required {required_domain}/{required_level}, got {dom}/{lvl}), "
                        f"adjusted confidence: {confidence:.3f}"
                    )
                elif not lvl:
                    # ✅ FIX #3: Penalize unclassified candidates (fallback method, multiplicative)
                    # Use same logic as posterior method: check if level is required
                    if required_level:
                        penalty_factor_mult = 0.15  # Fixed penalty when level is required
                    else:
                        penalty_factor_mult = 0.05  # Lower penalty when no requirement
                    confidence = max(0.0, confidence * (1.0 - min(penalty_factor_mult, 0.5)))
                    logger.debug(
                        f"[Gate] Granularity penalty (unclassified, point estimate, multiplicative) applied: penalty_factor={penalty_factor_mult:.2f} "
                        f"(required_level={required_level}), adjusted confidence: {confidence:.3f}"
                    )
            except Exception as e:
                logger.warning(f"Granularity check failed: {e}")
        
        # Re-check confidence threshold after granularity penalty
        if confidence < 0.5:
            return {
                "decision": "needs_more_evidence",
                "reason": "low_confidence_after_granularity_penalty",
                "action": "continue",
                "granularity_penalty": granularity_penalty,
            }

        # All checks passed
        return {
            "decision": "accept",
            "reason": "valid",
            "action": "proceed",
            "semantic_penalty": semantic_penalty,  # Include for debugging
            "evidence_term": evidence_term,  # Include for debugging
            "protected_match": protected_match,
            "slot": slot_id,
            "required_domain": required_domain,
            "required_level": required_level,
        }

