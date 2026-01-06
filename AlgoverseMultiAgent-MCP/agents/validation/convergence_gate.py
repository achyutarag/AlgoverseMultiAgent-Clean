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
    confirmations = {"yes", "no", "unknown", "none", "n/a", "na", ""}
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

        # 6) Granularity check (if required)
        if required_domain or required_level:
            try:
                from ..regulators.granularity_regulator import GranularityRegulator
                gran_reg = GranularityRegulator()
                dom, lvl, _ = gran_reg.classify_entity_level(answer_raw)
                if gran_reg.is_level_violation(required_domain, required_level, dom, lvl):
                    # Granularity violation: check if it's structural or just missing evidence
                    if evidence_count >= 2:
                        # Have evidence but wrong granularity → structural impossibility
                        return {
                            "decision": "reject",
                            "reason": "granularity_violation",
                            "action": "fallback",
                        }
                    else:
                        # Missing evidence → needs more evidence
                        return {
                            "decision": "needs_more_evidence",
                            "reason": "granularity_insufficient",
                            "action": "continue",
                        }
            except Exception as e:
                logger.warning(f"Granularity check failed: {e}")

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

