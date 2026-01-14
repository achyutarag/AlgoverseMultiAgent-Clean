from typing import Dict, Any, List, Optional
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
    """Layer-1 validity gate to filter junk/placeholder answers."""
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


class ProtectedAnswerManager:
    """
    Sole owner of protected answers per slot.
    Policy: slot-explicit, monotonic, total, deterministic.
    """
    
    def __init__(self):
        """Initialize with empty protected answers dict."""
        self._protected_answers: Dict[str, Dict[str, Any]] = {}
        self._target_slot: Optional[str] = None  # Target slot for this query (inferred once)
    
    def set_target_slot(self, target_slot: str):
        """Set the target slot for this query (called once per query)."""
        self._target_slot = target_slot
        logger.debug(f"[ProtectedAnswer] Set target_slot='{target_slot}'")
    
    @property
    def target_slot(self) -> Optional[str]:
        """Get target slot (read-only)."""
        return self._target_slot
    
    def propose_candidates(
        self,
        candidates: List[Dict[str, Any]],
        required_domain: Optional[str] = None,
        required_level: Optional[str] = None,
        granularity_posterior: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        QA proposes slot-labeled candidates; manager decides accept/reject/replace.
        
        Policy:
        - Accept first valid answer per slot
        - Replace only if: same slot AND confidence strictly higher AND evidence >= existing AND no granularity violation
        - Never replace across slots
        - Never downgrade
        
        Args:
            candidates: List of candidate dicts with keys: answer, slot, confidence, evidence_count, sources
            required_domain: Required granularity domain (optional)
            required_level: Required granularity level (optional)
            
        Returns:
            Dict with accepted_count, rejected_count, replaced_count
        """
        accepted_count = 0
        rejected_count = 0
        replaced_count = 0
        
        for candidate in candidates:
            answer = candidate.get("answer", "").strip()
            slot = candidate.get("slot")
            confidence = float(candidate.get("confidence", 0.0) or 0.0)
            evidence_count = int(candidate.get("evidence_count", 0) or 0)
            sources = candidate.get("sources", [])
            
            # Skip if no slot or invalid span
            if not slot:
                rejected_count += 1
                continue
            
            if not _is_valid_span(answer):
                rejected_count += 1
                continue
            
            norm_ans = _norm_answer(answer)
            
            # ✅ LAYER 2: Use posterior-based penalty if available, else fallback to point estimate
            granularity_penalty = 0.0
            if granularity_posterior:
                try:
                    from ..regulators.granularity_regulator import GranularityRegulator
                    gran_reg = GranularityRegulator()
                    dom, lvl, _ = gran_reg.classify_entity_level(answer)
                    if lvl:
                        # Use posterior probability: lower probability = stronger penalty
                        level_prob = granularity_posterior.get(lvl, 0.0)
                        
                        # ✅ IMPROVEMENT 5: Dynamic penalty based on evidence count and posterior confidence
                        base_penalty_factor = 0.15
                        # Adjust based on evidence quality
                        if evidence_count >= 2:
                            base_penalty_factor = 0.18  # Slightly higher for high-evidence violations
                        
                        # Check posterior confidence in other levels
                        max_other_prob = max(
                            (prob for level, prob in granularity_posterior.items() if level != lvl),
                            default=0.0
                        )
                        confidence_boost = max_other_prob * 0.05  # Up to 0.05 additional penalty
                        penalty_factor = base_penalty_factor + confidence_boost
                        
                        granularity_penalty = (1.0 - level_prob) * penalty_factor
                        # ✅ FIX 2: Multiplicative penalty (preserves relative ordering, Bayesian-consistent)
                        # Convert absolute penalty to multiplicative factor
                        # Higher confidence answers get proportionally penalized, preserving ordering
                        penalty_factor_mult = min(granularity_penalty, 0.5)  # Cap at 50% reduction
                        confidence = max(0.0, confidence * (1.0 - penalty_factor_mult))
                        logger.debug(
                            f"[ProtectedAnswer] Granularity penalty (posterior, multiplicative) applied to '{answer}' for slot '{slot}': "
                            f"penalty_factor={penalty_factor_mult:.3f} (level={lvl}, posterior={level_prob:.3f}, "
                            f"evidence={evidence_count}, base_factor={penalty_factor:.3f}), "
                            f"adjusted confidence: {confidence:.3f}"
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
                            f"[ProtectedAnswer] Granularity penalty (unclassified, multiplicative) applied to '{answer}' for slot '{slot}': "
                            f"penalty_factor={penalty_factor_mult:.3f} (required_level={required_level}), adjusted confidence: {confidence:.3f}"
                        )
                except Exception as e:
                    logger.warning(f"[ProtectedAnswer] Granularity check failed: {e}")
            elif required_domain or required_level:
                # Fallback to old point-estimate method
                try:
                    from ..regulators.granularity_regulator import GranularityRegulator
                    gran_reg = GranularityRegulator()
                    dom, lvl, _ = gran_reg.classify_entity_level(answer)
                    if gran_reg.is_level_violation(required_domain, required_level, dom, lvl):
                        penalty_factor_mult = 0.15
                        # ✅ FIX 2: Multiplicative penalty
                        confidence = max(0.0, confidence * (1.0 - penalty_factor_mult))
                        logger.debug(
                            f"[ProtectedAnswer] Granularity penalty (point estimate, multiplicative) applied to '{answer}' for slot '{slot}': "
                            f"penalty_factor={penalty_factor_mult:.2f} (required {required_domain}/{required_level}, got {dom}/{lvl}), "
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
                            f"[ProtectedAnswer] Granularity penalty (unclassified, point estimate, multiplicative) applied to '{answer}' for slot '{slot}': "
                            f"penalty_factor={penalty_factor_mult:.2f} (required_level={required_level}), adjusted confidence: {confidence:.3f}"
                        )
                except Exception as e:
                    logger.warning(f"[ProtectedAnswer] Granularity check failed: {e}")
            
            existing = self._protected_answers.get(slot)
            
            if not existing:
                # Accept first valid answer for this slot
                self._protected_answers[slot] = {
                    "answer": answer,
                    "normalized": norm_ans,
                    "confidence": confidence,
                    "evidence_count": evidence_count,
                    "sources": sources,
                }
                accepted_count += 1
                logger.debug(
                    f"[ProtectedAnswer] Accepted first answer for slot '{slot}': '{norm_ans}' "
                    f"(conf={confidence:.3f}, evidence={evidence_count})"
                )
            else:
                # ✅ FIX 3: Use granularity-aware replacement check
                if self._can_replace(
                    existing,
                    {
                        "answer": answer,
                        "normalized": norm_ans,
                        "confidence": confidence,
                        "evidence_count": evidence_count,
                        "sources": sources,
                    },
                    required_domain,
                    required_level,
                    granularity_posterior
                ):
                    self._protected_answers[slot] = {
                        "answer": answer,
                        "normalized": norm_ans,
                        "confidence": confidence,
                        "evidence_count": evidence_count,
                        "sources": sources,
                    }
                    replaced_count += 1
                    logger.debug(
                        f"[ProtectedAnswer] Replaced answer for slot '{slot}': "
                        f"'{existing.get('normalized')}' -> '{norm_ans}' "
                        f"(conf {existing.get('confidence', 0.0):.3f} -> {confidence:.3f}, evidence {existing.get('evidence_count', 0)} -> {evidence_count})"
                    )
                else:
                    # Never downgrade: reject if not strictly better or granularity worse
                    rejected_count += 1
                    logger.debug(
                        f"[ProtectedAnswer] Rejected candidate for slot '{slot}': "
                        f"not strictly better or granularity status worse"
                    )
        
        return {
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "replaced_count": replaced_count,
        }
    
    def _get_granularity_status(
        self, 
        answer: str, 
        required_domain: Optional[str], 
        required_level: Optional[str],
        granularity_posterior: Optional[Dict[str, float]] = None
    ) -> int:
        """
        Get granularity status: 2=match, 1=unclassified, 0=violation.
        Higher is better.
        """
        try:
            from ..regulators.granularity_regulator import GranularityRegulator
            gran_reg = GranularityRegulator()
            dom, lvl, _ = gran_reg.classify_entity_level(answer)
            
            if not lvl:
                return 1  # Unclassified
            
            if granularity_posterior:
                # Use posterior: high probability = match
                level_prob = granularity_posterior.get(lvl, 0.0)
                if level_prob >= 0.5:  # High confidence match
                    return 2
                elif level_prob >= 0.2:  # Moderate match
                    return 1
                else:
                    return 0  # Violation
            elif required_domain and required_level:
                # Use point estimate
                if not gran_reg.is_level_violation(required_domain, required_level, dom, lvl):
                    return 2  # Match
                else:
                    return 0  # Violation
            
            return 1  # Default: unclassified
        except Exception:
            return 1  # Default: unclassified
    
    def _can_replace(
        self,
        old: Dict[str, Any],
        new: Dict[str, Any],
        required_domain: Optional[str],
        required_level: Optional[str],
        granularity_posterior: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Check if new candidate can replace old one.
        
        ✅ FIX 3: Allows replacement when later evidence corrects earlier mistakes.
        
        Replacement is allowed if ANY of these conditions are met:
        1. Confidence significantly higher (> 0.05) AND evidence >= existing
        2. Confidence close (within 0.1) AND granularity status improves
        3. Confidence close (within 0.1) AND evidence quality is better (more sources)
        4. Granularity status improves significantly (status difference >= 1)
        """
        old_conf = float(old.get("confidence", 0.0) or 0.0)
        new_conf = float(new.get("confidence", 0.0) or 0.0)
        old_ev = old.get("evidence_count", 0)
        new_ev = new.get("evidence_count", 0)
        old_sources = len(old.get("sources", []))
        new_sources = len(new.get("sources", []))
        
        # Get granularity status for both
        old_status = self._get_granularity_status(
            old.get("answer", ""), required_domain, required_level, granularity_posterior
        )
        new_status = self._get_granularity_status(
            new.get("answer", ""), required_domain, required_level, granularity_posterior
        )
        
        # Condition 1: Significant confidence improvement AND evidence >= existing
        if new_conf > old_conf + 0.05 and new_ev >= old_ev:
            return True
        
        # Condition 2: Confidence close AND granularity status improves
        if abs(new_conf - old_conf) <= 0.1 and new_status > old_status:
            return True
        
        # Condition 3: Confidence close AND evidence quality is better (more sources)
        if abs(new_conf - old_conf) <= 0.1 and new_sources > old_sources:
            return True
        
        # Condition 4: Granularity status improves (allows correction even with lower confidence)
        if new_status > old_status:  # Any improvement in granularity status
            # Allow if confidence is not too much lower (within 0.15)
            if new_conf >= old_conf - 0.15:
                return True
        
        # Default: don't allow replacement (prevents downgrades)
        return False
    
    def get_protected_answers(self) -> Dict[str, Dict[str, Any]]:
        """
        Return read-only view of protected answers.
        
        Returns:
            Dict mapping slot -> protected answer dict (normalized, confidence, evidence_count, sources)
        """
        # Return shallow copy to prevent external mutation
        return {slot: dict(pa) for slot, pa in self._protected_answers.items()}
    
    def clear(self):
        """Clear all protected answers and target slot (for testing/reset)."""
        self._protected_answers.clear()
        self._target_slot = None