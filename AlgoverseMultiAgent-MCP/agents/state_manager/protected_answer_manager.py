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
            
            # Check granularity violation if required
            if required_domain or required_level:
                try:
                    from ..regulators.granularity_regulator import GranularityRegulator
                    gran_reg = GranularityRegulator()
                    dom, lvl, _ = gran_reg.classify_entity_level(answer)
                    if gran_reg.is_level_violation(required_domain, required_level, dom, lvl):
                        logger.debug(
                            f"[ProtectedAnswer] Reject candidate '{answer}' for slot '{slot}': "
                            f"granularity violation (required {required_domain}/{required_level}, got {dom}/{lvl})"
                        )
                        rejected_count += 1
                        continue
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
                # Replace only if: same slot AND confidence strictly higher AND evidence >= existing AND no granularity violation
                existing_conf = float(existing.get("confidence", 0.0) or 0.0)
                existing_ev = existing.get("evidence_count", 0)
                
                if (
                    confidence > existing_conf  # strictly higher
                    and evidence_count >= existing_ev  # evidence >= existing
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
                        f"(conf {existing_conf:.3f} -> {confidence:.3f}, evidence {existing_ev} -> {evidence_count})"
                    )
                else:
                    # Never downgrade: reject if not strictly better
                    rejected_count += 1
                    logger.debug(
                        f"[ProtectedAnswer] Rejected candidate for slot '{slot}': "
                        f"not strictly better (conf {confidence:.3f} <= {existing_conf:.3f} or "
                        f"evidence {evidence_count} < {existing_ev})"
                    )
        
        return {
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "replaced_count": replaced_count,
        }
    
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