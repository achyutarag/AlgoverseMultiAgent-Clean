# agents/regulators/evidence_regulator.py
from typing import Dict, Any, Optional, List
from .base_regulator import BaseRegulator, RegulatorConstraint
from .granularity_regulator import GranularityRegulator
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class EvidenceRegulator(BaseRegulator):
    """
    Evidence Regulator: P(x) ∝ e^(-V(x))
    
    Evidence acts as a potential well, stabilizing relevant retrieved chunks/docs
    and preventing distractor collapse.
    """
    
    def __init__(self, weight: float = 0.85):
        super().__init__("Evidence", weight)
    
    def apply_constraint(
        self,
        proposed_query: str,
        reasoning_state: Dict[str, Any],
        previous_answers: Dict[str, Any],
        plan_goal: Optional[str] = None
    ) -> RegulatorConstraint:
        """
        Apply evidence potential well constraint.
        
        Extracts key evidence terms from previous steps and reinforces
        them in the query to maintain focus on relevant information.
        
        ✅ FIRST PRINCIPLES: Respects hierarchical level constraints set by
        GranularityRegulator (initial condition). Filters out evidence terms
        that contain hierarchical level keywords from wrong level to prevent bias.
        """
        # Extract evidence terms from previous answers
        evidence_terms = self._extract_evidence_terms(
            previous_answers,
            reasoning_state
        )
        
        # ✅ FIX: Annotate instead of filter
        annotated_terms = self._annotate_by_hierarchical_level(
            evidence_terms,
            plan_goal,
            proposed_query
        )
        
        # Extract term strings for backward compatibility
        term_strings = [ann["term"] for ann in annotated_terms]
        
        # Calculate evidence strength (how well query aligns with evidence)
        evidence_alignment = self._calculate_evidence_alignment(
            proposed_query,
            term_strings
        )
        
        # Weight based on evidence strength
        weight = self.weight * evidence_alignment
        
        return RegulatorConstraint(
            regulator_name=self.name,
            constraint_type="potential_well",
            weight=weight,
            parameters={
                "evidence_terms": term_strings,  # For backward compatibility
                "annotated_terms": annotated_terms,  # NEW: Full annotations
                "top_terms": term_strings[:5],
                "evidence_alignment": evidence_alignment,
                "term_count": len(term_strings),
                "original_terms": evidence_terms
            }
        )
    
    def check_violation(
        self,
        query: str,
        constraint: RegulatorConstraint,
        current_state: Dict[str, Any]
    ) -> bool:
        """Check if query violates evidence potential well constraint."""
        evidence_terms = constraint.parameters.get("evidence_terms", [])
        if not evidence_terms:
            return False
        
        # Violation: query doesn't contain any key evidence terms
        query_lower = query.lower()
        has_evidence = any(
            term.lower() in query_lower
            for term in evidence_terms[:3]  # Check top 3 terms
        )
        
        return not has_evidence
    
    def _extract_evidence_terms(
        self,
        previous_answers: Dict[str, Any],
        reasoning_state: Dict[str, Any]
    ) -> List[str]:
        """
        Extract key evidence terms from previous answers.
        
        ✅ FIRST PRINCIPLES FIX: Filters out "unknown" and other non-evidence terms.
        "Unknown" is the absence of evidence, not evidence itself.
        """
        all_terms = []
        
        # Extract from previous answers
        for step_id, answer_data in previous_answers.items():
            if isinstance(answer_data, dict):
                answer = answer_data.get("answer", "")
                sources = answer_data.get("sources", [])
            else:
                answer = str(answer_data)
                sources = []
            
            # ✅ FIRST PRINCIPLES FIX: Skip "unknown" answers - they are not evidence
            # "Unknown" is the absence of evidence, not evidence itself
            answer_lower = answer.lower().strip()
            if self._is_invalid_evidence_term(answer_lower):
                logger.debug(
                    f"EvidenceRegulator: Skipping invalid evidence term '{answer}' from step {step_id}"
                )
                continue
            
            # Extract terms from answer
            terms = self._tokenize_and_filter(answer)
            all_terms.extend(terms)
        
        # Also get from reasoning state
        evidence_terms = reasoning_state.get("evidence_terms", [])
        all_terms.extend(evidence_terms)
        
        # Count term frequency and return top terms
        term_counts = Counter(all_terms)
        
        # Return top terms by frequency (excluding very common words and invalid terms)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        filtered_terms = [
            term for term, count in term_counts.most_common(20)
            if term.lower() not in stop_words 
            and len(term) > 2
            and not self._is_invalid_evidence_term(term)
        ]
        
        return filtered_terms[:10]  # Top 10 evidence terms
    
    def _is_invalid_evidence_term(self, term: str) -> bool:
        """
        Check if a term is invalid evidence (e.g., "unknown", "none", etc.).
        
        ✅ FIRST PRINCIPLES: Non-evidence terms should not be reinforced in queries.
        These terms indicate absence of evidence, not evidence itself.
        
        Args:
            term: Term to check
            
        Returns:
            True if term is invalid evidence, False otherwise
        """
        if not term:
            return True
        
        term_lower = term.lower().strip()
        invalid_terms = {
            "unknown", "none", "n/a", "na", "null", "nil", 
            "not found", "not available", "no answer", "no information",
            "unclear", "uncertain", "unsure", "ambiguous",
            "yes"
        }
        return term_lower in invalid_terms
    
    def _tokenize_and_filter(self, text: str) -> List[str]:
        """Tokenize text and filter for meaningful terms."""
        import re
        # Simple tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Filter: length > 2, not pure numbers
        filtered = [
            t for t in tokens
            if len(t) > 2 and not t.isdigit()
        ]
        return filtered
    
    def _calculate_evidence_alignment(
        self,
        query: str,
        evidence_terms: List[str]
    ) -> float:
        """Calculate how well query aligns with evidence terms."""
        if not evidence_terms:
            return 0.5  # Neutral if no evidence
        
        query_lower = query.lower()
        matching_terms = sum(
            1 for term in evidence_terms[:5]
            if term.lower() in query_lower
        )
        
        # Alignment score: 0.0 to 1.0
        alignment = min(1.0, matching_terms / 3.0)  # At least 3 terms = full alignment
        
        return alignment
    
    def _annotate_by_hierarchical_level(
        self,
        evidence_terms: List[str],
        plan_goal: Optional[str],
        proposed_query: str
    ) -> List[Dict[str, Any]]:
        """
        Annotate evidence terms with granularity metadata instead of filtering.
        
        ✅ FIX: Preserves all evidence terms, adds granularity_delta annotation.
        """
        if not evidence_terms:
            return []
        
        # ✅ FIRST PRINCIPLES FIX: Filter out invalid evidence terms first
        # This prevents "unknown" and similar non-evidence terms from polluting queries
        valid_terms = [term for term in evidence_terms if not self._is_invalid_evidence_term(term)]
        
        if not valid_terms:
            logger.debug("EvidenceRegulator: All evidence terms filtered as invalid")
            return []
        
        # Use GranularityRegulator to infer required level
        granularity_reg = GranularityRegulator()
        required_domain, required_level = granularity_reg._infer_required_level(plan_goal)
        
        annotated_terms = []
        for term in valid_terms:
            # Compute granularity metadata
            if required_domain and required_level:
                granularity_metadata = granularity_reg.compute_granularity_metadata(
                    term, required_domain, required_level
                )
            else:
                granularity_metadata = {
                    "granularity_delta": None,
                    "granularity_violation": False,
                    "is_unclassified": True,
                    "penalty_factor": 0.0
                }
            
            annotated_terms.append({
                "term": term,
                "granularity_metadata": granularity_metadata
            })
            
            logger.debug(
                f"EvidenceRegulator: Annotated term '{term}': "
                f"delta={granularity_metadata.get('granularity_delta')}, "
                f"violation={granularity_metadata.get('granularity_violation')}"
            )
        
        return annotated_terms
