# agents/regulators/evidence_regulator.py
from typing import Dict, Any, Optional, List
from .base_regulator import BaseRegulator, RegulatorConstraint
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
        """
        # Extract evidence terms from previous answers
        evidence_terms = self._extract_evidence_terms(
            previous_answers,
            reasoning_state
        )
        
        # Calculate evidence strength (how well query aligns with evidence)
        evidence_alignment = self._calculate_evidence_alignment(
            proposed_query,
            evidence_terms
        )
        
        # Weight based on evidence strength
        weight = self.weight * evidence_alignment
        
        return RegulatorConstraint(
            regulator_name=self.name,
            constraint_type="potential_well",
            weight=weight,
            parameters={
                "evidence_terms": evidence_terms,
                "top_terms": evidence_terms[:5],  # Top 5 terms
                "evidence_alignment": evidence_alignment,
                "term_count": len(evidence_terms)
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
        """Extract key evidence terms from previous answers."""
        all_terms = []
        
        # Extract from previous answers
        for step_id, answer_data in previous_answers.items():
            if isinstance(answer_data, dict):
                answer = answer_data.get("answer", "")
                sources = answer_data.get("sources", [])
            else:
                answer = str(answer_data)
                sources = []
            
            # Extract terms from answer
            terms = self._tokenize_and_filter(answer)
            all_terms.extend(terms)
        
        # Also get from reasoning state
        evidence_terms = reasoning_state.get("evidence_terms", [])
        all_terms.extend(evidence_terms)
        
        # Count term frequency and return top terms
        term_counts = Counter(all_terms)
        
        # Return top terms by frequency (excluding very common words)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        filtered_terms = [
            term for term, count in term_counts.most_common(20)
            if term.lower() not in stop_words and len(term) > 2
        ]
        
        return filtered_terms[:10]  # Top 10 evidence terms
    
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