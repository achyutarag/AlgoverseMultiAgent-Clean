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
        
        # ✅ FIRST PRINCIPLES FIX: Filter evidence terms by hierarchical level
        # If plan_goal requires a specific hierarchical level, don't add terms
        # that contain hierarchical level keywords from wrong level (prevents bias)
        filtered_terms = self._filter_by_hierarchical_level(
            evidence_terms,
            plan_goal,
            proposed_query
        )
        
        # Calculate evidence strength (how well query aligns with evidence)
        evidence_alignment = self._calculate_evidence_alignment(
            proposed_query,
            filtered_terms
        )
        
        # Weight based on evidence strength
        weight = self.weight * evidence_alignment
        
        return RegulatorConstraint(
            regulator_name=self.name,
            constraint_type="potential_well",
            weight=weight,
            parameters={
                "evidence_terms": filtered_terms,  # Use filtered terms
                "top_terms": filtered_terms[:5],  # Top 5 terms
                "evidence_alignment": evidence_alignment,
                "term_count": len(filtered_terms),
                "original_terms": evidence_terms  # Keep original for debugging
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
    
    def _filter_by_hierarchical_level(
        self,
        evidence_terms: List[str],
        plan_goal: Optional[str],
        proposed_query: str
    ) -> List[str]:
        """
        Filter evidence terms to respect hierarchical level requirements.
        
        ✅ GENERALIZED: Uses GranularityRegulator's hierarchical level system
        instead of hardcoded location terms. Works for any hierarchical domain
        (territorial, organizational, taxonomic, etc.).
        
        If a term violates the required hierarchical level, extracts the
        parent-level term name to help retrieval while respecting constraints.
        
        Args:
            evidence_terms: List of evidence terms to filter
            plan_goal: Overall plan goal/question (used to infer required level)
            proposed_query: Proposed query (for context)
            
        Returns:
            Filtered list of evidence terms that respect hierarchical constraints
        """
        if not evidence_terms:
            return []
        
        # ✅ FIRST PRINCIPLES FIX: Filter out invalid evidence terms first
        # This prevents "unknown" and similar non-evidence terms from polluting queries
        valid_terms = [term for term in evidence_terms if not self._is_invalid_evidence_term(term)]
        
        if not valid_terms:
            logger.debug("EvidenceRegulator: All evidence terms filtered as invalid")
            return []
        
        if not plan_goal:
            return valid_terms  # No hierarchical constraint, but still filter invalid terms
        
        # Use GranularityRegulator to infer required level (generalized, not hardcoded)
        granularity_reg = GranularityRegulator()
        required_domain, required_level = granularity_reg._infer_required_level(plan_goal)
        
        if not required_domain or not required_level:
            return valid_terms  # No hierarchical constraint detected
        
        # Filter evidence terms based on hierarchical level requirement
        filtered = []
        for term in valid_terms:
            # Classify term's hierarchical level using GranularityRegulator
            term_domain, term_level, _ = granularity_reg.classify_entity_level(term)
            
            # ✅ FIX: If term can't be classified (no level keywords), allow it through
            # It might be the correct answer (e.g., "Tamaulipas" without "state" keyword)
            # OR it might be in previous answers and needs to be reinforced for retrieval
            if not term_domain or not term_level:
                filtered.append(term)
                logger.debug(
                    f"EvidenceRegulator: Allowing unclassified term '{term}' through "
                    f"(no level keywords found - might be correct answer or needs reinforcement)"
                )
                continue
            
            # Check if term violates required level
            is_violation = granularity_reg.is_level_violation(
                required_domain, required_level,
                term_domain, term_level
            )
            
            if is_violation:
                # Try to extract parent-level term name (generalized extraction)
                parent_name = granularity_reg.extract_parent_level_name(
                    term, required_domain, required_level
                )
                
                if parent_name:
                    filtered.append(parent_name)
                    logger.debug(
                        f"EvidenceRegulator: Extracted parent-level term '{parent_name}' from '{term}' "
                        f"(required: {required_domain}/{required_level}, term: {term_domain}/{term_level}) "
                        f"to help retrieval while respecting hierarchical constraint"
                    )
                else:
                    logger.debug(
                        f"EvidenceRegulator: Skipping evidence term '{term}' "
                        f"(violates required level {required_domain}/{required_level}, "
                        f"could not extract parent-level name)"
                    )
                continue
            
            # Term respects hierarchical constraint
            filtered.append(term)
        
        # Fallback to original if all filtered (better than no terms)
        return filtered if filtered else valid_terms
