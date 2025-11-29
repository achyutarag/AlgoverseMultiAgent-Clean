# agents/regulators/plan_regulator.py
from typing import Dict, Any, Optional, List
from .base_regulator import BaseRegulator, RegulatorConstraint
import logging
import re

logger = logging.getLogger(__name__)

class PlanRegulator(BaseRegulator):
    """
    Plan Regulator: P(x, T) = δ(x - goal)
    
    Stabilizes next-hop goal or subtask and prevents wandering/incoherence.
    Defines final end-state boundary condition.
    """
    
    def __init__(self, weight: float = 0.9):
        super().__init__("Plan", weight)
    
    def apply_constraint(
        self,
        proposed_query: str,
        reasoning_state: Dict[str, Any],
        previous_answers: Dict[str, Any],
        plan_goal: Optional[str] = None
    ) -> RegulatorConstraint:
        """
        Apply plan goal boundary constraint.
        
        Ensures query aligns with the overall plan goal and maintains
        coherence with the intended reasoning trajectory.
        """
        if not plan_goal:
            # No plan goal → lower weight
            return RegulatorConstraint(
                regulator_name=self.name,
                constraint_type="boundary",
                weight=0.3,
                parameters={"has_goal": False}
            )
        
        # Extract goal keywords
        goal_keywords = self._extract_goal_keywords(plan_goal)
        
        # Check alignment with goal
        alignment = self._calculate_goal_alignment(
            proposed_query,
            plan_goal,
            goal_keywords
        )
        
        # Check if query is on track toward goal
        on_track = alignment > 0.3
        
        # Weight based on alignment
        weight = self.weight * alignment
        
        return RegulatorConstraint(
            regulator_name=self.name,
            constraint_type="boundary",
            weight=weight,
            parameters={
                "plan_goal": plan_goal,
                "goal_keywords": goal_keywords,
                "alignment": alignment,
                "on_track": on_track
            }
        )
    
    def check_violation(
        self,
        query: str,
        constraint: RegulatorConstraint,
        current_state: Dict[str, Any]
    ) -> bool:
        """Check if query violates plan goal boundary constraint."""
        on_track = constraint.parameters.get("on_track", True)
        return not on_track
    
    def _extract_goal_keywords(self, plan_goal: str) -> List[str]:
        """Extract key terms from plan goal."""
        # Remove common stop words
        stop_words = {"what", "is", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        # Tokenize
        words = re.findall(r'\b\w+\b', plan_goal.lower())
        
        # Filter and return meaningful keywords
        keywords = [
            word for word in words
            if word not in stop_words and len(word) > 2
        ]
        
        return keywords[:10]  # Top 10 keywords
    
    def _calculate_goal_alignment(
        self,
        query: str,
        plan_goal: str,
        goal_keywords: List[str]
    ) -> float:
        """Calculate how well query aligns with plan goal."""
        if not goal_keywords:
            return 0.5  # Neutral if no keywords
        
        query_lower = query.lower()
        goal_lower = plan_goal.lower()
        
        # Count matching keywords
        matching_keywords = sum(
            1 for keyword in goal_keywords
            if keyword in query_lower
        )
        
        # Check for goal-related concepts
        goal_concepts = self._extract_concepts(goal_lower)
        query_concepts = self._extract_concepts(query_lower)
        
        concept_overlap = len(set(goal_concepts) & set(query_concepts))
        max_concepts = max(len(goal_concepts), len(query_concepts), 1)
        concept_alignment = concept_overlap / max_concepts
        
        # Combined alignment score
        keyword_alignment = min(1.0, matching_keywords / 3.0)  # At least 3 keywords = full
        alignment = (keyword_alignment + concept_alignment) / 2.0
        
        return alignment
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        # Simple concept extraction (can be enhanced)
        # Look for noun phrases, entities, etc.
        concepts = []
        
        # Extract capitalized sequences (potential entities)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        concepts.extend(capitalized)
        
        # Extract important noun phrases (simple heuristic)
        # This is a simplified version - can be enhanced with NLP
        return concepts[:5]  # Top 5 concepts