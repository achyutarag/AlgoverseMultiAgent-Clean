# agents/regulators/relation_regulator.py
from typing import Dict, Any, Optional
from .base_regulator import BaseRegulator, RegulatorConstraint
import logging
import re

logger = logging.getLogger(__name__)

class RelationRegulator(BaseRegulator):
    """
    Relation Regulator: ∂P/∂x = 0 (Neumann constraint)
    
    Stabilizes relational direction between hops and prevents path deviation.
    """
    
    def __init__(self, weight: float = 0.8):
        super().__init__("Relation", weight)
    
    def apply_constraint(
        self,
        proposed_query: str,
        reasoning_state: Dict[str, Any],
        previous_answers: Dict[str, Any],
        plan_goal: Optional[str] = None
    ) -> RegulatorConstraint:
        """
        Apply relational direction constraint.
        
        Ensures query maintains the relational direction established
        in previous steps (e.g., "contains" vs "contained in").
        """
        # Detect relation direction from previous answers and reasoning state
        relation_direction = self._detect_relation_direction(
            proposed_query,
            previous_answers,
            reasoning_state
        )
        
        # Check if proposed query maintains relational direction
        maintains_direction = self._check_direction_consistency(
            proposed_query,
            relation_direction
        )
        
        return RegulatorConstraint(
            regulator_name=self.name,
            constraint_type="neumann",
            weight=self.weight if relation_direction else 0.5,
            parameters={
                "direction": relation_direction,
                "maintains_direction": maintains_direction,
                "relation_type": self._classify_relation_type(proposed_query)
            }
        )
    
    def check_violation(
        self,
        query: str,
        constraint: RegulatorConstraint,
        current_state: Dict[str, Any]
    ) -> bool:
        """Check if query violates relational direction constraint."""
        expected_direction = constraint.parameters.get("direction")
        if not expected_direction:
            return False
        
        maintains = self._check_direction_consistency(query, expected_direction)
        return not maintains
    
    def _detect_relation_direction(
        self,
        query: str,
        previous_answers: Dict[str, Any],
        reasoning_state: Dict[str, Any]
    ) -> Optional[str]:
        """Detect relational direction from context."""
        # Check reasoning state first
        relation_direction = reasoning_state.get("relation_direction")
        if relation_direction:
            return relation_direction
        
        # Detect from query patterns
        hierarchical_patterns = {
            "contains": ["contains", "includes", "has", "owns"],
            "contained_in": ["in", "within", "located in", "part of", "belongs to"],
            "parent_of": ["parent", "above", "higher"],
            "child_of": ["child", "below", "lower", "sub"]
        }
        
        query_lower = query.lower()
        for direction, patterns in hierarchical_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return direction
        
        return None
    
    def _check_direction_consistency(
        self,
        query: str,
        expected_direction: Optional[str]
    ) -> bool:
        """Check if query maintains expected relational direction."""
        if not expected_direction:
            return True
        
        query_lower = query.lower()
        
        # Check for direction keywords
        direction_keywords = {
            "contains": ["contains", "includes", "has"],
            "contained_in": ["in", "within", "located in"],
            "parent_of": ["parent", "above"],
            "child_of": ["child", "below", "sub"]
        }
        
        expected_keywords = direction_keywords.get(expected_direction, [])
        has_expected = any(kw in query_lower for kw in expected_keywords)
        
        # Check for contradictory keywords
        contradictory = {
            "contains": ["contained_in", "child_of"],
            "contained_in": ["contains", "parent_of"],
            "parent_of": ["child_of", "contained_in"],
            "child_of": ["parent_of", "contains"]
        }
        
        contradictory_directions = contradictory.get(expected_direction, [])
        has_contradictory = any(
            any(kw in query_lower for kw in direction_keywords.get(d, []))
            for d in contradictory_directions
        )
        
        return has_expected and not has_contradictory
    
    def _classify_relation_type(self, query: str) -> str:
        """Classify the type of relation in the query."""
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ["hierarchical", "administrative", "territorial"]):
            return "hierarchical"
        elif any(kw in query_lower for kw in ["compare", "versus", "vs", "difference"]):
            return "comparative"
        elif any(kw in query_lower for kw in ["contains", "includes", "has"]):
            return "containment"
        elif any(kw in query_lower for kw in ["in", "within", "located"]):
            return "location"
        else:
            return "factual"