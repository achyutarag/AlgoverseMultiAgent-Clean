# agents/regulators/plan_regulator.py
from typing import Dict, Any, Optional, List
from .base_regulator import BaseRegulator, RegulatorConstraint
import logging
import re

logger = logging.getLogger(__name__)


class PlanRegulator(BaseRegulator):
    """
    Plan Regulator: global goal boundary condition.

    Defines a soft boundary toward the plan goal. The boundary
    always exists; its strength is modulated by alignment.
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
        Emit a plan-alignment boundary constraint.

        IMPORTANT:
        - The boundary is always active if a plan goal exists.
        - Alignment modulates strength only (never gates behavior).
        """

        if not plan_goal:
            # No global goal → very weak boundary
            return RegulatorConstraint(
                regulator_name=self.name,
                constraint_type="boundary",
                weight=0.2,
                parameters={"has_goal": False}
            )

        goal_keywords = self._extract_goal_keywords(plan_goal)

        alignment = self._calculate_goal_alignment(
            proposed_query,
            plan_goal,
            goal_keywords
        )

        # Boundary strength scales smoothly with alignment
        weight = max(0.1, self.weight * alignment)

        return RegulatorConstraint(
            regulator_name=self.name,
            constraint_type="boundary",
            weight=weight,
            parameters={
                "plan_goal": plan_goal,
                "goal_keywords": goal_keywords,
                "alignment": alignment
            }
        )
    

    def check_violation(
        self,
        query: str,
        constraint: RegulatorConstraint,
        current_state: Dict[str, Any]
    ) -> bool:
        """Check if query violates plan goal boundary constraint."""
        # If no plan goal, no violation possible
        if not constraint.parameters.get("has_goal", True):
            return False
        
        plan_goal = constraint.parameters.get("plan_goal")
        if not plan_goal:
            return False
        
        goal_keywords = constraint.parameters.get("goal_keywords", [])
        if not goal_keywords:
            return False
        
        # Check alignment
        alignment = self._calculate_goal_alignment(query, plan_goal, goal_keywords)
        
        # Violation: very low alignment (< 0.2) indicates query has drifted from plan goal
        return alignment < 0.2
    # ---------------------------
    # Helpers
    # ---------------------------

    def _extract_goal_keywords(self, plan_goal: str) -> List[str]:
        """
        Extract coarse goal keywords.

        NOTE:
        - This is intentionally simple.
        - Fine-grained semantics do NOT belong here.
        """
        stop_words = {
            "what", "is", "the", "a", "an", "and", "or", "but",
            "in", "on", "at", "to", "for", "of", "with", "by"
        }

        words = re.findall(r"\b\w+\b", plan_goal.lower())
        keywords = [
            w for w in words
            if w not in stop_words and len(w) > 2
        ]

        return keywords[:8]  # coarse, bounded

    def _calculate_goal_alignment(
        self,
        query: str,
        plan_goal: str,
        goal_keywords: List[str]
    ) -> float:
        """
        Soft alignment score ∈ [0, 1].

        Used ONLY to scale boundary strength.
        """
        if not goal_keywords:
            return 0.5

        query_lower = query.lower()

        matches = sum(
            1 for kw in goal_keywords
            if kw in query_lower
        )

        # Normalize gently — avoid hard jumps
        return min(1.0, matches / max(len(goal_keywords), 3))
