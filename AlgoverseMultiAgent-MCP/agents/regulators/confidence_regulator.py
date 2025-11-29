# agents/regulators/confidence_regulator.py
from typing import Dict, Any, Optional
from .base_regulator import BaseRegulator, RegulatorConstraint
import logging

logger = logging.getLogger(__name__)

class ConfidenceRegulator(BaseRegulator):
    """
    Confidence Regulator: D(t) = f(1 - p_conf)
    
    Stabilizes uncertainty & hop reliability and prevents diffusion explosion.
    More uncertainty → more diffusion → need stronger constraints.
    """
    
    def __init__(self, weight: float = 0.75):
        super().__init__("Confidence", weight)
    
    def apply_constraint(
        self,
        proposed_query: str,
        reasoning_state: Dict[str, Any],
        previous_answers: Dict[str, Any],
        plan_goal: Optional[str] = None
    ) -> RegulatorConstraint:
        """
        Apply confidence-based diffusion control constraint.
        
        Adjusts constraint strength based on uncertainty/diffusion coefficient.
        Higher uncertainty requires stronger constraints.
        """
        # Get entropy and diffusion from reasoning state
        entropy = reasoning_state.get("entropy", 0.0)
        diffusion_coefficient = reasoning_state.get("diffusion_coefficient", 0.0)
        confidence = reasoning_state.get("confidence", 0.5)
        
        # Calculate uncertainty
        uncertainty = 1.0 - confidence
        
        # Adjust weight based on uncertainty
        # Higher uncertainty → higher weight (stronger constraint needed)
        adjusted_weight = self.weight * (1.0 + uncertainty)
        adjusted_weight = min(1.0, adjusted_weight)
        
        # Determine if diffusion is high (needs control)
        high_diffusion = diffusion_coefficient > 0.5
        high_entropy = entropy > 1.0
        
        return RegulatorConstraint(
            regulator_name=self.name,
            constraint_type="diffusion_control",
            weight=adjusted_weight,
            parameters={
                "entropy": entropy,
                "diffusion_coefficient": diffusion_coefficient,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "high_diffusion": high_diffusion,
                "high_entropy": high_entropy,
                "needs_control": high_diffusion or high_entropy
            }
        )
    
    def check_violation(
        self,
        query: str,
        constraint: RegulatorConstraint,
        current_state: Dict[str, Any]
    ) -> bool:
        """Check if query violates confidence/diffusion constraint."""
        needs_control = constraint.parameters.get("needs_control", False)
        if not needs_control:
            return False
        
        # If high diffusion/entropy, query should be more specific
        # Violation: query is too vague/generic
        query_lower = query.lower()
        
        # Check for vague terms
        vague_terms = ["what", "which", "where", "when", "who", "how"]
        has_vague = any(term in query_lower.split()[:3] for term in vague_terms)
        
        # Check for specificity (has specific terms, not just questions)
        specific_indicators = ["name", "type", "location", "date", "number", "value"]
        has_specific = any(indicator in query_lower for indicator in specific_indicators)
        
        # Violation: high diffusion but query is vague and not specific
        return needs_control and has_vague and not has_specific