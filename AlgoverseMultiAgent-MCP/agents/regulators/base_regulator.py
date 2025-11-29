# agents/regulators/base_regulator.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class RegulatorConstraint(BaseModel):
    """Constraint applied by a regulator."""
    regulator_name: str
    constraint_type: str  # "dirichlet", "neumann", "potential_well", "boundary"
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    parameters: Dict[str, Any] = Field(default_factory=dict)

class BaseRegulator(ABC):
    """
    Base class for all regulators in the diffusion-aware retrieval system.
    
    Regulators stabilize reasoning flow by applying boundary constraints
    that prevent drift, anchor entities, and guide retrieval.
    """
    
    def __init__(self, name: str, weight: float = 1.0):
        """
        Initialize the regulator.
        
        Args:
            name: Name of the regulator
            weight: Weight/strength of the regulator (0.0 to 1.0)
        """
        self.name = name
        self.weight = max(0.0, min(1.0, weight))
        logger.debug(f"Initialized {self.name} regulator with weight {self.weight}")
    
    @abstractmethod
    def apply_constraint(
        self,
        proposed_query: str,
        reasoning_state: Dict[str, Any],
        previous_answers: Dict[str, Any],
        plan_goal: Optional[str] = None
    ) -> RegulatorConstraint:
        """
        Apply regulator constraint to stabilize the query.
        
        Args:
            proposed_query: The raw query proposed by Step Definer
            reasoning_state: Current reasoning flow state (entropy, beliefs, etc.)
            previous_answers: Answers from previous steps
            plan_goal: The overall plan goal/question
            
        Returns:
            RegulatorConstraint with constraint details
        """
        pass
    
    @abstractmethod
    def check_violation(
        self,
        query: str,
        constraint: RegulatorConstraint,
        current_state: Dict[str, Any]
    ) -> bool:
        """
        Check if a query violates this regulator's constraint.
        
        Args:
            query: Query to check
            constraint: The constraint to check against
            current_state: Current reasoning state
            
        Returns:
            True if constraint is violated, False otherwise
        """
        pass
    
    def get_stabilized_query(
        self,
        proposed_query: str,
        reasoning_state: Dict[str, Any],
        previous_answers: Dict[str, Any],
        plan_goal: Optional[str] = None
    ) -> str:
        """
        Get stabilized version of the query after applying regulator.
        
        Default implementation returns the proposed query.
        Subclasses can override to modify the query.
        
        Args:
            proposed_query: Raw query from Step Definer
            reasoning_state: Current reasoning state
            previous_answers: Previous step answers
            plan_goal: Overall plan goal
            
        Returns:
            Stabilized query string
        """
        constraint = self.apply_constraint(
            proposed_query, reasoning_state, previous_answers, plan_goal
        )
        
        # Default: return original query (subclasses can override)
        return proposed_query