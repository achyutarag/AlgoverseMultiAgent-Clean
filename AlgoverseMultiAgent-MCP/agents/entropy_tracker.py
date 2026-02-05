"""
✅ EXPERIMENT 3: DEPRECATED - EntropyTracker removed
This file is kept for reference only. Entropy tracking has been removed
and replaced with simple heuristics (confidence-based compression, 
belief count checks, etc.)

Original purpose:
EntropyTracker: Tracks uncertainty and diffusion in multi-hop reasoning.

Models reasoning as a diffusion process where:
- Entropy H(t) measures uncertainty in beliefs
- Diffusion coefficient D(t) measures how fast uncertainty spreads
- Drift ||P(t) - P(t-1)|| measures how much beliefs changed
"""
import math
from typing import Dict, Optional, List
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class EntropyState(BaseModel):
    """State of entropy at a given reasoning hop."""
    hop: int
    entropy: float = Field(ge=0.0)  # H(t) - Shannon entropy
    diffusion_coefficient: float = Field(ge=0.0, le=1.0)  # D(t) - diffusion rate
    confidence: float = Field(ge=0.0, le=1.0)  # Confidence in current beliefs
    entity_distribution: Dict[str, float] = Field(default_factory=dict)  # P(interpretation)
    drift_from_previous: float = Field(default=0.0, ge=0.0)  # ||P(t) - P(t-1)||


class EntropyTracker:
    """
    Tracks entropy, diffusion, and drift across reasoning hops.
    
    Implements:
    - Shannon entropy: H = -Σ p_i * log2(p_i)
    - Diffusion coefficient: D(t) = base_diffusion * (1 - confidence) * (1 + hop * hop_factor)
    - Drift: L2 norm of distribution difference
    """
    
    def __init__(self, base_diffusion: float = 0.1, hop_factor: float = 0.1):
        """
        Initialize entropy tracker.
        
        Args:
            base_diffusion: Base diffusion coefficient
            hop_factor: Factor by which diffusion increases per hop
        """
        self.base_diffusion = base_diffusion
        self.hop_factor = hop_factor
        self.entropy_history: List[EntropyState] = []
        self.current_state: Optional[EntropyState] = None
    
    def calculate_entropy(self, distribution: Dict[str, float]) -> float:
        """
        Calculate Shannon entropy: H = -Σ p_i * log2(p_i)
        
        Args:
            distribution: Dictionary mapping entities/interpretations to probabilities
            
        Returns:
            Entropy value (bits). Higher = more uncertainty.
        """
        if not distribution:
            return 0.0
        
        # Normalize probabilities (in case they don't sum to 1)
        total = sum(distribution.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for prob in distribution.values():
            if prob > 0:
                normalized_prob = prob / total
                entropy -= normalized_prob * math.log2(normalized_prob)
        
        return entropy
    
    def calculate_diffusion_coefficient(
        self,
        confidence: float,
        hop: int,
        base_diffusion: Optional[float] = None
    ) -> float:
        """
        Calculate diffusion coefficient: D(t) = base * (1 - confidence) * (1 + hop * factor)
        
        Higher diffusion = faster uncertainty spread = harder to stabilize.
        
        Args:
            confidence: Confidence in current beliefs [0, 1]
            hop: Current reasoning hop number
            base_diffusion: Override base diffusion (uses self.base_diffusion if None)
            
        Returns:
            Diffusion coefficient [0, 1]
        """
        if base_diffusion is None:
            base_diffusion = self.base_diffusion
        
        # Diffusion increases with (1 - confidence) and with hop number
        diffusion = base_diffusion * (1.0 - confidence) * (1.0 + hop * self.hop_factor)
        
        # Cap at 1.0
        return min(diffusion, 1.0)
    
    def calculate_drift(
        self,
        current_dist: Dict[str, float],
        previous_dist: Dict[str, float]
    ) -> float:
        """
        Calculate L2 norm drift: ||P(t) - P(t-1)||
        
        Measures how much the belief distribution changed between hops.
        
        Args:
            current_dist: Current belief distribution
            previous_dist: Previous belief distribution
            
        Returns:
            L2 norm drift (non-negative)
        """
        if not previous_dist:
            return 0.0
        
        # Get all unique keys from both distributions
        all_keys = set(current_dist.keys()) | set(previous_dist.keys())
        
        if not all_keys:
            return 0.0
        
        # Normalize both distributions
        current_total = sum(current_dist.values())
        previous_total = sum(previous_dist.values())
        
        if current_total == 0 or previous_total == 0:
            return 0.0
        
        # Calculate L2 norm
        drift_squared = 0.0
        for key in all_keys:
            current_prob = current_dist.get(key, 0.0) / current_total
            previous_prob = previous_dist.get(key, 0.0) / previous_total
            diff = current_prob - previous_prob
            drift_squared += diff * diff
        
        return math.sqrt(drift_squared)
    
    def update_entropy_state(
        self,
        hop: int,
        entity_distribution: Dict[str, float],
        confidence: float,
        previous_state: Optional[EntropyState] = None
    ) -> EntropyState:
        """
        Update entropy state for a new reasoning hop.
        
        Args:
            hop: Current hop number
            entity_distribution: Current belief distribution
            confidence: Confidence in current beliefs
            previous_state: Previous entropy state (for drift calculation)
            
        Returns:
            New EntropyState
        """
        # Calculate entropy
        entropy = self.calculate_entropy(entity_distribution)
        
        # Calculate diffusion coefficient
        diffusion = self.calculate_diffusion_coefficient(confidence, hop)
        
        # Calculate drift from previous state
        drift = 0.0
        if previous_state:
            drift = self.calculate_drift(entity_distribution, previous_state.entity_distribution)
        
        # Create new state
        state = EntropyState(
            hop=hop,
            entropy=entropy,
            diffusion_coefficient=diffusion,
            confidence=confidence,
            entity_distribution=entity_distribution.copy(),
            drift_from_previous=drift
        )
        
        # Update history
        self.entropy_history.append(state)
        self.current_state = state
        
        return state
    
    def get_current_state(self) -> Optional[EntropyState]:
        """Get the current entropy state."""
        return self.current_state
    
    def get_entropy_history(self) -> List[EntropyState]:
        """Get all entropy states in history."""
        return self.entropy_history.copy()
    
    def should_terminate_early(
        self,
        current_state: EntropyState,
        plan_goal: str,
        min_confidence: float = 0.8
    ) -> bool:
        """
        Determine if reasoning should terminate early.
        
        Early termination conditions:
        - Low entropy (high certainty)
        - High confidence (above threshold)
        - Low drift (stable beliefs)
        
        Args:
            current_state: Current entropy state
            plan_goal: Goal of the reasoning plan
            min_confidence: Minimum confidence threshold for early termination
            
        Returns:
            True if should terminate early, False otherwise
        """
        # Must have high confidence
        if current_state.confidence < min_confidence:
            return False
        
        # Must have low entropy (high certainty)
        # Threshold: entropy < 0.5 (roughly equivalent to 70%+ probability on one interpretation)
        if current_state.entropy >= 0.5:
            return False
        
        # Must have low drift (stable)
        # Threshold: drift < 0.3 (beliefs haven't changed much)
        if current_state.drift_from_previous >= 0.3:
            return False
        
        # All conditions met - can terminate early
        return True