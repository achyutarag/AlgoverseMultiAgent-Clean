# agents/reasoning_flow.py
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)

class ReasoningState(BaseModel):
    """State of reasoning at a given hop."""
    hop: int
    beliefs: Dict[str, float] = Field(default_factory=dict)  # P(interpretation)
    entity_anchors: Dict[str, Any] = Field(default_factory=dict)  # Fixed anchors
    relation_direction: Optional[str] = None  # Direction of relational flow
    evidence_terms: List[str] = Field(default_factory=list)  # Key evidence terms
    plan_alignment: float = Field(default=0.0, ge=0.0, le=1.0)  # Alignment with goal
    # NEW: Entropy integration
    entropy: float = Field(default=0.0, ge=0.0)  # H(t) from entropy tracker
    diffusion_coefficient: float = Field(default=0.0, ge=0.0)  # D(t) from entropy tracker
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)  # Confidence from entropy tracker
    timestamp: float = Field(default_factory=time.time)

class BucketAnchor(BaseModel):
    """Fixed anchor point that prevents reasoning drift."""
    anchor_type: str  # "entity", "relation", "evidence", "plan"
    value: Any
    hop_created: int
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    context: Dict[str, Any] = Field(default_factory=dict)

class FlowSnapshot(BaseModel):
    """
    Unified flow snapshot for regulators and retrieval.
    
    This is the ONE structure that regulators use instead of
    accessing 5 separate fields. Plugged into stabilize_query()
    and entropy_aware_retrieve().
    """
    hop: int
    beliefs: Dict[str, float]  # P(interpretation) - corrected by anchors
    anchors: List[BucketAnchor]  # All bucket anchors
    entity_anchors: Dict[str, Any]  # Entity-specific anchors
    relation_direction: Optional[str]
    evidence_terms: List[str]
    plan_alignment: float
    # Entropy/diffusion state
    entropy: float  # H(t)
    diffusion_coefficient: float  # D(t)
    confidence: float
    drift_from_previous: float
    # Metadata
    timestamp: float

class ReasoningFlowIndex:
    """
    Structured memory that stores evolving reasoning state at each hop.
    
    Now acts as a CONTROLLER (not just memory) by:
    - Integrating entropy tracking (H(t), D(t))
    - Applying anchor corrections (potential wells)
    - Exporting unified flow snapshots for regulators
    """
    
    def __init__(self, max_history: int = 50, entropy_tracker=None):
        """
        Initialize the reasoning flow index.
        
        Args:
            max_history: Maximum number of reasoning states to keep
            entropy_tracker: EntropyTracker instance for integration
        """
        self.max_history = max_history
        self.flow_states: deque = deque(maxlen=max_history)
        self.bucket_anchors: List[BucketAnchor] = []
        self.entropy_tracker = entropy_tracker  # NEW: Integration point
        logger.info("Reasoning Flow Index initialized")
    
    def add_state(
        self,
        hop: int,
        beliefs: Dict[str, float],
        entity_anchors: Optional[Dict[str, Any]] = None,
        relation_direction: Optional[str] = None,
        evidence_terms: Optional[List[str]] = None,
        plan_alignment: float = 0.0,
        confidence: float = 0.5
    ) -> ReasoningState:
        """
        Add a new reasoning state for the current hop.
        
        NOW INTEGRATES WITH ENTROPY TRACKER:
        - Gets H(t) and D(t) from entropy tracker
        - Applies anchor corrections to beliefs (potential wells)
        - Creates unified state snapshot
        
        Args:
            hop: Current hop number
            beliefs: Probability distribution over interpretations
            entity_anchors: Fixed entity anchors
            relation_direction: Direction of relational flow
            evidence_terms: Key evidence terms found
            plan_alignment: Alignment with plan goal (0.0 to 1.0)
            confidence: Confidence in current state
            
        Returns:
            New ReasoningState with entropy integration
        """
        # NEW: Apply anchor corrections (potential wells)
        corrected_beliefs = self.apply_anchor_correction(beliefs.copy())
        
        # ✅ EXPERIMENT 3: Removed entropy tracking - using simple confidence instead
        # entropy_state = None
        # if self.entropy_tracker:
        #     previous_state = self.entropy_tracker.get_current_state()
        #     entropy_state = self.entropy_tracker.update_entropy_state(...)
        
        # Create reasoning state (without entropy integration)
        state = ReasoningState(
            hop=hop,
            beliefs=corrected_beliefs,  # Use corrected beliefs
            entity_anchors=(entity_anchors or {}).copy(),
            relation_direction=relation_direction,
            evidence_terms=(evidence_terms or []).copy(),
            plan_alignment=plan_alignment,
            # ✅ EXPERIMENT 3: Removed entropy tracking - set to 0.0
            entropy=0.0,
            diffusion_coefficient=0.0,
            confidence=confidence
        )
        
        self.flow_states.append(state)
        
        logger.debug(
            f"Hop {hop}: Added reasoning state with {len(corrected_beliefs)} beliefs, "
            f"confidence={confidence:.3f}, alignment={plan_alignment:.2f}"
        )
        
        return state
    
    def apply_anchor_correction(
        self,
        beliefs: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply anchor corrections to beliefs (potential well physics).
        
        Anchors act as potential wells: beliefs at anchor points
        are reinforced by (1 + anchor.strength), preventing drift.
        
        This is the DIFFUSION + anchor feedback mechanism that
        turns anchors from passive storage into active controllers.
        
        Args:
            beliefs: Original belief distribution
            
        Returns:
            Corrected belief distribution
        """
        corrected_beliefs = beliefs.copy()
        
        # Apply each anchor as a potential well
        for anchor in self.bucket_anchors:
            anchor_value = str(anchor.value)  # Convert to string for key matching
            
            # If anchor value exists in beliefs, reinforce it
            if anchor_value in corrected_beliefs:
                # Potential well: multiply by (1 + strength)
                # Stronger anchors = stronger reinforcement
                corrected_beliefs[anchor_value] = corrected_beliefs[anchor_value] * (1.0 + anchor.strength)
                logger.debug(
                    f"Anchor correction applied: {anchor.anchor_type}={anchor_value}, "
                    f"strength={anchor.strength:.2f}, "
                    f"belief={corrected_beliefs[anchor_value]:.3f}"
                )
            else:
                # If anchor not in beliefs, add it with base probability
                # This creates new anchor points in the distribution
                corrected_beliefs[anchor_value] = 0.1 * anchor.strength
        
        # Normalize to maintain probability distribution
        total = sum(corrected_beliefs.values())
        if total > 0:
            corrected_beliefs = {
                k: v / total
                for k, v in corrected_beliefs.items()
            }
        
        return corrected_beliefs
    
    def get_flow_snapshot(self) -> Optional[FlowSnapshot]:
        """
        Get unified flow snapshot for regulators and retrieval.
        
        This is the ONE structure that regulators use instead of
        accessing 5 separate fields. Plugged into:
        - stabilize_query() in RegulatorManager
        - entropy_aware_retrieve() in State Manager
        
        Returns:
            FlowSnapshot with all state unified, or None if no state
        """
        current_state = self.get_current_state()
        if not current_state:
            return None
        
        # ✅ EXPERIMENT 3: Removed entropy tracking - no drift calculation
        drift = 0.0  # Always 0.0 since entropy tracking removed
        
        # Build entity anchors dict from bucket anchors
        entity_anchors_dict = {}
        for anchor in self.get_anchors_by_type("entity"):
            entity_anchors_dict[str(anchor.value)] = {
                "strength": anchor.strength,
                "hop": anchor.hop_created,
                "context": anchor.context
            }
        
        snapshot = FlowSnapshot(
            hop=current_state.hop,
            beliefs=current_state.beliefs.copy(),  # Already corrected by anchors
            anchors=self.bucket_anchors.copy(),
            entity_anchors=entity_anchors_dict,
            relation_direction=current_state.relation_direction,
            evidence_terms=current_state.evidence_terms.copy(),
            plan_alignment=current_state.plan_alignment,
            # Entropy/diffusion from tracker
            entropy=current_state.entropy,
            diffusion_coefficient=current_state.diffusion_coefficient,
            confidence=current_state.confidence,
            drift_from_previous=drift,
            timestamp=current_state.timestamp
        )
        
        logger.debug(
            f"Flow snapshot exported: hop={snapshot.hop}, "
            f"H(t)={snapshot.entropy:.3f}, D(t)={snapshot.diffusion_coefficient:.3f}, "
            f"anchors={len(snapshot.anchors)}, beliefs={len(snapshot.beliefs)}"
        )
        
        return snapshot
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export current state as dictionary (for compatibility).
        
        Returns unified state dictionary that can be passed to regulators.
        """
        snapshot = self.get_flow_snapshot()
        if not snapshot:
            return {}
        
        return {
            "beliefs": snapshot.beliefs,
            "anchors": [anchor.dict() for anchor in snapshot.anchors],
            "entity_anchors": snapshot.entity_anchors,
            "relation_direction": snapshot.relation_direction,
            "evidence_terms": snapshot.evidence_terms,
            "plan_alignment": snapshot.plan_alignment,
            "entropy": snapshot.entropy,
            "diffusion_coefficient": snapshot.diffusion_coefficient,
            "confidence": snapshot.confidence,
            "drift_from_previous": snapshot.drift_from_previous,
            "hop": snapshot.hop
        }
    
    # ... (keep all existing methods: get_current_state, get_previous_state, 
    #      add_bucket_anchor, get_anchors_by_type, get_entity_anchors, 
    #      get_flow_history, detect_drift)
    
    def get_current_state(self) -> Optional[ReasoningState]:
        """Get the most recent reasoning state."""
        return self.flow_states[-1] if self.flow_states else None
    
    def get_previous_state(self) -> Optional[ReasoningState]:
        """Get the previous reasoning state."""
        return self.flow_states[-2] if len(self.flow_states) >= 2 else None
    
    def add_bucket_anchor(
        self,
        anchor_type: str,
        value: Any,
        hop: int,
        strength: float = 1.0,
        context: Optional[Dict[str, Any]] = None
    ) -> BucketAnchor:
        """Add a bucket anchor to prevent reasoning drift."""
        anchor = BucketAnchor(
            anchor_type=anchor_type,
            value=value,
            hop_created=hop,
            strength=strength,
            context=(context or {}).copy()
        )
        
        self.bucket_anchors.append(anchor)
        
        logger.debug(
            f"Added bucket anchor: {anchor_type}={value} at hop {hop}, "
            f"strength={strength:.2f}"
        )
        
        return anchor
    
    def get_anchors_by_type(self, anchor_type: str) -> List[BucketAnchor]:
        """Get all anchors of a specific type."""
        return [a for a in self.bucket_anchors if a.anchor_type == anchor_type]
    
    def get_entity_anchors(self) -> List[Any]:
        """Get all entity anchor values."""
        return [a.value for a in self.get_anchors_by_type("entity")]
    
    def get_flow_history(self) -> List[ReasoningState]:
        """Get all reasoning states."""
        return list(self.flow_states)
    
    def detect_drift(self, threshold: float = 0.3) -> bool:
        """Detect if reasoning has drifted significantly."""
        if len(self.flow_states) < 2:
            return False
        
        current = self.get_current_state()
        previous = self.get_previous_state()
        
        if not current or not previous:
            return False
        
        # Calculate belief drift
        all_keys = set(current.beliefs.keys()) | set(previous.beliefs.keys())
        drift_sum = 0.0
        
        for key in all_keys:
            current_prob = current.beliefs.get(key, 0.0)
            previous_prob = previous.beliefs.get(key, 0.0)
            drift_sum += abs(current_prob - previous_prob)
        
        drift_magnitude = drift_sum / max(len(all_keys), 1)
        
        is_drift = drift_magnitude > threshold
        
        if is_drift:
            logger.warning(
                f"Drift detected at hop {current.hop}: "
                f"magnitude={drift_magnitude:.3f} > threshold={threshold}"
            )
        
        return is_drift