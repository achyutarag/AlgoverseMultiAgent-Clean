# agents/tests/test_entropy_physics.py
"""
Physics tests for EntropyTracker: entropy, diffusion, and drift calculations.
No LLMs, no FAISS - pure mathematical checks.
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.entropy_tracker import EntropyTracker, EntropyState


class TestEntropyCalculations:
    """Test 1: Uniform vs. peaked distributions"""
    
    def test_uniform_distribution_has_higher_entropy(self):
        """Uniform distribution should have higher entropy than peaked."""
        tracker = EntropyTracker()
        
        # Uniform: 4 equal probabilities
        uniform_dist = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        entropy_uniform = tracker.calculate_entropy(uniform_dist)
        
        # Peaked: one dominant probability
        peaked_dist = {"A": 0.9, "B": 0.1}
        entropy_peaked = tracker.calculate_entropy(peaked_dist)
        
        assert entropy_uniform > entropy_peaked, \
            f"Uniform entropy ({entropy_uniform:.3f}) should be > peaked ({entropy_peaked:.3f})"
    
    def test_single_value_has_zero_entropy(self):
        """Single value distribution should have zero entropy (certainty)."""
        tracker = EntropyTracker()
        
        single_dist = {"A": 1.0}
        entropy = tracker.calculate_entropy(single_dist)
        
        assert entropy == 0.0, f"Single value should have zero entropy, got {entropy:.3f}"
    
    def test_empty_distribution_has_zero_entropy(self):
        """Empty distribution should have zero entropy."""
        tracker = EntropyTracker()
        
        empty_dist = {}
        entropy = tracker.calculate_entropy(empty_dist)
        
        assert entropy == 0.0, f"Empty distribution should have zero entropy, got {entropy:.3f}"


class TestDiffusionCoefficient:
    """Test 2: Diffusion coefficient D(t) = f(1 - confidence, hop)"""
    
    def test_low_confidence_has_higher_diffusion(self):
        """Lower confidence should result in higher diffusion coefficient."""
        tracker = EntropyTracker()
        
        # Low confidence
        diffusion_low = tracker.calculate_diffusion_coefficient(
            confidence=0.2,
            hop=1,
            base_diffusion=0.1
        )
        
        # High confidence
        diffusion_high = tracker.calculate_diffusion_coefficient(
            confidence=0.9,
            hop=1,
            base_diffusion=0.1
        )
        
        assert diffusion_low > diffusion_high, \
            f"Low confidence diffusion ({diffusion_low:.3f}) should be > high ({diffusion_high:.3f})"
    
    def test_later_hops_have_higher_diffusion(self):
        """Later hops should have higher base diffusion."""
        tracker = EntropyTracker()
        
        # Early hop
        diffusion_hop1 = tracker.calculate_diffusion_coefficient(
            confidence=0.5,
            hop=1,
            base_diffusion=0.1
        )
        
        # Later hop
        diffusion_hop3 = tracker.calculate_diffusion_coefficient(
            confidence=0.5,
            hop=3,
            base_diffusion=0.1
        )
        
        assert diffusion_hop3 > diffusion_hop1, \
            f"Hop 3 diffusion ({diffusion_hop3:.3f}) should be > hop 1 ({diffusion_hop1:.3f})"
    
    def test_diffusion_capped_at_one(self):
        """Diffusion coefficient should be capped at 1.0."""
        tracker = EntropyTracker()
        
        # Extreme case: very low confidence, high hop
        diffusion = tracker.calculate_diffusion_coefficient(
            confidence=0.01,
            hop=10,
            base_diffusion=0.1
        )
        
        assert diffusion <= 1.0, f"Diffusion should be capped at 1.0, got {diffusion:.3f}"


class TestDriftCalculation:
    """Test 3: Drift between distributions ||P(t) - P(t-1)||"""
    
    def test_identical_distributions_have_zero_drift(self):
        """Identical distributions should have zero drift."""
        tracker = EntropyTracker()
        
        dist1 = {"A": 0.5, "B": 0.5}
        dist2 = {"A": 0.5, "B": 0.5}
        
        drift = tracker.calculate_drift(dist1, dist2)
        
        assert abs(drift) < 1e-6, f"Identical distributions should have zero drift, got {drift:.6f}"
    
    def test_different_distributions_have_positive_drift(self):
        """Different distributions should have positive drift."""
        tracker = EntropyTracker()
        
        # t=1: all probability on A
        dist1 = {"A": 1.0}
        
        # t=2: split between A and B
        dist2 = {"A": 0.5, "B": 0.5}
        
        drift = tracker.calculate_drift(dist2, dist1)
        
        assert drift > 0.0, f"Different distributions should have positive drift, got {drift:.3f}"
        # L2 norm: sqrt((0.5-1.0)^2 + (0.5-0.0)^2) = sqrt(0.25 + 0.25) = sqrt(0.5) ≈ 0.707
        assert abs(drift - 0.707) < 0.1, f"Expected drift ≈ 0.707, got {drift:.3f}"
    
    def test_empty_previous_distribution_has_zero_drift(self):
        """Drift from empty previous distribution should be zero."""
        tracker = EntropyTracker()
        
        current = {"A": 0.5, "B": 0.5}
        previous = {}
        
        drift = tracker.calculate_drift(current, previous)
        
        assert drift == 0.0, f"Drift from empty previous should be zero, got {drift:.3f}"


class TestEntropyStateUpdate:
    """Test 4: Entropy state update and tracking"""
    
    def test_update_entropy_state_integrates_all_components(self):
        """update_entropy_state should integrate entropy, diffusion, and drift."""
        tracker = EntropyTracker()
        
        # First state
        state1 = tracker.update_entropy_state(
            hop=1,
            entity_distribution={"A": 1.0},
            confidence=0.9,
            previous_state=None
        )
        
        assert state1.entropy >= 0.0
        assert state1.diffusion_coefficient >= 0.0
        assert state1.confidence == 0.9
        assert state1.drift_from_previous == 0.0  # No previous state
        
        # Second state (different distribution)
        state2 = tracker.update_entropy_state(
            hop=2,
            entity_distribution={"A": 0.5, "B": 0.5},
            confidence=0.7,
            previous_state=state1
        )
        
        assert state2.entropy > state1.entropy  # More uncertainty
        assert state2.drift_from_previous > 0.0  # Drift detected
        assert state2.hop == 2
    
    def test_entropy_history_tracks_states(self):
        """Entropy history should accumulate states."""
        tracker = EntropyTracker()
        
        # Add multiple states
        for hop in range(1, 4):
            tracker.update_entropy_state(
                hop=hop,
                entity_distribution={"A": 0.5, "B": 0.5},
                confidence=0.8,
                previous_state=tracker.get_current_state()
            )
        
        history = tracker.get_entropy_history()
        assert len(history) == 3, f"Expected 3 states in history, got {len(history)}"
        assert history[-1].hop == 3


class TestEarlyTermination:
    """Test 5: Early termination logic"""
    
    def test_early_termination_with_low_entropy_high_confidence(self):
        """Should terminate early when entropy is low, confidence is high, drift is low."""
        tracker = EntropyTracker()
        
        # Create a state with low entropy, high confidence, low drift
        state = EntropyState(
            hop=2,
            entropy=0.1,  # Low entropy = high certainty
            diffusion_coefficient=0.1,
            confidence=0.9,  # High confidence
            entity_distribution={"A": 0.95, "B": 0.05},
            drift_from_previous=0.05  # Low drift = stable
        )
        
        should_terminate = tracker.should_terminate_early(
            current_state=state,
            plan_goal="What is A?",
            min_confidence=0.8
        )
        
        assert should_terminate is True, \
            "Should terminate early with low entropy, high confidence, low drift"
    
    def test_no_early_termination_with_high_entropy(self):
        """Should NOT terminate early when entropy is high."""
        tracker = EntropyTracker()
        
        # Create a state with high entropy
        state = EntropyState(
            hop=2,
            entropy=2.0,  # High entropy = high uncertainty
            diffusion_coefficient=0.8,
            confidence=0.3,  # Low confidence
            entity_distribution={"A": 0.33, "B": 0.33, "C": 0.34},
            drift_from_previous=0.5  # High drift
        )
        
        should_terminate = tracker.should_terminate_early(
            current_state=state,
            plan_goal="What is A?",
            min_confidence=0.8
        )
        
        assert should_terminate is False, \
            "Should NOT terminate early with high entropy, low confidence, high drift"
    
    def test_no_early_termination_with_high_drift(self):
        """Should NOT terminate early when drift is high (unstable)."""
        tracker = EntropyTracker()
        
        # Create a state with high drift
        state = EntropyState(
            hop=2,
            entropy=0.2,  # Low entropy
            diffusion_coefficient=0.2,
            confidence=0.9,  # High confidence
            entity_distribution={"A": 0.9, "B": 0.1},
            drift_from_previous=0.8  # High drift = unstable
        )
        
        should_terminate = tracker.should_terminate_early(
            current_state=state,
            plan_goal="What is A?",
            min_confidence=0.8
        )
        
        assert should_terminate is False, \
            "Should NOT terminate early with high drift (unstable state)"

