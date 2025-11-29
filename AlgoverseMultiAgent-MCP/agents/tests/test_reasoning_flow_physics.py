# agents/tests/test_reasoning_flow_physics.py
"""
Physics tests for ReasoningFlowIndex: anchor corrections and flow snapshots.
No LLMs, no FAISS - pure state management checks.
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.reasoning_flow import ReasoningFlowIndex, BucketAnchor
from agents.entropy_tracker import EntropyTracker


class TestAnchorCorrection:
    """Test 4: Anchor correction acts like a potential well"""
    
    def test_anchor_reinforces_belief(self):
        """Anchors should reinforce beliefs (potential well physics)."""
        flow = ReasoningFlowIndex()
        
        # Initial beliefs
        beliefs = {"A": 0.4, "B": 0.6}
        
        # Add anchor on "A" with strength 1.0
        flow.add_bucket_anchor(
            anchor_type="entity",
            value="A",
            hop=1,
            strength=1.0
        )
        
        # Apply anchor correction
        corrected = flow.apply_anchor_correction(beliefs)
        
        # "A" should be reinforced (multiplied by 1 + strength = 2.0)
        assert corrected["A"] > 0.4, \
            f"Anchor should reinforce A: {corrected['A']:.3f} > 0.4"
        
        # Distribution should still sum to 1.0 (normalized)
        total = sum(corrected.values())
        assert abs(total - 1.0) < 1e-6, \
            f"Corrected beliefs should sum to 1.0, got {total:.6f}"
    
    def test_anchor_creates_new_belief_point(self):
        """Anchors not in beliefs should create new belief points."""
        flow = ReasoningFlowIndex()
        
        # Initial beliefs (no "C")
        beliefs = {"A": 0.5, "B": 0.5}
        
        # Add anchor on "C" (not in beliefs)
        flow.add_bucket_anchor(
            anchor_type="entity",
            value="C",
            hop=1,
            strength=0.8
        )
        
        # Apply anchor correction
        corrected = flow.apply_anchor_correction(beliefs)
        
        # "C" should appear in corrected beliefs
        assert "C" in corrected, "Anchor should create new belief point for C"
        assert corrected["C"] > 0.0, "New anchor point should have positive probability"
        
        # Should still be normalized
        total = sum(corrected.values())
        assert abs(total - 1.0) < 1e-6, f"Beliefs should sum to 1.0, got {total:.6f}"
    
    def test_multiple_anchors_combine(self):
        """Multiple anchors should combine their effects."""
        flow = ReasoningFlowIndex()
        
        beliefs = {"A": 0.33, "B": 0.33, "C": 0.34}
        
        # Add two anchors
        flow.add_bucket_anchor("entity", "A", hop=1, strength=0.5)
        flow.add_bucket_anchor("entity", "B", hop=1, strength=0.5)
        
        corrected = flow.apply_anchor_correction(beliefs)
        
        # Both A and B should be reinforced
        assert corrected["A"] > 0.33, "Anchor A should be reinforced"
        assert corrected["B"] > 0.33, "Anchor B should be reinforced"
        
        # Still normalized
        total = sum(corrected.values())
        assert abs(total - 1.0) < 1e-6, f"Beliefs should sum to 1.0, got {total:.6f}"
    
    def test_stronger_anchor_has_greater_effect(self):
        """Stronger anchors should have greater reinforcement effect."""
        flow1 = ReasoningFlowIndex()
        flow2 = ReasoningFlowIndex()
        
        beliefs = {"A": 0.5, "B": 0.5}
        
        # Weak anchor
        flow1.add_bucket_anchor("entity", "A", hop=1, strength=0.3)
        weak_corrected = flow1.apply_anchor_correction(beliefs.copy())
        
        # Strong anchor
        flow2.add_bucket_anchor("entity", "A", hop=1, strength=0.9)
        strong_corrected = flow2.apply_anchor_correction(beliefs.copy())
        
        # Strong anchor should reinforce more
        assert strong_corrected["A"] > weak_corrected["A"], \
            f"Strong anchor ({strong_corrected['A']:.3f}) should reinforce more than weak ({weak_corrected['A']:.3f})"


class TestFlowSnapshot:
    """Test 5: Flow snapshot export"""
    
    def test_flow_snapshot_contains_all_fields(self):
        """Flow snapshot should contain all required fields."""
        entropy_tracker = EntropyTracker()
        flow = ReasoningFlowIndex(entropy_tracker=entropy_tracker)
        
        # Add state
        flow.add_state(
            hop=1,
            beliefs={"A": 0.6, "B": 0.4},
            entity_anchors={"A": {"strength": 0.8}},
            relation_direction="contains",
            evidence_terms=["term1", "term2"],
            plan_alignment=0.7,
            confidence=0.8
        )
        
        # Add anchor
        flow.add_bucket_anchor("entity", "A", hop=1, strength=0.8)
        
        # Get snapshot
        snapshot = flow.get_flow_snapshot()
        
        assert snapshot is not None, "Snapshot should not be None"
        assert snapshot.hop == 1
        assert "A" in snapshot.beliefs
        assert len(snapshot.anchors) == 1
        assert "A" in snapshot.entity_anchors
        assert snapshot.relation_direction == "contains"
        assert "term1" in snapshot.evidence_terms
        assert snapshot.plan_alignment == 0.7
        assert snapshot.entropy >= 0.0
        assert snapshot.diffusion_coefficient >= 0.0
        assert snapshot.confidence == 0.8
        assert snapshot.drift_from_previous >= 0.0
    
    def test_flow_snapshot_none_when_no_state(self):
        """Flow snapshot should be None when no state exists."""
        flow = ReasoningFlowIndex()
        
        snapshot = flow.get_flow_snapshot()
        
        assert snapshot is None, "Snapshot should be None when no state exists"
    
    def test_flow_snapshot_integrates_entropy_tracker(self):
        """Flow snapshot should integrate entropy tracker data."""
        entropy_tracker = EntropyTracker()
        flow = ReasoningFlowIndex(entropy_tracker=entropy_tracker)
        
        # Add state (this triggers entropy tracker update)
        flow.add_state(
            hop=1,
            beliefs={"A": 0.5, "B": 0.5},
            confidence=0.7
        )
        
        snapshot = flow.get_flow_snapshot()
        
        # Entropy should be calculated (uniform distribution = high entropy)
        assert snapshot.entropy > 0.0, "Entropy should be calculated"
        assert snapshot.diffusion_coefficient > 0.0, "Diffusion coefficient should be calculated"
        assert snapshot.confidence == 0.7, "Confidence should be preserved"


class TestStateManagement:
    """Test state addition and history"""
    
    def test_add_state_updates_current_state(self):
        """Adding state should update current state."""
        entropy_tracker = EntropyTracker()
        flow = ReasoningFlowIndex(entropy_tracker=entropy_tracker)
        
        state = flow.add_state(
            hop=1,
            beliefs={"A": 0.5, "B": 0.5},
            confidence=0.8
        )
        
        current = flow.get_current_state()
        assert current is not None
        assert current.hop == 1
        assert current.beliefs == state.beliefs
    
    def test_multiple_states_create_history(self):
        """Multiple states should create history."""
        entropy_tracker = EntropyTracker()
        flow = ReasoningFlowIndex(entropy_tracker=entropy_tracker)
        
        flow.add_state(hop=1, beliefs={"A": 1.0}, confidence=0.9)
        flow.add_state(hop=2, beliefs={"A": 0.5, "B": 0.5}, confidence=0.7)
        
        history = flow.get_flow_history()
        assert len(history) == 2
        
        previous = flow.get_previous_state()
        assert previous is not None
        assert previous.hop == 1

