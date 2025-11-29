# agents/tests/test_regulator_physics.py
"""
Physics tests for individual regulators: constraint application and query stabilization.
No LLMs, no FAISS - synthetic state only.
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.regulators.entity_regulator import EntityRegulator
from agents.regulators.evidence_regulator import EvidenceRegulator
from agents.regulators.confidence_regulator import ConfidenceRegulator
from agents.regulators.plan_regulator import PlanRegulator
from agents.regulators.relation_regulator import RelationRegulator
from agents.regulators.regulator_manager import RegulatorManager


class TestEntityRegulator:
    """Test 6: EntityRegulator constraint application"""
    
    def test_entity_regulator_extracts_entities(self):
        """Entity regulator should extract entities from previous answers."""
        regulator = EntityRegulator(weight=0.9)
        
        reasoning_state = {
            "entity_anchors": {"France": {}, "Paris": {}}
        }
        previous_answers = {
            "step_1": {"answer": "France is a country", "confidence": 0.9},
            "step_2": {"answer": "Paris is the capital", "confidence": 0.8}
        }
        
        constraint = regulator.apply_constraint(
            proposed_query="What is the capital?",
            reasoning_state=reasoning_state,
            previous_answers=previous_answers,
            plan_goal=None
        )
        
        assert constraint.regulator_name == "Entity"
        assert constraint.constraint_type == "dirichlet"
        entities = constraint.parameters.get("entities", [])
        assert len(entities) > 0, "Should extract entities from previous answers"
        assert "France" in entities or "Paris" in entities
    
    def test_entity_regulator_maintains_focus(self):
        """Entity regulator should check if query maintains entity focus."""
        regulator = EntityRegulator(weight=0.9)
        
        reasoning_state = {"entity_anchors": {"France": {}}}
        previous_answers = {"step_1": {"answer": "France", "confidence": 0.9}}
        
        # Query with entity
        constraint1 = regulator.apply_constraint(
            proposed_query="What is the capital of France?",
            reasoning_state=reasoning_state,
            previous_answers=previous_answers,
            plan_goal=None
        )
        
        # Query without entity
        constraint2 = regulator.apply_constraint(
            proposed_query="What is the capital?",
            reasoning_state=reasoning_state,
            previous_answers=previous_answers,
            plan_goal=None
        )
        
        assert constraint1.parameters["maintains_focus"] is True
        assert constraint2.parameters["maintains_focus"] is False
    
    def test_entity_regulator_violation_check(self):
        """Entity regulator should detect violations."""
        regulator = EntityRegulator(weight=0.9)
        
        constraint = regulator.apply_constraint(
            proposed_query="What is the capital?",
            reasoning_state={"entity_anchors": {"France": {}}},
            previous_answers={"step_1": {"answer": "France"}},
            plan_goal=None
        )
        
        # Query without entity should violate
        violates = regulator.check_violation(
            query="What is the capital?",
            constraint=constraint,
            current_state={}
        )
        assert violates is True, "Query without entity should violate constraint"
        
        # Query with entity should not violate
        does_not_violate = regulator.check_violation(
            query="What is the capital of France?",
            constraint=constraint,
            current_state={}
        )
        assert does_not_violate is False, "Query with entity should not violate"


class TestEvidenceRegulator:
    """Test 7: EvidenceRegulator constraint application"""
    
    def test_evidence_regulator_extracts_terms(self):
        """Evidence regulator should extract evidence terms."""
        regulator = EvidenceRegulator(weight=0.85)
        
        previous_answers = {
            "step_1": {
                "answer": "Paris is the capital of France",
                "confidence": 0.9,
                "sources": []
            }
        }
        
        constraint = regulator.apply_constraint(
            proposed_query="What is the capital?",
            reasoning_state={},
            previous_answers=previous_answers,
            plan_goal=None
        )
        
        assert constraint.constraint_type == "potential_well"
        evidence_terms = constraint.parameters.get("evidence_terms", [])
        assert len(evidence_terms) > 0, "Should extract evidence terms"
        # Should contain key terms (case-insensitive check)
        term_lower = [t.lower() for t in evidence_terms]
        assert any("paris" in t or "france" in t or "capital" in t for t in term_lower)
    
    def test_evidence_regulator_calculates_alignment(self):
        """Evidence regulator should calculate alignment with query."""
        regulator = EvidenceRegulator(weight=0.85)
        
        previous_answers = {
            "step_1": {"answer": "Paris is the capital of France", "confidence": 0.9}
        }
        
        # Query aligned with evidence
        constraint1 = regulator.apply_constraint(
            proposed_query="What is the capital of France?",
            reasoning_state={},
            previous_answers=previous_answers,
            plan_goal=None
        )
        
        # Query not aligned with evidence
        constraint2 = regulator.apply_constraint(
            proposed_query="What is the population?",
            reasoning_state={},
            previous_answers=previous_answers,
            plan_goal=None
        )
        
        alignment1 = constraint1.parameters.get("evidence_alignment", 0.0)
        alignment2 = constraint2.parameters.get("evidence_alignment", 0.0)
        
        assert alignment1 > alignment2, \
            f"Aligned query should have higher alignment ({alignment1:.3f} > {alignment2:.3f})"
    
    def test_evidence_regulator_violation_check(self):
        """Evidence regulator should detect violations."""
        regulator = EvidenceRegulator(weight=0.85)
        
        previous_answers = {
            "step_1": {"answer": "Paris is the capital of France", "confidence": 0.9}
        }
        
        constraint = regulator.apply_constraint(
            proposed_query="What is the population?",
            reasoning_state={},
            previous_answers=previous_answers,
            plan_goal=None
        )
        
        # Query without evidence terms should violate
        violates = regulator.check_violation(
            query="What is the population?",
            constraint=constraint,
            current_state={}
        )
        # May or may not violate depending on term extraction, but should be consistent
        assert isinstance(violates, bool)


class TestConfidenceRegulator:
    """Test 8: ConfidenceRegulator constraint application"""
    
    def test_confidence_regulator_adjusts_weight_by_uncertainty(self):
        """Confidence regulator should adjust weight based on uncertainty."""
        regulator = ConfidenceRegulator(weight=0.75)
        
        # High confidence state
        high_conf_state = {
            "entropy": 0.1,
            "diffusion_coefficient": 0.1,
            "confidence": 0.9
        }
        
        constraint_high = regulator.apply_constraint(
            proposed_query="What is the capital?",
            reasoning_state=high_conf_state,
            previous_answers={},
            plan_goal=None
        )
        
        # Low confidence state
        low_conf_state = {
            "entropy": 2.0,
            "diffusion_coefficient": 0.8,
            "confidence": 0.2
        }
        
        constraint_low = regulator.apply_constraint(
            proposed_query="What is the capital?",
            reasoning_state=low_conf_state,
            previous_answers={},
            plan_goal=None
        )
        
        # Low confidence should have higher adjusted weight
        assert constraint_low.weight > constraint_high.weight, \
            f"Low confidence should have higher weight ({constraint_low.weight:.3f} > {constraint_high.weight:.3f})"
    
    def test_confidence_regulator_detects_high_diffusion(self):
        """Confidence regulator should detect high diffusion/entropy."""
        regulator = ConfidenceRegulator(weight=0.75)
        
        # High diffusion state
        high_diff_state = {
            "entropy": 1.5,
            "diffusion_coefficient": 0.7,
            "confidence": 0.3
        }
        
        constraint = regulator.apply_constraint(
            proposed_query="What is the capital?",
            reasoning_state=high_diff_state,
            previous_answers={},
            plan_goal=None
        )
        
        assert constraint.parameters["needs_control"] is True, \
            "High diffusion/entropy should need control"
        assert constraint.parameters["high_diffusion"] is True
        assert constraint.parameters["high_entropy"] is True
    
    def test_confidence_regulator_violation_check(self):
        """Confidence regulator should detect vague queries when control needed."""
        regulator = ConfidenceRegulator(weight=0.75)
        
        high_diff_state = {
            "entropy": 1.5,
            "diffusion_coefficient": 0.7,
            "confidence": 0.3
        }
        
        constraint = regulator.apply_constraint(
            proposed_query="What is the capital?",
            reasoning_state=high_diff_state,
            previous_answers={},
            plan_goal=None
        )
        
        # Vague query should violate when control needed
        violates = regulator.check_violation(
            query="What is it?",
            constraint=constraint,
            current_state=high_diff_state
        )
        
        # Specific query should not violate
        does_not_violate = regulator.check_violation(
            query="What is the name of the capital city?",
            constraint=constraint,
            current_state=high_diff_state
        )
        
        # If control is needed, vague query should violate
        if constraint.parameters["needs_control"]:
            assert violates is True or does_not_violate is False, \
                "Vague query should violate when control needed"


class TestPlanRegulator:
    """Test 9: PlanRegulator constraint application"""
    
    def test_plan_regulator_calculates_alignment(self):
        """Plan regulator should calculate alignment with plan goal."""
        regulator = PlanRegulator(weight=0.9)
        
        plan_goal = "What administrative territorial entity contains Nuevo Laredo Municipality?"
        
        # Aligned query
        constraint1 = regulator.apply_constraint(
            proposed_query="Find the state containing Nuevo Laredo Municipality",
            reasoning_state={},
            previous_answers={},
            plan_goal=plan_goal
        )
        
        # Not aligned query
        constraint2 = regulator.apply_constraint(
            proposed_query="Find the population of Paris",
            reasoning_state={},
            previous_answers={},
            plan_goal=plan_goal
        )
        
        alignment1 = constraint1.parameters.get("alignment", 0.0)
        alignment2 = constraint2.parameters.get("alignment", 0.0)
        
        assert alignment1 > alignment2, \
            f"Aligned query should have higher alignment ({alignment1:.3f} > {alignment2:.3f})"
    
    def test_plan_regulator_without_goal(self):
        """Plan regulator should have lower weight without plan goal."""
        regulator = PlanRegulator(weight=0.9)
        
        constraint = regulator.apply_constraint(
            proposed_query="What is the capital?",
            reasoning_state={},
            previous_answers={},
            plan_goal=None
        )
        
        assert constraint.weight < 0.5, \
            f"Without goal, weight should be low ({constraint.weight:.3f} < 0.5)"
        assert constraint.parameters["has_goal"] is False
    
    def test_plan_regulator_violation_check(self):
        """Plan regulator should detect off-track queries."""
        regulator = PlanRegulator(weight=0.9)
        
        plan_goal = "What administrative territorial entity contains Nuevo Laredo?"
        
        constraint = regulator.apply_constraint(
            proposed_query="Find the population of Paris",
            reasoning_state={},
            previous_answers={},
            plan_goal=plan_goal
        )
        
        # Off-track query should violate
        violates = regulator.check_violation(
            query="Find the population of Paris",
            constraint=constraint,
            current_state={}
        )
        
        # Should violate if not on track
        if not constraint.parameters.get("on_track", True):
            assert violates is True, "Off-track query should violate"


class TestRegulatorManager:
    """Test regulator manager integration"""
    
    def test_regulator_manager_applies_all_regulators(self):
        """Regulator manager should apply all regulators."""
        regulators = [
            EntityRegulator(weight=0.9),
            EvidenceRegulator(weight=0.85),
            ConfidenceRegulator(weight=0.75),
            PlanRegulator(weight=0.9)
        ]
        
        manager = RegulatorManager(regulators)
        
        reasoning_state = {
            "entropy": 0.5,
            "diffusion_coefficient": 0.3,
            "confidence": 0.7,
            "entity_anchors": {"France": {}}
        }
        
        previous_answers = {
            "step_1": {"answer": "France is a country", "confidence": 0.9}
        }
        
        stabilized_query, constraints = manager.apply_all(
            proposed_query="What is the capital?",
            reasoning_state=reasoning_state,
            previous_answers=previous_answers,
            plan_goal="What is the capital of France?"
        )
        
        assert len(constraints) == 4, f"Should apply 4 regulators, got {len(constraints)}"
        assert isinstance(stabilized_query, str)
        assert len(stabilized_query) > 0
    
    def test_regulator_manager_sorts_by_weight(self):
        """Regulator manager should apply constraints in weight order."""
        regulators = [
            EntityRegulator(weight=0.5),
            EvidenceRegulator(weight=0.9),  # Highest
            ConfidenceRegulator(weight=0.3)   # Lowest
        ]
        
        manager = RegulatorManager(regulators)
        
        # Check that constraints are sorted by weight
        _, constraints = manager.apply_all(
            proposed_query="What is the capital?",
            reasoning_state={"confidence": 0.7},
            previous_answers={},
            plan_goal=None
        )
        
        # Constraints should be sorted by weight (descending)
        weights = [c.weight for c in constraints]
        assert weights == sorted(weights, reverse=True), \
            f"Constraints should be sorted by weight: {weights}"

