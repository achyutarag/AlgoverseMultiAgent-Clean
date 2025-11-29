# agents/regulators/entity_regulator.py
from typing import Dict, Any, Optional, List
from .base_regulator import BaseRegulator, RegulatorConstraint
import logging

logger = logging.getLogger(__name__)

class EntityRegulator(BaseRegulator):
    """
    Entity Regulator: P(x, t) = P_entity (Dirichlet anchor)
    
    Stabilizes the current entity focus and prevents entity drift.
    """
    
    def __init__(self, weight: float = 0.9):
        super().__init__("Entity", weight)
    
    def apply_constraint(
        self,
        proposed_query: str,
        reasoning_state: Dict[str, Any],
        previous_answers: Dict[str, Any],
        plan_goal: Optional[str] = None
    ) -> RegulatorConstraint:
        """
        Apply entity anchoring constraint.
        
        Extracts entity names from previous answers and anchors them
        to prevent drift to unrelated entities.
        """
        # Extract entities from previous answers
        entities = self._extract_entities(previous_answers, reasoning_state)
        
        # Get entity anchors from reasoning flow
        entity_anchors = reasoning_state.get("entity_anchors", {})
        
        # Combine extracted entities with anchors
        all_entities = list(set(entities + list(entity_anchors.keys())))
        
        # Check if proposed query maintains entity focus
        maintains_focus = any(
            entity.lower() in proposed_query.lower()
            for entity in all_entities
        ) if all_entities else True
        
        # Calculate constraint weight based on entity presence
        weight = self.weight if all_entities else 0.3
        
        return RegulatorConstraint(
            regulator_name=self.name,
            constraint_type="dirichlet",
            weight=weight,
            parameters={
                "entities": all_entities,
                "main_entity": all_entities[0] if all_entities else None,
                "maintains_focus": maintains_focus,
                "entity_count": len(all_entities)
            }
        )
    
    def check_violation(
        self,
        query: str,
        constraint: RegulatorConstraint,
        current_state: Dict[str, Any]
    ) -> bool:
        """Check if query violates entity anchor constraint."""
        entities = constraint.parameters.get("entities", [])
        if not entities:
            return False
        
        # Violation: query doesn't contain any anchored entities
        has_entity = any(
            entity.lower() in query.lower()
            for entity in entities
        )
        
        return not has_entity
    
    
    def _extract_entities(
        self,
        previous_answers: Dict[str, Any],
        reasoning_state: Dict[str, Any]
    ) -> List[str]:
        """Extract entity names from previous answers."""
        entities = []
        
        # Extract from previous step answers
        for step_id, answer_data in previous_answers.items():
            if isinstance(answer_data, dict):
                answer = answer_data.get("answer", "")
            else:
                answer = str(answer_data)
            
            # First, try to preserve the full answer if it looks like a single entity
            # (e.g., "Nuevo Laredo", "New York", "San Francisco")
            answer_cleaned = answer.strip(".,!?")
            words = answer_cleaned.split()
            
            # If answer is 1-3 capitalized words, treat as a single entity
            if len(words) <= 3 and all(word and word[0].isupper() for word in words):
                # Check it's not a common phrase
                if answer_cleaned.lower() not in ["the", "and", "or", "for", "with"]:
                    entities.append(answer_cleaned)
            else:
                # Extract individual capitalized words
                for word in words:
                    if word and word[0].isupper() and len(word) > 2:
                        if word.lower() not in ["the", "and", "or", "for", "with"]:
                            entities.append(word.strip(".,!?"))
        
        # Also check reasoning state for entity anchors
        entity_anchors = reasoning_state.get("entity_anchors", {})
        entities.extend(entity_anchors.keys())
        
        # Deduplicate while preserving order (keep longer entities first)
        seen = set()
        unique_entities = []
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower not in seen:
                seen.add(entity_lower)
                unique_entities.append(entity)
        
        # Sort by length (longer first) to prefer multi-word entities
        unique_entities.sort(key=len, reverse=True)
        return unique_entities