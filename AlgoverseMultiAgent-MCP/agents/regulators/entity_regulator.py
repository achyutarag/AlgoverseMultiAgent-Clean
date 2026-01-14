# agents/regulators/entity_regulator.py
from typing import Dict, Any, Optional, List, Tuple
from .base_regulator import BaseRegulator, RegulatorConstraint
from .granularity_regulator import GranularityRegulator
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
        
        ✅ FIRST PRINCIPLES: Respects hierarchical level constraints set by
        GranularityRegulator (initial condition). Filters out entities that
        violate hierarchical level requirements to prevent bias at the source.
        """
        # Extract entities from previous answers
        entities = self._extract_entities(previous_answers, reasoning_state)
        
        # Get entity anchors from reasoning flow
        entity_anchors = reasoning_state.get("entity_anchors", {})
        
        # Combine extracted entities with anchors
        all_entities = list(set(entities + list(entity_anchors.keys())))
        
        # ✅ FIX: Annotate instead of filter
        annotated_entities = self._annotate_by_hierarchical_level(
            all_entities,
            plan_goal,
            proposed_query
        )
        
        # Extract entity strings for backward compatibility (all entities preserved)
        entity_strings = [ann["entity"] for ann in annotated_entities]
        
        # Check if proposed query maintains entity focus
        maintains_focus = any(
            entity.lower() in proposed_query.lower()
            for entity in entity_strings
        ) if entity_strings else True
        
        # Calculate constraint weight based on entity presence
        weight = self.weight if entity_strings else 0.3
        
        return RegulatorConstraint(
            regulator_name=self.name,
            constraint_type="dirichlet",
            weight=weight,
            parameters={
                "entities": entity_strings,  # For backward compatibility
                "annotated_entities": annotated_entities,  # NEW: Full annotations
                "main_entity": entity_strings[0] if entity_strings else None,
                "maintains_focus": maintains_focus,
                "entity_count": len(entity_strings),
                "original_entities": all_entities
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
    
    def _annotate_by_hierarchical_level(
        self,
        entities: List[str],
        plan_goal: Optional[str],
        proposed_query: str
    ) -> List[Dict[str, Any]]:
        """
        Annotate entities with granularity metadata instead of filtering.
        
        ✅ FIX: Preserves all entities, adds granularity_delta annotation.
        Downstream components can use this for scoring, not elimination.
        
        Args:
            entities: List of entity names
            plan_goal: Overall plan goal/question
            proposed_query: Proposed query (for context)
            
        Returns:
            List of dicts: {"entity": str, "granularity_metadata": Dict[str, Any]}
        """
        if not entities:
            return []
        
        # Use GranularityRegulator to infer required level
        granularity_reg = GranularityRegulator()
        required_domain, required_level = granularity_reg._infer_required_level(plan_goal)
        
        annotated_entities = []
        for entity in entities:
            entity_lower = entity.lower().strip()
            if entity_lower in {"yes", "unknown", "none", "n/a"}:
                logger.debug(f"EntityRegulator: Skipping non-entity token '{entity}'")
                continue
            
            # Compute granularity metadata
            if required_domain and required_level:
                granularity_metadata = granularity_reg.compute_granularity_metadata(
                    entity, required_domain, required_level
                )
            else:
                granularity_metadata = {
                    "granularity_delta": None,
                    "granularity_violation": False,
                    "entity_domain": None,
                    "entity_level": None,
                    "is_unclassified": True,
                    "penalty_factor": 0.0
                }
            
            annotated_entities.append({
                "entity": entity,
                "granularity_metadata": granularity_metadata
            })
            
            logger.debug(
                f"EntityRegulator: Annotated entity '{entity}': "
                f"delta={granularity_metadata.get('granularity_delta')}, "
                f"violation={granularity_metadata.get('granularity_violation')}"
            )
        
        return annotated_entities