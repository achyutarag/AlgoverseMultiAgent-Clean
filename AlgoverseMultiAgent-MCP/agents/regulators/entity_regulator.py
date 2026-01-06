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
        
        # ✅ FIRST PRINCIPLES FIX: Filter entities by hierarchical level
        # If plan_goal requires a specific hierarchical level, don't anchor
        # entities that violate that requirement (prevents bias at source)
        filtered_entities = self._filter_by_hierarchical_level(
            all_entities,
            plan_goal,
            proposed_query
        )
        
        # Check if proposed query maintains entity focus
        maintains_focus = any(
            entity.lower() in proposed_query.lower()
            for entity in filtered_entities
        ) if filtered_entities else True
        
        # Calculate constraint weight based on entity presence
        weight = self.weight if filtered_entities else 0.3
        
        return RegulatorConstraint(
            regulator_name=self.name,
            constraint_type="dirichlet",
            weight=weight,
            parameters={
                "entities": filtered_entities,  # Use filtered entities
                "main_entity": filtered_entities[0] if filtered_entities else None,
                "maintains_focus": maintains_focus,
                "entity_count": len(filtered_entities),
                "original_entities": all_entities  # Keep original for debugging
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
    
    def _filter_by_hierarchical_level(
        self,
        entities: List[str],
        plan_goal: Optional[str],
        proposed_query: str
    ) -> List[str]:
        """
        Filter entities to respect hierarchical level requirements.
        
        ✅ GENERALIZED: Uses GranularityRegulator's hierarchical level system
        instead of hardcoded location terms. Works for any hierarchical domain
        (territorial, organizational, taxonomic, etc.).
        
        If an entity violates the required hierarchical level, extracts the
        parent-level entity name to help retrieval while respecting constraints.
        
        This removes the precision-recall trade-off: we maintain retrieval recall
        (retrieve relevant documents) while preserving answer precision (extract correct-level entity).
        
        Args:
            entities: List of entity names to filter
            plan_goal: Overall plan goal/question (used to infer required level)
            proposed_query: Proposed query (for context)
            
        Returns:
            Filtered list of entities that respect hierarchical constraints
        """
        if not entities or not plan_goal:
            return entities
        
        # Use GranularityRegulator to infer required level (generalized, not hardcoded)
        granularity_reg = GranularityRegulator()
        required_domain, required_level = granularity_reg._infer_required_level(plan_goal)
        
        if not required_domain or not required_level:
            return entities  # No hierarchical constraint detected
        
        # Filter entities based on hierarchical level requirement
        filtered = []
        for entity in entities:
            entity_lower = entity.lower().strip()
            if entity_lower in {"yes", "unknown", "none", "n/a"}:
                logger.debug(f"EntityRegulator: Skipping non-entity token '{entity}'")
                continue
            # Classify entity's hierarchical level using GranularityRegulator
            entity_domain, entity_level, _ = granularity_reg.classify_entity_level(entity)
            
            # ✅ FIX: If entity can't be classified (no level keywords), allow it through
            # It might be the correct answer (e.g., "Tamaulipas" without "state" keyword)
            # OR it might be in previous answers and needs to be anchored for retrieval
            if not entity_domain or not entity_level:
                filtered.append(entity)
                logger.debug(
                    f"EntityRegulator: Allowing unclassified entity '{entity}' through "
                    f"(no level keywords found - might be correct answer or needs anchoring)"
                )
                continue
            
            # Check if entity violates required level
            is_violation = granularity_reg.is_level_violation(
                required_domain, required_level,
                entity_domain, entity_level
            )
            
            if is_violation:
                # Try to extract parent-level entity name (generalized extraction)
                parent_name = granularity_reg.extract_parent_level_name(
                    entity, required_domain, required_level
                )
                
                if parent_name:
                    filtered.append(parent_name)
                    logger.debug(
                        f"EntityRegulator: Extracted parent-level entity '{parent_name}' from '{entity}' "
                        f"(required: {required_domain}/{required_level}, entity: {entity_domain}/{entity_level}) "
                        f"to help retrieval while respecting hierarchical constraint"
                    )
                else:
                    logger.debug(
                        f"EntityRegulator: Skipping entity '{entity}' "
                        f"(violates required level {required_domain}/{required_level}, "
                        f"could not extract parent-level name)"
                    )
                continue
            
            # Entity respects hierarchical constraint
            filtered.append(entity)
        
        # Fallback to original if all filtered (better than no entities)
        return filtered if filtered else entities