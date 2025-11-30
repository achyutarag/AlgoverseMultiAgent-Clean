# agents/regulators/regulator_manager.py
from typing import List, Dict, Any, Optional, Tuple
from .base_regulator import BaseRegulator, RegulatorConstraint
import logging
import re

logger = logging.getLogger(__name__)

class RegulatorManager:
    """
    Composes multiple regulator constraints into a single stabilized query.
    
    This is the central controller that merges individual regulator outputs
    into a coherent query transformation for entropy-aware retrieval.
    """
    
    def __init__(self, regulators: List[BaseRegulator]):
        """
        Initialize the regulator manager.
        
        Args:
            regulators: List of regulator instances to apply
        """
        self.regulators = regulators
        logger.info(f"RegulatorManager initialized with {len(regulators)} regulators")
    
    def apply_all(
        self,
        proposed_query: str,
        reasoning_state: Dict[str, Any],
        previous_answers: Dict[str, Any],
        plan_goal: Optional[str] = None
    ) -> Tuple[str, List[RegulatorConstraint]]:
        """
        Apply all regulators and merge their constraints.
        
        Args:
            proposed_query: Raw query from Step Definer
            reasoning_state: Current reasoning flow state
            previous_answers: Previous step answers
            plan_goal: Overall plan goal
            
        Returns:
            Tuple of (stabilized_query, list_of_constraints)
        """
        
        # Collect constraints from all regulators
        constraints = []
        for regulator in self.regulators:
            try:
                constraint = regulator.apply_constraint(
                    proposed_query,
                    reasoning_state,
                    previous_answers,
                    plan_goal
                )
                constraints.append(constraint)
                logger.debug(
                    f"{regulator.name} applied: type={constraint.constraint_type}, "
                    f"weight={constraint.weight:.2f}"
                )
            except Exception as e:
                logger.warning(f"{regulator.name} failed to apply constraint: {e}")
                continue
        
        # Sort constraints by weight (descending - highest weight first)
        sorted_constraints = sorted(
            constraints,
            key=lambda c: c.weight,
            reverse=True
        )
        
        # Merge constraints into final query
        final_query = self.merge_constraints(proposed_query, sorted_constraints, reasoning_state)
        
        return final_query, sorted_constraints
    
    def merge_constraints(
        self,
        query: str,
        constraints: List[RegulatorConstraint],
        reasoning_state: Dict[str, Any]
    ) -> str:
        """
        Merge multiple regulator constraints into a single stabilized query.
        
        Strategy:
        1. Sort by weight (strongest constraints first)
        2. Apply entity anchoring (Dirichlet - fixes entities)
        3. Apply relation direction (Neumann - prevents drift)
        4. Apply evidence reinforcement (potential well - boosts relevant terms)
        5. Apply plan alignment (boundary - ensures goal consistency)
        6. Apply confidence-based adjustments (diffusion control)
        
        Args:
            query: Original proposed query
            constraints: List of constraints from all regulators
            reasoning_state: Current reasoning state (entropy, etc.)
            
        Returns:
            Stabilized query string
        """
        if not constraints:
            return query
        
        # Sort by weight (strongest first)
        sorted_constraints = sorted(
            constraints,
            key=lambda c: c.weight,
            reverse=True
        )
        
        stabilized_query = query
        
        # ✅ FIRST PRINCIPLES: Find GranularityRegulator constraint (initial condition)
        # This is used as a safeguard when applying other regulators
        granularity_constraint = None
        for constraint in sorted_constraints:
            if "granularity" in constraint.regulator_name.lower():
                granularity_constraint = constraint
                break
        
        # Apply constraints in order of strength
        for constraint in sorted_constraints:
            stabilized_query = self._apply_single_constraint(
                stabilized_query,
                constraint,
                reasoning_state,
                granularity_constraint  # Pass as safeguard
            )
        
        logger.debug(
            f"Query stabilized: '{query}' -> '{stabilized_query}' "
            f"(applied {len(constraints)} constraints)"
        )
        
        return stabilized_query
    
    def _apply_single_constraint(
        self,
        query: str,
        constraint: RegulatorConstraint,
        reasoning_state: Dict[str, Any],
        granularity_constraint: Optional[RegulatorConstraint] = None
    ) -> str:
        """
        Apply a single constraint to the query.
        
        Args:
            query: Current query state
            constraint: Constraint to apply
            reasoning_state: Reasoning state for context
            
        Returns:
            Modified query
        """
        constraint_type = constraint.constraint_type
        params = constraint.parameters
        weight = constraint.weight
        
        
        # Entity Regulator: Dirichlet anchor (fix entities)
        if constraint_type == "dirichlet" and "entity" in constraint.regulator_name.lower():
            # Prefer explicitly provided main_entity, fall back to first in entities list
            entity = params.get("main_entity")
            if not entity:
                entities = params.get("entities", [])
                entity = entities[0] if entities else None
            
            if entity:
                # ✅ GENERALIZED SAFEGUARD: Check if entity violates hierarchical constraints
                # This is a defense-in-depth check (EntityRegulator already filters, but this ensures)
                # Uses GranularityRegulator's level system instead of hardcoded location terms
                if granularity_constraint:
                    from .granularity_regulator import GranularityRegulator
                    granularity_reg = GranularityRegulator()
                    required_domain = granularity_constraint.parameters.get("required_domain")
                    required_level = granularity_constraint.parameters.get("required_level")
                    
                    if required_domain and required_level:
                        # Classify entity's hierarchical level
                        entity_domain, entity_level, _ = granularity_reg.classify_entity_level(entity)
                        
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
                                entity = parent_name
                                logger.info(
                                    f"EntityRegulator (safeguard): Extracted parent-level entity '{entity}' "
                                    f"(required: {required_domain}/{required_level}, "
                                    f"entity: {entity_domain}/{entity_level}) "
                                    f"to help retrieval while respecting hierarchical constraint"
                                )
                            else:
                                logger.info(
                                    f"EntityRegulator (safeguard): Skipping entity anchor '{entity}' "
                                    f"because it violates required level {required_domain}/{required_level} "
                                    f"(could not extract parent-level name)"
                                )
                                return query  # Don't anchor, return query as-is
                
                # Check if entity (or its parts) is already in query
                entity_lower = entity.lower()
                query_lower = query.lower()
                
                # Check if full entity or all its words are in query
                entity_words = entity_lower.split()
                entity_in_query = (
                    entity_lower in query_lower or
                    all(word in query_lower for word in entity_words)
                )
                
                if not entity_in_query:
                    # Anchor entity in query (prepend for visibility)
                    query = f"{entity} {query}"
                    logger.debug(f"Entity anchor applied: {entity}")
        
        # Relation Regulator: Neumann constraint (prevent drift)
        elif constraint_type == "neumann":
            relation_direction = params.get("direction")
            if relation_direction:
                # Ensure query maintains relational direction
                # Example: "contains" vs "contained in"
                if relation_direction not in query.lower():
                    # Could prepend relation context
                    pass  # Relation preservation logic
        
        # Evidence Regulator: Potential well (boost relevant terms)
        elif constraint_type == "potential_well":
            evidence_terms = params.get("evidence_terms", [])
            if evidence_terms:
                # ✅ FIRST PRINCIPLES SAFEGUARD: Filter evidence terms by hierarchical level
                # This is a defense-in-depth check (EvidenceRegulator already filters, but this ensures)
                filtered_terms = evidence_terms[:3]  # Top 3 terms
                if granularity_constraint:
                    required_level = granularity_constraint.parameters.get("required_level")
                    municipality_keywords = ["municipality", "city", "town", "county", "district"]
                    
                    if required_level == "state_province":
                        # Filter out municipality keywords
                        filtered_terms = [
                            term for term in filtered_terms
                            if not any(keyword in term.lower() for keyword in municipality_keywords)
                        ]
                        if len(filtered_terms) < len(evidence_terms[:3]):
                            logger.info(
                                f"EvidenceRegulator: Filtered out municipality keywords "
                                f"because GranularityRegulator requires state/province level "
                                f"(safeguard check in regulator_manager)"
                            )
                
                # Reinforce evidence-based terminology
                for term in filtered_terms:
                    if term.lower() not in query.lower():
                        query = f"{query} {term}"
                logger.debug(f"Evidence reinforcement: {filtered_terms}")
        
        
        
        # Granularity Regulator: Initial boundary condition (enforce hierarchical level with domain)
        elif constraint_type == "boundary" and "granularity" in constraint.regulator_name.lower():
            level_keywords = params.get("level_keywords", [])
            needs_modification = params.get("needs_modification", False)
            required_domain = params.get("required_domain")
            required_level = params.get("required_level")
            query_has_level_keyword = params.get("query_has_level_keyword", False)
            
            # ✅ FIRST PRINCIPLES: Explicitly add required level keyword to initial condition
            # This ensures u(x,0) is explicit, not assumed
            if needs_modification and level_keywords:
                query_lower = query.lower()
                # Add level keyword if missing (explicit initial condition)
                best_keyword = level_keywords[0]  # Use most specific keyword
                
                # Check if keyword is already present (double-check)
                if best_keyword.lower() not in query_lower:
                    # Strategy 1: Insert after "what" or "which" if present (more natural)
                    pattern = r'\b(what|which)\s+([a-z]+)\s+'
                    match = re.search(pattern, query_lower)
                    if match:
                        pos = match.end()
                        query = query[:pos] + f"{best_keyword} " + query[pos:]
                        logger.info(
                            f"GranularityRegulator (u(x,0)): Added level keyword '{best_keyword}' "
                            f"to query (domain={required_domain}, level={required_level})"
                        )
                    else:
                        # Strategy 2: Prepend for visibility (ensures it's in the query)
                        query = f"{best_keyword} {query}"
                        logger.info(
                            f"GranularityRegulator (u(x,0)): Prepended level keyword '{best_keyword}' "
                            f"to query (domain={required_domain}, level={required_level})"
                        )
                else:
                    logger.debug(
                        f"GranularityRegulator: Level keyword '{best_keyword}' already present in query"
                    )
        
        # Plan Regulator: Boundary condition (goal alignment)
        elif constraint_type == "boundary":
            goal_keywords = params.get("goal_keywords", [])
            if goal_keywords:
                # Add goal keywords to query if missing to help retrieval
                query_lower = query.lower()
                for keyword in goal_keywords[:5]:  # Top 5 keywords (increased from 3)
                    if keyword.lower() not in query_lower:
                        query = f"{query} {keyword}"
                        logger.debug(f"Added goal keyword to query: {keyword}")
        
        # Confidence Regulator: Diffusion control
        elif "confidence" in constraint.regulator_name.lower():
            # Adjust query based on confidence/diffusion
            diffusion = reasoning_state.get("diffusion_coefficient", 0.0)
            if diffusion > 0.5:  # High uncertainty
                # Add more specific terms to reduce ambiguity
                pass  # Diffusion control logic
        
        return query
    
    def get_constraint_summary(self, constraints: List[RegulatorConstraint]) -> Dict[str, Any]:
        """
        Get summary of applied constraints for logging/debugging.
        
        Args:
            constraints: List of applied constraints
            
        Returns:
            Summary dictionary
        """
        return {
            "total_constraints": len(constraints),
            "by_type": {
                c.constraint_type: sum(1 for x in constraints if x.constraint_type == c.constraint_type)
                for c in constraints
            },
            "total_weight": sum(c.weight for c in constraints),
            "regulators": [c.regulator_name for c in constraints]
        }