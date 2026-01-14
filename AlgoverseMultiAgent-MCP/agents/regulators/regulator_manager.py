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
                # ✅ FIX: Annotate instead of filter
                entity_metadata = None
                if granularity_constraint:
                    from .granularity_regulator import GranularityRegulator
                    granularity_reg = GranularityRegulator()
                    required_domain = granularity_constraint.parameters.get("required_domain")
                    required_level = granularity_constraint.parameters.get("required_level")
                    
                    if required_domain and required_level:
                        # Compute granularity metadata
                        entity_metadata = granularity_reg.compute_granularity_metadata(
                                entity, required_domain, required_level
                            )
                            
                        logger.debug(
                            f"EntityRegulator (safeguard): Annotated entity '{entity}': "
                            f"delta={entity_metadata.get('granularity_delta')}, "
                            f"violation={entity_metadata.get('granularity_violation')}"
                        )
                        
                        # ✅ FIX: Always preserve entity, but log violation
                        if entity_metadata.get("granularity_violation"):
                                logger.info(
                                f"EntityRegulator (safeguard): Entity '{entity}' violates granularity "
                                f"(required: {required_domain}/{required_level}, "
                                f"got: {entity_metadata.get('entity_domain')}/{entity_metadata.get('entity_level')}), "
                                f"but preserving for downstream scoring"
                            )
                
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
                
                # ✅ FIX: Reinforce evidence-based terminology (limit additions to prevent query bloat)
                query_lower = query.lower()
                added_count = 0
                added_terms = []
                for term in filtered_terms:
                    if term.lower() not in query_lower and added_count < 2:  # Limit to 2 additions
                        query = f"{query} {term}"
                        query_lower = query.lower()  # Update for next iteration
                        added_count += 1
                        added_terms.append(term)
                logger.debug(f"Evidence reinforcement: {added_terms if added_terms else 'none (already in query or limit reached)'}")
        
        
        
        # Granularity Regulator: Initial boundary condition (track but don't inject)
        elif constraint_type == "boundary" and "granularity" in constraint.regulator_name.lower():
            required_domain = params.get("required_domain")
            required_level = params.get("required_level")
            # ✅ FIX: Don't inject keywords - granularity is a soft constraint
            # The requirement is tracked in the constraint parameters for downstream scoring
            logger.debug(
                f"GranularityRegulator constraint applied: domain={required_domain}, "
                f"level={required_level} (tracked for scoring, not injected into query)"
            )
            # Query remains unchanged - granularity will be applied as penalty downstream
        
        # Plan Regulator: Boundary condition (goal alignment)
        elif constraint_type == "boundary":
            goal_keywords = params.get("goal_keywords", [])
            if goal_keywords:
                # ✅ FIX: Filter out generic framing words that don't help retrieval
                generic_framing_words = {
                    "identify", "determine", "find", "locate", "provide", "within",
                    "provided", "context", "specific", "designated", "particular",
                    "given", "available", "information", "details", "regarding",
                    "about", "concerning", "related", "relevant", "primary",
                    "first", "overarching", "main", "key", "important"
                }
                
                # Only add semantically meaningful keywords (nouns, entities, specific terms)
                meaningful_keywords = [
                    kw for kw in goal_keywords[:8]  # Check more keywords but filter aggressively
                    if kw.lower() not in generic_framing_words
                    and len(kw) > 3  # Filter very short words
                ]
                
                # Add goal keywords to query if missing to help retrieval
                query_lower = query.lower()
                added_count = 0
                for keyword in meaningful_keywords:
                    if keyword.lower() not in query_lower and added_count < 2:  # Limit to 2 additions
                        query = f"{query} {keyword}"
                        query_lower = query.lower()  # Update for next iteration
                        added_count += 1
                        logger.debug(f"Added meaningful goal keyword to query: {keyword}")
        
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