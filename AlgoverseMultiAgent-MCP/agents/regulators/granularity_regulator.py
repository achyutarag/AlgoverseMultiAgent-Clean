# agents/regulators/granularity_regulator.py
from typing import Dict, Any, Optional, List, Tuple
from .base_regulator import BaseRegulator, RegulatorConstraint
import logging
import re

logger = logging.getLogger(__name__)

class GranularityRegulator(BaseRegulator):
    """
    Granularity Regulator: u(x, 0) - Initial Condition
    
    Enforces hierarchical level requirements at the START (initial condition)
    to prevent hierarchical leakage and level-mismatch cascades.
    
    This is the INITIAL CONDITION in the diffusion process (u(x,0)) that sets
    the correct hierarchical level before retrieval begins, preventing cascading
    errors that would require downstream regulators to correct.
    
    Features:
    1. Domain disambiguation (territorial, organizational, taxonomic)
    2. Monotonic hierarchy constraint (coarse→fine allowed, fine→coarse forbidden)
    3. Multi-hierarchy conflict resolution (domain priority order)
    4. Keyword scoring * level weighting for accurate inference
    """
    
    # Hierarchical level definitions with DOMAIN field
    # Structure: domain -> level_name -> {keywords, level}
    # This prevents "level 1 appears twice" issue by distinguishing domains
    HIERARCHICAL_LEVELS = {
        "territorial": {
            "country": {"keywords": ["country", "nation", "national"], "level": 1},
            "state_province": {
                "keywords": ["state", "province", "administrative territorial entity", 
                            "administrative entity", "territorial entity", "region"],
                "level": 2
            },
            "municipality": {
                "keywords": ["municipality", "city", "town", "county", "district"],
                "level": 3
            },
            "neighborhood": {
                "keywords": ["neighborhood", "ward", "precinct", "locality"],
                "level": 4
            }
        },
        "organizational": {
            "company": {"keywords": ["company", "corporation", "organization", "firm"], "level": 1},
            "division": {"keywords": ["division", "department", "unit", "branch"], "level": 2},
            "team": {"keywords": ["team", "group", "squad"], "level": 3},
            "individual": {"keywords": ["ceo", "president", "director", "manager", "person", "individual"], "level": 4}
        },
        "taxonomic": {
            "kingdom": {"keywords": ["kingdom"], "level": 1},
            "phylum": {"keywords": ["phylum"], "level": 2},
            "class": {"keywords": ["class"], "level": 3},
            "order": {"keywords": ["order"], "level": 4},
            "family": {"keywords": ["family"], "level": 5},
            "genus": {"keywords": ["genus"], "level": 6},
            "species": {"keywords": ["species"], "level": 7}
        }
    }
    
    # Domain priority order for disambiguation (higher = more priority)
    # Used when multiple hierarchies match at once (Issue 1)
    DOMAIN_PRIORITY = {
        "territorial": 3,  # Highest priority (most common in queries)
        "organizational": 2,
        "taxonomic": 1
    }
    
    def __init__(self, weight: float = 1.0):
        """
        Initialize Granularity Regulator (initial condition).
        
        Args:
            weight: Weight/strength of the regulator (default 1.0 for initial condition)
        """
        super().__init__("Granularity", weight)
        # Cache for level inference (optimization)
        self._level_cache: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        logger.debug(f"GranularityRegulator initialized (initial condition regulator)")
    
    def apply_constraint(
        self,
        proposed_query: str,
        reasoning_state: Dict[str, Any],
        previous_answers: Dict[str, Any],
        plan_goal: Optional[str] = None
    ) -> RegulatorConstraint:
        """
        Apply granularity boundary constraint (initial condition).
        
        This is the INITIAL CONDITION (u(x, 0)) that sets the correct
        hierarchical level before retrieval begins.
        
        Implements:
        1. Domain disambiguation (domain first → then level)
        2. Monotonic hierarchy constraint (coarse→fine allowed, fine→coarse forbidden)
        
        Args:
            proposed_query: Raw query from Step Definer
            reasoning_state: Current reasoning flow state
            previous_answers: Previous step answers
            plan_goal: Overall plan goal/question (used to infer required level)
            
        Returns:
            RegulatorConstraint with granularity enforcement details
        """
        # Use plan_goal for level inference (more reliable than subquery)
        source_text = plan_goal or proposed_query
        required_domain, required_level = self._infer_required_level(source_text)
        
        # Check if proposed query matches required level
        query_domain, query_level = self._infer_required_level(proposed_query)
        
        # Get level keywords for checking and modification
        level_keywords = self._get_level_keywords(required_domain, required_level) if required_level else []
        
        # ====================================================================
        # FIRST PRINCIPLES FIX: Initial Condition Must Be Explicit
        # ====================================================================
        # The initial condition (u(x,0)) must be explicit, not assumed.
        # If the query doesn't explicitly mention the required level keyword,
        # it MUST be modified to include it. This prevents hierarchical leakage.
        # ====================================================================
        query_has_level_keyword = False
        if required_level and level_keywords:
            query_lower = proposed_query.lower()
            # Check if any level keyword is present in the query
            query_has_level_keyword = any(
                keyword.lower() in query_lower 
                for keyword in level_keywords
            )
        
        # Monotonic consistency check (not strict equality) - Issue 3 fix
        is_monotonic = self._check_monotonic_consistency(
            required_domain, required_level,
            query_domain, query_level
        )
        
        # ✅ FIRST PRINCIPLES: Needs modification if:
        # 1. Required level exists AND
        # 2. (Query doesn't have level keyword OR monotonic violation)
        # This ensures the initial condition is explicit, not assumed
        needs_modification = (
            required_level and 
            (not query_has_level_keyword or not is_monotonic)
        )
        
        # High weight if modification needed (critical to fix at initial condition)
        weight = self.weight if needs_modification else 0.7
        
        return RegulatorConstraint(
            regulator_name=self.name,
            constraint_type="boundary",  # Initial boundary condition
            weight=weight,
            parameters={
                "required_domain": required_domain,
                "required_level": required_level,
                "query_domain": query_domain,
                "query_level": query_level,
                "is_monotonic": is_monotonic,
                "query_has_level_keyword": query_has_level_keyword,
                "needs_modification": needs_modification,
                "level_keywords": level_keywords
            }
        )
    
    def check_violation(
        self,
        query: str,
        constraint: RegulatorConstraint,
        current_state: Dict[str, Any]
    ) -> bool:
        """
        Check if query violates granularity constraint.
        
        Uses monotonic consistency (not strict equality) - Issue 3 fix.
        
        Args:
            query: Query to check
            constraint: Granularity constraint
            current_state: Current reasoning state
            
        Returns:
            True if constraint is violated, False otherwise
        """
        required_domain = constraint.parameters.get("required_domain")
        required_level = constraint.parameters.get("required_level")
        if not required_domain or not required_level:
            return False  # No requirement = no violation
        
        query_domain, query_level = self._infer_required_level(query)
        
        # Check monotonic consistency
        is_monotonic = self._check_monotonic_consistency(
            required_domain, required_level,
            query_domain, query_level
        )
        
        return not is_monotonic
    
    def _check_monotonic_consistency(
        self,
        required_domain: Optional[str],
        required_level: Optional[str],
        query_domain: Optional[str],
        query_level: Optional[str]
    ) -> bool:
        """
        Check monotonic hierarchy constraint (Issue 3 fix).
        
        Rules:
        1. Allow: coarse → fine (e.g., country → state, company → division)
        2. Forbid: fine → coarse (e.g., state → country, division → company)
        3. Forbid: cross-domain jumps (e.g., territorial → organizational)
        
        This implements monotonic consistency instead of strict equality,
        allowing the system to accept more specific answers when a general
        level is required, but preventing acceptance of general answers
        when a specific level is required.
        
        Args:
            required_domain: Required domain from plan goal
            required_level: Required level from plan goal
            query_domain: Domain detected in query
            query_level: Level detected in query
            
        Returns:
            True if constraint is satisfied (monotonic), False otherwise
        """
        # If no requirements, always consistent
        if not required_domain or not required_level:
            return True
        
        # Cross-domain violation: different domains (forbid cross-domain jumps)
        if query_domain and query_domain != required_domain:
            return False
        
        # If query has no domain/level, assume it's okay (will be fixed by modification)
        if not query_domain or not query_level:
            return True
        
        # Get level numbers for comparison
        required_level_num = self._get_level_number(required_domain, required_level)
        query_level_num = self._get_level_number(query_domain, query_level)
        
        if required_level_num is None or query_level_num is None:
            return True  # Can't compare, assume okay
        
        # Monotonic constraint: query level must be >= required level (coarse→fine allowed)
        # This means:
        # - If required is level 1 (coarse), query can be level 1, 2, 3, 4... (fine) ✅
        # - If required is level 3 (fine), query cannot be level 1 or 2 (coarse) ❌
        return query_level_num >= required_level_num
    
    def _get_level_number(self, domain: str, level_name: str) -> Optional[int]:
        """Get numeric level for a given domain and level name."""
        if domain not in self.HIERARCHICAL_LEVELS:
            return None
        if level_name not in self.HIERARCHICAL_LEVELS[domain]:
            return None
        return self.HIERARCHICAL_LEVELS[domain][level_name]["level"]
    
    def classify_evidence_level(self, evidence_text: str) -> Optional[Tuple[str, str]]:
        """
        Classify the hierarchical level of retrieved evidence.
        
        This is function 2: Classify retrieved evidence level.
        Used to filter or rerank evidence based on granularity requirements.
        
        Args:
            evidence_text: Text from retrieved document
            
        Returns:
            Tuple of (domain, level_name) or None if unclear
        """
        if not evidence_text:
            return None
        
        evidence_lower = evidence_text.lower()
        
        # Score all domains and levels (keyword scoring * level weighting)
        domain_scores = {}
        for domain, levels in self.HIERARCHICAL_LEVELS.items():
            domain_score = 0
            best_level = None
            best_level_score = 0
            
            for level_name, level_info in levels.items():
                keywords = level_info["keywords"]
                matches = sum(1 for keyword in keywords if keyword in evidence_lower)
                if matches > 0:
                    # Score = matches * level (keyword scoring * level weighting)
                    level_score = matches * level_info["level"]
                    domain_score += level_score
                    if level_score > best_level_score:
                        best_level_score = level_score
                        best_level = level_name
            
            if domain_score > 0:
                # Weight by domain priority
                domain_scores[domain] = {
                    "score": domain_score * self.DOMAIN_PRIORITY.get(domain, 1),
                    "level": best_level
                }
        
        if not domain_scores:
            return None
        
        # Select domain with highest priority-weighted score
        best_domain = max(domain_scores.items(), key=lambda x: x[1]["score"])[0]
        best_level = domain_scores[best_domain]["level"]
        
        return (best_domain, best_level)
    
    def _infer_required_level(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Infer required hierarchical level from question.
        
        This is function 1: Infer required level from question.
        
        Implements domain disambiguation (Issue 1 fix):
        1. Score all domains and levels (keyword scoring * level weighting)
        2. Apply domain priority order (domain first → then level)
        3. Resolve multi-hierarchy conflicts
        
        Args:
            question: Question text to analyze
            
        Returns:
            Tuple of (domain, level_name) or (None, None) if unclear
        """
        if not question:
            return (None, None)
        
        # Check cache first (optimization)
        if question in self._level_cache:
            return self._level_cache[question]
        
        question_lower = question.lower()
        
        # ====================================================================
        # STEP 1: Score all domains and levels
        # ====================================================================
        # Keyword scoring * level weighting for accurate inference
        domain_scores = {}
        for domain, levels in self.HIERARCHICAL_LEVELS.items():
            domain_score = 0
            best_level = None
            best_level_score = 0
            
            for level_name, level_info in levels.items():
                keywords = level_info["keywords"]
                matches = sum(1 for keyword in keywords if keyword in question_lower)
                if matches > 0:
                    # Score = matches * level (keyword scoring * level weighting)
                    level_score = matches * level_info["level"]
                    domain_score += level_score
                    if level_score > best_level_score:
                        best_level_score = level_score
                        best_level = level_name
            
            if domain_score > 0:
                # Weight by domain priority (domain first → then level)
                domain_scores[domain] = {
                    "score": domain_score * self.DOMAIN_PRIORITY.get(domain, 1),
                    "level": best_level,
                    "raw_score": domain_score
                }
        
        if not domain_scores:
            result = (None, None)
            self._level_cache[question] = result
            return result
        
        # ====================================================================
        # STEP 2: Domain disambiguation (domain first → then level)
        # ====================================================================
        # Issue 1 fix: Multiple hierarchies can match at once
        # Use domain priority order to resolve conflicts
        if len(domain_scores) > 1:
            # Sort by priority-weighted score (highest first)
            sorted_domains = sorted(
                domain_scores.items(),
                key=lambda x: (x[1]["score"], self.DOMAIN_PRIORITY.get(x[0], 0)),
                reverse=True
            )
            
            # Check if top two are close (within 20% of each other)
            top_score = sorted_domains[0][1]["score"]
            second_score = sorted_domains[1][1]["score"] if len(sorted_domains) > 1 else 0
            
            if top_score > 0 and (top_score - second_score) / top_score < 0.2:
                # Close scores: use domain priority as tiebreaker
                best_domain = sorted_domains[0][0]
                logger.debug(
                    f"GranularityRegulator: Multiple domains matched. "
                    f"Using domain priority: {best_domain}"
                )
            else:
                # Clear winner
                best_domain = sorted_domains[0][0]
        else:
            best_domain = list(domain_scores.keys())[0]
        
        best_level = domain_scores[best_domain]["level"]

        

        # ✅ FIX: Context-aware guard - only downgrade if question explicitly asks for municipality-level answer
        # Check if municipality/city keywords appear in answer type position (after "what"/"which")
        if best_domain == "territorial" and best_level == "state_province":
            # Check if municipality-level keywords appear as answer type (not just mentioned in context)
            # Answer type position: immediately after "what" or "which"
            answer_type_pattern = r'\b(what|which)\s+(city|municipality|ciudad|town|county|district)\b'
            if re.search(answer_type_pattern, question_lower):
                best_level = "municipality"
                logger.debug(
                    f"GranularityRegulator: Downgraded to municipality (question asks for municipality-level answer)"
                )
        
        result = (best_domain, best_level)
        
        # Cache result (optimization)
        self._level_cache[question] = result
        
        if result[0] and result[1]:
            logger.debug(
                f"GranularityRegulator: Inferred domain='{best_domain}', level='{best_level}' "
                f"from: '{question[:80]}...'"
            )
        
        return result
    
    def _get_level_keywords(self, domain: Optional[str], level_name: Optional[str]) -> List[str]:
        """Get keywords for a given domain and hierarchical level."""
        if not domain or not level_name:
            return []
        if domain not in self.HIERARCHICAL_LEVELS:
            return []
        if level_name not in self.HIERARCHICAL_LEVELS[domain]:
            return []
        return self.HIERARCHICAL_LEVELS[domain][level_name]["keywords"]
    
    def classify_entity_level(self, entity_text: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """
        Classify the hierarchical level of an entity/term using keyword matching.
        
        This is a generalized method that works for any hierarchical domain,
        not just territorial/administrative structures.
        
        Args:
            entity_text: Entity or term text to classify
            
        Returns:
            Tuple of (domain, level_name, level_number) or (None, None, None) if unclear
        """
        if not entity_text:
            return (None, None, None)
        
        entity_lower = entity_text.lower()
        best_match = None
        best_score = 0
        
        # Score all domains and levels
        for domain, levels in self.HIERARCHICAL_LEVELS.items():
            for level_name, level_info in levels.items():
                keywords = level_info.get("keywords", [])
                level_num = level_info.get("level", 0)
                
                # Count keyword matches
                matches = sum(1 for keyword in keywords if keyword in entity_lower)
                if matches > 0:
                    score = matches * level_num  # Weight by level
                    if score > best_score:
                        best_score = score
                        best_match = (domain, level_name, level_num)
        
        return best_match if best_match else (None, None, None)
    
    def get_level_number(self, domain: Optional[str], level_name: Optional[str]) -> Optional[int]:
        """Get the numeric level for a given domain and level name."""
        if not domain or not level_name:
            return None
        if domain not in self.HIERARCHICAL_LEVELS:
            return None
        if level_name not in self.HIERARCHICAL_LEVELS[domain]:
            return None
        return self.HIERARCHICAL_LEVELS[domain][level_name].get("level")
    
    def is_level_violation(self, 
                          required_domain: Optional[str],
                          required_level: Optional[str],
                          entity_domain: Optional[str],
                          entity_level: Optional[str]) -> bool:
        """
        Check if an entity's hierarchical level violates the required level.
        
        ✅ FIX #2: Uses monotonic logic (consistent with _check_monotonic_consistency).
        Allows coarse→fine (e.g., state → city) but forbids fine→coarse (e.g., city → state).
        
        Args:
            required_domain: Required hierarchical domain
            required_level: Required hierarchical level name
            entity_domain: Entity's hierarchical domain
            entity_level: Entity's hierarchical level name
            
        Returns:
            True if entity violates required level (wrong domain or too coarse), False otherwise
        """
        if not required_domain or not required_level:
            return False  # No constraint
        
        if not entity_domain or not entity_level:
            return False  # Can't determine entity level
        
        # ✅ FIX #6: Cross-domain is a violation (different hierarchies)
        if entity_domain != required_domain:
            return True  # Violation: different domain
        
        # Same domain: compare levels using monotonic logic
        required_level_num = self.get_level_number(required_domain, required_level)
        entity_level_num = self.get_level_number(entity_domain, entity_level)
        
        if required_level_num is None or entity_level_num is None:
            return False  # Can't compare
        
        # ✅ FIX #2: Violation only if entity is coarser than required (fine→coarse forbidden)
        # Allows: coarse→fine (e.g., state level 2 → city level 3) ✅
        # Forbids: fine→coarse (e.g., city level 3 → state level 2) ❌
        return entity_level_num < required_level_num

    # Public helper to avoid reaching into private methods externally
    @staticmethod
    def infer_required(plan_goal: Optional[str], query: Optional[str]) -> Tuple[Optional[str], Optional[str], List[str]]:
        """
        Infer required granularity with priority: question target > plan goal.
        
        Priority order:
        1. Explicit answer type in question (WH-target)
        2. Role words in question
        3. Plan goal text
        """
        reg = GranularityRegulator()
        
        # ✅ FIX 1: Try question first (answer type priority)
        if query:
            # Extract answer type from WH-questions
            answer_type = reg._extract_answer_type_from_question(query)
            if answer_type:
                domain, level = reg._infer_required_level(query)
                # If we found explicit answer type, use it
                if level == answer_type:
                    keywords = reg._get_level_keywords(domain, level) if domain and level else []
                    return domain, level, keywords
        
        # Try question without answer type extraction
        if query:
            domain, level = reg._infer_required_level(query)
            if domain and level:
                keywords = reg._get_level_keywords(domain, level) if domain and level else []
                return domain, level, keywords
        
        # Fallback to plan goal
        if plan_goal:
            domain, level = reg._infer_required_level(plan_goal)
            keywords = reg._get_level_keywords(domain, level) if domain and level else []
            return domain, level, keywords
        
        return None, None, []
    
    def _extract_answer_type_from_question(self, question: str) -> Optional[str]:
        """
        Extract explicit answer type from WH-questions.
        
        Examples:
        - "What province is X located in?" → "state_province"
        - "What city is X in?" → "municipality"
        - "What company owns X?" → "company"
        """
        if not question:
            return None
        
        q_lower = question.lower()
        
        # Pattern: "What [ANSWER_TYPE] is/does/..."
        answer_type_patterns = {
            "province": "state_province",
            "state": "state_province",
            "city": "municipality",
            "municipality": "municipality",
            "district": "municipality",  # District is municipality-level
            "country": "country",
            "company": "company",
            "corporation": "company",
            "organization": "company",
        }
        
        for keyword, level in answer_type_patterns.items():
            # Check for "What [keyword]" or "Which [keyword]" pattern
            if f"what {keyword}" in q_lower or f"which {keyword}" in q_lower:
                return level
        
        return None
    
    # ✅ FIX: Build granularity prior from query only (no candidate feedback loop)
    @staticmethod
    def build_granularity_posterior(
        candidates: List[Dict[str, Any]], 
        query: str,
        gran_regulator: Optional['GranularityRegulator'] = None
    ) -> Optional[Dict[str, float]]:
        """
        Build granularity prior distribution from question only (not from candidates).
        
        This prevents circular dependency where candidate confidences update the posterior,
        which then penalizes those same candidates. The prior is fixed based on what the
        question asks for, independent of candidate answers.
        
        Args:
            candidates: List of candidate dicts (unused, kept for API compatibility)
            query: Original query/question
            gran_regulator: GranularityRegulator instance (creates one if None)
            
        Returns:
            Dict[str, float]: Prior probability for each level (normalized), or None if unavailable
        """
        if gran_regulator is None:
            gran_regulator = GranularityRegulator()
        
        # Initialize prior distribution
        prior = {}
        all_levels = set()
        
        # Collect all possible levels from the regulator
        for domain, levels in gran_regulator.HIERARCHICAL_LEVELS.items():
            for level_name in levels.keys():
                all_levels.add(level_name)
                prior[level_name] = 0.0
        
        # Build prior based on query/question only (not from candidates)
        try:
            prior_domain, prior_level = gran_regulator._infer_required_level(query)
            if prior_level:
                # Give inferred level a boost (but not too strong to allow exploration)
                prior[prior_level] = 0.4
                # Distribute remaining probability uniformly across other levels
                remaining = 0.6 / max(1, len(all_levels) - 1)
                for level in all_levels:
                    if level != prior_level:
                        prior[level] = remaining
            else:
                # Uniform prior if no level can be inferred from query
                uniform = 1.0 / max(1, len(all_levels))
                for level in all_levels:
                    prior[level] = uniform
        except Exception:
            # Uniform prior on error
            uniform = 1.0 / max(1, len(all_levels))
            for level in all_levels:
                prior[level] = uniform
        
        # ✅ FIX: No candidate-based updates - prior is fixed from query only
        # This breaks the circular dependency where candidates update posterior,
        # which then penalizes those same candidates
        
        # Final normalization check (defensive)
        total = sum(prior.values())
        if abs(total - 1.0) > 0.001:  # Allow small floating point errors
            logger.warning(f"[GranularityPrior] Prior not normalized (sum={total:.3f}), normalizing")
            if total > 0:
                for level in prior:
                    prior[level] /= total
            else:
                uniform = 1.0 / max(1, len(all_levels))
                for level in all_levels:
                    prior[level] = uniform
        
        return prior
    
    def compute_granularity_delta(
        self,
        required_domain: Optional[str],
        required_level: Optional[str],
        entity_domain: Optional[str],
        entity_level: Optional[str]
    ) -> Optional[int]:
        """
        Compute granularity delta: entity_level_num - required_level_num
        
        Returns:
            - Positive: entity is finer than required (e.g., city when state required)
            - Zero: exact match
            - Negative: entity is coarser than required (e.g., country when city required)
            - None: unclassified or can't compare
        """
        if not required_domain or not required_level:
            return None
        
        if not entity_domain or not entity_level:
            return None  # Unclassified
        
        if entity_domain != required_domain:
            return None  # Cross-domain, can't compare numerically
        
        required_level_num = self.get_level_number(required_domain, required_level)
        entity_level_num = self.get_level_number(entity_domain, entity_level)
        
        if required_level_num is None or entity_level_num is None:
            return None
        
        return entity_level_num - required_level_num

    def compute_granularity_metadata(
        self,
        entity_text: str,
        required_domain: Optional[str],
        required_level: Optional[str]
    ) -> Dict[str, Any]:
        """
        Compute complete granularity metadata for an entity.
        
        Returns:
            {
                "granularity_delta": int | None,
                "granularity_violation": bool,
                "entity_domain": str | None,
                "entity_level": str | None,
                "entity_level_num": int | None,
                "required_domain": str | None,
                "required_level": str | None,
                "required_level_num": int | None,
                "is_unclassified": bool,
                "is_cross_domain": bool,
                "penalty_factor": float  # Suggested penalty (0.0 to 0.5)
            }
        """
        entity_domain, entity_level, entity_level_num = self.classify_entity_level(entity_text)
        required_level_num = self.get_level_number(required_domain, required_level) if required_domain and required_level else None
        
        granularity_delta = self.compute_granularity_delta(
            required_domain, required_level,
            entity_domain, entity_level
        )
        
        is_violation = self.is_level_violation(
            required_domain, required_level,
            entity_domain, entity_level
        )
        
        is_unclassified = not entity_domain or not entity_level
        is_cross_domain = (entity_domain and required_domain and 
                          entity_domain != required_domain)
        
        # Compute suggested penalty factor
        penalty_factor = 0.0
        if is_violation:
            if granularity_delta is not None:
                # Stronger penalty for coarser (negative delta)
                if granularity_delta < 0:
                    penalty_factor = min(0.3, abs(granularity_delta) * 0.1)
                else:
                    penalty_factor = min(0.15, granularity_delta * 0.05)
            else:
                penalty_factor = 0.15  # Default violation penalty
        elif is_unclassified:
            # Moderate penalty for unclassified when level is required
            penalty_factor = 0.15 if required_level else 0.05
        elif is_cross_domain:
            penalty_factor = 0.2
        
        return {
            "granularity_delta": granularity_delta,
            "granularity_violation": is_violation,
            "entity_domain": entity_domain,
            "entity_level": entity_level,
            "entity_level_num": entity_level_num,
            "required_domain": required_domain,
            "required_level": required_level,
            "required_level_num": required_level_num,
            "is_unclassified": is_unclassified,
            "is_cross_domain": is_cross_domain,
            "penalty_factor": penalty_factor
        }

    def extract_parent_level_name(self, 
                                  entity_text: str,
                                  required_domain: Optional[str],
                                  required_level: Optional[str]) -> Optional[str]:
        """
        Extract parent-level entity name from entity text.
        
        This is a generalized method that removes level-specific keywords
        to extract the base entity name, which helps retrieve documents
        that mention both the entity and its parent.
        
        Args:
            entity_text: Entity text (e.g., "Nuevo Laredo Municipality")
            required_domain: Required hierarchical domain
            required_level: Required hierarchical level name
            
        Returns:
            Extracted parent-level entity name or None if extraction fails
        """
        if not entity_text or not required_domain or not required_level:
            return None
        
        # Get all level keywords for the domain (to remove them)
        all_keywords = []
        if required_domain in self.HIERARCHICAL_LEVELS:
            for level_info in self.HIERARCHICAL_LEVELS[required_domain].values():
                all_keywords.extend(level_info.get("keywords", []))
        
        # Remove level keywords from entity text
        entity_lower = entity_text.lower()
        extracted = entity_lower
        
        for keyword in all_keywords:
            # Remove keyword with spaces around it
            extracted = extracted.replace(f" {keyword}", "").replace(keyword, "").strip()
        
        # Only return if we successfully extracted something meaningful
        if extracted and extracted != entity_lower and len(extracted) > 2:
            # Capitalize properly (handle multi-word entities)
            extracted_capitalized = " ".join(
                word.capitalize() for word in extracted.split()
            )
            return extracted_capitalized
        
        return None

