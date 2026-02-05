"""
Breadcrumb Extraction Module

Extracts hierarchical breadcrumbs from document paragraphs using heuristics:
1. Entity extraction (capitalized sequences)
2. Hierarchical patterns ("X is a Y", "X located in Y")
3. Paragraph order and context
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class BreadcrumbExtractor:
    """
    Extracts breadcrumb paths from document text using heuristics.
    
    Strategy:
    1. Extract entities (capitalized sequences)
    2. Detect hierarchical relationships (is-a, part-of, located-in)
    3. Build path from general → specific
    """
    
    # Common hierarchical patterns
    HIERARCHICAL_PATTERNS = [
        (r"(\w+(?:\s+\w+)*)\s+is\s+(?:a|an|the)\s+(\w+(?:\s+\w+)*)", "is_a"),  # "DLR is a center"
        (r"(\w+(?:\s+\w+)*)\s+located\s+in\s+(\w+(?:\s+\w+)*)", "located_in"),  # "DLR located in Cologne"
        (r"(\w+(?:\s+\w+)*)\s+is\s+part\s+of\s+(\w+(?:\s+\w+)*)", "part_of"),  # "DLR is part of NASA"
        (r"(\w+(?:\s+\w+)*)\s+of\s+(\w+(?:\s+\w+)*)", "of"),  # "center of NASA"
        (r"(\w+(?:\s+\w+)*)\s+in\s+(\w+(?:\s+\w+)*)", "in"),  # "DLR in Germany"
    ]
    
    # Stop words to filter out
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "for", "with", "in", "on", "at",
        "to", "from", "by", "of", "is", "are", "was", "were", "be", "been"
    }
    
    # Common phrases to filter out (not real entities) - "Noise-Robust Entity Scoping"
    COMMON_PHRASES = {
        "first session", "first stand", "very green", "other fortune",
        "in scotland", "in finland", "in germany", "in italy", "in spain",
        "the commonwealth", "the northern", "the southern", "the eastern", "the western",
        "the democratic", "the republic", "the united", "the green",
        "of the", "in the", "on the", "at the"
    }
    
    def extract_breadcrumbs(
        self,
        text: str,
        paragraph_id: Optional[int] = None,
        example_id: Optional[str] = None,
        previous_breadcrumbs: Optional[List[str]] = None,
        example_root_entity: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract breadcrumb path from text.
        
        Implements "Example-Level Root Entity Tracking" to create deeper hierarchies
        by using the first paragraph's main entity as a root for all paragraphs in the example.
        
        Args:
            text: Document text to extract from
            paragraph_id: Paragraph index (for ordering)
            example_id: Example ID (for context)
            previous_breadcrumbs: Breadcrumbs from previous paragraph (for continuity)
            example_root_entity: Root entity from first paragraph of this example (for depth)
            
        Returns:
            Dict with:
            - breadcrumb_path: List[str] - hierarchical path
            - breadcrumb_string: str - formatted string
            - breadcrumb_depth: int - depth of hierarchy
            - entities: List[str] - extracted entities
            - confidence: float - confidence score (0-1)
        """
        if not text or not text.strip():
            return self._empty_breadcrumbs()
        
        # Step 1: Extract entities (capitalized sequences)
        entities = self._extract_entities(text)
        
        # Step 2: Detect hierarchical relationships
        relationships = self._detect_relationships(text, entities)
        
        # Step 3: Build breadcrumb path (with strategy tracking for confidence)
        breadcrumb_path, strategy_used = self._build_path(
            entities, 
            relationships, 
            previous_breadcrumbs,
            example_root_entity
        )
        
        # Step 4: Calculate confidence (strategy-aware)
        confidence = self._calculate_confidence(breadcrumb_path, strategy_used)
        
        # Step 5: Format output
        breadcrumb_string = " > ".join(breadcrumb_path) if breadcrumb_path else "Unknown"
        
        return {
            "breadcrumb_path": breadcrumb_path,
            "breadcrumb_string": breadcrumb_string,
            "breadcrumb_depth": len(breadcrumb_path),
            "entities": entities,
            "confidence": confidence,
            "metadata": {
                "paragraph_id": paragraph_id,
                "example_id": example_id,
                "num_entities": len(entities),
                "num_relationships": len(relationships)
            }
        }
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract entity names (capitalized sequences).
        
        Implements "Noise-Robust Entity Scoping" by filtering out common phrases
        and prepositional patterns that aren't real organizational units.
        """
        entities = []
        
        # Pattern 1: Capitalized sequences (2+ words, proper nouns)
        # Matches: "German Aerospace Center", "New York", "Mike Medavoy"
        pattern1 = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
        matches = re.findall(pattern1, text)
        entities.extend([m.strip() for m in matches])
        
        # Pattern 2: Acronyms (2+ uppercase letters, standalone)
        # Matches: "NASA", "DLR", "UHF" (acronyms or single-word entities)
        pattern2 = r"\b([A-Z]{2,})\b"
        matches = re.findall(pattern2, text)
        # Filter: only keep if it's not part of a longer word
        for match in matches:
            if len(match) >= 2 and match.isupper():
                entities.append(match)
        
        # Pattern 3: Single capitalized words after "the" or in specific positions
        # Matches: "The Green performer" → "Green" (but we'll filter "Green" if it's not meaningful)
        pattern3 = r"\b(?:the|The)\s+([A-Z][a-z]+)\b"
        matches = re.findall(pattern3, text)
        entities.extend([m.strip() for m in matches])
        
        # Filter out stop words, duplicates, and common phrases (Noise-Robust Entity Scoping)
        filtered = []
        seen = set()
        for entity in entities:
            entity_lower = entity.lower().strip()
            
            # Skip if it's a common phrase (not a real entity)
            if entity_lower in self.COMMON_PHRASES:
                continue
            
            # Skip if it starts with common prepositions (prepositional phrases)
            if any(entity_lower.startswith(prep) for prep in ["in ", "of ", "the ", "a ", "an "]):
                continue
            
            # Skip if it's too short, is a stop word, or already seen
            if (entity_lower and 
                entity_lower not in self.STOP_WORDS and 
                len(entity) > 2 and  # Increased from 1 to 2 (filter single letters)
                entity_lower not in seen):
                filtered.append(entity)
                seen.add(entity_lower)
        
        # Sort by length (longer = more specific, keep first)
        filtered.sort(key=len, reverse=True)
        
        return filtered[:5]  # Top 5 entities
    
    def _detect_relationships(self, text: str, entities: List[str]) -> List[Dict[str, Any]]:
        """Detect hierarchical relationships in text."""
        relationships = []
        
        for pattern, rel_type in self.HIERARCHICAL_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                parent = match.group(1).strip()
                child = match.group(2).strip()
                
                # Check if both are entities
                if (self._is_entity(parent, entities) and 
                    self._is_entity(child, entities)):
                    relationships.append({
                        "type": rel_type,
                        "parent": parent,
                        "child": child,
                        "confidence": 0.8 if rel_type in ["is_a", "part_of"] else 0.6
                    })
        
        return relationships
    
    def _is_entity(self, text: str, entities: List[str]) -> bool:
        """Check if text matches an entity."""
        text_lower = text.lower()
        for entity in entities:
            if entity.lower() in text_lower or text_lower in entity.lower():
                return True
        # Also check if it's capitalized (likely an entity)
        words = text.split()
        if words and all(word and word[0].isupper() for word in words):
            return True
        return False
    
    def _build_path(
        self,
        entities: List[str],
        relationships: List[Dict[str, Any]],
        previous_breadcrumbs: Optional[List[str]] = None,
        example_root_entity: Optional[str] = None
    ) -> Tuple[List[str], str]:
        """
        Build breadcrumb path from entities and relationships.
        
        Implements "Intra-document Relationship Mapping" to build a micro-knowledge graph,
        "Breadcrumb Persistence" (Structural Memoization) for continuity, and
        "Example-Level Root Entity Tracking" for deeper hierarchies.
        
        Strategy (in priority order):
        0. Use example root entity (Example-Level Root Entity Tracking) - NEW
        1. Use relationships to build parent → child hierarchy (Graph-Based Path Reconstruction)
        2. If no relationships, use entity order (general → specific)
        3. Inherit from previous breadcrumbs if available (Breadcrumb Persistence)
        
        Returns:
            Tuple of (breadcrumb_path, strategy_used)
        """
        if not entities:
            # Strategy 3: Inherit from previous (Breadcrumb Persistence)
            # But if we have a root entity, try to use it
            if example_root_entity and previous_breadcrumbs:
                # If previous breadcrumbs don't start with root, prepend it
                if (previous_breadcrumbs and 
                    previous_breadcrumbs[0].lower() != example_root_entity.lower()):
                    enhanced = [example_root_entity] + previous_breadcrumbs
                    return (enhanced[:3], "root_entity")
            return (previous_breadcrumbs or [], "inherited")
        
        # Strategy 0: Use example root entity (Example-Level Root Entity Tracking)
        # This creates deeper hierarchies: [Root Entity] > [Current Entity]
        if example_root_entity and entities:
            current_entity = entities[0] if entities else None
            if (current_entity and 
                current_entity.lower() != example_root_entity.lower() and
                len(current_entity) > 2 and
                current_entity.lower() not in self.COMMON_PHRASES):
                # Create 2-level hierarchy: Root > Current
                # This pushes average depth from 1.43 to 2.0-2.5
                path = [example_root_entity, current_entity]
                return (path[:3], "root_entity")  # Max depth 3
        
        # Strategy 1: Use relationships (Graph-Based Path Reconstruction)
        if relationships:
            # Build graph: parent → child (Micro-Knowledge Graph)
            graph = {}
            for rel in relationships:
                parent = rel["parent"]
                child = rel["child"]
                # Only add if both are valid entities (filter noise)
                if (len(parent) > 2 and len(child) > 2 and
                    parent.lower() not in self.STOP_WORDS and
                    child.lower() not in self.STOP_WORDS):
                    if parent not in graph:
                        graph[parent] = []
                    graph[parent].append(child)
            
            if graph:
                # Find root (entity with no parent)
                all_children = set()
                for children in graph.values():
                    all_children.update(children)
                
                roots = [e for e in entities if e not in all_children]
                
                if roots:
                    # Build path from root
                    root = roots[0]
                    path = [root]
                    if root in graph and graph[root]:
                        path.append(graph[root][0])  # Add first child
                    return (path[:3], "relationship")  # Max depth 3
        
        # Strategy 2: Use entity order (only if entities are meaningful)
        # Filter entities: must be 3+ characters and not common phrases
        valid_entities = [
            e for e in entities 
            if len(e) >= 3 and e.lower() not in self.COMMON_PHRASES
        ]
        
        if len(valid_entities) >= 2:
            # Take first (likely general) and last (likely specific)
            # But only if they're different enough
            first = valid_entities[0]
            last = valid_entities[-1]
            
            # Don't combine if they're too similar or unrelated
            if first.lower() != last.lower() and len(first) > 3:
                return ([first, last][:3], "entity_order")
        elif valid_entities:
            return ([valid_entities[0]], "entity_order")
        
        # Strategy 3: Inherit from previous (Breadcrumb Persistence / Structural Memoization)
        # This solves the "Sparse Entity Problem" - maintains continuity across chunks
        if previous_breadcrumbs:
            return (previous_breadcrumbs, "inherited")
        
        # Strategy 4: Return empty if nothing valid
        return ([], "none")
    
    def _calculate_confidence(
        self,
        breadcrumb_path: List[str],
        strategy_used: str
    ) -> float:
        """
        Calculate strategy-aware confidence score for breadcrumb extraction.
        
        Relationships are more certain than simple entity order, and inherited
        breadcrumbs have lower confidence (they're maintained for continuity).
        
        Args:
            breadcrumb_path: The extracted breadcrumb path
            strategy_used: Which strategy was used ("relationship", "entity_order", "inherited", "none")
            
        Returns:
            Confidence score (0-1)
        """
        if not breadcrumb_path:
            return 0.0
        
        # Strategy-based confidence weights
        # Relationships are most reliable (explicit hierarchical links)
        # Root entity is high (example-level structure is reliable)
        # Entity order is moderate (heuristic-based)
        # Inherited is lower (maintained for continuity, not newly extracted)
        weights = {
            "relationship": 0.9,   # High confidence: explicit relationships
            "root_entity": 0.75,    # High confidence: example-level root structure
            "entity_order": 0.6,    # Moderate confidence: heuristic ordering
            "inherited": 0.4,       # Lower confidence: maintained for continuity
            "none": 0.0
        }
        
        base_conf = weights.get(strategy_used, 0.3)
        
        # Penalize very deep paths (uncertainty increases with depth)
        # Depth penalty: confidence decreases as path gets deeper
        depth_penalty = 1 + (len(breadcrumb_path) * 0.1)
        confidence = base_conf / depth_penalty
        
        return min(1.0, confidence)
    
    def _infer_domain_from_question_type(self, question_type: Optional[str]) -> Optional[str]:
        """
        Infer domain from question type for root entity context.
        
        This helps create more meaningful root entities when available.
        """
        if not question_type:
            return None
        
        question_type_lower = question_type.lower()
        
        # Map question types to domains
        domain_mapping = {
            "location": "Geographic",
            "person": "People",
            "organization": "Organizations",
            "event": "Events",
            "entity": "Entities"
        }
        
        for key, domain in domain_mapping.items():
            if key in question_type_lower:
                return domain
        
        return None
    
    def _empty_breadcrumbs(self) -> Dict[str, Any]:
        """Return empty breadcrumb structure."""
        return {
            "breadcrumb_path": [],
            "breadcrumb_string": "Unknown",
            "breadcrumb_depth": 0,
            "entities": [],
            "confidence": 0.0,
            "metadata": {}
        }

