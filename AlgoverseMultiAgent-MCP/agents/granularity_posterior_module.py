# agents/granularity_posterior_module.py
"""
Granularity Posterior Module: Bayesian filtering of retrieved documents.

Implements posterior probability update:
P(level|doc) ∝ P(doc|level) * P(level)

Where:
- P(level) = Prior probability from GranularityRegulator (uniform or query-based)
- P(doc|level) = Likelihood computed from document content
- P(level|doc) = Posterior (used for filtering/weighting)

This module filters documents at the evidence gate (retrieval → extraction),
preventing hierarchical level mismatches from propagating downstream.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from .regulators.granularity_regulator import GranularityRegulator

logger = logging.getLogger(__name__)


class GranularityPosteriorModule:
    """
    Bayesian posterior module for granularity-aware document filtering.
    
    Filters retrieved documents based on hierarchical level alignment,
    preventing level-mismatch errors from propagating to extraction/QA.
    """
    
    def __init__(self, filter_threshold: float = 0.3, weight_by_posterior: bool = True):
        """
        Initialize the Granularity Posterior Module.
        
        Args:
            filter_threshold: Minimum posterior score to keep document (0.0-1.0).
                Currently uses absolute threshold. Future improvement: could use
                relative threshold (e.g., keep top-k by posterior, or keep docs
                within ε of max posterior) to handle edge cases where all documents
                have moderate but still informative posterior scores.
            weight_by_posterior: If True, weight documents by posterior; if False, binary filter
        """
        self.granularity_regulator = GranularityRegulator()
        self.filter_threshold = filter_threshold
        self.weight_by_posterior = weight_by_posterior
        logger.debug("Granularity Posterior Module initialized")
    
    def filter_documents(
        self,
        documents: List[Dict[str, Any]],
        prior_domain: Optional[str],
        prior_level: Optional[str],
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Filter and weight documents based on granularity posterior.
        
        Bayesian Update:
        P(level|doc) = P(doc|level) * P(level) / P(doc)
        
        Where:
        - P(level) = Prior (uniform or query-based, default 0.5)
        - P(doc|level) = Likelihood from classify_evidence_level()
        - P(level|doc) = Posterior (used for filtering)
        
        Args:
            documents: List of retrieved documents (each with 'page_content', 'metadata', etc.)
            prior_domain: Required domain from GranularityRegulator
            prior_level: Required level from GranularityRegulator
            query: Optional query text for context
            
        Returns:
            Dict with:
            - 'filtered_documents': List of documents that passed filter
            - 'filtered_count': Number of documents filtered out
            - 'posterior_scores': List of posterior scores for each document
            - 'metadata': Filtering statistics
        """
        if not documents:
            return {
                'filtered_documents': [],
                'filtered_count': 0,
                'posterior_scores': [],
                'metadata': {
                    'total_docs': 0,
                    'filtered_docs': 0,
                    'kept_docs': 0,
                    'prior_domain': prior_domain,
                    'prior_level': prior_level
                }
            }
        
        # DESIGN CHOICE: When no granularity prior is available, we default to 
        # recall-oriented behavior (pass-through all documents) rather than 
        # attempting to infer granularity from the query.
        #
        # Rationale:
        # 1. Granularity inference from queries alone is error-prone and can
        #    introduce false negatives (filtering relevant documents)
        # 2. Downstream agents (ExtractorAgent, QAAgent) can handle level
        #    mismatches with their own filtering mechanisms
        # 3. Recall is prioritized over precision when uncertainty is high
        # 4. The GranularityRegulator already provides prior inference at the
        #    query stabilization stage; if it fails, we avoid cascading errors
        #
        # Future work: Could implement query-based granularity inference here
        # as a fallback, but would require careful validation to avoid over-filtering.
        if not prior_domain or not prior_level:
            logger.debug(
                f"Granularity Posterior: No prior granularity provided, "
                f"defaulting to recall-oriented pass-through for all {len(documents)} documents"
            )
            return {
                'filtered_documents': documents,
                'filtered_count': 0,
                'posterior_scores': [1.0] * len(documents),
                'metadata': {
                    'total_docs': len(documents),
                    'filtered_docs': 0,
                    'kept_docs': len(documents),
                    'prior_domain': None,
                    'prior_level': None,
                    'filtering_applied': False,
                    'reason': 'no_prior_available_recall_oriented'
                }
            }
        
        # Compute posterior for each document
        doc_scores = []
        for doc in documents:
            doc_text = doc.get('page_content', '') or doc.get('text', '')
            if not doc_text:
                # No text → low posterior
                doc_scores.append((doc, 0.0, None, None))
                continue
            
            # Compute likelihood: P(doc|level)
            likelihood = self._compute_likelihood(
                doc_text=doc_text,
                required_domain=prior_domain,
                required_level=prior_level
            )
            
            # Prior: P(level) = 0.5 (uniform prior, can be made query-based later)
            prior = 0.5
            
            # Posterior: P(level|doc) ∝ P(doc|level) * P(level)
            # Normalize to [0, 1] range
            posterior = min(1.0, likelihood * prior * 2.0)  # *2.0 for normalization
            
            doc_scores.append((doc, posterior, likelihood, prior))
        
        # Filter by threshold
        filtered_docs = []
        filtered_scores = []
        filtered_count = 0
        
        for doc, posterior, likelihood, prior in doc_scores:
            if posterior >= self.filter_threshold:
                # Apply posterior weighting if enabled
                if self.weight_by_posterior:
                    # Weight document score by posterior
                    original_score = doc.get('score', 1.0)
                    weighted_score = original_score * posterior
                    doc['score'] = weighted_score
                    doc['granularity_posterior'] = posterior
                    doc['granularity_metadata'] = {
                        'posterior': posterior,
                        'likelihood': likelihood,
                        'prior': prior,
                        'aligned': True
                    }
                else:
                    # Binary: just add metadata
                    doc['granularity_posterior'] = posterior
                    doc['granularity_metadata'] = {
                        'posterior': posterior,
                        'likelihood': likelihood,
                        'prior': prior,
                        'aligned': True
                    }
                
                filtered_docs.append(doc)
                filtered_scores.append(posterior)
            else:
                filtered_count += 1
                logger.debug(
                    f"Granularity Posterior: Filtered document (posterior={posterior:.3f} < {self.filter_threshold})"
                )
        
        logger.info(
            f"Granularity Posterior: Filtered {filtered_count}/{len(documents)} documents "
            f"(kept {len(filtered_docs)}, threshold={self.filter_threshold})"
        )
        
        return {
            'filtered_documents': filtered_docs,
            'filtered_count': filtered_count,
            'posterior_scores': filtered_scores,
            'metadata': {
                'total_docs': len(documents),
                'filtered_docs': filtered_count,
                'kept_docs': len(filtered_docs),
                'prior_domain': prior_domain,
                'prior_level': prior_level,
                'filter_threshold': self.filter_threshold,
                'filtering_applied': True,
                'avg_posterior': sum(filtered_scores) / len(filtered_scores) if filtered_scores else 0.0
            }
        }
    
    def _compute_likelihood(
        self,
        doc_text: str,
        required_domain: str,
        required_level: str
    ) -> float:
        """
        Compute likelihood P(doc|level) using GranularityRegulator.
        
        Likelihood is based on:
        1. Evidence level classification (doc_domain, doc_level)
        2. Monotonic constraint check (coarse→fine allowed, fine→coarse forbidden)
        3. Domain match check
        
        Args:
            doc_text: Document text content
            required_domain: Required domain
            required_level: Required level
            
        Returns:
            Likelihood score [0.0, 1.0]
        """
        # Classify document's granularity level
        doc_classification = self.granularity_regulator.classify_evidence_level(doc_text)
        
        if not doc_classification:
            # Unclassified document → low likelihood
            return 0.3  # Neutral likelihood for unclassified
        
        doc_domain, doc_level = doc_classification
        
        # Check domain match
        if doc_domain != required_domain:
            # Cross-domain → very low likelihood
            return 0.1
        
        # Check level alignment (with monotonic constraint)
        required_level_num = self.granularity_regulator.get_level_number(required_domain, required_level)
        doc_level_num = self.granularity_regulator.get_level_number(doc_domain, doc_level)
        
        if required_level_num is None or doc_level_num is None:
            # Can't compare → neutral likelihood
            return 0.5
        
        # Monotonic constraint: coarse→fine allowed, fine→coarse forbidden
        if doc_level_num >= required_level_num:
            # Document is at same or finer level → high likelihood
            # Exact match = 1.0, finer = 0.9, much finer = 0.8
            level_delta = doc_level_num - required_level_num
            if level_delta == 0:
                return 1.0  # Exact match
            elif level_delta == 1:
                return 0.9  # One level finer (acceptable)
            else:
                return max(0.7, 1.0 - (level_delta - 1) * 0.1)  # Much finer
        else:
            # Document is coarser than required → low likelihood (violation)
            level_delta = required_level_num - doc_level_num
            return max(0.0, 0.5 - level_delta * 0.2)  # Penalty for coarser
    
    def compute_posterior_for_entities(
        self,
        entities: List[str],
        prior_domain: Optional[str],
        prior_level: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Compute posterior scores for entities (for filtering irrelevant entities).
        
        Used by downstream agents to filter entities that don't match granularity.
        
        Args:
            entities: List of entity names
            prior_domain: Required domain
            prior_level: Required level
            
        Returns:
            List of dicts with entity, posterior, and metadata
        """
        if not prior_domain or not prior_level:
            # No prior → return all entities with neutral posterior
            return [
                {
                    'entity': entity,
                    'posterior': 0.5,
                    'metadata': {'aligned': True, 'reason': 'no_prior'}
                }
                for entity in entities
            ]
        
        annotated_entities = []
        for entity in entities:
            # Use GranularityRegulator's compute_granularity_metadata
            metadata = self.granularity_regulator.compute_granularity_metadata(
                entity_text=entity,
                required_domain=prior_domain,
                required_level=prior_level
            )
            
            # Convert violation status to posterior
            if metadata.get('granularity_violation'):
                posterior = 0.2  # Low posterior for violations
            elif metadata.get('is_unclassified'):
                posterior = 0.4  # Medium-low for unclassified
            elif metadata.get('is_cross_domain'):
                posterior = 0.1  # Very low for cross-domain
            else:
                posterior = 0.9  # High for aligned entities
            
            annotated_entities.append({
                'entity': entity,
                'posterior': posterior,
                'metadata': metadata
            })
        
        return annotated_entities

