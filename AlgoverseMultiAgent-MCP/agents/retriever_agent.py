from typing import Dict, Any, List, Optional, Union, Tuple
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from .base_agent import BaseAgent, AgentResponse
import numpy as np
import json
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class LocalEmbeddings(Embeddings):
    """FAISS-compatible wrapper for local embedding models with preprocessing."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize the local embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            device: Device to run the model on ('cuda', 'mps', 'cpu')
        """
        super().__init__()
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.model.eval()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better embedding quality."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that might hurt embedding quality
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        return text
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query with preprocessing."""
        processed_text = self._preprocess_text(text)
        embeddings = self.model.encode([processed_text], convert_to_numpy=True)
        return embeddings[0].tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with preprocessing."""
        if not texts:
            return []
        
        # Preprocess all texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Return numpy array for FAISS compatibility
        embeddings = self.model.encode(processed_texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def __call__(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Alias for embed_documents for compatibility."""
        if isinstance(texts, str):
            return [self.embed_query(texts)]
        return self.embed_documents(texts)

class RetrieverAgent(BaseAgent):
    """
    Enhanced Retriever Agent using FAISS for fast, scalable search over large corpora.
    Implements the MA-RAG retrieval tool with proper preprocessing and embedding.
    
    **Entropy-Aware Retrieval:**
    Retrieval parameters (k, min_similarity) are dynamically adjusted based on entropy and diffusion:
    - High entropy (uncertainty) → broaden retrieval (increase k, lower similarity threshold)
    - Low entropy (certainty) → narrow retrieval (decrease k, higher similarity threshold)
    - This makes retrieval part of the diffusion process, reducing scatter at root cause
    - Mathematically consistent: retrieval adapts to reasoning state uncertainty
    """
    
    def __init__(
        self, 
        documents: List[Document] = None, 
        model_config: Optional[Dict[str, Any]] = None,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = None,
        top_k: int = 5,
        min_similarity: float = 0.2,
        batch_size: int = 32
    ):
        """
        Initialize the Enhanced Retriever Agent.
        
        Args:
            documents: List of Document objects to create the vector store
            model_config: Configuration for the embedding model
            model_name: Name of the local embedding model to use
            device: Device to run the model on ('cuda', 'mps', 'cpu')
            top_k: Default number of documents to retrieve
            min_similarity: Minimum similarity score threshold
            batch_size: Batch size for embedding processing
        """
        super().__init__("retriever_agent", model_config, model_name)
        
        # Initialize local embeddings with preprocessing
        self.embeddings = LocalEmbeddings(
            model_name=model_name,
            device=device or ("cuda" if (model_config and model_config.get("use_cuda", False)) else None)
        )
        self.vector_store = None
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.batch_size = batch_size
        self._granularity_regulator = None  # ✅ Cached GranularityRegulator instance for performance
        
        if documents:
            self._create_vector_store(documents)
    
    def _create_vector_store(self, documents: List[Document]):
        """Create or update the vector store with the given documents."""
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        # Preprocess documents for better retrieval
        processed_docs = []
        for doc in documents:
            # Clean and preprocess document content
            cleaned_content = self.embeddings._preprocess_text(doc.page_content)
            
            # Create new document with cleaned content
            processed_doc = Document(
                page_content=cleaned_content,
                metadata=doc.metadata
            )
            processed_docs.append(processed_doc)
        
        self.vector_store = FAISS.from_documents(
            documents=processed_docs,
            embedding=self.embeddings
        )
        
        logger.info(f"Vector store created successfully with {len(processed_docs)} documents")
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to the vector store."""
        if not self.vector_store:
            self._create_vector_store(documents)
        else:
            # Preprocess new documents
            processed_docs = []
            for doc in documents:
                cleaned_content = self.embeddings._preprocess_text(doc.page_content)
                processed_doc = Document(
                    page_content=cleaned_content,
                    metadata=doc.metadata
                )
                processed_docs.append(processed_doc)
            
            self.vector_store.add_documents(processed_docs)
            logger.info(f"Added {len(processed_docs)} new documents to vector store")
    

    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process the input sub-query and retrieve relevant documents using FAISS.
        
        **Entropy-Aware Retrieval with Regulator Integration:**
        - Adaptively adjusts thresholds based on entropy/diffusion
        - Uses regulator constraints to filter and rerank documents
        - Integrates with flow snapshot for context-aware retrieval
        - Maintains backward compatibility with basic retrieval
        
        Args:
            input_data: Dictionary containing:
                - 'query': The sub-query to retrieve documents for
                - 'k': Number of documents to retrieve (default: top_k)
                - 'min_similarity': Minimum similarity score threshold
                - 'filter': Filter criteria for the documents
                - 'include_scores': Whether to include similarity scores (default: True)
                - 'entropy_penalty': Entropy value H(t) from flow snapshot (optional)
                - 'diffusion_penalty': Diffusion coefficient D(t) from flow snapshot (optional)
                - 'regulator_constraints': List of regulator constraints (optional)
                - 'flow_snapshot': Flow snapshot dict with beliefs, confidence, etc. (optional)
                
        Returns:
            AgentResponse containing the retrieved documents and metadata
        """
        if not self.vector_store:
            return AgentResponse(
                content="Error: No vector store initialized",
                metadata={"error": "Vector store not initialized"}
            )
            
        query = input_data.get('query', '').strip()
        if not query:
            return AgentResponse(
                content="Error: No query provided",
                metadata={"error": "No query provided"}
            )
            
        try:
            # Get base retrieval parameters
            base_k = int(input_data.get('k', self.top_k))
            base_min_similarity = float(input_data.get('min_similarity', self.min_similarity))
            filter_criteria = input_data.get('filter', {})
            include_scores = bool(input_data.get('include_scores', True))
            
            # ====================================================================
            # ENTROPY-AWARE RETRIEVAL: Adaptive parameters based on uncertainty
            # ====================================================================
            entropy_penalty = float(input_data.get('entropy_penalty', 0.0))  # H(t) from flow snapshot
            diffusion_penalty = float(input_data.get('diffusion_penalty', 0.0))  # D(t) from flow snapshot
            regulator_constraints = input_data.get('regulator_constraints', [])
            flow_snapshot = input_data.get('flow_snapshot', {})
            
            # Check if entropy-aware retrieval is enabled
            is_entropy_aware = (entropy_penalty > 0.0 or diffusion_penalty > 0.0 or 
                               regulator_constraints or flow_snapshot)
            
            # Combine entropy and diffusion to measure total uncertainty
            # Both contribute to uncertainty: entropy = belief uncertainty, diffusion = query instability
            total_uncertainty = min(1.0, entropy_penalty + diffusion_penalty * 0.5) if is_entropy_aware else 0.0
            
            # Adaptive k: High uncertainty → retrieve more documents (broaden search)
            if is_entropy_aware:
                uncertainty_factor_k = 1.5  # How much uncertainty affects k
                k_adjusted = int(base_k * (1.0 + uncertainty_factor_k * total_uncertainty))
                k = min(k_adjusted, 30)  # Cap at 30 for scattered scenarios
            else:
                k = min(base_k, 20)  # Standard cap for basic retrieval
            
            # Adaptive min_similarity: High uncertainty → lower threshold (accept more candidates)
            if is_entropy_aware:
                uncertainty_factor_sim = 0.4  # How much uncertainty affects similarity threshold
                min_similarity_adjusted = base_min_similarity * (1.0 - uncertainty_factor_sim * total_uncertainty)
                min_similarity = max(0.1, min_similarity_adjusted)  # Floor at 0.1
            else:
                min_similarity = base_min_similarity
            
            # Log entropy-aware adjustments
            if is_entropy_aware:
                logger.debug(
                    f"🎯 Entropy-aware retrieval: H(t)={entropy_penalty:.3f}, D(t)={diffusion_penalty:.3f}, "
                    f"uncertainty={total_uncertainty:.3f} → k={base_k}→{k}, min_sim={base_min_similarity:.3f}→{min_similarity:.3f}"
                )

            logger.info(f"🔍 Retrieval: query='{query[:80]}...', k={k}, min_sim={min_similarity}")
            
            # Extract breadcrumb_scope from input (from StepDefinerAgent Search Schema)
            breadcrumb_scope = input_data.get('breadcrumb_scope')
            
            # Retrieve with larger k if breadcrumb_scope is provided (for re-ranking)
            # Use a larger candidate pool to make Bayesian reranking meaningful.
            retrieval_k = max(k * 2, 50) if breadcrumb_scope else k
            if breadcrumb_scope:
                logger.info(f"🔍 Structural Intent Detected: {breadcrumb_scope}")
                logger.info(f"📈 Retrieval pool expanded to {retrieval_k} for Bayesian auditing")
            
            # Perform similarity search with FAISS
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=retrieval_k,  # Retrieve more if we have breadcrumb scope
                filter=filter_criteria
            )
            
            # Convert FAISS scores to similarity scores (FAISS returns distance, we need similarity)
            # FAISS returns (distance, doc) where lower distance = more similar
            # Convert to similarity: similarity = 1 / (1 + distance)
            docs_and_similarities = [
                (doc, 1.0 / (1.0 + score)) for doc, score in docs_and_scores
            ]
            
            # Apply Bayesian re-ranking by breadcrumb scope (if provided)
            if breadcrumb_scope:
                logger.debug(f"🔄 Applying Bayesian re-ranking with breadcrumb scope: {breadcrumb_scope}")
                docs_and_similarities = self._bayesian_rerank_by_breadcrumb(
                    docs_and_similarities,
                    breadcrumb_scope,
                    heuristic_conf=0.62  # From empirical evaluation
                )
                # Limit to original k after re-ranking
                docs_and_similarities = docs_and_similarities[:k]
            
            # Process and apply entropy-aware filtering/reranking
            # Note: docs_and_similarities already has re-ranked scores if breadcrumb_scope was provided
            if is_entropy_aware:
                results, similarities, filtered_count = self._apply_entropy_aware_processing(
                    docs_and_similarities,  # Use re-ranked results
                    query,
                    min_similarity,
                    regulator_constraints,
                    flow_snapshot,
                    entropy_penalty,
                    diffusion_penalty,
                    include_scores
                )
            else:
                # Basic retrieval processing with granularity bias
                results, similarities, filtered_count = self._apply_basic_processing(
                    docs_and_similarities,  # Use re-ranked results
                    min_similarity,
                    include_scores,
                    regulator_constraints=regulator_constraints if regulator_constraints else None
                )
            
            # Log results
            if similarities:
                logger.info(f"✅ Retrieved {len(results)} docs (filtered: {filtered_count}), sim: {min(similarities):.3f}-{max(similarities):.3f} (avg: {np.mean(similarities):.3f})")
                if logger.isEnabledFor(logging.DEBUG):
                    for i, doc_data in enumerate(results[:3]):
                        preview = doc_data["page_content"][:80].replace('\n', ' ')
                        logger.debug(f"   [{i+1}] (sim={doc_data['score']:.3f}) {preview}...")
            else:
                logger.warning(f"⚠️  No documents passed threshold (min_sim={min_similarity})")

            # --- Non-zero docs floor (deterministic) ---
            metadata_floor = {
                "low_similarity_floor": False,
                "filtered_count": filtered_count,
                "k_requested": k,
                "k_actual": len(results),
                "min_similarity_effective": min_similarity,
            }
            if not results and docs_and_similarities:
                # Take top by base similarity (ignore min_similarity/regulator filters)
                fallback = sorted(
                    docs_and_similarities,  # Already has (doc, similarity) format
                    key=lambda x: x[1],
                    reverse=True
                )
                # Align floor with K_MIN-style floor (5)
                take = min(len(fallback), max(int(input_data.get("k", self.top_k) or 0), 5))
                fallback = fallback[:take]
                for doc, base_sim in fallback:
                    doc_id = doc.metadata.get('id', f"doc_{len(results)+1}")
                    doc.metadata['id'] = doc_id
                    doc_data = {
                        "id": doc_id,
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(base_sim) if include_scores else None
                    }
                    results.append(doc_data)
                    similarities.append(base_sim)
                metadata_floor.update({
                    "low_similarity_floor": True,
                    "k_actual": len(results),
                })
            
            # Update history
            self._update_history("user", f"Retrieve documents for: {query}")
            self._update_history(
                "assistant", 
                f"Retrieved {len(results)} documents "
                f"(avg similarity: {np.mean(similarities) if similarities else 0:.2f})"
            )
            
            # Prepare response data
            response_data = {
                "query": query,
                "documents": [
                    {k: v for k, v in doc.items() if k != 'score' or include_scores}
                    for doc in results
                ]
            }
            
            if include_scores and similarities:
                response_data["scores"] = [float(s) for s in similarities]
            
            # Build metadata
            metadata = {
                "query": query,
                "num_documents": len(results),
                "average_score": float(np.mean(similarities)) if similarities else 0.0,
                "min_score": float(min(similarities)) if similarities else 0.0,
                "max_score": float(max(similarities)) if similarities else 0.0,
                "retrieval_parameters": {
                    "k": k,
                    "min_similarity": min_similarity,
                    "filter": filter_criteria,
                    "model": self.embeddings.model_name,
                    "batch_size": self.batch_size,
                    "entropy_aware": is_entropy_aware
                },
                "documents": results
            }
            metadata.update(metadata_floor)
            
            # Add entropy-aware metadata if enabled
            if is_entropy_aware:
                metadata["entropy_metadata"] = {
                    "entropy_penalty": entropy_penalty,
                    "diffusion_penalty": diffusion_penalty,
                    "total_uncertainty": total_uncertainty,
                    "regulator_constraints_count": len(regulator_constraints),
                    "flow_snapshot_used": bool(flow_snapshot)
                }
            
            return AgentResponse(
                content=json.dumps(response_data, ensure_ascii=False),
                metadata=metadata
            )
            
        except Exception as e:
            error_msg = f"Error retrieving documents: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return AgentResponse(
                content=error_msg,
                metadata={
                    "error": str(e),
                    "error_type": e.__class__.__name__
                }
            )
    
    def _get_granularity_regulator(self):
        """
        Get or create GranularityRegulator instance (cached for performance).
        
        Returns:
            GranularityRegulator instance or None if import fails
        """
        if self._granularity_regulator is None:
            try:
                from .regulators.granularity_regulator import GranularityRegulator
                self._granularity_regulator = GranularityRegulator()
            except ImportError:
                return None
        return self._granularity_regulator
    
    def _compute_granularity_likelihood(
        self,
        doc_text: str,
        required_domain: Optional[str],
        required_level: Optional[str]
    ) -> float:
        """
        Compute granularity likelihood P(doc|granularity) for a document.
        
        This is used to bias retrieval toward documents matching the expected
        hierarchical level, addressing retrieval quality at the source.
        
        Args:
            doc_text: Document text content
            required_domain: Required domain from GranularityRegulator
            required_level: Required level from GranularityRegulator
            
        Returns:
            Likelihood score [0.0, 1.0]. Returns 1.0 if no granularity prior.
        """
        # If no granularity prior, return neutral likelihood (no bias)
        if not required_domain or not required_level:
            return 1.0
        
        # ✅ Performance fix: Use cached GranularityRegulator instance
        granularity_reg = self._get_granularity_regulator()
        if not granularity_reg:
            return 1.0
        
        # Classify document's granularity level
        doc_classification = granularity_reg.classify_evidence_level(doc_text)
        
        if not doc_classification:
            # Unclassified document → neutral likelihood (no penalty, no boost)
            return 0.5
        
        doc_domain, doc_level = doc_classification
        
        # Check domain match
        if doc_domain != required_domain:
            # Cross-domain → very low likelihood (matches GranularityPosteriorModule)
            return 0.1
        
        # Check level alignment (with monotonic constraint)
        required_level_num = granularity_reg.get_level_number(required_domain, required_level)
        doc_level_num = granularity_reg.get_level_number(doc_domain, doc_level)
        
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
            return max(0.1, 0.5 - level_delta * 0.2)  # Penalty for coarser
    
    def _apply_basic_processing(
        self,
        docs_and_scores: List[tuple],
        min_similarity: float,
        include_scores: bool,
        regulator_constraints: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[Dict[str, Any]], List[float], int]:
        """
        Basic document processing with optional granularity bias.
        
        Args:
            docs_and_scores: List of (Document, score) tuples from FAISS
            min_similarity: Minimum similarity threshold
            include_scores: Whether to include scores
            regulator_constraints: Optional regulator constraints for granularity bias
            
        Returns:
            Tuple of (results, similarities, filtered_count)
        """
        # Extract granularity prior from constraints
        required_domain = None
        required_level = None
        if regulator_constraints:
            for constraint in regulator_constraints:
                if isinstance(constraint, dict):
                    if constraint.get("regulator_name") == "Granularity":
                        params = constraint.get("parameters", {})
                        required_domain = params.get("required_domain")
                        required_level = params.get("required_level")
                        break
        
        # Log granularity bias application
        if required_domain and required_level:
            logger.debug(
                f"🎯 Granularity bias in retrieval: prior={required_domain}/{required_level}, "
                f"will bias docs toward this level"
            )
        
        results = []
        similarities = []
        filtered_count = 0
        doc_scores = []  # Store (doc, combined_score, base_similarity) for re-ranking
        
        for doc, score in docs_and_scores:
            # Convert score to similarity (higher is better)
            base_similarity = 1.0 / (1.0 + score)
            
            # Skip if below threshold
            if base_similarity < min_similarity:
                filtered_count += 1
                continue
            
            # ✅ GRANULARITY BIAS: Compute likelihood and combine with similarity
            granularity_likelihood = self._compute_granularity_likelihood(
                doc_text=doc.page_content,
                required_domain=required_domain,
                required_level=required_level
            )
            
            # Combined score: similarity × granularity_likelihood
            # This biases retrieval toward documents matching expected granularity
            combined_score = base_similarity * granularity_likelihood
            
            doc_scores.append((doc, combined_score, base_similarity))
        
        # Re-rank by combined score (granularity-biased)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        for doc, combined_score, base_similarity in doc_scores:
            # Add document ID if not present
            doc_id = doc.metadata.get('id', f"doc_{len(results)+1}")
            doc.metadata['id'] = doc_id

            # Add to results
            doc_data = {
                "id": doc_id,
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(combined_score) if include_scores else None
            }
            results.append(doc_data)
            similarities.append(combined_score)
        
        # Log granularity bias summary
        if required_domain and required_level:
            avg_likelihood = np.mean([
                self._compute_granularity_likelihood(
                    doc["page_content"],
                    required_domain,
                    required_level
                ) for doc in results[:5]  # Sample first 5
            ]) if results else 0.0
            logger.debug(
                f"✅ Granularity bias applied: {len(results)} docs ranked "
                f"(prior: {required_domain}/{required_level}, avg likelihood: {avg_likelihood:.3f})"
            )
        
        return results, similarities, filtered_count
    
    def _apply_entropy_aware_processing(
        self,
        docs_and_scores: List[tuple],
        query: str,
        min_similarity: float,
        regulator_constraints: List[Dict[str, Any]],
        flow_snapshot: Dict[str, Any],
        entropy_penalty: float,
        diffusion_penalty: float,
        include_scores: bool
    ) -> Tuple[List[Dict[str, Any]], List[float], int]:
        """
        Apply entropy-aware processing with regulator constraints and flow snapshot.
        
        This method:
        1. Filters documents based on similarity threshold
        2. Applies regulator constraint filtering (plan alignment, entity anchors)
        3. Reranks documents based on evidence terms, entity anchors, and beliefs
        4. Adjusts scores based on diffusion penalty
        
        Args:
            docs_and_scores: List of (Document, score) tuples from FAISS
            query: The query string
            min_similarity: Minimum similarity threshold
            regulator_constraints: List of regulator constraint dicts
            flow_snapshot: Flow snapshot dict with beliefs, confidence, etc.
            entropy_penalty: Entropy value H(t)
            diffusion_penalty: Diffusion coefficient D(t)
            include_scores: Whether to include scores
            
        Returns:
            Tuple of (results, similarities, filtered_count)
        """
        # Extract regulator constraint information (with hierarchical context)
        constraint_info = self._extract_constraint_info(regulator_constraints, flow_snapshot)
        
        # Extract flow snapshot information
        snapshot_info = self._extract_snapshot_info(flow_snapshot)
        
        # Process documents with entropy-aware scoring
        doc_scores = []
        filtered_count = 0
        
        for doc, faiss_score in docs_and_scores:
            # Convert FAISS distance to similarity
            base_similarity = 1.0 / (1.0 + faiss_score)
            
            # Skip if below base threshold
            if base_similarity < min_similarity:
                filtered_count += 1
                continue
            
            # ✅ GRANULARITY BIAS: Compute likelihood and apply to base similarity first
            # Extract granularity prior from constraints (reuse from constraint_info if available)
            required_domain = constraint_info.get("required_domain")
            required_level = constraint_info.get("required_level")
            
            # If not in constraint_info, extract from constraints directly
            if not required_domain or not required_level:
                for constraint in regulator_constraints:
                    if isinstance(constraint, dict):
                        if constraint.get("regulator_name") == "Granularity":
                            params = constraint.get("parameters", {})
                            required_domain = params.get("required_domain")
                            required_level = params.get("required_level")
                            break
            
            granularity_likelihood = self._compute_granularity_likelihood(
                doc_text=doc.page_content,
                required_domain=required_domain,
                required_level=required_level
            )
            
            # Apply granularity bias to base similarity
            granularity_biased_similarity = base_similarity * granularity_likelihood
            
            # Calculate entropy-aware score (now on granularity-biased similarity)
            enhanced_score = self._calculate_entropy_aware_score(
                doc=doc,
                query=query,
                base_similarity=granularity_biased_similarity,  # Use granularity-biased similarity
                constraint_info=constraint_info,
                snapshot_info=snapshot_info,
                entropy_penalty=entropy_penalty,
                diffusion_penalty=diffusion_penalty
            )
            
            # Filter based on regulator constraints
            if not self._passes_constraint_filters(doc, query, constraint_info, snapshot_info):
                filtered_count += 1
                continue
            
            doc_scores.append((doc, enhanced_score, base_similarity))
        
        # Rerank by enhanced score
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        similarities = []
        
        for doc, enhanced_score, base_similarity in doc_scores:
            doc_id = doc.metadata.get('id', f"doc_{len(results)+1}")
            doc.metadata['id'] = doc_id
            
            doc_data = {
                "id": doc_id,
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(enhanced_score) if include_scores else None
            }
            results.append(doc_data)
            similarities.append(enhanced_score)
        
        logger.debug(
            f"Entropy-aware processing: {len(results)} docs after filtering/reranking "
            f"(filtered: {filtered_count}, avg enhanced score: {np.mean(similarities) if similarities else 0:.3f})"
        )
        
        return results, similarities, filtered_count
    
    def _extract_constraint_info(
        self,
        regulator_constraints: List[Dict[str, Any]],
        flow_snapshot: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract useful information from regulator constraints.
        
        ✅ FIRST PRINCIPLES: Extracts parent-level entities for hierarchical expansion.
        When lower-level entities are found but higher-level is required, extracts
        parent entity names to improve retrieval.
        
        Args:
            regulator_constraints: List of regulator constraint dicts
            flow_snapshot: Optional flow snapshot with hierarchical context
            
        Returns:
            Dict with evidence_terms, entity_anchors, plan_alignment, etc.
        """
        info = {
            "evidence_terms": [],
            "entity_anchors": [],
            "plan_alignment": 0.0,
            "plan_keywords": [],
            "relation_direction": None,
            "constraint_weights": {},
            "required_domain": None,  # ✅ Store granularity prior for reuse
            "required_level": None
        }
        
        # Extract required hierarchical level from flow snapshot or GranularityRegulator constraint
        required_domain = None
        required_level = None
        granularity_reg = None
        
        for constraint in regulator_constraints:
            if isinstance(constraint, dict):
                regulator_name = constraint.get("regulator_name", "")
                params = constraint.get("parameters", {})
                weight = constraint.get("weight", 0.0)
                
                info["constraint_weights"][regulator_name] = weight
                
                # Extract hierarchical level requirement from GranularityRegulator
                if regulator_name == "Granularity":
                    required_domain = params.get("required_domain")
                    required_level = params.get("required_level")
                    # ✅ Store in info for reuse
                    info["required_domain"] = required_domain
                    info["required_level"] = required_level
                    if required_domain and required_level:
                        from .regulators.granularity_regulator import GranularityRegulator
                        granularity_reg = GranularityRegulator()
                
                # ✅ EXPERIMENT 1b: Adding back EntityRegulator constraint extraction
                elif regulator_name == "Entity":
                    entities = params.get("entities", [])
                    main_entity = params.get("main_entity")
                    if main_entity:
                        info["entity_anchors"].append(main_entity)
                    info["entity_anchors"].extend(entities[:3])  # Top 3 entities
                    
                    # ✅ FIRST PRINCIPLES FIX: Extract parent-level entities for hierarchical expansion
                    # If we have lower-level entities but need higher-level, extract parent names
                    if granularity_reg and required_domain and required_level:
                        required_level_num = granularity_reg.get_level_number(required_domain, required_level)
                        if required_level_num:
                            for entity in entities[:3]:
                                entity_domain, entity_level, entity_level_num = granularity_reg.classify_entity_level(entity)
                                if entity_domain == required_domain and entity_level_num:
                                    # If entity is at lower level than required, extract parent name
                                    if entity_level_num > required_level_num:
                                        parent_name = granularity_reg.extract_parent_level_name(
                                            entity, required_domain, required_level
                                        )
                                        if parent_name and parent_name not in info["entity_anchors"]:
                                            info["entity_anchors"].append(parent_name)
                                            logger.debug(
                                                f"Retriever: Added parent entity '{parent_name}' "
                                                f"for hierarchical expansion (entity: {entity}, "
                                                f"required: {required_domain}/{required_level})"
                                            )
                
                # ✅ FINAL: Removed EvidenceRegulator, RelationRegulator, ConfidenceRegulator constraint extraction
                # These regulators removed - no benefit, cause issues. Basic extraction still happens in flow_update.py
                
                # Extract from PlanRegulator
                elif regulator_name == "Plan":
                    info["plan_alignment"] = params.get("alignment", 0.0)
                    info["plan_keywords"] = params.get("goal_keywords", [])
        
        # Deduplicate
        info["evidence_terms"] = list(set(info["evidence_terms"]))[:10]
        info["entity_anchors"] = list(set(info["entity_anchors"]))[:5]
        
        return info
    
    def _calculate_breadcrumb_match_level(
        self,
        chunk_breadcrumb: List[str],
        target_scope: List[str]
    ) -> float:
        """
        Calculate prefix match level between chunk breadcrumb and target scope.
        
        Uses prefix matching: scope must be a prefix of chunk breadcrumb.
        Examples:
        - scope=["NASA"], chunk=["NASA", "DLR"] → 1.0 (perfect match)
        - scope=["NASA", "DLR"], chunk=["NASA", "DLR", "Engines"] → 1.0 (perfect match)
        - scope=["NASA"], chunk=["DLR"] → 0.0 (no match)
        - scope=["NASA", "DLR"], chunk=["NASA"] → 0.5 (partial match)
        
        Args:
            chunk_breadcrumb: Breadcrumb path from document metadata
            target_scope: Target breadcrumb scope from StepDefinerAgent
            
        Returns:
            Match level (0.0 to 1.0)
        """
        if not target_scope or not chunk_breadcrumb:
            return 0.0
        
        # Prefix matching: scope must be a prefix of chunk breadcrumb
        min_len = min(len(target_scope), len(chunk_breadcrumb))
        matching_levels = 0
        
        for i in range(min_len):
            if chunk_breadcrumb[i].lower() == target_scope[i].lower():
                matching_levels += 1
            else:
                break
        
        # Normalize by target scope length (how much of scope matched)
        return matching_levels / len(target_scope) if target_scope else 0.0
    
    def _calculate_structural_prior(
        self,
        breadcrumb_match_level: float,
        breadcrumb_confidence: float,
        heuristic_conf: float = 0.62
    ) -> float:
        """
        Calculate structural prior P(R|S) based on match quality.
        
        Combines:
        - breadcrumb_match_level: How well the path matches (0.0-1.0)
        - breadcrumb_confidence: Confidence in breadcrumb extraction (0.0-1.0)
        - heuristic_conf: Overall confidence in heuristic (0.62 from empirical evaluation)
        
        Returns:
            Prior probability P(R|S) (0.0 to 1.0)
        """
        # Perfect match (1.0) → use full heuristic confidence
        # No match (0.0) → use epsilon (small probability)
        epsilon = 0.1
        
        # Weighted by both match quality and extraction confidence
        match_quality = breadcrumb_match_level * breadcrumb_confidence
        
        # Interpolate between epsilon and heuristic_conf
        prior = epsilon + (heuristic_conf - epsilon) * match_quality
        
        return prior
    
    def _bayesian_rerank_by_breadcrumb(
        self,
        docs_and_scores: List[Tuple[Document, float]],
        breadcrumb_scope: Optional[List[str]],
        heuristic_conf: float = 0.62
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank documents using Bayesian update with breadcrumb scope.
        
        Bayesian Update: P(R|D) ∝ P(D|R) * P(R|S)
        
        Where:
        - P(R|D) = Posterior: Probability document is relevant given evidence
        - P(D|R) = Likelihood: Semantic similarity (evidence from vector search)
        - P(R|S) = Prior: Probability relevant given structural scope match
        
        Args:
            docs_and_scores: List of (Document, semantic_score) tuples
            breadcrumb_scope: Target breadcrumb scope from StepDefinerAgent
            heuristic_conf: Confidence in breadcrumb heuristic (0.62 from tests)
            
        Returns:
            Re-ranked list of (Document, posterior_score) tuples
        """
        if not breadcrumb_scope:
            # Bayesian interpretation: empty scope implies a uniform prior.
            # P(R|D) ∝ P(D|R) * 1.0, so no re-ranking is required.
            return docs_and_scores
        
        reranked = []
        for doc, semantic_score in docs_and_scores:
            # Get breadcrumb from document metadata
            chunk_breadcrumb = doc.metadata.get('breadcrumb_path', [])
            breadcrumb_confidence = doc.metadata.get('breadcrumb_confidence', 0.5)
            
            # Calculate match level (prefix matching)
            match_level = self._calculate_breadcrumb_match_level(
                chunk_breadcrumb,
                breadcrumb_scope
            )
            
            # Calculate structural prior P(R|S)
            prior = self._calculate_structural_prior(
                match_level,
                breadcrumb_confidence,
                heuristic_conf
            )
            
            # Bayesian update: Posterior ∝ Likelihood * Prior
            # P(R|D) ∝ P(D|R) * P(R|S)
            posterior = semantic_score * prior
            
            reranked.append((doc, posterior))
        
        # Sort by posterior (descending)
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked
    
    def _extract_snapshot_info(self, flow_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract useful information from flow snapshot.
        
        Returns:
            Dict with beliefs, confidence, entity_anchors, evidence_terms, etc.
        """
        info = {
            "beliefs": {},
            "confidence": 0.5,
            "entity_anchors": {},
            "evidence_terms": [],
            "plan_alignment": 0.0
        }
        
        if not flow_snapshot:
            return info
        
        info["beliefs"] = flow_snapshot.get("beliefs", {})
        info["confidence"] = flow_snapshot.get("confidence", 0.5)
        info["entity_anchors"] = flow_snapshot.get("entity_anchors", {})
        info["evidence_terms"] = flow_snapshot.get("evidence_terms", [])
        info["plan_alignment"] = flow_snapshot.get("plan_alignment", 0.0)
        
        return info
    
    def _calculate_entropy_aware_score(
        self,
        doc: Document,
        query: str,
        base_similarity: float,
        constraint_info: Dict[str, Any],
        snapshot_info: Dict[str, Any],
        entropy_penalty: float,
        diffusion_penalty: float
    ) -> float:
        """
        Calculate entropy-aware score for a document.
        
        Combines:
        - Base similarity from FAISS
        - Evidence term boosts (potential well)
        - Entity anchor boosts (Dirichlet anchor)
        - Belief distribution weighting
        - Diffusion penalty adjustment
        
        Args:
            doc: Document to score
            query: Query string
            base_similarity: Base similarity from FAISS
            constraint_info: Extracted constraint information
            snapshot_info: Extracted snapshot information
            entropy_penalty: Entropy value H(t)
            diffusion_penalty: Diffusion coefficient D(t)
            
        Returns:
            Enhanced similarity score
        """
        score = base_similarity
        doc_text = doc.page_content.lower()
        doc_metadata = doc.metadata
        
        # 1. Evidence term boost (potential well)
        # Documents containing evidence terms get boosted
        evidence_terms = constraint_info.get("evidence_terms", [])
        if evidence_terms:
            evidence_matches = sum(
                1 for term in evidence_terms[:5]
                if term.lower() in doc_text
            )
            evidence_boost = min(0.15, evidence_matches * 0.03)  # Max 15% boost
            score += evidence_boost
        
        # 2. Entity anchor boost (Dirichlet anchor)
        # Documents containing anchored entities get boosted
        entity_anchors = constraint_info.get("entity_anchors", [])
        if entity_anchors:
            entity_matches = sum(
                1 for entity in entity_anchors[:3]
                if entity.lower() in doc_text
            )
            entity_boost = min(0.2, entity_matches * 0.07)  # Max 20% boost
            score += entity_boost
        
        # Also check snapshot entity anchors
        snapshot_entities = list(snapshot_info.get("entity_anchors", {}).keys())
        if snapshot_entities:
            snapshot_matches = sum(
                1 for entity in snapshot_entities[:3]
                if entity.lower() in doc_text
            )
            snapshot_boost = min(0.15, snapshot_matches * 0.05)
            score += snapshot_boost
        
        # 3. Belief distribution weighting
        # Documents matching high-probability beliefs get boosted
        beliefs = snapshot_info.get("beliefs", {})
        if beliefs:
            # Check if document content matches any high-probability belief
            for belief_entity, belief_prob in sorted(beliefs.items(), key=lambda x: x[1], reverse=True)[:3]:
                if belief_entity.lower() in doc_text and belief_prob > 0.3:
                    belief_boost = min(0.1, belief_prob * 0.2)  # Max 10% boost
                    score += belief_boost
                    break  # Only boost for top matching belief
        
        # 4. Plan alignment boost
        # Documents aligned with plan goal get boosted
        plan_alignment = constraint_info.get("plan_alignment", 0.0)
        plan_keywords = constraint_info.get("plan_keywords", [])
        if plan_keywords:
            plan_matches = sum(
                1 for keyword in plan_keywords[:5]
                if keyword.lower() in doc_text
            )
            plan_boost = min(0.1, plan_alignment * plan_matches * 0.02)
            score += plan_boost
        
        # 5. Diffusion penalty adjustment
        # High diffusion (instability) → slightly penalize to prefer stable documents
        if diffusion_penalty > 0.3:
            diffusion_penalty_factor = min(0.05, diffusion_penalty * 0.1)
            score *= (1.0 - diffusion_penalty_factor)
        
        # 6. Confidence-based adjustment
        # Low confidence → boost slightly to explore more
        confidence = snapshot_info.get("confidence", 0.5)
        if confidence < 0.5:
            confidence_boost = (0.5 - confidence) * 0.05  # Max 2.5% boost when very uncertain
            score += confidence_boost
        
        # Normalize to [0, 1] range
        score = min(1.0, max(0.0, score))
        
        return score
    
    def _passes_constraint_filters(
        self,
        doc: Document,
        query: str,
        constraint_info: Dict[str, Any],
        snapshot_info: Dict[str, Any]
    ) -> bool:
        """
        Check if document passes regulator constraint filters.
        
        Filters out documents that:
        - Don't align with plan goal (if plan alignment is strong)
        - Violate relation direction constraints
        
        Args:
            doc: Document to check
            query: Query string
            constraint_info: Extracted constraint information
            snapshot_info: Extracted snapshot information
            
        Returns:
            True if document passes filters, False otherwise
        """
        doc_text = doc.page_content.lower()
        query_lower = query.lower()
        
        # 1. Plan alignment filter
        # If plan alignment is strong and document doesn't contain plan keywords, filter out
        plan_alignment = constraint_info.get("plan_alignment", 0.0)
        plan_keywords = constraint_info.get("plan_keywords", [])
        if plan_alignment > 0.6 and plan_keywords:
            # Strong plan alignment → require at least one plan keyword
            has_plan_keyword = any(
                keyword.lower() in doc_text
                for keyword in plan_keywords[:5]
            )
            if not has_plan_keyword:
                return False
        
        # 2. Relation direction filter (if applicable)
        # This is more complex and would require checking semantic relations
        # For now, we skip this filter as it's handled by query stabilization
        
        # 3. Entity anchor filter (soft)
        # If we have strong entity anchors but document doesn't mention them, 
        # we don't filter out (just don't boost) - this maintains recall
        
        return True
    
    def save_index(self, path: Union[str, Path]):
        """
        Save the vector store to disk.
        
        Args:
            path: Directory path to save the index
        """
        if not self.vector_store:
            raise ValueError("No vector store to save")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(path))
        
        # Save model config
        config = {
            "model_name": self.embeddings.model_name,
            "class_name": self.__class__.__name__,
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
            "batch_size": self.batch_size
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)
        
        logger.info(f"Vector store saved to {path}")
    
    @classmethod
    def load_index(
        cls, 
        path: Union[str, Path], 
        model_config: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None
    ) -> 'RetrieverAgent':
        """
        Load a vector store from disk.
        
        Args:
            path: Directory path containing the saved index
            model_config: Configuration for the embedding model
            model_name: Override the model name from saved config
            
        Returns:
            An instance of RetrieverAgent with the loaded index
        """
        path = Path(path)
        
        # Load config if exists
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            model_name = model_name or config.get("model_name", "all-MiniLM-L6-v2")
            top_k = config.get("top_k", 5)
            min_similarity = config.get("min_similarity", 0.6)
            batch_size = config.get("batch_size", 32)
        else:
            model_name = model_name or "all-MiniLM-L6-v2"
            top_k = 5
            min_similarity = 0.6
            batch_size = 32
        
        # Initialize the agent
        instance = cls(
            documents=None,
            model_config=model_config,
            model_name=model_name,
            device=model_config.get("device") if model_config else None,
            top_k=top_k,
            min_similarity=min_similarity,
            batch_size=batch_size
        )
        
        # Load the vector store
        instance.vector_store = FAISS.load_local(
            folder_path=str(path),
            embeddings=instance.embeddings
        )
        
        logger.info(f"Vector store loaded from {path}")
        return instance
