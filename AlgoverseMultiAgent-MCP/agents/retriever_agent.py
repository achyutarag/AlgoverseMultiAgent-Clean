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
            
            # Perform similarity search with FAISS
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_criteria
            )
            
            # Process and apply entropy-aware filtering/reranking
            if is_entropy_aware:
                results, similarities, filtered_count = self._apply_entropy_aware_processing(
                    docs_and_scores,
                    query,
                    min_similarity,
                    regulator_constraints,
                    flow_snapshot,
                    entropy_penalty,
                    diffusion_penalty,
                    include_scores
                )
            else:
                # Basic retrieval processing (backward compatible)
                results, similarities, filtered_count = self._apply_basic_processing(
                    docs_and_scores,
                    min_similarity,
                    include_scores
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
            if not results and docs_and_scores:
                # Take top by base similarity (ignore min_similarity/regulator filters)
                fallback = sorted(
                    [(doc, 1.0 / (1.0 + score)) for doc, score in docs_and_scores],
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
    
    def _apply_basic_processing(
        self,
        docs_and_scores: List[tuple],
        min_similarity: float,
        include_scores: bool
    ) -> Tuple[List[Dict[str, Any]], List[float], int]:
        """
        Basic document processing (backward compatible).
        
        Args:
            docs_and_scores: List of (Document, score) tuples from FAISS
            min_similarity: Minimum similarity threshold
            include_scores: Whether to include scores
            
        Returns:
            Tuple of (results, similarities, filtered_count)
        """
        results = []
        similarities = []
        filtered_count = 0
        
        for doc, score in docs_and_scores:
            # Convert score to similarity (higher is better)
            similarity = 1.0 / (1.0 + score)
            
            # Skip if below threshold
            if similarity < min_similarity:
                filtered_count += 1
                continue
                
            # Add document ID if not present
            doc_id = doc.metadata.get('id', f"doc_{len(results)+1}")
            doc.metadata['id'] = doc_id

            # Add to results
            doc_data = {
                "id": doc_id,
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(similarity) if include_scores else None
            }
            results.append(doc_data)
            similarities.append(similarity)
        
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
            
            # Calculate entropy-aware score
            enhanced_score = self._calculate_entropy_aware_score(
                doc=doc,
                query=query,
                base_similarity=base_similarity,
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
            "constraint_weights": {}
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
                    if required_domain and required_level:
                        from .regulators.granularity_regulator import GranularityRegulator
                        granularity_reg = GranularityRegulator()
                
                # Extract from EvidenceRegulator
                elif regulator_name == "Evidence":
                    evidence_terms = params.get("evidence_terms", [])
                    top_terms = params.get("top_terms", [])
                    info["evidence_terms"].extend(top_terms or evidence_terms[:5])
                
                # Extract from EntityRegulator
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
                
                # Extract from PlanRegulator
                elif regulator_name == "Plan":
                    info["plan_alignment"] = params.get("alignment", 0.0)
                    info["plan_keywords"] = params.get("goal_keywords", [])
                
                # Extract from RelationRegulator
                elif regulator_name == "Relation":
                    info["relation_direction"] = params.get("direction")
        
        # Deduplicate
        info["evidence_terms"] = list(set(info["evidence_terms"]))[:10]
        info["entity_anchors"] = list(set(info["entity_anchors"]))[:5]
        
        return info
    
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
