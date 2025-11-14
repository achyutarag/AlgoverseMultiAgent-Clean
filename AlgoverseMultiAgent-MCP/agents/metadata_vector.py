"""
Metadata Vector

Encodes reasoning intent into 3 interpretable dimensions that guide
retrieval and reasoning operations. Acts as a directional constraint
in reasoning space.

R = a_s · S + a_e · E + a_r · R

Where:
- S (Structural): Format/units constraints
- E (Existential): Trust/recency requirements  
- R (Relational): Relationship type (compare, infer, join, etc.)
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class MetadataVector(BaseModel):
    """
    Metadata vector encoding reasoning direction.
    
    Three dimensions guide retrieval and reasoning:
    - Structural: How much format/units matter
    - Existential: How much trust/recency matters
    - Relational: What type of relationship (compare, infer, etc.)
    """
    
    # Three core dimensions (0.0 to 1.0)
    structural: float = Field(..., ge=0.0, le=1.0, description="Format/units importance (0=low, 1=high)")
    existential: float = Field(..., ge=0.0, le=1.0, description="Trust/recency importance (0=low, 1=high)")
    relational: float = Field(..., ge=0.0, le=1.0, description="Relationship type strength (0=low, 1=high)")
    
    # Relational type interpretation
    relational_type: str = Field("factual", description="Type: compare, infer, join, temporal, factual")
    
    # Metadata
    query_type: str = Field("unknown", description="Original query type")
    step_id: Optional[str] = Field(None, description="Step this vector is for")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy passing to agents."""
        return {
            "structural": self.structural,
            "existential": self.existential,
            "relational": self.relational,
            "relational_type": self.relational_type,
            "query_type": self.query_type,
            "step_id": self.step_id
        }
    
    def get_retrieval_filters(self) -> Dict[str, Any]:
        """
        Generate filter criteria for retriever based on vector dimensions.
        
        Returns:
            Dictionary of filter criteria for document retrieval
        """
        filters = {}
        
        # Structural filters
        if self.structural > 0.7:
            # High structural = need specific format/type
            filters["require_structured"] = True
        
        # Existential filters  
        if self.existential > 0.7:
            # High existential = need recent/trusted sources
            filters["min_recency"] = 0.8
            filters["require_verified"] = True
        elif self.existential < 0.3:
            # Low existential = any source is fine
            filters["any_source"] = True
        
        # Relational filters
        if self.relational_type == "compare":
            filters["has_comparison"] = True
        elif self.relational_type == "temporal":
            filters["has_dates"] = True
        elif self.relational_type == "cause-effect":
            filters["has_causation"] = True
        
        return filters
    
    def get_reasoning_operator(self) -> str:
        """
        Select reasoning operator based on relational dimension.
        
        Returns:
            Operator name: "compare", "infer", "join", "lookup"
        """
        if self.relational_type == "compare":
            return "compare"
        elif self.relational_type == "infer" or self.relational_type == "cause-effect":
            return "infer"
        elif self.relational_type == "temporal" or self.relational_type == "join":
            return "join"
        else:
            return "lookup"  # Default for factual queries


class MetadataVectorGenerator:
    """
    Generates metadata vectors from MCP reasoning state.
    Uses rule-based classification for now (can be enhanced with LLM later).
    """
    
    @staticmethod
    def generate_from_mcp_state(
        mcp_state: Any,  # MCPReasoningState
        step: Optional[Dict[str, Any]] = None
    ) -> MetadataVector:
        """
        Generate metadata vector from MCP reasoning state.
        
        Args:
            mcp_state: The MCP reasoning state
            step: Optional current step context
            
        Returns:
            MetadataVector with appropriate dimensions
        """
        query_type = mcp_state.query_type.lower()
        reasoning_intent = mcp_state.reasoning_intent.lower()
        main_question = mcp_state.main_question.lower()
        
        # Determine relational type from query characteristics
        relational_type = MetadataVectorGenerator._classify_relational_type(
            query_type, reasoning_intent, main_question
        )
        
        # Generate dimensions based on query type and characteristics
        structural = MetadataVectorGenerator._calculate_structural(query_type, main_question)
        existential = MetadataVectorGenerator._calculate_existential(query_type, main_question)
        relational = MetadataVectorGenerator._calculate_relational(relational_type, query_type)
        
        step_id = step.get("id") if step else None
        
        vector = MetadataVector(
            structural=structural,
            existential=existential,
            relational=relational,
            relational_type=relational_type,
            query_type=mcp_state.query_type,
            step_id=step_id
        )
        
        logger.debug(f"Generated metadata vector: S={structural:.2f}, E={existential:.2f}, R={relational:.2f}, Type={relational_type}")
        
        return vector
    
    @staticmethod
    def _classify_relational_type(
        query_type: str,
        reasoning_intent: str,
        main_question: str
    ) -> str:
        """Classify the relational type from query characteristics."""
        
        # Check for comparison keywords
        comparison_keywords = ["compare", "difference", "versus", "vs", "better", "worse", "similar", "different"]
        if any(keyword in main_question or keyword in reasoning_intent for keyword in comparison_keywords):
            return "compare"
        
        # Check for temporal keywords
        temporal_keywords = ["when", "before", "after", "first", "earlier", "later", "chronology", "timeline"]
        if any(keyword in main_question or keyword in reasoning_intent for keyword in temporal_keywords):
            return "temporal"
        
        # Check for cause-effect keywords
        causation_keywords = ["why", "because", "cause", "effect", "result", "led to", "due to", "reason"]
        if any(keyword in main_question or keyword in reasoning_intent for keyword in causation_keywords):
            return "cause-effect"
        
        # Check for inference keywords
        inference_keywords = ["how", "what if", "would", "could", "should", "predict", "infer"]
        if any(keyword in main_question or keyword in reasoning_intent for keyword in inference_keywords):
            return "infer"
        
        # Check query type
        if query_type == "comparative":
            return "compare"
        elif query_type == "analytical":
            return "infer"
        elif query_type == "multi-hop":
            return "join"  # Multi-hop often requires joining information
        
        # Default
        return "factual"
    
    @staticmethod
    def _calculate_structural(query_type: str, main_question: str) -> float:
        """Calculate structural dimension (format/units importance)."""
        
        # Questions about numbers, dates, measurements need high structural
        structural_keywords = ["how many", "how much", "when", "date", "year", "number", "count", 
                             "percentage", "ratio", "measurement", "unit"]
        if any(keyword in main_question for keyword in structural_keywords):
            return 0.8
        
        # Comparative queries need some structure
        if query_type == "comparative":
            return 0.7
        
        # Multi-hop might need structure for joining
        if query_type == "multi-hop":
            return 0.6
        
        # Default
        return 0.5
    
    @staticmethod
    def _calculate_existential(query_type: str, main_question: str) -> float:
        """Calculate existential dimension (trust/recency importance)."""
        
        # Questions about current events, recent things need high existential
        recency_keywords = ["current", "recent", "latest", "new", "now", "today", "2024", "2025"]
        if any(keyword in main_question for keyword in recency_keywords):
            return 0.9
        
        # Factual queries need some trust
        if query_type == "simple":
            return 0.6
        
        # Multi-hop might need trusted sources
        if query_type == "multi-hop":
            return 0.7
        
        # Analytical/comparative can use various sources
        if query_type in ["analytical", "comparative"]:
            return 0.5
        
        # Default
        return 0.6
    
    @staticmethod
    def _calculate_relational(relational_type: str, query_type: str) -> float:
        """Calculate relational dimension strength."""
        
        # High relational for comparison, temporal, cause-effect
        if relational_type in ["compare", "temporal", "cause-effect"]:
            return 0.9
        
        # Medium for inference, joining
        if relational_type in ["infer", "join"]:
            return 0.7
        
        # Low for simple factual
        if relational_type == "factual":
            return 0.3
        
        # Default
        return 0.5


# Global instance for easy access
metadata_vector_generator = MetadataVectorGenerator()