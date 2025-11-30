from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from .base_agent import BaseAgent, AgentResponse
from .tokenization_utils import tokenization_utils, TokenizationUtils
import json
import logging

logger = logging.getLogger(__name__)

class Evidence(BaseModel):
    """Supporting evidence for an answer."""
    text: str = Field(..., description="The text of the evidence")
    source: str = Field(..., description="Source document ID for this evidence")
    relevance: float = Field(0.0, description="Relevance score of this evidence")

class Answer(BaseModel):
    """Structured answer with supporting evidence."""
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The generated answer")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    sources: List[str] = Field(default_factory=list, description="List of source document IDs")
    supporting_evidence: List[Evidence] = Field(
        default_factory=list,
        description="List of supporting evidence with text and source"
    )
    reasoning: str = Field("", description="Reasoning process for the answer")

class QAAgent(BaseAgent):
    """
    Enhanced QA Agent that synthesizes answers using in-context learning with 
    step-specific context. It produces responses for each step which are passed 
    to the next iteration, enabling grounded reasoning throughout the trajectory.
    """
    
    def __init__(
        self, 
        model_config: Optional[Dict[str, Any]] = None,
        model_name: str = "gemini-2.5-flash",  # LLM for answer synthesis
        temperature: float = 0.3,
        max_tokens: int = 1024
    ):
        """
        Initialize the Enhanced QA Agent.
        
        Args:
            model_config: Configuration for the LLM
            model_name: Name of the model to use
            temperature: Temperature for text generation (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
        """
        super().__init__("qa_agent", model_config, model_name)
        self.temperature = max(0.0, min(1.0, temperature))
        self.max_tokens = max(100, min(4096, max_tokens))
        self.conversation_history: List[Dict[str, str]] = []
        
        self.system_prompt = """You are an extractive question-answering system. Your task is to extract answers directly from provided evidence.

Your task is to:
1. Extract the answer EXACTLY as it appears in the evidence
2. Do NOT explain, paraphrase, or modify the wording
3. Do NOT infer anything not explicitly in the evidence
4. If the answer is not directly present, return "unknown"
5. Return valid JSON

Return a JSON object with this structure:
{
    "answer": "Extracted answer from evidence",
    "confidence": 0.95,
    "sources": ["source1", "source2"],
    "supporting_evidence": [
        {
            "text": "Relevant text from evidence",
            "source": "source_id",
            "relevance": 0.9
        }
    ]
}

**IMPORTANT**: Always return valid JSON. Do not include any text before or after the JSON object."""
    
    def _attempt_hierarchical_inference(
        self,
        question: str,
        context: List[Dict[str, Any]],
        required_domain: Optional[str],
        required_level: Optional[str]
    ) -> Optional[str]:
        """
        Attempt to infer parent-level entity when evidence contains lower-level entities.
        
        ✅ FIRST PRINCIPLES: When documents contain lower-level entities but query requires
        higher-level, we should infer the parent entity from context rather than returning "unknown".
        
        This is generalized and works for any hierarchical domain (territorial, organizational, taxonomic).
        
        Args:
            question: The question being answered
            context: List of extracted passages/evidence
            required_domain: Required hierarchical domain
            required_level: Required hierarchical level name
            
        Returns:
            Inferred parent entity name if found, None otherwise
        """
        if not required_domain or not required_level:
            return None
        
        try:
            from .regulators.granularity_regulator import GranularityRegulator
            granularity_reg = GranularityRegulator()
            
            # Get required level number
            required_level_num = granularity_reg.get_level_number(required_domain, required_level)
            if not required_level_num:
                return None
            
            # Get level keywords for the required level
            level_keywords = granularity_reg._get_level_keywords(required_domain, required_level)
            if not level_keywords:
                return None
            
            # Search context for entities at required level
            # Look for patterns where entities are mentioned with required level keywords
            import re
            for evidence_item in context:
                evidence_text = evidence_item.get("text", "") if isinstance(evidence_item, dict) else str(evidence_item)
                if not evidence_text:
                    continue
                
                evidence_lower = evidence_text.lower()
                
                # Check if evidence contains required level keywords
                for keyword in level_keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in evidence_lower:
                        # Try to extract entity name near the keyword
                        # Pattern: look for capitalized words/phrases near the keyword
                        # This works for patterns like "X is in Y state" or "Y state contains X"
                        patterns = [
                            # Pattern 1: "Entity [keyword]" or "[keyword] Entity"
                            rf'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+{re.escape(keyword_lower)}\b',
                            rf'\b{re.escape(keyword_lower)}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                            # Pattern 2: "Entity, [keyword]" or "[keyword], Entity"
                            rf'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*{re.escape(keyword_lower)}\b',
                            rf'\b{re.escape(keyword_lower)},\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                            # Pattern 3: "X is in Y [keyword]" or "Y [keyword] contains X"
                            rf'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+{re.escape(keyword_lower)}\s+(?:contains|has|owns|governs)',
                            rf'\b(?:located|situated|found)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+{re.escape(keyword_lower)}\b',
                        ]
                        
                        for pattern in patterns:
                            matches = re.findall(pattern, evidence_text, re.IGNORECASE)
                            if matches:
                                # Return the first match (most likely the parent entity)
                                inferred = matches[0].strip()
                                # Validate it's not a common word
                                if inferred and len(inferred) > 2 and inferred.lower() not in ["the", "a", "an", "this", "that"]:
                                    logger.debug(
                                        f"QA Agent: Hierarchical inference found '{inferred}' "
                                        f"using pattern with keyword '{keyword}'"
                                    )
                                    return inferred
        except Exception as e:
            logger.debug(f"QA Agent: Hierarchical inference failed: {str(e)}")
        
        return None


    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Generate a step-specific answer using entropy-aware compression in a stabilized belief field.
        
        ====================================================================
        DIFFUSION-TO-CONVERGENCE MODEL:
        ====================================================================
        Multi-hop reasoning is modeled as a diffusion process where beliefs
        P(x,t) spread through document space. This QA agent performs the
        compression step that collapses probability mass into anchors:
        
        P(x,t+1) = compress(P(x,t), anchors, H(t), D(t))
        
        Where:
        - P(x,t): Belief distribution at hop t
        - anchors: Fixed points (potential wells) from previous hops
        - H(t): Entropy (uncertainty measure)
        - D(t): Diffusion coefficient (drift measure)
        
        Compression Strategy:
        - High entropy (H(t) > 0.5): Low compression → explore evidence broadly
        - Low entropy (H(t) < 0.3) + high confidence: High compression → collapse to anchors
        - Medium entropy: Balanced compression
        
        The output becomes new anchors for the next hop, completing the
        diffusion → compression → anchor cycle.
        ====================================================================
        
        Args:
            input_data: Dictionary containing:
                - 'question': The subquery to answer
                - 'context': List of extracted passages with their sources and relevance
                - 'flow_snapshot': FlowSnapshot with H(t), D(t), anchors, beliefs (NEW)
                - 'regulator_constraints': List of regulator constraints (NEW)
                - 'stabilized_query': The stabilized query used for retrieval (NEW)
                - 'hop': Current hop number (NEW)
                - Optional 'history': Previous interactions for context
                - Optional 'step_context': Information about the current step
                - Optional 'overall_query': The main question being answered
                - Optional 'previous_answers': Answers from previous steps
                - Optional 'max_history_items': Max history items to include (default: 4)
                - Optional 'min_confidence': Minimum confidence threshold (0.0-1.0)
                
        Returns:
            AgentResponse containing:
                - Synthesized answer (collapsed probability mass)
                - New anchors for next hop
                - Diffusion metadata (entropy, diffusion, anchor consistency)
        """
        question = input_data.get('question', '').strip()
        context = input_data.get('context', [])
        flow_snapshot = input_data.get('flow_snapshot')
        regulator_constraints = input_data.get('regulator_constraints', [])
        stabilized_query = input_data.get('stabilized_query', '')
        hop = input_data.get('hop', 1)
        history = input_data.get('history', [])
        step_context = input_data.get('step_context', {})
        overall_query = input_data.get('overall_query', '')
        previous_answers = input_data.get('previous_answers', {})
        max_history = int(input_data.get('max_history_items', 4))
        min_confidence = max(0.0, min(1.0, float(input_data.get('min_confidence', 0.0))))
        
        # ====================================================================
        # EXTRACT DIFFUSION STATE FROM STABILIZED BELIEF FIELD
        # ====================================================================
        # Extract entropy H(t), diffusion D(t), confidence, and anchors
        # These determine the compression strategy
        if flow_snapshot:
            if isinstance(flow_snapshot, dict):
                entropy = flow_snapshot.get('entropy', 0.5)
                diffusion = flow_snapshot.get('diffusion_coefficient', 0.5)
                confidence = flow_snapshot.get('confidence', 0.5)
                anchors = flow_snapshot.get('anchors', [])
                entity_anchors = flow_snapshot.get('entity_anchors', {})
            else:
                # FlowSnapshot object
                entropy = flow_snapshot.entropy if hasattr(flow_snapshot, 'entropy') else 0.5
                diffusion = flow_snapshot.diffusion_coefficient if hasattr(flow_snapshot, 'diffusion_coefficient') else 0.5
                confidence = flow_snapshot.confidence if hasattr(flow_snapshot, 'confidence') else 0.5
                anchors = flow_snapshot.anchors if hasattr(flow_snapshot, 'anchors') else []
                entity_anchors = flow_snapshot.entity_anchors if hasattr(flow_snapshot, 'entity_anchors') else {}
        else:
            # Fallback if no flow_snapshot (backward compatibility)
            entropy = 0.5
            diffusion = 0.5
            confidence = 0.5
            anchors = []
            entity_anchors = {}
        
        # ====================================================================
        # EXTRACT HIERARCHICAL LEVEL REQUIREMENT (Initial Condition)
        # ====================================================================
        # Extract hierarchical level requirement from GranularityRegulator constraint
        # This is the initial condition (u(x,0)) that must be respected during compression
        required_hierarchical_level = None
        required_domain = None
        for constraint in regulator_constraints:
            constraint_dict = constraint if isinstance(constraint, dict) else constraint.dict() if hasattr(constraint, 'dict') else {}
            constraint_name = constraint_dict.get('regulator_name', '')
            if 'granularity' in constraint_name.lower():
                params = constraint_dict.get('parameters', {})
                required_level = params.get('required_level')
                required_domain = params.get('required_domain')
                if required_level:
                    required_hierarchical_level = required_level
                    break
        
        # ====================================================================
        # ENTROPY-AWARE COMPRESSION DECISION
        # ====================================================================
        # Compression level determines how much to collapse probability mass:
        # - High compression: Low entropy + high confidence → collapse to anchors
        # - Medium compression: Moderate uncertainty → balanced approach
        # - Low compression: High entropy → explore evidence broadly
        if entropy < 0.3 and confidence > 0.7:
            compression_level = "high"
            compression_strategy = "Collapse probability mass to most anchor-consistent answer. High precision, low exploration."
        elif entropy < 0.5:
            compression_level = "medium"
            compression_strategy = "Balance anchor consistency with evidence exploration."
        else:
            compression_level = "low"
            compression_strategy = "Explore evidence broadly while maintaining anchor consistency."
        
        logger.debug(
            f"QA Agent (Hop {hop}): Entropy={entropy:.3f}, Diffusion={diffusion:.3f}, "
            f"Confidence={confidence:.3f}, Compression={compression_level}"
        )
        
        # Normalize question for consistent processing
        question = tokenization_utils.normalize_query(question)
        
        if not question:
            return AgentResponse(
                content=json.dumps({
                    "question": "",
                    "answer": "Error: No question provided",
                    "confidence": 0.0,
                    "reasoning": "No question provided",
                    "sources": [],
                    "supporting_evidence": []
                }),
                metadata={
                    "error": "No question provided",
                    "question": "",
                    "has_context": bool(context),
                    "num_sources": 0
                }
            )
            
        if not context:
            return AgentResponse(
                content=json.dumps({
                    "question": question,
                    "answer": "No context provided to answer this question.",
                    "confidence": 0.0,
                    "reasoning": "No evidence available",
                    "sources": [],
                    "supporting_evidence": []
                }),
                metadata={
                    "question": question,
                    "has_context": False,
                    "num_sources": 0,
                    "confidence": 0.0
                }
            )
        
        try:
            # ====================================================================
            # BUILD PROMPT WITH STABILIZED BELIEF FIELD CONTEXT
            # ====================================================================
            # The prompt includes diffusion state (H(t), D(t)), anchors, and constraints
            # to guide entropy-aware compression
            prompt = f"""{self.system_prompt}
            
### ====================================================================
### DIFFUSION-AWARE CONTEXT (Stabilized Belief Field)
### ====================================================================
### Current Hop: {hop}
### Entropy H(t): {entropy:.3f} ({'Low uncertainty' if entropy < 0.3 else 'Medium uncertainty' if entropy < 0.5 else 'High uncertainty'})
### Diffusion D(t): {diffusion:.3f} ({'Low drift' if diffusion < 0.3 else 'Medium drift' if diffusion < 0.5 else 'High drift'})
### Confidence: {confidence:.3f}
### Compression Level: {compression_level.upper()}
### Active Anchors: {len(anchors)} bucket anchors, {len(entity_anchors)} entity-specific anchors

### Stabilized Query Used for Retrieval:
{stabilized_query if stabilized_query else question}
### ====================================================================

### Subquery to Answer:
{tokenization_utils.preprocess_llm_input(question)}

### Step Context:
{json.dumps(step_context, indent=2) if step_context else "No specific step context"}
"""
            
            # Add anchor context for consistency (potential wells)
            if anchors or entity_anchors:
                prompt += "\n\n### ===================================================================="
                prompt += "\n### ACTIVE ANCHORS (Fixed Points - Maintain Consistency)"
                prompt += "\n### ===================================================================="
                prompt += "\n**Anchors are fixed points (potential wells) that stabilize reasoning.**"
                prompt += "\nYour answer should be consistent with these anchors to prevent drift.\n"
                
                for anchor in (anchors if isinstance(anchors, list) else []):
                    anchor_entity = anchor.get('entity', '') if isinstance(anchor, dict) else (anchor.entity if hasattr(anchor, 'entity') else '')
                    anchor_type = anchor.get('type', '') if isinstance(anchor, dict) else (anchor.type if hasattr(anchor, 'type') else '')
                    if anchor_entity:
                        prompt += f"- {anchor_entity} ({anchor_type})\n"
                
                for entity, anchor_data in (entity_anchors.items() if isinstance(entity_anchors, dict) else []):
                    prompt += f"- {entity}: {str(anchor_data)[:100]}\n"
                
                prompt += "\n**CRITICAL**: Your answer must be consistent with these anchors.\n"
                prompt += "If multiple answers exist, choose the one that aligns with anchors.\n"
            
            # Add regulator constraints (boundary conditions)
            if regulator_constraints:
                prompt += "\n### ===================================================================="
                prompt += "\n### REGULATOR CONSTRAINTS (Boundary Conditions)"
                prompt += "\n### ===================================================================="
                prompt += "\n**Constraints from regulators guide reasoning boundaries.**\n"
                for constraint in regulator_constraints[:3]:  # Top 3 constraints
                    constraint_name = constraint.get('regulator_name', '') if isinstance(constraint, dict) else (constraint.regulator_name if hasattr(constraint, 'regulator_name') else '')
                    constraint_type = constraint.get('constraint_type', '') if isinstance(constraint, dict) else (constraint.constraint_type if hasattr(constraint, 'constraint_type') else '')
                    if constraint_name:
                        prompt += f"- {constraint_name} ({constraint_type})\n"
                prompt += "\n"
            
            # ✅ FIRST PRINCIPLES: Hierarchical Level Awareness in Entropy-Aware Compression
            # ====================================================================
            # The global boundary condition and initial condition (u(x,0)) sets the required hierarchical level.
            # Entropy-aware compression must respect this constraint to minimize
            # ambiguity. If evidence contains entities at different hierarchical
            # levels, compress to the entity at the REQUIRED level.
            # ====================================================================
            

            if required_hierarchical_level:
                prompt += "\n### ===================================================================="
                prompt += "\n### HIERARCHICAL LEVEL CONSTRAINT (Initial Condition - MUST RESPECT)"
                prompt += "\n### ===================================================================="
                prompt += f"\n**CRITICAL**: The query requires a {required_hierarchical_level} level entity"
                if required_domain:
                    prompt += f" (domain: {required_domain})"
                prompt += ".\n\n**Entropy-Aware Compression Rule**:"
                prompt += "\n- If evidence contains entities at DIFFERENT hierarchical levels, "
                prompt += f"extract the entity at the {required_hierarchical_level} level (required level)."
                prompt += "\n- DO NOT extract entities at wrong hierarchical levels."
                prompt += "\n- This minimizes entropy by ensuring the answer respects the initial condition constraint."
                prompt += "\n\n**How to Identify Entities at Required Level**:"
                prompt += "\n- **KEY INSIGHT**: The entity name itself doesn't need to contain level keywords"
                prompt += "\n- Look for entities mentioned WITH the required level keywords in the evidence"
                prompt += "\n- Example: If state/province level is required and evidence says 'X Municipality is located in Y state' → Extract 'Y' (mentioned with 'state' keyword)"
                prompt += "\n- Example: If state/province level is required and evidence says 'Y is the administrative territorial entity containing X' → Extract 'Y' (mentioned with 'administrative territorial entity')"
                prompt += "\n- Example: If state/province level is required and evidence says 'City Municipality, State Name' → Extract 'State Name' (the one without municipality keyword)"
                prompt += "\n- **Pattern Recognition**: Entities described with required level keywords (even if the entity name itself lacks those keywords) are at that level"
                prompt += "\n\n**General Principle**:"
                prompt += "\n- Hierarchical structures have levels (e.g., country > state > municipality, "
                prompt += "company > division > team, kingdom > phylum > species, etc.)"
                prompt += "\n- If the query requires level X, extract entities at level X, not level Y (higher or lower)."
                prompt += "\n- When evidence contains multiple entities at different levels, identify which is at the required level by looking at how they're described in the evidence."
                prompt += "\n\n**Examples (Generic Patterns)**:"
                prompt += "\n- Evidence contains 'Entity A (lower level), Entity B (higher level)' → Extract Entity B if higher level is required"
                prompt += "\n- Evidence contains 'Entity A (higher level), Entity B (lower level)' → Extract Entity A if higher level is required"
                prompt += "\n- Evidence contains 'X is located in Y' where Y is described with required level keywords → Extract Y"
                prompt += "\n- Evidence contains 'X is part of Y' where Y is described with required level keywords → Extract Y"
                prompt += "\n\n**This is entropy-aware compression**: Compressing evidence to the correct hierarchical level "
                prompt += "minimizes ambiguity (min H(final answer; evidence)) while respecting the initial condition.\n"
            
            # Add overall query context prominently at the top (if available)
            if overall_query:
                prompt += f"\n\n### ORIGINAL QUESTION (USE TO GUIDE ANSWER FORMAT):\n{overall_query}\n"
            
            # Add previous answers if available (for context only, NOT to include in answer)
            if previous_answers:
                prompt += "\n\n### Previous Step Answers (FOR CONTEXT ONLY - DO NOT include these in your answer):"
                for step_id, answer in previous_answers.items():
                    answer_preview = str(answer)[:200] + "..." if len(str(answer)) > 200 else str(answer)
                    prompt += f"\n- Step {step_id}: {answer_preview}"
                prompt += "\n\nIMPORTANT: The previous step answers above are provided for context to help you understand the question. DO NOT include them in your answer. Answer ONLY what the current subquery asks for."
            
            # Add conversation history if available
            if history:
                history_str = "\n".join(
                    f"{h.get('role', 'user').upper()}: {h.get('content', '')}"
                    for h in history[-max_history:]
                )
                prompt += f"\n### Previous Conversation (most recent last):\n{history_str}\n\n"
            
            # Add context documents with enhanced formatting
            prompt += "\n### Extracted Evidence:"
            for i, doc in enumerate(context):
                doc_id = doc.get('document_id', f'doc_{i+1}')
                text = doc.get('text', '').strip()
                relevance = doc.get('relevance', 0.0)
                reasoning = doc.get('reasoning', '')
                
                prompt += (
                    f"\n[Evidence {i+1}, Source: {doc_id}, Relevance: {relevance:.2f}]\n"
                    f"Text: {text}\n"
                    f"Reasoning: {reasoning}\n"
                )
            
            
            # Add general instruction for using evidence as-provided
            prompt += "\n\n### ⚠️ IMPORTANT: Using Evidence As-Provided"
            prompt += "\n- Extract answers DIRECTLY from the evidence provided above"
            prompt += "\n- DO NOT reject or modify answers based on the year, date, or whether they match your world knowledge"
            prompt += "\n- Use the information as it appears in the evidence, regardless of when it's from or whether it seems 'correct'"
            prompt += "\n- If multiple pieces of evidence exist, use the one that most directly answers the question"
            prompt += "\n- Your job is to extract the answer from the evidence, not to judge its recency, accuracy, or plausibility"
            prompt += "\n- Trust the evidence's context over your general knowledge - the evidence is the source of truth"
            prompt += "\n\n### ⚠️ CRITICAL: Trust Extractor's Contextual Matching"
            prompt += "\n- The evidence provided above has ALREADY been contextually matched by the Extractor Agent"
            prompt += "\n- DO NOT return 'unknown' just because words don't exactly match - the Extractor already handled contextual matching"
            prompt += "\n- Extract the answer following the Answer Format Rules below"
            
            
            # Add instructions for extraction
            prompt += f"""

### EXTRACTION INSTRUCTIONS:
Using the question and evidence provided above:

**CRITICAL: Detect Yes/No Questions Based on SUB-QUERY Format**
- Check ONLY the subquery format (NOT the original question) to determine if this is a yes/no question
- If the SUBQUERY starts with "Is", "Are", "Was", "Were", "Do", "Does", "Did", "Can", "Could", "Would", "Should", "Has", "Have", "Had" → This is a YES/NO question
- For yes/no subqueries, you MUST return "Yes" or "No" based on whether the evidence confirms or denies the statement
- For factual subqueries (e.g., "What is X's nationality?", "Find the use of Y"), extract the factual answer (e.g., "American", "real estate use") - DO NOT convert to yes/no
- Example: Subquery "Is X from Y?" with evidence "X is from Y" → Answer: "Yes"
- Example: Subquery "What is X's nationality?" with evidence "X is American" → Answer: "American" (NOT "Yes", even if original question is yes/no format)
- Example: Subquery "Find the primary use of X" with evidence "X is used for real estate" → Answer: "real estate use" (factual extraction)

**CRITICAL: Entity Extraction Priority**
- If the subquery asks for an ENTITY NAME (e.g., "What is the name of...", "Who founded...", "What company..."), and the entity name appears in the evidence, extract the entity name
- Do NOT return "unknown" just because related information (e.g., founder, date) is not found - if the entity itself is in the evidence, extract it
- Example: Subquery "What is the distribution company for film X?" Evidence mentions "YG Entertainment" but doesn't mention founder → Answer: "YG Entertainment" (NOT "unknown")
- Example: Subquery "Who founded YG Entertainment?" Evidence mentions "YG Entertainment" but doesn't mention founder → Answer: "unknown" (founder not found, which is what was asked)
- Only return "unknown" if the SPECIFIC information asked for is not in the evidence

**✅ CRITICAL: Hierarchical Level Awareness (Entropy-Aware Compression)**
- If a hierarchical level constraint is specified above (e.g., "state_province" level required), you MUST extract the entity at that level
- This is entropy-aware compression: min H(final answer; evidence) while respecting the initial condition constraint
- If evidence contains entities at DIFFERENT hierarchical levels, extract the entity at the REQUIRED level (not a different level)
- **IMPORTANT**: If evidence only contains entities at the WRONG hierarchical level (e.g., municipality when state is required), 
  you may return "unknown" for THIS step, but the system will continue searching in other documents
- **However**: If evidence contains BOTH wrong-level and correct-level entities, extract the correct-level entity
- Examples (Generic Patterns):
  * If higher level is required and evidence says "Lower-Level Entity, Higher-Level Entity" → Extract "Higher-Level Entity" (required level)
  * If higher level is required and evidence says "X is located in Y [higher-level term]" → Extract "Y" (higher level)
  * If lower level is required and evidence says "Lower-Level Entity, Higher-Level Entity" → Extract "Lower-Level Entity" (required level)
  * For territorial hierarchies: If state/province level required and evidence says "City Municipality, State Name" → Extract "State Name"
  * For organizational hierarchies: If company level required and evidence says "Department Name, Company Name" → Extract "Company Name"
  * For taxonomic hierarchies: If genus level required and evidence says "Species Name (Genus Name)" → Extract "Genus Name"
- This ensures compression respects the global boundary condition as well as the initial condition (u(x,0)) set by GranularityRegulator
- DO NOT extract entities at wrong hierarchical levels - this violates the constraint and increases entropy

### ====================================================================
### ENTROPY-AWARE COMPRESSION INSTRUCTIONS
### ====================================================================
**You are collapsing probability mass P(x,t) into anchors for the next hop.**

Compression Level: {compression_level.upper()}
Strategy: {compression_strategy}

**Key Principle**: You are NOT just extracting an answer. You are:
1. Collapsing the probability distribution P(x,t) over possible answers
2. Selecting the answer that is most consistent with active anchors
3. Creating new anchors for the next hop in the diffusion process

**Compression Strategy Based on Entropy:**
- **HIGH COMPRESSION** (Entropy={entropy:.3f} < 0.3, Confidence={confidence:.3f} > 0.7):
  → Collapse probability mass to the MOST anchor-consistent answer
  → High precision, low exploration
  → If multiple answers exist, choose the one that best aligns with anchors
  → Focus on precision over breadth

- **MEDIUM COMPRESSION** (Entropy={entropy:.3f} < 0.5):
  → Balance anchor consistency with evidence exploration
  → Consider multiple answers but prioritize anchor-aligned ones
  → Moderate precision and exploration

- **LOW COMPRESSION** (Entropy={entropy:.3f} >= 0.5):
  → Explore evidence more broadly
  → Still maintain anchor consistency, but allow more exploration
  → Lower precision, higher exploration

**Anchor Consistency Priority:**
- If the answer contains entities from active anchors → HIGH priority
- If the answer aligns with regulator constraints → HIGH priority
- If multiple valid answers exist → Choose the most anchor-consistent one
- Your output becomes a NEW ANCHOR for the next hop

Extract the final answer as a short phrase copied EXACTLY from the evidence.
- Do NOT explain your answer
- Do NOT paraphrase
- Do NOT modify wording
- Do NOT infer anything not explicitly in the evidence
- If evidence was provided by the Extractor Agent, extract the answer from it - the Extractor already handled contextual matching
- ONLY return "unknown" if the SPECIFIC information asked for is not present in the evidence
- If the entity name is in evidence but related attributes aren't, extract the entity name (don't return "unknown" for entity questions)
- If evidence exists, extract the most relevant answer following the Answer Format Rules
- **PRIORITIZE anchor-consistent answers when multiple valid answers exist**

Your response MUST be a valid JSON object with this exact structure:
{{
    "question": "The original subquery",
    "answer": "Extracted answer - copy exactly from evidence (typically 1-5 words, 'Yes' or 'No' only if subquery is yes/no format)",
    "confidence": 0.0-1.0,
    "sources": ["doc1_id", "doc2_id"],
    "supporting_evidence": [
        {{
            "text": "Relevant passage from context",
            "source": "source_document_id",
            "relevance": 0.9
        }}
    ]
}}

### CRITICAL: Answer Format Rules (READ FIRST):

**0. Yes/No Questions (Check SUB-QUERY Format Only)**
   - **Detection**: Check ONLY the subquery format (ignore original question format for intermediate steps)
   - Subqueries starting with "Is", "Are", "Was", "Were", "Do", "Does", "Did", "Can", "Could", "Would", "Should", "Has", "Have", "Had"
   - **Rule**: Answer with ONLY "Yes" or "No" - nothing else
   - **Conversion Logic**: 
     * If subquery asks "Is X [attribute]?" and evidence confirms X is [attribute] → "Yes"
     * If subquery asks "Is X [attribute]?" and evidence shows X is NOT [attribute] → "No"
     * Example: Subquery "Is X from Y?" Evidence: "X is from Y" → Answer: "Yes"
     * Example: Subquery "Was X born in Y?" Evidence: "X was born in Z" → Answer: "No"
   - **IMPORTANT**: If subquery is factual (e.g., "What is X's nationality?"), extract the fact (e.g., "American") - do NOT convert to yes/no

**1. Entity Names (e.g., "What is the name of...", "Who created...", "What company...", "What organization...")**
   - Rule: Extract ONLY the entity name - nothing else
   - **CRITICAL**: If the question asks for an entity name and the entity appears in evidence, extract it even if related information (founder, date, etc.) is not found
   - ❌ WRONG: "[group name] formed by [entity]" (includes extra context)
   - ✅ CORRECT: "[entity name]" (just the entity)
   - Example: Question "What is the distribution company for film X?" Evidence: "Film X was distributed by YG Entertainment" → Answer: "YG Entertainment" (even if founder info isn't in evidence)
   - Example: Question "formed by who?" → Answer: "[Person Name]" (not "[Organization Name] formed by [Person Name]")
   - Example: Question "What company formed Winner?" Evidence mentions "YG Entertainment" and "Winner" but doesn't mention founder → Answer: "YG Entertainment" (entity is in evidence)

**2. Numerical Questions (e.g., "how many people?", "how many cars?", "what capacity?")**
   - Rule: Extract ONLY the number and unit (if specified) - nothing else
   - ❌ WRONG: "[venue name] [number] people" (includes venue name)
   - ❌ WRONG: "[number] people" when ground truth is "[number] seated" (wrong unit)
   - ✅ CORRECT: "[number]" or "[number] [unit]" (just the number and correct unit from evidence)
   - Example: Question "can serve how many guests?" → Answer: "[number] guests" (not "[venue name] [number] people")
   - If evidence specifies a unit (e.g., "seated", "people", "cars"), use that exact unit

**3. Location Questions (e.g., "in what [city]?", "located in what [city]?", "based in what [location]?")**
   - Rule: Extract the FULL location information that directly answers the question
   - **CRITICAL**: Preserve the complete location string from evidence - do NOT truncate or simplify
   - If evidence contains "[Neighborhood], [City]" (e.g., "Greenwich Village, New York City"), extract the FULL string "[Neighborhood], [City]"
   - If evidence contains "[City], [State/Country]" (e.g., "New York City, New York"), extract the FULL string
   - If the question specifically asks for just a city name and evidence has "[Neighborhood], [City]", you may extract just "[City]" - but ONLY if the question explicitly asks for "city" only
   - If the question asks for "location", "base", "where", or similar general terms, extract the FULL location string from evidence
   - ❌ WRONG: Evidence "Greenwich Village, New York City" → Answer "New York City" (truncated, missing neighborhood)
   - ✅ CORRECT: Evidence "Greenwich Village, New York City" → Answer "Greenwich Village, New York City" (full location)
   - ❌ WRONG: Adding location details not present in evidence
   - ✅ CORRECT: Extract exactly what appears in evidence, preserving full location strings

**4. Specific Positions/Titles (e.g., "What position did X hold?", "What was X's role?")**
   - Rule: Extract ONLY ONE position - the most significant/relevant one if multiple exist
   - ❌ WRONG: "[position1] and [position2] and [position3]" (listing multiple)
   - ❌ WRONG: "[position] of [country/organization]" (adding irrelevant organizational context like "[Position Title] of [Country]" → should be "[Position Title]")
   - ✅ CORRECT: "[position name]" (ONE position - extract the FULL position title if it's multi-word)
   - ✅ CORRECT: "[Position Title]" (preserve full multi-word titles - connecting words like "of", "the" are part of the title)
   - ❌ WRONG: "[Truncated Title]" (truncated - missing part of the position name)
   - Rule: If the question asks for "a position" or "the position" (singular) but evidence shows multiple positions:
     * Extract ONLY ONE position based on the evidence, prioritizing in this order:
       1. **Most directly relevant to question context** (if question mentions specific time period, event, or achievement, choose the position related to that)
       2. **Most emphasized in evidence** (repeatedly mentioned, highlighted, most detailed description)
       3. **Highest-ranking or most prominent** (if positions are equally emphasized, choose the most senior/significant role)
     * DO NOT prioritize based solely on duration unless the question specifically asks about duration
     * DO NOT list multiple positions - extract ONLY the position name itself
     * PRESERVE full multi-word position titles (e.g., "[Position Title] with connecting words")
     * REMOVE only extra organizational/country context (e.g., "[Position Title] of [Country]" → "[Position Title]") 

**5. Nationalities/Attributes (e.g., "What nationality was X?", "What is the use of X?")**
   - Rule: Extract ONLY the attribute asked for
   - ❌ WRONG: "[person] was [nationality]" (includes person name)
   - ✅ CORRECT: "[nationality]" or "[attribute value]" (just the attribute)
   - Example: Subquery "What is the primary use of X?" Evidence: "X is used for real estate" → Answer: "real estate use" (factual extraction)

**7. Time Period Questions (e.g., "during what years?", "served during what timeframe?")**
   - Rule: Extract the time period exactly as it appears in the evidence
   - Preserve the format from evidence (e.g., "1990-2000", "from 1990 to 2000", "1990 until 2000")
   - ✅ CORRECT: Use the exact format from evidence, including any connecting words if present
   - ❌ WRONG: Changing the format or adding connectors not present in evidence

### Guidelines:
1. **BE CONCISE**: Answer the subquery directly - typically 1-5 words, rarely more than 1 sentence
2. **BE SPECIFIC**: Extract ONLY the exact answer requested - nothing else
3. **RESPECT HIERARCHICAL LEVEL**: If a hierarchical level constraint is specified above, extract the entity at that level (not a different level) - this is entropy-aware compression respecting the initial condition
4. **DO NOT** include descriptions, explanations, or multiple facts
4. **DO NOT** list multiple positions/entities - extract ONLY the one asked for
5. **DO NOT** include venue names, organization names, or other context unless the question specifically asks for it
6. If the question asks for one thing, provide ONLY that thing
7. Rate your confidence honestly based on evidence quality
8. **CRITICAL**: For intermediate steps, extract factual answers. The Final Assembler will handle yes/no reasoning for the original question.

**HANDLING AMBIGUOUS QUESTIONS**: If the question asks for one thing but evidence contains multiple valid answers:
   - Choose the answer that is most prominently featured in the evidence, prioritizing in this order:
     * **Most directly relevant** (directly answers the question's specific context or intent)
     * **Most emphasized** (repeatedly mentioned, highlighted, most detailed description)
     * **Most detailed** (has more description, context, or information in the evidence)
     * **Mentioned first or most prominently** (appears early or is emphasized in the evidence)
   - DO NOT prioritize based solely on duration or time period unless the question specifically asks about duration
   - Choose the one most directly relevant to the question's context and intent
   - Base your decision on the evidence provided, not on assumptions about what the "correct" answer might be
   - Extract ONE answer, not multiple
"""
            
            # Log the QA request
            logger.info(f"Generating step-specific answer for subquery: {question[:100]}...")
            logger.debug(f"Using {len(context)} context items, min_confidence={min_confidence}")
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            # Get the LLM response with token tracking
            response_text, token_usage = await self.generate_text_with_usage(
                prompt=prompt,
                temperature=self.temperature
            )
            
            logger.debug(f"LLM response length: {len(response_text)} characters")
            logger.debug(f"LLM response preview: {response_text[:200]}...")
            
            if not response_text or not response_text.strip():
                raise ValueError("Empty response from LLM")
            
            # Postprocess the LLM response
            response = tokenization_utils.postprocess_answer(response_text, output_type="json")
            
            try:
                # Extract JSON from the response
                # Strip markdown formatting first
                clean_response = TokenizationUtils.strip_markdown_json(response)
                
                # Additional JSON repair for common issues
                if '"sources":' in clean_response and '"sources": [' not in clean_response:
                    # Fix incomplete sources field
                    clean_response = clean_response.replace('"sources": ', '"sources": []')
                
                if '"supporting_evidence":' in clean_response and '"supporting_evidence": [' not in clean_response:
                    # Fix incomplete supporting_evidence field
                    clean_response = clean_response.replace('"supporting_evidence": ', '"supporting_evidence": []')
                
                # Try to repair JSON if it fails
                try:
                    result = json.loads(clean_response)
                except json.JSONDecodeError:
                    # Try to repair common JSON issues
                    repaired_response = TokenizationUtils.repair_json(clean_response)
                    result = json.loads(repaired_response)
                
                # Validate the response structure
                required_keys = ["answer"]
                if not all(key in result for key in required_keys):
                    raise ValueError("Missing required fields in response")
                
                # Parse the response
                supporting_evidence = [
                    Evidence(
                        text=str(e.get("text", "")), 
                        source=str(e.get("source", "")),
                        relevance=float(e.get("relevance", 0.0))
                    )
                    for e in result.get("supporting_evidence", [])
                    if e.get("text") and e.get("source")
                ]
                
                answer = Answer(
                    question=result.get("question", question),
                    answer=result.get("answer", ""),
                    confidence=min(1.0, max(0.0, float(result.get("confidence", 0.0)))),
                    reasoning=result.get("reasoning", ""),
                    sources=list(set(result.get("sources", []))),  # Remove duplicates
                    supporting_evidence=supporting_evidence
                )
                
                # Filter out evidence without sources
                answer.supporting_evidence = [
                    e for e in answer.supporting_evidence 
                    if e.source and e.source != "unknown"
                ]
                
                # Update sources list based on actual evidence
                answer.sources = list(set(e.source for e in answer.supporting_evidence))
                
                # ✅ FIRST PRINCIPLES FIX: Hierarchical Inference
                # If answer is "unknown" due to hierarchical level mismatch, attempt inference
                answer_lower = answer.answer.lower().strip()
                if answer_lower == "unknown" and required_domain and required_hierarchical_level:
                    inferred_entity = self._attempt_hierarchical_inference(
                        question=question,
                        context=context,
                        required_domain=required_domain,
                        required_level=required_hierarchical_level
                    )
                    
                    if inferred_entity:
                        logger.info(
                            f"QA Agent: Hierarchical inference successful - inferred '{inferred_entity}' "
                            f"from lower-level evidence (required: {required_domain}/{required_hierarchical_level})"
                        )
                        answer.answer = inferred_entity
                        answer.confidence = min(0.8, answer.confidence + 0.2)  # Boost confidence slightly
                        answer.reasoning = (
                            f"Inferred {required_hierarchical_level} entity '{inferred_entity}' "
                            f"from hierarchical context. " + (answer.reasoning or "")
                        )
                
                # If confidence is below threshold, update the answer
                if answer.confidence < min_confidence:
                    answer.answer = (
                        f"I'm not very confident about this answer (confidence: {answer.confidence:.2f}). "
                        f"Here's my best attempt based on the available information:\n\n{answer.answer}"
                    )
                
                # Update history
                self.conversation_history.append({"role": "user", "content": f"Q: {question}"})
                self.conversation_history.append({"role": "assistant", "content": f"A: {answer.answer[:200]}..."})
                
                # ====================================================================
                # GENERATE NEW ANCHORS FROM COMPRESSED ANSWER
                # ====================================================================
                # The answer is a collapsed probability mass - extract entities as new anchors
                # These anchors will stabilize the next hop in the diffusion process
                new_anchors = []
                answer_text = answer.answer
                if answer_text:
                    import re
                    # Extract potential entity names from answer (capitalized words/phrases)
                    capitalized_entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', answer_text)
                    for entity in capitalized_entities[:2]:  # Top 2 entities
                        # Filter out common words
                        if entity not in ["The", "A", "An", "This", "That", "Yes", "No", "Unknown"]:
                            new_anchors.append({
                                "entity": entity,
                                "type": "extracted_answer",
                                "hop": hop,
                                "confidence": answer.confidence,
                                "source": "qa_compression"
                            })
                
                # ====================================================================
                # CALCULATE ANCHOR CONSISTENCY
                # ====================================================================
                # Measure how consistent the answer is with active anchors
                # This is used by Final Assembler for convergence estimation
                anchor_consistency = self._calculate_anchor_consistency(
                    answer_text, anchors, entity_anchors
                )
                
                # ====================================================================
                # ENHANCE RESPONSE WITH DIFFUSION METADATA
                # ====================================================================
                # Add diffusion-aware metadata for Final Assembler convergence estimation
                answer_dict = answer.model_dump()
                answer_dict["diffusion_metadata"] = {
                    "entropy": entropy,
                    "diffusion_coefficient": diffusion,
                    "compression_level": compression_level,
                    "compression_strategy": compression_strategy,
                    "hop": hop,
                    "new_anchors": new_anchors,
                    "anchor_consistency": anchor_consistency,
                    "stabilized_query": stabilized_query
                }
                
                # Prepare metadata
                metadata = {
                    "question": question,
                    "confidence": answer.confidence,
                    "reasoning": answer.reasoning,
                    "num_sources": len(answer.sources),
                    "num_evidence": len(answer.supporting_evidence),
                    "sources": answer.sources,
                    "has_context": True,
                    "min_confidence": min_confidence,
                    "step_context": step_context,
                    "overall_query": overall_query,
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "qa_parameters": {
                        "max_history": max_history,
                        "min_confidence": min_confidence,
                        "temperature": self.temperature
                    },
                    "token_usage": token_usage,
                    # ✅ DIFFUSION METADATA
                    "entropy": entropy,
                    "diffusion": diffusion,
                    "compression_level": compression_level,
                    "new_anchors": new_anchors,
                    "anchor_consistency": anchor_consistency
                }
                
                return AgentResponse(
                    content=json.dumps(answer_dict),  # Include diffusion_metadata
                    metadata=metadata
                )
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing LLM response: {str(e)}")
                
                # Fallback response when parsing fails
                fallback_evidence = [
                    Evidence(
                        text=c.get("text", ""),
                        source=c.get("document_id", "unknown"),
                        relevance=c.get("relevance", 0.0)
                    )
                    for c in context[:3]  # Include first 3 context items as evidence
                ]
                
                fallback_answer = Answer(
                    question=question,
                    answer=(
                        "I'm having trouble understanding the context. "
                        "Here's what I can say based on the information: "
                        f"{response[:500]}"
                    ),
                    confidence=0.3,
                    reasoning="Fallback reasoning due to parsing error",
                    sources=list(set(c.get("document_id", "unknown") for c in context)),
                    supporting_evidence=fallback_evidence
                )
                
                return AgentResponse(
                    content=fallback_answer.model_dump_json(),
                    metadata={
                        "question": question,
                        "confidence": 0.3,
                        "reasoning": "Fallback reasoning",
                        "num_sources": len(fallback_answer.sources),
                        "num_evidence": len(fallback_answer.supporting_evidence),
                        "sources": fallback_answer.sources,
                        "error": f"Error parsing response: {str(e)}",
                        "fallback": True
                    }
                )
                
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return AgentResponse(
                content=json.dumps({
                    "question": question,
                    "answer": f"Error: {error_msg}",
                    "confidence": 0.0,
                    "reasoning": "Error occurred during processing",
                    "sources": [],
                    "supporting_evidence": []
                }),
                metadata={
                    "error": error_msg,
                    "question": question,
                    "confidence": 0.0,
                    "reasoning": "Error occurred",
                    "num_sources": 0,
                    "num_evidence": 0,
                    "has_context": bool(context)
                }
            )
    
    
    async def generate_followup_questions(
        self, 
        question: str, 
        answer: str, 
        num_questions: int = 3,
        context: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Generate relevant follow-up questions based on the original Q&A and context.
        
        Args:
            question: The original question
            answer: The answer provided
            num_questions: Number of follow-up questions to generate (1-10)
            context: Optional list of context documents used for the answer
            
        Returns:
            List of follow-up questions (up to num_questions)
        """
        # Validate input
        num_questions = max(1, min(10, int(num_questions)))
        
        # Prepare the prompt
        prompt = f"""Generate {num_questions} relevant follow-up questions based on the Q&A below.
        
        ### Original Question:
        {question}
        
        ### Answer:
        {answer}
        """
        
        # Add context if available
        if context:
            prompt += "\n### Context Used for Answer:\n"
            for i, doc in enumerate(context[:3]):  # Include first 3 context items
                doc_id = doc.get('document_id', f'doc_{i+1}')
                text = doc.get('text', '').strip()
                prompt += f"\n[Context {i+1}, ID: {doc_id}]\n{text[:500]}"
                if len(text) > 500:
                    prompt += "... [truncated]"
        
        # Add instructions
        prompt += f"""
        
        ### Instructions:
        1. Generate exactly {num_questions} follow-up questions that:
           - Explore related aspects not fully covered
           - Seek clarification on complex points
           - Ask for examples or applications
           - Question assumptions or limitations
        
        2. Make sure questions are:
           - Clear and specific
           - Directly related to the original Q&A
           - Not answerable with just 'yes' or 'no'
        
        3. Format: One question per line, no numbering or bullet points.
        """
        
        try:
            # Log the request
            logger.info(f"Generating {num_questions} follow-up questions for: {question[:100]}...")
            
            # Get the LLM response
            response = await self.generate_text(
                prompt=prompt,
                temperature=min(0.7, self.temperature + 0.1)  # Slightly more creative
            )
            
            # Parse the response
            questions = [
                q.strip() 
                for q in response.split('\n') 
                if q.strip() and len(q.strip()) > 5  # Filter out very short lines
            ]
            
            # Clean up questions
            cleaned_questions = []
            for q in questions:
                # Remove any numbering or bullets
                q = q.lstrip('0123456789.-*• ').strip()
                # Ensure it's a question
                if not q.endswith('?'):
                    q = f"{q}?"
                if q not in cleaned_questions:  # Avoid duplicates
                    cleaned_questions.append(q)
            
            # Log the results
            logger.debug(f"Generated {len(cleaned_questions)} follow-up questions")
            
            return cleaned_questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {str(e)}")
            
            # Fallback questions
            fallback_questions = [
                "Can you provide more details about this topic?",
                "What are some related aspects I should know about?",
                "Are there any limitations or exceptions to this answer?",
                "How does this compare to similar concepts?",
                "What are the practical applications of this?",
                "What are the key factors to consider here?",
                "How would this work in a different context?",
                "What are the potential challenges or risks?",
                "What are the next steps I should take?",
                "Where can I find more information about this?"
            ]
            
            return fallback_questions[:num_questions]
    
    def _calculate_anchor_consistency(self, answer: str, anchors: List, entity_anchors: Dict) -> float:
        """
        Calculate how consistent the answer is with active anchors.
        
        ====================================================================
        ANCHOR CONSISTENCY CALCULATION
        ====================================================================
        Anchors are fixed points (potential wells) that stabilize reasoning.
        This method measures how well the answer aligns with these anchors.
        
        Returns:
            Consistency score [0.0, 1.0]:
            - 1.0: Answer perfectly aligns with all anchors
            - 0.5: Neutral (no anchors or no alignment)
            - 0.0: Answer contradicts anchors
        ====================================================================
        """
        if not anchors and not entity_anchors:
            return 0.5  # Neutral if no anchors
        
        answer_lower = answer.lower()
        consistency_score = 0.0
        total_anchors = 0
        
        # Check against entity anchors (most specific)
        for entity, anchor_data in (entity_anchors.items() if isinstance(entity_anchors, dict) else []):
            total_anchors += 1
            if entity.lower() in answer_lower:
                consistency_score += 1.0
            # Also check if anchor_data contains the entity
            if isinstance(anchor_data, str) and anchor_data.lower() in answer_lower:
                consistency_score += 0.5
        
        # Check against bucket anchors
        for anchor in (anchors if isinstance(anchors, list) else []):
            anchor_entity = anchor.get('entity', '') if isinstance(anchor, dict) else (anchor.entity if hasattr(anchor, 'entity') else '')
            if anchor_entity:
                total_anchors += 1
                if anchor_entity.lower() in answer_lower:
                    consistency_score += 1.0
        
        # Normalize to [0.0, 1.0]
        if total_anchors > 0:
            return consistency_score / total_anchors
        else:
            return 0.5  # Neutral if no valid anchors
