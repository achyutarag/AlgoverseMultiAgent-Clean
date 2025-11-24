from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from .base_agent import BaseAgent, AgentResponse
from .tokenization_utils import tokenization_utils, TokenizationUtils
import json
import logging

logger = logging.getLogger(__name__)

class SubQuery(BaseModel):
    """A sub-query for a specific step."""
    id: str = Field(..., description="Unique identifier for the sub-query")
    query: str = Field(..., description="The specific question to answer")
    purpose: str = Field(..., description="What this sub-query aims to accomplish")
    priority: int = Field(1, description="Priority level (1=highest)")
    context_needed: List[str] = Field(default_factory=list, description="Types of context needed for this sub-query")

class StepDefinerAgent(BaseAgent):
    """
    The Step Definer Agent makes abstract steps executable by generating detailed 
    subqueries tailored for retrieval. It conditions on the original query, plan, 
    current step, and accumulated history to bridge high-level intent and low-level execution.
    """
    
    def __init__(
        self, 
        model_config: Optional[Dict[str, Any]] = None,
        model_name: str = "gemini-2.5-flash",  # LLM for step definition
        max_subqueries: int = 3
    ):
        """
        Initialize the Step Definer Agent.
        
        Args:
            model_config: Configuration for the LLM
            model_name: Name of the model to use (if model_config not provided)
            max_subqueries: Maximum number of sub-queries to generate per step
        """
        super().__init__("step_definer_agent", model_config, model_name)
        self.max_subqueries = max_subqueries
        
        self.system_prompt = """You are an expert at converting abstract reasoning steps into specific, executable sub-queries for retrieval-augmented generation. Your task is to:

1. **Context Grounding**: Condition on the original query q, the overall plan P, the current step si, and accumulated history Hi-1 = {(s1, a1), ..., (si-1, ai-1)}

2. **Subquery Generation**: Generate detailed subqueries tailored for retrieval that bridge high-level intent and low-level execution

3. **Precision Focus**: Create subqueries that enable precise and relevant document retrieval by being specific about what information is needed

Guidelines for subquery generation:
- Make subqueries specific and actionable for retrieval (must be searchable questions)
- Each subquery should be a clear, answerable question that can be used to search documents
- Include specific entity names, dates, locations from previous steps when available
- Ensure subqueries are focused enough to retrieve precise information
- Include necessary context and constraints in the subquery
- Prioritize subqueries based on their importance to the step objective
- Consider what types of documents or information sources would be most relevant
- **CRITICAL: Avoid synthesis language** - DO NOT use words like "synthesize", "combine", "merge", "integrate", "assemble". Subqueries must be retrieval-focused questions, not synthesis instructions

Return your response as a JSON object with this structure:
{
    "step_id": "ID of the current step",
    "step_description": "Description of the current step",
    "reasoning": "Your reasoning for these specific sub-queries",
    "context_analysis": "Analysis of relevant context from previous steps",
    "sub_queries": [
        {
            "id": "subquery_1",
            "query": "Specific, focused question for retrieval",
            "purpose": "What this sub-query aims to accomplish",
            "priority": 1,
            "context_needed": ["factual", "statistical", "comparative"]
        }
    ]
}

Examples of good subquery generation:

**Step**: "Research environmental benefits of renewable energy"
- Subquery 1: "What are the specific environmental benefits of solar energy production?"
- Subquery 2: "What environmental advantages does wind energy have over fossil fuels?"
- Subquery 3: "How does hydroelectric power impact local ecosystems?"

**Step**: "Compare economic policies between countries"
- Subquery 1: "What are Japan's current monetary policy rates and targets?"
- Subquery 2: "What fiscal policies has South Korea implemented in the last 5 years?"
- Subquery 3: "How do Japan and South Korea differ in their trade policy approaches?"
"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process the current step and generate detailed subqueries conditioned on context and history.
        
        Args:
            input_data: Dictionary containing:
                - 'step': Dict with 'id', 'description', 'objective', 'dependencies', 'critical'
                - 'plan': The overall plan with all steps
                - 'history': List of previous interactions and their results
                - 'context': Additional context for the task
                - 'previous_answers': Dict of {step_id: answer} for completed steps
                - 'max_subqueries': Override default max subqueries
                
        Returns:
            AgentResponse containing the generated sub-queries and metadata
        """
        step = input_data.get('step', {})
        plan = input_data.get('plan', {})
        history = input_data.get('history', [])
        context = input_data.get('context', {})
        previous_answers = input_data.get('previous_answers', {})
        max_subqueries = min(input_data.get('max_subqueries', self.max_subqueries), 5)  # Cap at 5
        
        if not step or 'id' not in step:
            return AgentResponse(
                content="Error: Invalid or missing step information",
                metadata={"error": "Invalid step data"}
            )
        
        try:
            # Prepare the enhanced prompt with full context (preprocess for LLM)
            prompt = f"""{self.system_prompt}
            
            ### Main Question:
            {tokenization_utils.preprocess_llm_input(plan.get('main_question', 'Not specified'))}
            
            ### Disambiguated Query:
            {tokenization_utils.preprocess_llm_input(plan.get('disambiguated_query', plan.get('main_question', 'Not specified')))}
            
            ### Query Type:
            {plan.get('query_type', 'unknown')}
            
            ### Current Step:
            ID: {step.get('id', 'unknown')}
            Description: {tokenization_utils.preprocess_llm_input(step.get('description', 'No description'))}
            Objective: {tokenization_utils.preprocess_llm_input(step.get('objective', 'No objective'))}
            Dependencies: {', '.join(step.get('dependencies', [])) or 'None'}
            Critical: {step.get('critical', False)}
            Expected Output: {tokenization_utils.preprocess_llm_input(step.get('expected_output', 'Not specified'))}
            """
            
            # Add context if available
            if context:
                prompt += f"\n### Additional Context:\n{json.dumps(context, indent=2)}"
            
            # Add information about previous steps and their answers
            if previous_answers:
                prompt += "\n\n### Previous Steps and Answers:"
                for step_id, answer in previous_answers.items():
                    # Truncate long answers for context
                    answer_preview = str(answer)[:300] + "..." if len(str(answer)) > 300 else str(answer)
                    prompt += f"\n- Step {step_id}: {answer_preview}"
            
            # Add relevant conversation history
            if history:
                prompt += "\n\n### Conversation History:"
                for i, h in enumerate(history[-5:]):  # Last 5 messages for more context
                    role = h.get('role', 'unknown').upper()
                    content = h.get('content', '')[:400]  # Truncate long content
                    prompt += f"\n{role}: {content}"
            
            # Extract step information
            step_id = step.get('id', 'unknown')
            step_description = step.get('description', 'No description')
            step_objective = step.get('objective', 'No objective')
            original_query = plan.get('main_question', 'Unknown query')
            
            # Format previous answers for prompts
            previous_answers_text = ""
            if previous_answers:
                previous_answers_text = "\n\n### Previous Steps and Answers (CRITICAL for multi-hop queries):"
                for step_id_prev, answer in previous_answers.items():
                    answer_str = str(answer)[:200] + "..." if len(str(answer)) > 200 else str(answer)
                    previous_answers_text += f"\n- Step {step_id_prev}: {answer_str}"
                previous_answers_text += "\n\nIMPORTANT: If this step depends on previous steps, you MUST use the specific entity names, values, or information from the answers above."
            
            # First, get basic step info
            basic_prompt = f"""Step: {step_description}
Objective: {step_objective}
Original Query: {original_query}
{previous_answers_text}

Return a JSON object with basic info:
{{
    "step_id": "{step_id}",
    "step_description": "{step_description}",
    "reasoning": "brief reasoning about how to accomplish this step"
}}

Return ONLY the JSON."""
            
            # Generate basic step info with token tracking
            basic_response_text, basic_token_usage = await self.generate_text_with_usage(
                basic_prompt,
                temperature=0.2,
                max_new_tokens=512
            )
            total_token_usage = {
                "prompt_tokens": basic_token_usage.get("prompt_tokens", 0),
                "generated_tokens": basic_token_usage.get("generated_tokens", 0),
                "total_tokens": basic_token_usage.get("total_tokens", 0)
            }
            
            basic_response = tokenization_utils.postprocess_answer(basic_response_text, output_type="json")
            
            try:
                clean_basic = TokenizationUtils.repair_json(basic_response)
                result = json.loads(clean_basic)
                logger.info(f"Successfully parsed basic step info")
            except Exception as e:
                logger.error(f"Failed to parse basic step info: {e}")
                logger.error(f"Raw response: {basic_response[:200]}")
                result = {
                    "step_id": step_id,
                    "step_description": step_description,
                    "reasoning": "Generate sub-queries to accomplish this step"
                }
            
            # Check if step can be answered directly from previous answers (before generating sub-queries)
            if previous_answers:
                check_direct_answer_prompt = f"""Step: {step_description}
Objective: {step_objective}
Original Query: {original_query}
{previous_answers_text}

**CRITICAL CHECK**: Can this step be answered DIRECTLY from the previous answers above WITHOUT needing to retrieve new documents?

**IMPORTANT: Comparative Questions Require All Entities**
- If the step asks to COMPARE two entities (e.g., "Who is older, X or Y?", "Compare X and Y", "Which is larger, X or Y?"), you need information about BOTH entities
- If previous steps only have information about ONE entity, you CANNOT answer directly - need to retrieve information about the other entity
- Example: Step 1 found "Annie Morton's birth date: October 8, 1970" and Step 3 asks "Who is older, Annie Morton or Terry Richardson?" → NO, need to retrieve Terry Richardson's birth date (only have Annie's)
- Example: Step 1 found "X's attribute: A" and Step 2 found "Y's attribute: B" and Step 3 asks "Compare X and Y" → MAYBE, depends on what comparison is needed

Examples:
- If Step 1 found "[Location A], [Location B]" and Step 2 asks "Identify the [Location B] from step 1" → YES, answer is "[Location B]" (already available in previous answer)
- If Step 1 found "[Entity Name]" and Step 2 asks "Find information about [Entity Name]'s [Attribute]" → NO, need to retrieve new documents about the entity's attribute
- If Step 1 found "[Person Name]" and Step 2 asks "What is [Person Name]'s [Attribute]?" → NO, need to retrieve documents about the person's attribute
- If Step 1 found "[Value]" and Step 2 asks "What is the [Value] from step 1?" → YES, answer is "[Value]" (already available in previous answer)
- If Step 1 found "[Entity A]" and Step 2 asks "What is the [Component] of [Entity A]?" → NO, need to retrieve documents about the entity's component
- If Step 1 found "[Complete Answer]" and Step 2 asks "Extract the [Part] from step 1" → YES, answer can be extracted from "[Complete Answer]" (already available)
- **CRITICAL**: If Step 1 found "[Entity X's attribute: Value]" and Step 3 asks "Compare Entity X and Entity Y" or "Who is older, X or Y?" → NO, need Entity Y's information

Key principle: If the step asks to EXTRACT, IDENTIFY, or SELECT a specific part from a previous answer that already contains that information, it can be answered directly. If it asks to FIND, SEARCH, RETRIEVE, or COMPARE information about entities, it requires retrieval. For comparative questions, ALL entities must have their information available in previous steps.

Return ONLY a JSON object:
{{
    "can_answer_directly": true/false,
    "reasoning": "brief explanation",
    "direct_answer": "the answer if can_answer_directly is true, otherwise null"
}}

Return ONLY the JSON."""

                check_response_text, check_token_usage = await self.generate_text_with_usage(
                    check_direct_answer_prompt,
                    temperature=0.1,  # Low temperature for deterministic check
                    max_new_tokens=256
                )
                total_token_usage["prompt_tokens"] += check_token_usage.get("prompt_tokens", 0)
                total_token_usage["generated_tokens"] += check_token_usage.get("generated_tokens", 0)
                total_token_usage["total_tokens"] += check_token_usage.get("total_tokens", 0)
                
                check_response = tokenization_utils.postprocess_answer(check_response_text, output_type="json")
                try:
                    clean_check = TokenizationUtils.repair_json(check_response)
                    check_result = json.loads(clean_check)
                    
                    # If step can be answered directly, return a single "extraction" sub-query
                    if check_result.get("can_answer_directly", False):
                        direct_answer = check_result.get("direct_answer", "")
                        logger.info(f"Step {step_id} can be answered directly from previous steps: {direct_answer}")
                        
                        # Return a single sub-query that extracts from previous answers
                        return AgentResponse(
                            content=json.dumps({
                                "sub_queries": [{
                                    "id": "direct_answer_sq",
                                    "query": f"Extract the answer from previous step results: {step_objective}",
                                    "purpose": f"Answer directly from previous steps: {direct_answer}",
                                    "priority": 1,
                                    "context_needed": ["previous_answers"],
                                    "direct_answer": direct_answer  # Include the direct answer
                                }]
                            }, indent=2),
                            metadata={
                                "step_id": step_id,
                                "num_subqueries": 1,
                                "direct_answer": True,
                                "token_usage": total_token_usage
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to parse direct answer check, proceeding with normal sub-query generation: {e}")
            
            # Now generate sub-queries one by one
            sub_queries = []
            num_subqueries = min(2, max_subqueries)  # Generate 2 sub-queries
            
            # Extract entities from step description for multi-entity handling
            step_lower = step_description.lower()
            step_objective_lower = step_objective.lower() if step_objective else ""
            
            # Detect if step involves multiple entities (look for "and", "both", entity patterns)
            has_multiple_entities = any(keyword in step_lower or keyword in step_objective_lower 
                                       for keyword in [" and ", "both", " and ", "both"])
            
            # Try to extract entity names from step description
            import re
            # Pattern: Look for capitalized words that might be entity names
            potential_entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', step_description)
            # Filter out common words
            common_words = {"What", "Where", "When", "Who", "Which", "Find", "Locate", "Identify", 
                          "Determine", "Check", "Verify", "For", "The", "From", "To", "Step"}
            potential_entities = [e for e in potential_entities if e not in common_words]
            
            for i in range(num_subqueries):
                sq_num = i + 1
                previous_sqs = "\n".join([f"Sub-query {j+1}: {sq['query']}" for j, sq in enumerate(sub_queries)])
                
                # Build entity context for multi-entity steps
                entity_context = ""
                if has_multiple_entities and potential_entities:
                    entity_context = f"\n\n**MULTI-ENTITY STEP DETECTED**: This step involves multiple entities: {', '.join(potential_entities[:4])}"
                    entity_context += f"\n- Generate ONE sub-query per entity if the step asks about attributes of multiple entities"
                    entity_context += f"\n- Sub-query {sq_num} should focus on: {potential_entities[i % len(potential_entities)] if i < len(potential_entities) else 'the remaining entity'}"
                
                sq_prompt = f"""Step: {step_description}
Objective: {step_objective}
Original Query: {original_query}
{previous_answers_text}
{entity_context}

Previous sub-queries:
{previous_sqs if previous_sqs else "None yet"}

**CRITICAL: RETRIEVAL-FOCUSED SUBQUERIES ONLY**
Your sub-query MUST be a retrieval-focused question that can be used to search documents. It should:
1. Be a clear, answerable question (e.g., "What is X?", "Who founded Y?", "When did Z happen?")
2. Include specific entity names from previous steps when available (use exact names, dates, locations from previous answers)
3. Be searchable - a document search engine should be able to find relevant documents using this query
4. Avoid synthesis language - DO NOT use words like "synthesize", "combine", "merge", "integrate", "assemble", "put together"
5. **CRITICAL**: DO NOT generate a sub-query that is identical or very similar to previous sub-queries

**❌ FORBIDDEN SYNTHESIS LANGUAGE:**
- ❌ "Synthesize the distribution company and its founder" (synthesis, not retrieval)
- ❌ "Combine information from step 1 and step 2" (synthesis, not retrieval)
- ❌ "Merge the results to find..." (synthesis, not retrieval)
- ❌ "Integrate the answers to determine..." (synthesis, not retrieval)

**✅ CORRECT RETRIEVAL-FOCUSED SUBQUERIES:**
- ✅ "What is the distribution company for UHF film?" (retrieval-focused, searchable)
- ✅ "Who is a prominent founder of The Union during the 18th century?" (retrieval-focused, uses entity from previous step)
- ✅ "What is Abraham Lincoln's role as a Union officer?" (retrieval-focused, includes entities)

**CRITICAL: Multi-Entity Steps**
- If the step asks about multiple entities (e.g., "both X and Y", "X and Y"), generate ONE sub-query per entity
- Example: Step "Find nationality of Local H and For Against" → Sub-query 1: "What is Local H's nationality?", Sub-query 2: "What is For Against's nationality?"
- Example: Step "Find primary use of Random House Tower and 888 7th Avenue" → Sub-query 1: "What is the primary use of Random House Tower?", Sub-query 2: "What is the primary use of 888 7th Avenue?"
- **DO NOT generate duplicate sub-queries** - each entity should get its own unique sub-query
- If you've already generated a sub-query for an entity, generate one for a different entity

**CRITICAL: Prioritize Evidence-Based Terminology from Previous Steps**
- **Use EXACT terminology from previous step answers when available, even if it differs from the original query**
- Previous step answers represent what was ACTUALLY found in evidence, which is more reliable than original query wording
- If previous step answers use specific terminology (entity types, location formats, classifications), use that EXACT terminology in your sub-query
- Only fall back to original query terminology if previous steps don't provide it
- **Key principle**: Evidence-based terminology > Original query wording when they conflict
- **General rule**: 
  * If previous step answer contains a specific entity type/classification (e.g., administrative division type, location hierarchy level, entity category), use that exact type in your sub-query
  * If previous step answer contains a specific entity name/value, use that exact name/value in your sub-query
  * If previous step answer contains a specific format (e.g., location format, date format), preserve that exact format
  * If original query uses different terminology than previous step answers, prefer the terminology from previous step answers
- **Preserve original query's intent**: Keep asking for the same attribute (population, CEO, nationality, location, etc.) but use evidence-based entity types/names/formats from previous steps

**IMPORTANT INSTRUCTIONS FOR MULTI-HOP QUERIES:**
- If this step depends on previous steps, you MUST include the specific entity names, values, or information from previous answers in your sub-query
- Use the EXACT entity names, dates, locations, entity types, classifications, or other specific values from previous answers
- **Prioritize terminology from previous step answers over original query when they differ**
- Extract and use the exact terminology, entity types, and formats that appear in previous step answers
- If the current step is independent, create a focused retrieval sub-query without forcing in previous information

**CRITICAL: Preserve Question Structure (Entity Matching vs Attribute Extraction)**
- If the original question asks for an ENTITY NAME (e.g., "Which X... or Y?", "Who...?", "What is the name of...?"), generate subqueries that help IDENTIFY the entity, not extract attributes
- If the original question asks for an ATTRIBUTE (e.g., "What is the nationality of X?", "How many...?"), generate subqueries that extract the attribute
- ❌ WRONG: Question asks "Which writer was from England, X or Y?" → Subquery "Compare the nationality of X and Y" (extracts attribute, changes question structure)
- ✅ CORRECT: Question asks "Which writer was from England, X or Y?" → Subqueries "What is the nationality of X?" and "What is the nationality of Y?" (helps identify which entity matches)
- Preserve the original question's intent: entity selection vs attribute extraction

**CRITICAL: Semantic Interpretation of "Used For" Phrases**
- When the step or original query contains "used for Y", interpret Y as function/purpose, NOT category
- Generate sub-queries that ask about functional/purpose use, not property type/category
- **Key principle**: "X is used for Y" means "X serves Y-related functions/purposes", NOT "X is a Y-type entity"
- Example: Step "Find if X is used for real estate" → Sub-query: "What is X used for?" or "What purpose does X serve?" (NOT "What is the property type of X?")
- Example: Step "Find if X is used for country" → Sub-query: "What is X used for?" (country-related purposes), NOT "What country is X?"
- Example: Step "Find if X is used for luxury" → Sub-query: "What is X used for?" (luxury purposes), NOT "Is X a luxury item?"
- This applies to: addresses, buildings, tools, products, roles, documents, objects, and ambiguous nouns
- When generating sub-queries for "used for" questions, focus on extracting functional/purpose information, not category/type information

Generate sub-query {sq_num} to help accomplish this step. Return a JSON object:

**Example 1 (Simple)**: For step "Find the nationality of Scott Derrickson", return:
{{
    "id": "sq_{sq_num}",
    "query": "What is Scott Derrickson's nationality?",
    "purpose": "Find Scott Derrickson's nationality or country of origin",
    "priority": {sq_num},
    "context_needed": ["factual"]
}}

**Example 2 (Multi-hop)**: If Step 1 found "Orion Pictures" and Step 2 is "Find its founder", return:
{{
    "id": "sq_{sq_num}",
    "query": "Who founded Orion Pictures?",
    "purpose": "Find the founder of Orion Pictures",
    "priority": {sq_num},
    "context_needed": ["factual"]
}}

**Example 3 (Multi-Entity)**: For step "Find nationality of Local H and For Against", return:
- Sub-query 1: {{"query": "What is Local H's nationality?", ...}}
- Sub-query 2: {{"query": "What is For Against's nationality?", ...}}
- **DO NOT** generate duplicate queries like "What is Local H's nationality?" twice

**Example 4 (Avoiding Synthesis)**: If Step 1 found "UHF film" and Step 2 is "Find distribution company and founder", return TWO separate retrieval subqueries:
- Subquery 1: "What is the distribution company for UHF film?"
- Subquery 2: "Who founded [distribution company name from Step 1]?" (use entity from previous step)

Return ONLY the JSON object, no other text."""
                
                sq_response_text, sq_token_usage = await self.generate_text_with_usage(
                    sq_prompt,
                    temperature=0.3,
                    max_new_tokens=512
                )
                # Aggregate token usage
                total_token_usage["prompt_tokens"] += sq_token_usage.get("prompt_tokens", 0)
                total_token_usage["generated_tokens"] += sq_token_usage.get("generated_tokens", 0)
                total_token_usage["total_tokens"] += sq_token_usage.get("total_tokens", 0)
                
                sq_response = tokenization_utils.postprocess_answer(sq_response_text, output_type="json")
                
                try:
                    # Clean and repair JSON
                    clean_sq = TokenizationUtils.repair_json(sq_response)
                    
                    # Additional JSON repair for common issues
                    if '"context_needed":' in clean_sq and '"context_needed": [' not in clean_sq:
                        # Fix incomplete context_needed field
                        clean_sq = clean_sq.replace('"context_needed": ', '"context_needed": ["factual"]')
                    
                    sq = json.loads(clean_sq)
                    
                    # Validate sub-query has required fields
                    if all(k in sq for k in ["id", "query", "purpose"]):
                        # Ensure context_needed is a list
                        if "context_needed" not in sq or not isinstance(sq["context_needed"], list):
                            sq["context_needed"] = ["factual"]
                        
                        # Check for duplicates before appending
                        query_text = sq.get("query", "").strip().lower()
                        is_duplicate = False
                        for existing_sq in sub_queries:
                            existing_query = existing_sq.get("query", "").strip().lower()
                            # Check if queries are too similar (same or one contains the other)
                            if (query_text == existing_query or 
                                (len(query_text) > 10 and query_text in existing_query) or
                                (len(existing_query) > 10 and existing_query in query_text)):
                                is_duplicate = True
                                logger.warning(f"Sub-query {sq_num} is duplicate of previous sub-query, skipping")
                                break
                        
                        if not is_duplicate:
                            sub_queries.append(sq)
                            logger.info(f"Generated sub-query {sq_num}: {sq['query'][:50]}")
                        else:
                            # If duplicate, try one more time with explicit instruction
                            if i < num_subqueries - 1:  # Only retry if not last iteration
                                logger.info(f"Duplicate detected, will retry in next iteration")
                    else:
                        logger.warning(f"Sub-query {sq_num} missing required fields: {list(sq.keys())}")
                except Exception as e:
                    logger.error(f"Failed to parse sub-query {sq_num}: {e}")
                    logger.error(f"Raw sub-query response: {sq_response[:200]}")
                    continue
            
            # Add sub-queries to result
            result["sub_queries"] = sub_queries
            logger.info(f"Generated {len(sub_queries)} sub-queries for step {step_id}")
            
            # Add context_analysis if not present
            if "context_analysis" not in result:
                result["context_analysis"] = "Analysis of relevant context from previous steps"
            
            # Update history
            self._update_history("user", f"Define sub-queries for step: {step_id}")
            self._update_history("assistant", json.dumps(result, indent=2))
            
            # Parse sub-queries into SubQuery objects
            parsed_sub_queries = [
                SubQuery(
                    id=sq.get("id", f"subq_{i+1}"),
                    query=sq["query"],
                    purpose=sq["purpose"],
                    priority=int(sq.get("priority", i + 1)),
                    context_needed=sq.get("context_needed", ["factual"])
                )
                for i, sq in enumerate(result["sub_queries"])
            ]
            
            # Sort sub-queries by priority (ascending, so 1 comes first)
            parsed_sub_queries.sort(key=lambda x: x.priority)
            
            return AgentResponse(
                content=json.dumps({
                    "step_id": result["step_id"],
                    "step_description": result["step_description"],
                    "reasoning": result["reasoning"],
                    "context_analysis": result["context_analysis"],
                    "sub_queries": [sq.dict() for sq in parsed_sub_queries]
                }),
                metadata={
                    "step_id": result["step_id"],
                    "step_description": result["step_description"],
                    "token_usage": total_token_usage,
                    "reasoning": result["reasoning"],
                    "context_analysis": result["context_analysis"],
                    "sub_queries": [sq.dict() for sq in parsed_sub_queries],
                    "num_subqueries": len(parsed_sub_queries),
                    "step_definer_parameters": {
                        "max_subqueries": max_subqueries,
                        "model": self.model_name,
                        "temperature": 0.3
                    }
                }
            )
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            
            # Fallback: create a simple subquery
            fallback_subquery = {
                "step_id": step_id,
                "step_description": step_description,
                "reasoning": "Unable to parse detailed subqueries, using fallback approach",
                "context_analysis": "Limited context analysis due to parsing error",
                "sub_queries": [
                    {
                        "id": "subquery_1",
                        "query": step_description,
                        "purpose": f"Accomplish step objective: {step_objective}",
                        "priority": 1,
                        "context_needed": ["factual"]
                    }
                ]
            }
            
            return AgentResponse(
                content=json.dumps(fallback_subquery),
                metadata={
                    "step_id": step_id,
                    "step_description": step_description,
                    "reasoning": "Fallback subquery due to parsing error",
                    "context_analysis": "Limited analysis",
                    "sub_queries": fallback_subquery["sub_queries"],
                    "error": f"Failed to parse response: {str(e)}",
                    "fallback": True
                }
            )
                
        except Exception as e:
            logger.error(f"Error processing LLM response: {e}", exc_info=True)
            logger.error(f"Step ID: {step_id}, Description: {step_description}")
            return AgentResponse(
                content=f"Error processing LLM response: {str(e)}",
                metadata={
                    "error": "Response processing error",
                    "exception": str(e),
                    "exception_type": type(e).__name__
                }
            )
