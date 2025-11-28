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
- Make subqueries specific and actionable for retrieval
- Consider what context from previous steps might be relevant
- Ensure subqueries are focused enough to retrieve precise information
- Include necessary context and constraints in the subquery
- Prioritize subqueries based on their importance to the step objective
- Consider what types of documents or information sources would be most relevant

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
                temperature=0.3,
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
            
            # Now generate sub-queries one by one
            sub_queries = []
            num_subqueries = min(2, max_subqueries)  # Generate 2 sub-queries
            
            for i in range(num_subqueries):
                sq_num = i + 1
                previous_sqs = "\n".join([f"Sub-query {j+1}: {sq['query']}" for j, sq in enumerate(sub_queries)])
                
                sq_prompt = f"""Step: {step_description}
Objective: {step_objective}
Original Query: {original_query}
{previous_answers_text}

Previous sub-queries:
{previous_sqs if previous_sqs else "None yet"}

**IMPORTANT INSTRUCTIONS FOR MULTI-HOP QUERIES:**
- If this step depends on previous steps, you should include the specific entity names, values, or information from previous answers in your sub-query when relevant
- For example: If Step 1 found "John Smith" and Step 2 asks about John Smith's position, query "What position did John Smith hold?" rather than just "What position?"
- When previous steps provide relevant entities (names, dates, locations, etc.), incorporate them into your sub-query to improve retrieval accuracy
- Use the exact entity names, dates, locations, or other specific values from previous answers when they are directly relevant to the current step
- However, if the current step is independent or doesn't need previous context, create a focused sub-query without forcing in previous information

**CRITICAL: Preserve Question Structure (Entity Matching vs Attribute Extraction)**
- If the original question asks for an ENTITY NAME (e.g., "Which X... or Y?", "Who...?", "What is the name of...?"), generate subqueries that help IDENTIFY the entity, not extract attributes
- If the original question asks for an ATTRIBUTE (e.g., "What is the nationality of X?", "How many...?"), generate subqueries that extract the attribute
- ❌ WRONG: Question asks "Which writer was from England, X or Y?" → Subquery "Compare the nationality of X and Y" (extracts attribute, changes question structure)
- ✅ CORRECT: Question asks "Which writer was from England, X or Y?" → Subqueries "What is the nationality of X?" and "What is the nationality of Y?" (helps identify which entity matches)
- Preserve the original question's intent: entity selection vs attribute extraction

Generate sub-query {sq_num} to help accomplish this step. Return a JSON object:

**Example 1 (Simple)**: For step "Find the nationality of Scott Derrickson", return:
{{
    "id": "sq_{sq_num}",
    "query": "Scott Derrickson nationality American director",
    "purpose": "Find Scott Derrickson's nationality or country of origin",
    "priority": {sq_num},
    "context_needed": ["factual"]
}}

**Example 2 (Multi-hop)**: If Step 1 found "Jane Doe" and Step 2 is "Find her government position", return:
{{
    "id": "sq_{sq_num}",
    "query": "Jane Doe government position held",
    "purpose": "Find the government position held by Jane Doe",
    "priority": {sq_num},
    "context_needed": ["factual"]
}}

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
                        
                        sub_queries.append(sq)
                        logger.info(f"Generated sub-query {sq_num}: {sq['query'][:50]}")
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
