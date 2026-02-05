from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from .base_agent import BaseAgent, AgentResponse
from .tokenization_utils import tokenization_utils, TokenizationUtils
import json
import logging

logger = logging.getLogger(__name__)

class PlanStep(BaseModel):
    """A single step in the plan."""
    id: str = Field(..., description="Unique identifier for the step")
    description: str = Field(..., description="Description of the step")
    objective: str = Field(..., description="Objective of this step")
    dependencies: List[str] = Field(default_factory=list, description="List of step IDs this step depends on")
    critical: bool = Field(False, description="Whether this step is critical for the overall plan")

class PlannerAgent(BaseAgent):
    """
    The Planner Agent performs query disambiguation and task decomposition.
    It analyzes input queries to identify ambiguities and creates structured 
    reasoning plans with chain-of-thought prompting.
    """
    
    def __init__(
        self, 
        model_config: Optional[Dict[str, Any]] = None,
        model_name: str = "gemini-2.5-flash",  # LLM for complex planning
        max_steps: int = 5
    ):
        """
        Initialize the Planner Agent.
        
        Args:
            model_config: Configuration for the LLM
            model_name: Name of the model to use (if model_config not provided)
            max_steps: Maximum number of steps in the plan
        """
        super().__init__("planner_agent", model_config, model_name)
        self.max_steps = max_steps
        
        self.system_prompt = """You are a query planner. Create a step-by-step plan to answer the user's question.

CRITICAL: You MUST return a complete JSON object with ALL required fields, especially the "steps" array.

**IMPORTANT: Question Intent Analysis FIRST**
- Before generating steps, determine what the question is actually asking for
- Does it ask FOR an entity name? (e.g., "What company is X?", "What is the name of...")
- Does it ask ABOUT an entity? (e.g., "Who is CEO of X?", "What is X's attribute?")
- Generate only the steps needed - simple questions need 1 step, complex questions need 2-3 steps
- If question asks FOR entity name, DO NOT generate steps asking for CEO/founder/etc.

**IMPORTANT: Step descriptions must be retrieval-focused, NOT synthesis instructions**
- Step descriptions should describe what information to FIND or RETRIEVE
- DO NOT use synthesis language: "synthesize", "combine", "merge", "integrate", "assemble", "put together"
- Use action verbs: "Find", "Identify", "Search for", "Retrieve", "Locate", "Determine"
- Each step should be a clear information retrieval task, not a synthesis task

Return your response as a JSON object with this EXACT structure:
{
    "main_question": "The original question",
    "disambiguated_query": "Clarified version of the query if needed",
    "reasoning": "Your chain-of-thought reasoning for the plan",
    "query_type": "simple|multi-hop|comparative|analytical",
    "steps": [
        {
            "id": "step_1",
            "description": "Clear description of what this step does",
            "objective": "Specific objective this step aims to accomplish",
            "dependencies": [],
            "critical": true,
            "expected_output": "What kind of information this step should produce"
        }
    ]
}

IMPORTANT RULES:
1. **Determine question complexity FIRST** - Generate only the number of steps actually needed:
   - Simple entity lookup (e.g., "What is X?", "What company is Y?") → 1 step
   - Attribute lookup (e.g., "Who is CEO of X?", "What is X's nationality?") → 1-2 steps
   - Multi-hop (e.g., "Who founded the company that distributed film X?") → 2-3 steps
   - Comparative (e.g., "Are X and Y both Z?") → 2-3 steps (one per entity)
2. **Question Intent Analysis** - Before generating steps, determine:
   - Does the question ask FOR an entity name? (e.g., "What company...", "What is the name of...") → Generate step to find entity name
   - Does the question ask ABOUT an entity? (e.g., "Who is CEO of X?", "What is X's attribute?") → Generate step to find that specific attribute
   - **CRITICAL**: If question asks FOR entity name, DO NOT generate steps asking for CEO/founder/etc. - only find the entity name
3. Each step must have "id", "description", "objective", "dependencies", "critical", and "expected_output"
4. Make steps simple and actionable
5. The JSON must be complete and valid
6. Step descriptions must be retrieval-focused (what to find), not synthesis instructions

**❌ FORBIDDEN in step descriptions:**
- ❌ "Synthesize the information gathered in step 2"
- ❌ "Combine information from step 1 and step 2"
- ❌ "Merge the results to find..."

**✅ CORRECT step descriptions:**
- ✅ "Find the distribution company for the film"
- ✅ "Identify the founder of [company name from step 1]"
- ✅ "Locate information about [entity] and its [attribute]"

Example for "Who created X?":
{
    "main_question": "Who created X?",
    "disambiguated_query": "Identify the creator of X",
    "reasoning": "Simple factual lookup requiring identification of creator",
    "query_type": "simple",
    "steps": [
        {
            "id": "step_1",
            "description": "Search for information about X and its creator",
            "objective": "Find who created X",
            "dependencies": [],
            "critical": true,
            "expected_output": "Name of creator"
        },
        {
            "id": "step_2",
            "description": "Verify the creator information",
            "objective": "Confirm accuracy",
            "dependencies": ["step_1"],
            "critical": true,
            "expected_output": "Verified creator name"
        }
    ]
}"""
        
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process the input query and create a structured reasoning plan with disambiguation.
        
        Args:
            input_data: Dictionary containing:
                - 'query': The main question to plan for
                - Optional 'context': Additional context for planning
                - Optional 'max_steps': Override default max steps
                
        Returns:
            AgentResponse containing the structured plan with disambiguation
        """
        query = input_data.get('query', '')
        context = input_data.get('context', {})
        max_steps = min(input_data.get('max_steps', self.max_steps), 10)  # Cap at 10 steps
        
        # Normalize query for consistent processing
        query = tokenization_utils.normalize_query(query)
        
        if not query:
            return AgentResponse(
                content="Error: No query provided",
                metadata={"error": "No query provided"}
            )
        
        try:
            # First, analyze question intent and complexity
            intent_prompt = f"""Analyze this question: {query}

**CRITICAL: Question Intent Analysis**
Determine what the question is actually asking for:
1. Does it ask FOR an entity name? (e.g., "What company is X?", "What is the name of...", "What organization...")
2. Does it ask ABOUT an entity? (e.g., "Who is CEO of X?", "What is X's attribute?", "Who founded X?")
3. Is it a simple lookup (1 step) or multi-hop/comparative (2+ steps)?

**Examples:**
- "What company is YG Entertainment?" → Asks FOR entity name, simple (1 step)
- "Who is the CEO of YG Entertainment?" → Asks ABOUT entity (for person), simple (1 step)
- "Who founded the company that distributed film X?" → Multi-hop (2 steps: find company, then founder)
- "Are X and Y both from Z?" → Comparative (2 steps: one per entity)

Return a JSON object with these fields:
{{
    "main_question": "the original question",
    "disambiguated_query": "clarified version",
    "reasoning": "brief reasoning about how to answer",
    "query_type": "simple|multi-hop|comparative|analytical",
    "question_intent": "entity_name|entity_attribute|comparison|other",
    "estimated_steps_needed": 1-5,
    "is_simple_lookup": true/false
}}

Return ONLY the JSON, nothing else."""
            
            # Generate basic plan info with token tracking
            basic_response_text, basic_token_usage = await self.generate_text_with_usage(
                intent_prompt,
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
            except Exception as e:
                logger.error(f"Failed to parse basic plan: {e}")
                # Create fallback
                result = {
                    "main_question": query,
                    "disambiguated_query": query,
                    "reasoning": "Direct factual lookup",
                    "query_type": "simple",
                    "question_intent": "other",
                    "estimated_steps_needed": 1,
                    "is_simple_lookup": True
                }
            
            # Determine number of steps based on question analysis
            question_intent = result.get("question_intent", "other")
            estimated_steps = result.get("estimated_steps_needed", 1)
            is_simple = result.get("is_simple_lookup", True)
            
            # Adjust number of steps based on question complexity
            if is_simple and question_intent == "entity_name":
                # Simple entity lookup - only 1 step needed
                num_steps = 1
            elif is_simple:
                # Simple attribute lookup - 1-2 steps
                num_steps = min(2, max_steps)
            else:
                # Multi-hop or comparative - use estimated or default to 2-3
                num_steps = min(max(estimated_steps, 2), max_steps)
            
            logger.info(f"Question intent: {question_intent}, generating {num_steps} steps")
            
            # Now generate steps one by one
            steps = []
            
            for i in range(num_steps):
                step_num = i + 1
                previous_steps = "\n".join([f"Step {j+1}: {s['description']}" for j, s in enumerate(steps)])
                
                step_prompt = f"""Question: {query}
Question Intent: {question_intent}
Is Simple Lookup: {is_simple}
Previous steps:
{previous_steps if previous_steps else "None yet"}

**CRITICAL: Step descriptions must be retrieval-focused**
- Use action verbs: "Find", "Identify", "Search for", "Retrieve", "Locate", "Determine"
- DO NOT use synthesis language: "synthesize", "combine", "merge", "integrate", "assemble", "put together"
- Describe what information to FIND, not what to synthesize
- If step depends on previous steps, describe what to find using information from previous steps (e.g., "Find the founder of [company name from step 1]")

**CRITICAL: Question Intent Alignment**
- If question intent is "entity_name" (question asks FOR entity name), generate step to find the entity name ONLY
- DO NOT generate steps asking for CEO/founder/attributes if question just asks for entity name
- Example: Question "What company is YG Entertainment?" → Step: "Find information about YG Entertainment to identify what company it is" (NOT "Find CEO of YG Entertainment")
- If question intent is "entity_attribute" (question asks ABOUT entity), generate step to find that specific attribute
- Example: Question "Who is CEO of YG Entertainment?" → Step: "Find the CEO of YG Entertainment"

Generate step {step_num} to answer this question. Return a JSON object:
{{
    "id": "step_{step_num}",
    "description": "what to do in this step (retrieval-focused, not synthesis)",
    "objective": "goal of this step",
    "dependencies": {json.dumps([f"step_{j+1}" for j in range(i)]) if i > 0 else "[]"},
    "critical": true,
    "expected_output": "what this step should produce"
}}

Return ONLY the JSON."""
                
                step_response_text, step_token_usage = await self.generate_text_with_usage(
                    step_prompt,
                    temperature=0.2,
                    max_new_tokens=512
                )
                # Aggregate token usage
                total_token_usage["prompt_tokens"] += step_token_usage.get("prompt_tokens", 0)
                total_token_usage["generated_tokens"] += step_token_usage.get("generated_tokens", 0)
                total_token_usage["total_tokens"] += step_token_usage.get("total_tokens", 0)
                
                step_response = tokenization_utils.postprocess_answer(step_response_text, output_type="json")
                
                try:
                    clean_step = TokenizationUtils.repair_json(step_response)
                    step = json.loads(clean_step)
                    
                    # Validate step has required fields
                    if all(k in step for k in ["id", "description", "objective"]):
                        steps.append(step)
                        logger.info(f"Generated step {step_num}: {step['description'][:50]}")
                    else:
                        logger.warning(f"Step {step_num} missing required fields, skipping")
                except Exception as e:
                    logger.warning(f"Failed to parse step {step_num}: {e}")
                    continue
            
            # Add steps to result
            result["steps"] = steps
            logger.info(f"Generated plan with {len(steps)} steps")
            
            # Validate the response structure
            required_keys = ["main_question", "reasoning", "steps"]
            if not all(key in result for key in required_keys):
                raise ValueError(f"Missing required keys in response. Expected: {required_keys}")
            
            # Validate steps structure
            if not isinstance(result["steps"], list):
                raise ValueError("Steps must be a list")
            
            # Enhance each step (they're already validated in the loop above)
            for i, step in enumerate(result["steps"]):
                # Ensure dependencies is a list
                if "dependencies" not in step:
                    step["dependencies"] = []
                elif not isinstance(step["dependencies"], list):
                    step["dependencies"] = [step["dependencies"]]
                
                # Add critical flag if not present
                if "critical" not in step:
                    step["critical"] = True
                
                # Add expected output if not present
                if "expected_output" not in step:
                    step["expected_output"] = f"Information needed for: {step['objective']}"
            
            # Add disambiguated query if not present
            if "disambiguated_query" not in result:
                result["disambiguated_query"] = result["main_question"]
            
            # Add query type if not present
            if "query_type" not in result:
                query_lower = query.lower()
                if any(word in query_lower for word in ["compare", "difference", "vs", "versus"]):
                    result["query_type"] = "comparative"
                elif any(word in query_lower for word in ["how", "why", "what", "when", "where"]):
                    result["query_type"] = "multi-hop"
                elif any(word in query_lower for word in ["analyze", "analysis", "evaluate"]):
                    result["query_type"] = "analytical"
                else:
                    result["query_type"] = "simple"
            
            # Update history
            self._update_history("user", f"Plan for: {query}")
            self._update_history("assistant", json.dumps(result, indent=2))
            
            return AgentResponse(
                content=json.dumps(result, indent=2),
                metadata={
                    "main_question": result["main_question"],
                    "disambiguated_query": result["disambiguated_query"],
                    "query_type": result["query_type"],
                    "num_steps": len(result["steps"]),
                    "reasoning": result["reasoning"],
                    "steps": result["steps"],
                    "planning_parameters": {
                        "max_steps": max_steps,
                        "model": self.model_name,
                        "temperature": 0.2
                    },
                    "token_usage": total_token_usage
                }
            )
                
        except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing LLM response: {str(e)}")
                
                # Fallback: create a simple plan
                fallback_plan = {
                    "main_question": query,
                    "disambiguated_query": query,
                    "reasoning": "Unable to parse detailed plan, using fallback approach",
                    "query_type": "simple",
                    "steps": [
                        {
                            "id": "step_1",
                            "description": f"Research information about: {query}",
                            "objective": f"Gather relevant information to answer: {query}",
                            "dependencies": [],
                            "critical": True,
                            "expected_output": "Relevant facts and information"
                        }
                    ]
                }
                
                return AgentResponse(
                    content=json.dumps(fallback_plan, indent=2),
                    metadata={
                        "main_question": query,
                        "disambiguated_query": query,
                        "query_type": "simple",
                        "num_steps": 1,
                        "reasoning": "Fallback plan due to parsing error",
                        "steps": fallback_plan["steps"],
                        "error": f"Failed to parse detailed plan: {str(e)}",
                        "fallback": True
                    }
                )
                
        except Exception as e:
            logger.error(f"Error in PlannerAgent: {str(e)}", exc_info=True)
            return AgentResponse(
                content=f"Error processing query: {str(e)}",
                metadata={
                    "error": str(e),
                    "query": query,
                    "context": context
                }
            )

