from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import json
import logging
import asyncio
from datetime import datetime

# Import existing agents
from .planner_agent import PlannerAgent
from .step_definer_agent import StepDefinerAgent
from .retriever_agent import RetrieverAgent
from .extractor_agent import ExtractorAgent
from .qa_agent import QAAgent
from .state_manager import StateManager
from .final_assembler import FinalAssembler
from .tokenization_utils import TokenizationUtils
from .tokenization_utils import tokenization_utils
from .mcp_reasoning_state import mcp_state_manager
from .metadata_vector import metadata_vector_generator
from .mcp_reasoning_state import MCPReasoningStateManager

logger = logging.getLogger(__name__)

class PipelineResult(BaseModel):
    """Result of the complete MA-RAG pipeline execution."""
    main_query: str = Field(..., description="The original user query")
    disambiguated_query: str = Field(..., description="Disambiguated version of the query")
    query_type: str = Field(..., description="Type of query (simple, multi-hop, etc.)")
    execution_time: float = Field(..., description="Total execution time in seconds")
    steps_completed: int = Field(..., description="Number of steps completed")
    final_answer: str = Field(..., description="The final synthesized answer")
    confidence: float = Field(..., description="Overall confidence score")
    reasoning_trajectory: List[Dict[str, Any]] = Field(..., description="Step-by-step reasoning")
    sources: List[str] = Field(..., description="All source documents used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    token_usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "generated_tokens": 0, "total_tokens": 0}, description="Total token usage across all agents")

class MARAGOrchestrator:
    """
    Main orchestrator that manages the dynamic, modular agent invocation
    based on the structure of the reasoning plan. Implements the MA-RAG pipeline
    with seamless integration to existing agents.
    """
    
    def __init__(
        self,
        planner_agent: Optional[PlannerAgent] = None,
        step_definer_agent: Optional[StepDefinerAgent] = None,
        retriever_agent: Optional[RetrieverAgent] = None,
        extractor_agent: Optional[ExtractorAgent] = None,
        qa_agent: Optional[QAAgent] = None,
        state_manager: Optional[StateManager] = None,
        final_assembler: Optional[FinalAssembler] = None,
        mcp_state_manager: Optional[MCPReasoningStateManager] = None,
        max_concurrent_steps: int = 1,
        timeout_seconds: int = 300
    ):
        """
        Initialize the MA-RAG Orchestrator.
        
        Args:
            planner_agent: Planner agent instance
            step_definer_agent: Step definer agent instance
            retriever_agent: Retriever agent instance
            extractor_agent: Extractor agent instance
            qa_agent: QA agent instance
            state_manager: State manager instance
            final_assembler: Final assembler instance
            max_concurrent_steps: Maximum concurrent steps (default: 1 for sequential)
            timeout_seconds: Timeout for pipeline execution
        """
        # Initialize agents (create defaults if not provided)
        self.planner = planner_agent or PlannerAgent()
        self.step_definer = step_definer_agent or StepDefinerAgent()
        self.retriever = retriever_agent or RetrieverAgent()
        self.extractor = extractor_agent or ExtractorAgent()
        self.qa = qa_agent or QAAgent()
        
        # Initialize supporting components
        self.state_manager = state_manager or StateManager()
        self.final_assembler = final_assembler or FinalAssembler()
        
        # Pipeline configuration
        self.max_concurrent_steps = max_concurrent_steps
        self.timeout_seconds = timeout_seconds
        
        # Execution tracking
        self.current_execution_id: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.token_usage: Dict[str, int] = {"prompt_tokens": 0, "generated_tokens": 0, "total_tokens": 0}
        
        logger.info("MA-RAG Orchestrator initialized with all components")
    
    def _extract_and_aggregate_token_usage(self, agent_response: Any) -> None:
        """
        Extract token usage from agent response and aggregate it.
        
        Args:
            agent_response: AgentResponse object that may contain token usage in metadata
        """
        if hasattr(agent_response, 'metadata') and agent_response.metadata:
            token_usage = agent_response.metadata.get("token_usage", {})
            if token_usage:
                self.token_usage["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
                self.token_usage["generated_tokens"] += token_usage.get("generated_tokens", 0)
                self.token_usage["total_tokens"] += token_usage.get("total_tokens", 0)
    
    async def execute_pipeline(self, query: str, context: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """
        Execute the complete MA-RAG pipeline following the paper's methodology.
        
        Args:
            query: The user's question to answer
            context: Optional additional context
            
        Returns:
            PipelineResult containing the complete execution results
        """
        # Initialize execution tracking
        self.current_execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now()
        # Reset token usage for this execution
        self.token_usage = {"prompt_tokens": 0, "generated_tokens": 0, "total_tokens": 0}

        
        logger.info(f"Starting MA-RAG pipeline execution: {self.current_execution_id}")
        logger.info(f"Query: {query[:100]}...")

        
        # Normalize question for consistent processing
        question = tokenization_utils.normalize_query(query)
        
        try:
            # Step 1: Initialize state
            await self.state_manager.initialize_execution(
                execution_id=self.current_execution_id,
                main_query=query,
                context=context or {}
            )
            
            # Step 2: Planner Agent (once at beginning)
            plan_result = await self._execute_planner(query, context)
            if not plan_result:
                raise Exception("Failed to generate plan")
            
            # NEW: Step 2.5: Initialize MCP Reasoning State
            mcp_state = mcp_state_manager.create_state(
                execution_id=self.current_execution_id,
                main_question=query,
                disambiguated_query=plan_result.get("disambiguated_query", query),
                reasoning_intent=plan_result.get("reasoning", "Answer the user's question"),
                query_type=plan_result.get("query_type", "unknown"),
                original_plan=plan_result
            )
            logger.info("MCP reasoning state initialized")


            
            
            
            # Step 3: Execute each step in the plan
            step_results = await self._execute_plan_steps(plan_result)
            
            # Step 4: Assemble final answer
            final_result = await self._assemble_final_answer(plan_result, step_results)
            
            # Step 5: Create pipeline result
            execution_time = (datetime.now() - self.start_time).total_seconds()
            
            pipeline_result = PipelineResult(
                main_query=query,
                disambiguated_query=plan_result.get("disambiguated_query", query),
                query_type=plan_result.get("query_type", "unknown"),
                execution_time=execution_time,
                steps_completed=len(step_results),
                final_answer=final_result["final_answer"],
                confidence=final_result.get("confidence", 0.0),
                reasoning_trajectory=step_results,
                sources=final_result.get("sources", []),
                token_usage=self.token_usage.copy(),
                metadata={
                    "execution_id": self.current_execution_id,
                    "plan": plan_result,
                    "state_snapshots": await self.state_manager.get_execution_snapshots(),
                    "performance_metrics": {
                        "total_time": execution_time,
                        "avg_step_time": execution_time / len(step_results) if step_results else 0,
                        "steps_per_second": len(step_results) / execution_time if execution_time > 0 else 0
                    }
                }
            )
            
            logger.info(f"Pipeline execution completed successfully in {execution_time:.2f}s")
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise Exception(f"MA-RAG pipeline execution failed: {str(e)}")
        
        finally:
            # Cleanup
            await self.state_manager.cleanup_execution(self.current_execution_id)
            self.current_execution_id = None
            self.start_time = None
    
    async def _execute_planner(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute the Planner Agent to generate the reasoning plan.
        
        Args:
            query: The user query
            context: Optional context
            
        Returns:
            Plan result from planner
        """
        logger.info("Executing Planner Agent...")
        
        try:
            planner_input = {
                "query": query,
                "context": context or {},
                "max_steps": 5  # Default max steps
            }
            
            planner_response = await self.planner.process(planner_input)
            self._extract_and_aggregate_token_usage(planner_response)
            
            if planner_response.metadata.get("error"):
                raise Exception(f"Planner error: {planner_response.metadata.get('error')}")
            
            # Parse the plan from response
            clean_plan = TokenizationUtils.strip_markdown_json(planner_response.content)
            plan_data = json.loads(clean_plan)
            
            # Update state with plan
            await self.state_manager.update_plan(plan_data)
            
            logger.info(f"Plan generated with {len(plan_data.get('steps', []))} steps")
            return plan_data
            
        except Exception as e:
            logger.error(f"Planner execution failed: {str(e)}")
            raise Exception(f"Failed to generate plan: {str(e)}")
    
    

    async def _execute_plan_steps(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute all steps in the plan following MA-RAG methodology.
        Includes early stopping if answer is found.
        """
        steps = plan.get("steps", [])
        if not steps:
            logger.warning("No steps in plan")
            return []
        
        logger.info(f"Executing {len(steps)} plan steps...")
        
        # Resolve step dependencies
        ordered_steps = await self.state_manager.resolve_step_dependencies(steps)
        
        step_results = []
        main_question = plan.get("main_question", "")
        
        for i, step in enumerate(ordered_steps):
            try:
                logger.info(f"Executing step {i+1}/{len(ordered_steps)}: {step.get('id', 'unknown')}")
                
                # Execute single step following MA-RAG sequence
                step_result = await self._execute_single_step(step, plan)
                
                # Update state with step result
                await self.state_manager.add_step_result(step["id"], step_result)
                
                step_results.append({
                    "step_id": step["id"],
                    "step_description": step.get("description", ""),
                    "result": step_result,
                    "execution_order": i + 1,
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"Step {step['id']} completed successfully")
                
                # EARLY STOPPING: Check if answer is sufficient
                # Extract answer from qa_result
                qa_result = step_result.get("qa_result", {})
                answer = qa_result.get("answer", "")
                
                # Check if answer exists and semantically aligns with main question
                # (confidence is step-local for subquery, not global for main question, so we rely on semantic alignment)
                if answer:
                    # Semantic alignment check: Does this answer match what the main question asks for?
                    if self._is_answer_sufficient(answer, main_question, step_result):
                        logger.info(f"✅ Answer found in step {i+1}: '{answer}'")
                        logger.info(f"Stopping early - skipping remaining {len(ordered_steps) - i - 1} steps")
                        break  # Stop executing remaining steps
                
            except Exception as e:
                logger.error(f"Step {step.get('id', 'unknown')} failed: {str(e)}")
                
                # Add error result to maintain trajectory
                step_results.append({
                    "step_id": step["id"],
                    "step_description": step.get("description", ""),
                    "result": {"error": str(e), "success": False},
                    "execution_order": i + 1,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Continue with next step (don't fail entire pipeline)
                continue
        
        logger.info(f"Completed {len(step_results)} steps")
        return step_results

    def _is_answer_sufficient(self, answer: str, main_question: str, step_result: Dict) -> bool:
        """
        Check if the current answer is sufficient to answer the main question.
        Uses semantic alignment to verify answer type matches question type.
        
        Args:
            answer: The answer from current step
            main_question: The original question
            step_result: Full step result with context
            
        Returns:
            True if answer is sufficient, False otherwise
        """
        if not answer or answer.lower() in ["unknown", "not found", "no information"]:
            return False
        
        # Check if answer is a valid entity/name (not a placeholder)
        if len(answer) < 2:  # Too short to be meaningful
            return False
        
        question_lower = main_question.lower()
        answer_lower = answer.lower()
        
        # Classify question type based on semantic patterns
        question_type = self._classify_question_type(main_question)
        answer_type = self._classify_answer_type(answer, main_question)
        
        # Check semantic alignment: answer type must match question type
        if not self._check_semantic_alignment(question_type, answer_type, main_question, answer):
            return False
        
        # Additional validation: for entity selection questions, answer should be one of the mentioned entities
        if question_type == "entity_selection":
            # Check if answer matches one of the entities in the question
            if not self._answer_matches_question_entities(answer, main_question):
                return False
        
        return True
    
    def _classify_question_type(self, question: str) -> str:
        """Classify what type of answer the question is asking for."""
        question_lower = question.lower()
        
        # Entity name questions
        if any(phrase in question_lower for phrase in [
            "which", "who", "what is the name", "what administrative", 
            "what person", "what organization", "what place"
        ]):
            # Check if it's entity selection (which X or Y)
            if " or " in question_lower or " vs " in question_lower:
                return "entity_selection"
            return "entity_name"
        
        # Attribute questions
        if any(phrase in question_lower for phrase in [
            "what is the", "what was the", "how many", "how much",
            "what nationality", "what country", "what year", "when",
            "where was", "where did"
        ]):
            return "attribute"
        
        # Location questions
        if any(phrase in question_lower for phrase in [
            "where is", "where are", "where was", "where did", "located"
        ]):
            return "location"
        
        # Default: assume entity or attribute
        return "unknown"
    
    def _classify_answer_type(self, answer: str, question: str) -> str:
        """Classify the type of answer provided."""
        answer_lower = answer.lower()
        question_lower = question.lower()
        
        # Check if answer looks like an entity name (proper noun, capitalized)
        if answer[0].isupper() and len(answer.split()) <= 5:
            # Check if it's a location (common location indicators)
            if any(indicator in answer_lower for indicator in ["city", "state", "county", "country", "province"]):
                return "location"
            # Check if it's an attribute value (nationality, date, etc.)
            if any(attr in question_lower for attr in ["nationality", "country of origin", "born in"]):
                # If answer is a nationality/attribute, not an entity name
                if any(nationality in answer_lower for nationality in [
                    "american", "british", "english", "french", "german", "spanish",
                    "born", "native"
                ]):
                    return "attribute"
            return "entity_name"
        
        # Check if answer is a number/quantity
        if answer.replace(",", "").replace(".", "").isdigit():
            return "attribute"
        
        # Check if answer is a date/year
        if any(year in answer for year in ["19", "20"]) and len(answer) <= 10:
            return "attribute"
        
        # Check if answer is a location indicator
        if any(indicator in answer_lower for indicator in [
            "city", "state", "county", "country", "province", "region"
        ]):
            return "location"
        
        # Default: assume attribute or entity
        return "unknown"
    
    def _check_semantic_alignment(self, question_type: str, answer_type: str, question: str, answer: str) -> bool:
        """Check if answer type semantically aligns with question type."""
        # Direct matches
        if question_type == answer_type:
            return True
        
        # Compatible matches
        if question_type == "entity_name" and answer_type == "entity_name":
            return True
        
        if question_type == "entity_selection" and answer_type == "entity_name":
            return True
        
        if question_type == "location" and answer_type == "location":
            return True
        
        if question_type == "attribute" and answer_type == "attribute":
            return True
        
        # Mismatches: question asks for entity but got attribute (or vice versa)
        if question_type == "entity_name" and answer_type == "attribute":
            return False
        
        if question_type == "entity_selection" and answer_type == "attribute":
            return False
        
        if question_type == "attribute" and answer_type == "entity_name":
            # Sometimes entity names can be answers to attribute questions
            # But if question explicitly asks for attribute, entity is wrong
            question_lower = question.lower()
            if any(attr in question_lower for attr in ["nationality", "country of origin", "born"]):
                return False  # Question asks for nationality, got entity name
        
        # Unknown types: be conservative, allow it
        if question_type == "unknown" or answer_type == "unknown":
            return True
        
        return False
    
    def _answer_matches_question_entities(self, answer: str, question: str) -> bool:
        """Check if answer matches one of the entities mentioned in the question."""
        # Extract entities from question (capitalized words, proper nouns)
        import re
        # Find capitalized words/phrases (likely entities)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
        
        # Check if answer matches any entity (case-insensitive, partial match)
        answer_lower = answer.lower()
        for entity in entities:
            if entity.lower() in answer_lower or answer_lower in entity.lower():
                return True
        
        # Also check for "X or Y" pattern
        if " or " in question.lower():
            parts = question.lower().split(" or ")
            for part in parts:
                # Extract entity from part
                entity_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', part)
                if entity_match:
                    entity = entity_match.group(1)
                    if entity.lower() in answer_lower or answer_lower in entity.lower():
                        return True
        
        # If we can't find entities, be lenient (return True)
        return True
    
    async def _execute_single_step(self, step: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step following MA-RAG sequence:
        1. Step Definer → subqueries
        2. Retrieval Tool → documents
        3. Extractor Agent → evidence
        4. QA Agent → answer
        
        Args:
            step: The current step to execute
            plan: The overall plan
            
        Returns:
            Step execution result
        """
        try:
            # Get accumulated history for context
            history = await self.state_manager.get_accumulated_history()
            previous_answers = await self.state_manager.get_previous_answers()

            # New: Generate metadata vector from MCP state for this step
            mcp_state = mcp_state_manager.get_state(self.current_execution_id)
            if mcp_state:
                metadata_vector = metadata_vector_generator.generate_from_mcp_state(
                    mcp_state=mcp_state,
                    step=step
                )
                logger.debug(f"Step {step['id']}: Generated metadata vector - "
                           f"S={metadata_vector.structural:.2f}, "
                           f"E={metadata_vector.existential:.2f}, "
                           f"R={metadata_vector.relational:.2f}, "
                           f"Type={metadata_vector.relational_type}")
            else:
                # Fallback if MCP state not available
                logger.warning(f"No MCP state found, using default metadata vector")
                from .metadata_vector import MetadataVector
                metadata_vector = MetadataVector(
                    structural=0.5,
                    existential=0.6,
                    relational=0.5,
                    relational_type="factual",
                    query_type="unknown"
                )
            
            # 1. Step Definer Agent
            logger.debug(f"Step {step['id']}: Executing Step Definer...")
            step_definer_input = {
                "step": step,
                "plan": plan,
                "history": history,
                "previous_answers": previous_answers
            }
            
            step_definer_response = await self.step_definer.process(step_definer_input)
            self._extract_and_aggregate_token_usage(step_definer_response)
            if step_definer_response.metadata.get("error"):
                raise Exception(f"Step definer failed: {step_definer_response.metadata.get('error')}")
            
            clean_subqueries = TokenizationUtils.strip_markdown_json(step_definer_response.content)
            subqueries_data = json.loads(clean_subqueries)
            subqueries = subqueries_data.get("sub_queries", [])
            
            if not subqueries:
                # For simple steps, use the step description as a single query
                logger.warning(f"No subqueries generated for step {step['id']}, using step description as query")
                subqueries = [{
                    "id": "default_sq",
                    "query": step.get('description', step.get('objective', 'search for relevant information')),
                    "purpose": step.get('objective', 'accomplish step goal'),
                    "priority": 1
                }]
            
            # 2. Retrieval Tool (for each subquery)
            logger.debug(f"Step {step['id']}: Executing retrieval for {len(subqueries)} subqueries...")
            all_retrieved_docs = []
            seen_doc_ids = set()  # Track seen document IDs for deduplication
            
            for subquery in subqueries:

                
                # Don't use metadata filters as FAISS filters (documents don't have these fields)
                # Metadata filtering happens AFTER retrieval in retriever_agent
                
                retrieval_input = {
                    "query": subquery["query"],
                    "k": getattr(self.retriever, 'top_k', 10),
                    "min_similarity": 0.3,
                    "metadata_vector": metadata_vector.to_dict(),
                    "filter": {}  # ✅ Empty filter - let FAISS return all matches
                }
                
                retrieval_response = await self.retriever.process(retrieval_input)
                self._extract_and_aggregate_token_usage(retrieval_response)
                if retrieval_response.metadata.get("error"):
                    logger.warning(f"Retrieval failed for subquery: {subquery['query']}")
                    continue
                
                # Use metadata first (more efficient, already a list)
                docs_from_metadata = retrieval_response.metadata.get("documents", [])
                
                if docs_from_metadata:
                    # Deduplicate by document ID and add to collection
                    for doc in docs_from_metadata:
                        doc_id = doc.get("id") or doc.get("metadata", {}).get("id")
                        if doc_id and doc_id not in seen_doc_ids:
                            seen_doc_ids.add(doc_id)
                            all_retrieved_docs.append(doc)
                        elif not doc_id:
                            # If no ID, use content hash or add anyway
                            all_retrieved_docs.append(doc)
                else:
                    # Fallback to JSON parsing if metadata not available
                    try:
                        clean_retrieval = TokenizationUtils.strip_markdown_json(retrieval_response.content)
                        retrieved_docs = json.loads(clean_retrieval)
                        docs_from_json = retrieved_docs.get("documents", [])
                        for doc in docs_from_json:
                            doc_id = doc.get("id") or doc.get("metadata", {}).get("id")
                            if doc_id and doc_id not in seen_doc_ids:
                                seen_doc_ids.add(doc_id)
                                all_retrieved_docs.append(doc)
                            elif not doc_id:
                                all_retrieved_docs.append(doc)
                    except (json.JSONDecodeError, Exception) as e:
                        logger.warning(f"Failed to parse retriever response for subquery '{subquery['query']}': {e}")
                        continue
            
            if not all_retrieved_docs:
                raise Exception("No documents retrieved")
            
            # Sort documents by relevance score (highest first) if scores available
            all_retrieved_docs.sort(
                key=lambda x: x.get("score", 0.0) if isinstance(x.get("score"), (int, float)) else 0.0,
                reverse=True
            )
            
            # Limit documents before passing to extractor (reduce load)
            max_docs_for_extractor = min(len(all_retrieved_docs), 15)  # Cap at 15 documents
            limited_docs = all_retrieved_docs[:max_docs_for_extractor]
            
            logger.info(f"Step {step['id']}: Collected {len(all_retrieved_docs)} unique documents, "
                       f"limiting to {len(limited_docs)} for extractor")
            
            # 3. Extractor Agent
            logger.debug(f"Step {step['id']}: Executing extraction on {len(limited_docs)} documents...")
            
            # Extract subquery texts for the extractor
            subquery_texts = [sq["query"] for sq in subqueries]
            
            extractor_input = {
                "query": step["description"],  # Keep for context
                "subqueries": subquery_texts,  # CRITICAL: Pass the actual subqueries used for retrieval
                "documents": limited_docs,  # Use limited, sorted, deduplicated docs
                "history": history,
                "max_documents": max_docs_for_extractor,  # Pass the limit explicitly
                "min_relevance": 0.2  # Lower threshold for extraction
            }

            # DEBUG: Log extractor input details
            logger.debug(f"[EXTRACTOR DEBUG] Step {step['id']}: Extractor Input Details:")
            logger.debug(f"  - Query: {step['description'][:200]}...")  # Truncate if too long
            logger.debug(f"  - Subqueries: {subquery_texts}")  # ADD THIS LINE to verify subqueries
            logger.debug(f"  - Number of documents: {len(all_retrieved_docs)}")
            logger.debug(f"  - Max documents to process: {extractor_input['max_documents']}")
            logger.debug(f"  - Min relevance: {extractor_input['min_relevance']}")
            logger.debug(f"  - History length: {len(history)}")
            
            # Show sample document structure (first 2-3 documents)
            logger.debug(f"[EXTRACTOR DEBUG] Sample documents (first {min(3, len(all_retrieved_docs))}):")
            for i, doc in enumerate(all_retrieved_docs[:3]):
                doc_id = doc.get('id', 'NO_ID')
                page_content_preview = doc.get('page_content', 'NO_CONTENT')[:200]  # First 200 chars
                score = doc.get('score', 'NO_SCORE')
                logger.debug(f"  Document {i+1}:")
                logger.debug(f"    - ID: {doc_id}")
                logger.debug(f"    - Score: {score}")
                logger.debug(f"    - Content preview: {page_content_preview}...")
                logger.debug(f"    - Content length: {len(doc.get('page_content', ''))}")
                logger.debug(f"    - Metadata keys: {list(doc.get('metadata', {}).keys())}")
            
            # Check if documents have content
            docs_with_content = sum(1 for doc in all_retrieved_docs if doc.get('page_content', '').strip())
            docs_empty = len(all_retrieved_docs) - docs_with_content
            logger.debug(f"[EXTRACTOR DEBUG] Document content status:")
            logger.debug(f"  - Documents with content: {docs_with_content}")
            logger.debug(f"  - Empty documents: {docs_empty}")
            
            extractor_response = await self.extractor.process(extractor_input)
            self._extract_and_aggregate_token_usage(extractor_response)
            
            if extractor_response.metadata.get("error"):
                raise Exception(f"Extractor failed: {extractor_response.metadata.get('error')}")
            
            clean_extraction = TokenizationUtils.strip_markdown_json(extractor_response.content)
            extracted_data = json.loads(clean_extraction)
            extracted_passages = extracted_data.get("extracted_passages", [])
            
            if not extracted_passages:
                raise Exception("No passages extracted")
            

            # 4. QA Agent
            logger.debug(f"Step {step['id']}: Executing QA synthesis...")
            qa_input = {
                "question": step["description"],
                "context": extracted_passages,
                "step_context": step,
                "overall_query": plan.get("main_question", ""),
                "previous_answers": previous_answers,
                "metadata_vector": metadata_vector.to_dict()  # ADD THIS
            }
            
            qa_response = await self.qa.process(qa_input)
            self._extract_and_aggregate_token_usage(qa_response)
            if qa_response.metadata.get("error"):
                raise Exception(f"QA agent failed: {qa_response.metadata.get('error')}")
            
            clean_qa = TokenizationUtils.strip_markdown_json(qa_response.content)
            qa_result = json.loads(clean_qa)

            #Update MCP state after step completes
            step_result_dict = {
                "step_id": step["id"],
                "answer": qa_result.get("answer", ""),
                "sources": qa_result.get("sources", []),
                "confidence": qa_result.get("confidence", 0.0)
            }

            mcp_state_manager.update_state(
                step_id=step["id"],
                step_result=step_result_dict,
                execution_id=self.current_execution_id
            )
            
            # Return comprehensive step result
            return {
                "step_id": step["id"],
                "step_description": step.get("description", ""),
                "subqueries": subqueries,
                "retrieved_documents": all_retrieved_docs,
                "extracted_passages": extracted_passages,
                "qa_result": qa_result,
                "metadata_vector": metadata_vector.to_dict(),
                "success": True,
                "timestamp": datetime.now().isoformat()
                
            }
            
        except Exception as e:
            logger.error(f"Step execution failed: {str(e)}")
            return {
                "step_id": step["id"],
                "step_description": step.get("description", ""),
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _assemble_final_answer(self, plan: Dict[str, Any], step_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assemble the final answer from all step results.
        
        Args:
            plan: The original plan
            step_results: Results from all executed steps
            
        Returns:
            Final assembled answer
        """
        logger.info("Assembling final answer...")
        
        try:
            assembler_input = {
                "main_query": plan.get("main_question", ""),
                "disambiguated_query": plan.get("disambiguated_query", ""),
                "query_type": plan.get("query_type", "unknown"),
                "step_results": step_results,
                "plan": plan
            }
            
            final_result = await self.final_assembler.assemble_final_answer(assembler_input)
            
            logger.info("Final answer assembled successfully")
            return final_result
            
        except Exception as e:
            logger.error(f"Final assembly failed: {str(e)}")
            # Return fallback answer
            return {
                "final_answer": f"Error assembling final answer: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "error": str(e)
            }
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline execution status.
        
        Returns:
            Status information
        """
        return {
            "execution_id": self.current_execution_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "elapsed_time": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "state": await self.state_manager.get_current_state() if self.state_manager else {},
            "agents_initialized": {
                "planner": self.planner is not None,
                "step_definer": self.step_definer is not None,
                "retriever": self.retriever is not None,
                "extractor": self.extractor is not None,
                "qa": self.qa is not None
            }
        }
    
    async def cancel_execution(self) -> bool:
        """
        Cancel current pipeline execution.
        
        Returns:
            True if cancelled successfully
        """
        if not self.current_execution_id:
            return False
        
        try:
            logger.info(f"Cancelling execution: {self.current_execution_id}")
            await self.state_manager.cleanup_execution(self.current_execution_id)
            self.current_execution_id = None
            self.start_time = None
            return True
        except Exception as e:
            logger.error(f"Failed to cancel execution: {str(e)}")
            return False


# Convenience function for easy usage
async def run_marag_pipeline(
    query: str,
    retriever_agent: Optional[RetrieverAgent] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> PipelineResult:
    """
    Convenience function to run the complete MA-RAG pipeline.
    
    Args:
        query: The user's question
        retriever_agent: Optional retriever agent (must be initialized with documents)
        context: Optional additional context
        **kwargs: Additional arguments for orchestrator
        
    Returns:
        PipelineResult with complete execution results
    """
    orchestrator = MARAGOrchestrator(retriever_agent=retriever_agent, **kwargs)
    return await orchestrator.execute_pipeline(query, context)

