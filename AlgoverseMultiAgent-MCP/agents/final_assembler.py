from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import json
import logging
from datetime import datetime
from collections import defaultdict

from .tokenization_utils import tokenization_utils, TokenizationUtils

logger = logging.getLogger(__name__)

class FinalAnswer(BaseModel):
    """Structured final answer with comprehensive metadata."""
    main_query: str = Field(..., description="The original user query")
    disambiguated_query: str = Field(..., description="Disambiguated version of the query")
    final_answer: str = Field(..., description="The synthesized final answer")
    confidence: float = Field(..., description="Overall confidence score (0.0-1.0)")
    reasoning_summary: str = Field(..., description="Summary of the reasoning process")
    step_summaries: List[Dict[str, Any]] = Field(..., description="Summary of each step")
    all_sources: List[str] = Field(..., description="All source documents used")
    evidence_quality: Dict[str, Any] = Field(..., description="Quality metrics of evidence")
    execution_metadata: Dict[str, Any] = Field(..., description="Execution metadata")

class FinalAssembler:
    """
    Assembles the final answer from all step results, providing comprehensive
    synthesis and quality assessment following MA-RAG methodology.
    """
    
    def __init__(
        self,
        min_confidence_threshold: float = 0.3,
        max_answer_length: int = 2000,
        include_reasoning: bool = True,
        include_sources: bool = True
    ):
        """
        Initialize the Final Assembler.
        
        Args:
            min_confidence_threshold: Minimum confidence to include step results
            max_answer_length: Maximum length of final answer
            include_reasoning: Whether to include reasoning summary
            include_sources: Whether to include source information
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.max_answer_length = max_answer_length
        self.include_reasoning = include_reasoning
        self.include_sources = include_sources
        
        logger.info("Final Assembler initialized")
    
    async def assemble_final_answer(self, assembler_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assemble the final answer from all step results.
        
        Args:
            assembler_input: Dictionary containing:
                - 'main_query': The original user query
                - 'disambiguated_query': Disambiguated version
                - 'query_type': Type of query
                - 'step_results': List of step execution results
                - 'plan': The original plan
                
        Returns:
            Dictionary with final assembled answer and metadata
        """
        logger.info("Starting final answer assembly...")
        
        try:
            main_query = assembler_input.get("main_query", "")
            disambiguated_query = assembler_input.get("disambiguated_query", main_query)
            query_type = assembler_input.get("query_type", "unknown")
            step_results = assembler_input.get("step_results", [])
            plan = assembler_input.get("plan", {})
            
            # Process step results
            processed_steps = await self._process_step_results(step_results)
            
            # Generate reasoning summary
            reasoning_summary = await self._generate_reasoning_summary(
                main_query, processed_steps, query_type
            )
            
            # Synthesize final answer
            final_answer = await self._synthesize_final_answer(
                main_query, processed_steps, query_type, reasoning_summary
            )
            
            # Calculate overall confidence
            overall_confidence = await self._calculate_overall_confidence(processed_steps)
            
            # Collect all sources
            all_sources = await self._collect_all_sources(processed_steps)
            
            # Assess evidence quality
            evidence_quality = await self._assess_evidence_quality(processed_steps)
            
            # Create step summaries
            step_summaries = await self._create_step_summaries(processed_steps)
            
            # Create final answer object
            final_answer_obj = FinalAnswer(
                main_query=main_query,
                disambiguated_query=disambiguated_query,
                final_answer=final_answer,
                confidence=overall_confidence,
                reasoning_summary=reasoning_summary,
                step_summaries=step_summaries,
                all_sources=all_sources,
                evidence_quality=evidence_quality,
                execution_metadata={
                    "query_type": query_type,
                    "total_steps": len(step_results),
                    "successful_steps": len([s for s in processed_steps if s.get("success", False)]),
                    "assembly_timestamp": datetime.now().isoformat(),
                    "plan_summary": {
                        "steps": [s.get("step_id", "") for s in processed_steps],
                        "critical_steps": [s.get("step_id", "") for s in processed_steps if s.get("critical", False)]
                    }
                }
            )
            
            logger.info("Final answer assembly completed successfully")
            
            return {
                "final_answer": final_answer_obj.final_answer,
                "confidence": final_answer_obj.confidence,
                "reasoning_summary": final_answer_obj.reasoning_summary,
                "sources": final_answer_obj.all_sources,
                "step_summaries": final_answer_obj.step_summaries,
                "evidence_quality": final_answer_obj.evidence_quality,
                "metadata": final_answer_obj.execution_metadata,
                "structured_answer": final_answer_obj.dict()
            }
            
        except Exception as e:
            logger.error(f"Final answer assembly failed: {str(e)}", exc_info=True)
            
            # Return fallback answer
            return {
                "final_answer": f"I encountered an error while assembling the final answer: {str(e)}. Please try rephrasing your question or providing more specific information.",
                "confidence": 0.0,
                "reasoning_summary": "Assembly error occurred",
                "sources": [],
                "step_summaries": [],
                "evidence_quality": {"error": str(e)},
                "metadata": {"error": str(e), "assembly_failed": True}
            }
    
    async def _process_step_results(self, step_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and clean step results for assembly.
        
        Args:
            step_results: Raw step results from execution
            
        Returns:
            Processed step results
        """
        processed_steps = []
        
        for step in step_results:
            try:
                # Extract key information
                step_id = step.get("step_id", "unknown")
                step_description = step.get("step_description", "")
                result = step.get("result", {})
                
                # Check if step was successful
                success = result.get("success", False)
                
                # Extract QA result if available
                qa_result = result.get("qa_result", {})
                
                # Process the step
                processed_step = {
                    "step_id": step_id,
                    "step_description": step_description,
                    "success": success,
                    "answer": qa_result.get("answer", ""),
                    "confidence": qa_result.get("confidence", 0.0),
                    "sources": qa_result.get("sources", []),
                    "reasoning": qa_result.get("reasoning", ""),
                    "evidence": qa_result.get("supporting_evidence", []),
                    "error": result.get("error", "") if not success else "",
                    "execution_order": step.get("execution_order", 0),
                    "timestamp": step.get("timestamp", "")
                }
                
                # Only include steps above confidence threshold
                if success and processed_step["confidence"] >= self.min_confidence_threshold:
                    processed_steps.append(processed_step)
                elif not success:
                    # Include failed steps for transparency
                    processed_steps.append(processed_step)
                
            except Exception as e:
                logger.warning(f"Error processing step {step.get('step_id', 'unknown')}: {str(e)}")
                continue
        
        return processed_steps
    
    async def _generate_reasoning_summary(
        self, 
        main_query: str, 
        processed_steps: List[Dict[str, Any]], 
        query_type: str
    ) -> str:
        """
        Generate a summary of the reasoning process.
        
        Args:
            main_query: The original query
            processed_steps: Processed step results
            query_type: Type of query
            
        Returns:
            Reasoning summary
        """
        try:
            successful_steps = [s for s in processed_steps if s.get("success", False)]
            failed_steps = [s for s in processed_steps if not s.get("success", False)]
            
            summary_parts = []
            
            # Introduction
            if query_type == "multi-hop":
                summary_parts.append("This multi-hop question required breaking down into several sequential steps:")
            elif query_type == "comparative":
                summary_parts.append("This comparative question required analyzing multiple aspects:")
            elif query_type == "analytical":
                summary_parts.append("This analytical question required systematic investigation:")
            else:
                summary_parts.append("This question was addressed through the following steps:")
            
            # Successful steps
            if successful_steps:
                summary_parts.append(f"Successfully completed {len(successful_steps)} steps:")
                for i, step in enumerate(successful_steps, 1):
                    summary_parts.append(f"{i}. {step['step_description']} (confidence: {step['confidence']:.2f})")
            
            # Failed steps
            if failed_steps:
                summary_parts.append(f"Encountered issues with {len(failed_steps)} steps:")
                for step in failed_steps:
                    summary_parts.append(f"- {step['step_description']}: {step.get('error', 'Unknown error')}")
            
            # Overall assessment
            if successful_steps:
                avg_confidence = sum(s['confidence'] for s in successful_steps) / len(successful_steps)
                summary_parts.append(f"Overall confidence in the reasoning process: {avg_confidence:.2f}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating reasoning summary: {str(e)}")
            return f"Reasoning summary generation failed: {str(e)}"
    
    async def _synthesize_final_answer(
        self,
        main_query: str,
        processed_steps: List[Dict[str, Any]],
        query_type: str,
        reasoning_summary: str
    ) -> str:
        """
        Synthesize the final answer from all step results.
        
        Args:
            main_query: The original query
            processed_steps: Processed step results
            query_type: Type of query
            reasoning_summary: Reasoning summary
            
        Returns:
            Synthesized final answer
        """
        try:
            successful_steps = [s for s in processed_steps if s.get("success", False)]
            
            if not successful_steps:
                return "I was unable to find sufficient information to answer your question. The reasoning steps encountered errors or insufficient evidence."
            
            # Collect all answers
            step_answers = []
            for step in successful_steps:
                answer = step.get("answer", "").strip()
                if answer:
                    step_answers.append({
                        "step": step["step_description"],
                        "answer": answer,
                        "confidence": step["confidence"]
                    })
            
            # Synthesize based on query type
            # First, check if this is entity selection (regardless of query_type classification)
            query_lower = main_query.lower()
            # More generalizable entity selection detection
            # Catches: "Which X or Y?", "Who is older, X or Y?", "What company..., X or Y?"
            has_selection_word = any(word in query_lower for word in ["which", "who", "what"])
            has_options = " or " in query_lower or "either" in query_lower
            is_entity_selection = has_selection_word and has_options
            
            if is_entity_selection:
                # Entity selection question - extract just the entity name from final step
                # The final step should identify which entity matches the criteria
                final_answer = step_answers[-1]["answer"] if step_answers else "No information found."
                final_answer = self._extract_concise_answer(main_query, final_answer)
            elif query_type == "comparative":
                final_answer = await self._synthesize_comparative_answer(main_query, step_answers)
            elif query_type == "multi-hop":
                final_answer = await self._synthesize_multihop_answer(main_query, step_answers)
            elif query_type == "analytical":
                final_answer = await self._synthesize_analytical_answer(main_query, step_answers)
            else:
                final_answer = await self._synthesize_simple_answer(main_query, step_answers)
            
            # Post-process the answer
            final_answer = tokenization_utils.postprocess_answer(final_answer, output_type="text")
            
            # Truncate if too long
            if len(final_answer) > self.max_answer_length:
                final_answer = final_answer[:self.max_answer_length] + "..."
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Error synthesizing final answer: {str(e)}")
            return f"Error synthesizing final answer: {str(e)}"
    
    async def _synthesize_comparative_answer(self, query: str, step_answers: List[Dict[str, Any]]) -> str:
        """Synthesize answer for comparative questions."""
        if len(step_answers) < 2:
            return step_answers[0]["answer"] if step_answers else "Insufficient information for comparison."
        
        # Combine answers with comparison structure
        synthesis = f"Based on the analysis, here's a comparison:\n\n"
        
        for i, step_answer in enumerate(step_answers, 1):
            synthesis += f"{i}. {step_answer['step']}: {step_answer['answer']}\n\n"
        
        synthesis += "This comparison provides a comprehensive view of the differences and similarities."
        return synthesis
    
    async def _synthesize_multihop_answer(self, query: str, step_answers: List[Dict[str, Any]]) -> str:
        """Synthesize answer for multi-hop questions."""
        if not step_answers:
            return "No information found to answer your question."
        
        # For multi-hop, use the final step answer and extract concise version
        final_answer = step_answers[-1]["answer"]
        final_answer = self._extract_concise_answer(query, final_answer)
        
        return final_answer
    
    async def _synthesize_analytical_answer(self, query: str, step_answers: List[Dict[str, Any]]) -> str:
        """Synthesize answer for analytical questions."""
        synthesis = f"Analysis results:\n\n"
        
        for step_answer in step_answers:
            synthesis += f"• {step_answer['answer']}\n"
        
        synthesis += f"\nThis analytical approach provides a systematic investigation of your question."
        return synthesis
    
    async def _synthesize_simple_answer(self, query: str, step_answers: List[Dict[str, Any]]) -> str:
        """Synthesize answer for simple questions - produce concise answers."""
        if not step_answers:
            return "No information found to answer your question."
        
        query_lower = query.lower()
        # More precise yes/no detection: check if question STARTS with yes/no words or is structured as yes/no
        # Not just if it contains these words (which would catch "What did X do?" as yes/no incorrectly)
        yes_no_starters = ["is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "would ", "should ", "has ", "have ", "had "]
        starts_with_yes_no = any(query_lower.startswith(starter) for starter in yes_no_starters)
        contains_yes_no_pattern = any(keyword in query_lower for keyword in [
            "yes or no", "is it", "are they", "do they", 
            "same", "different", "compare", "both", "either", "neither"
        ])
        is_yes_no_question = starts_with_yes_no or contains_yes_no_pattern
        
        # Check if original question asks for entity name (even if subqueries were yes/no)
        # This handles cases where original question asks "Who is X?" but subquery was yes/no
        is_entity_name_question = any(keyword in query_lower for keyword in [
            "who is", "who was", "who are", "what is the name", "what is called", 
            "what person", "what individual", "identify the person", "identify who",
            "what is the identity", "what is the real name"
        ])
        
        # Check if this is a "both" question requiring multi-entity reasoning
        is_both_question = any(keyword in query_lower for keyword in ["both", "and"]) and is_yes_no_question
        
        if is_both_question and len(step_answers) >= 2:
            # This is a "both" question - need to reason about multiple entities
            # Extract the attribute/criteria from the question
            # Example: "Are X and Y both from Z?" → check if both answers match "Z"
            # Example: "Are X and Y both Z?" → check if both answers indicate "Z"
            
            # Try to extract the criteria/attribute from the question
            import re
            
            # Pattern 1: "Are X and Y both [attribute]?" or "Are X and Y both [criteria]?"
            both_match = re.search(r'both\s+([^?]+)', query_lower)
            if both_match:
                criteria = both_match.group(1).strip()
            else:
                # Pattern 2: "Are X and Y [attribute]?" (without "both" but with "and")
                # Extract attribute after the entities
                # This is more complex, so we'll use a simpler approach
                criteria = None
            
            # Get answers from steps (should be factual answers, not yes/no)
            step1_answer = step_answers[0].get("answer", "").lower().strip()
            step2_answer = step_answers[1].get("answer", "").lower().strip() if len(step_answers) > 1 else ""
            
            # Normalize answers for comparison with semantic equivalence
            def normalize_for_comparison(text):
                """Normalize text for comparison by removing common variations and handling semantic equivalence."""
                if not text:
                    return ""
                text = text.lower().strip()
                # Remove common prefixes/suffixes
                text = re.sub(r'^(a|an|the)\s+', '', text)
                text = text.strip()
                
                # Handle semantic equivalence (synonyms and variations)
                # Nationality/Country equivalence
                nationality_synonyms = {
                    r'\bunited states\b': 'united states',
                    r'\busa\b': 'united states',
                    r'\bus\b': 'united states',
                    r'\bamerican\b': 'united states',
                    r'\bunited kingdom\b': 'united kingdom',
                    r'\buk\b': 'united kingdom',
                    r'\bbritish\b': 'united kingdom',
                    r'\bengland\b': 'united kingdom',  # Common but not always accurate
                    r'\bcanada\b': 'canada',
                    r'\bcanadian\b': 'canada',
                    r'\baustralia\b': 'australia',
                    r'\baustralian\b': 'australia',
                    r'\bfrance\b': 'france',
                    r'\bfrench\b': 'france',
                    r'\bgermany\b': 'germany',
                    r'\bgerman\b': 'germany',
                    r'\bspain\b': 'spain',
                    r'\bspanish\b': 'spain',
                    r'\bitaly\b': 'italy',
                    r'\bitalian\b': 'italy',
                    r'\bjapan\b': 'japan',
                    r'\bjapanese\b': 'japan',
                    r'\bchina\b': 'china',
                    r'\bchinese\b': 'china',
                    r'\bindia\b': 'india',
                    r'\bindian\b': 'india',
                    r'\bmexico\b': 'mexico',
                    r'\bmexican\b': 'mexico',
                    r'\bbrazil\b': 'brazil',
                    r'\bbrazilian\b': 'brazil',
                }
                
                # Apply nationality/country normalization
                for pattern, normalized in nationality_synonyms.items():
                    if re.search(pattern, text):
                        return normalized
                
                # Handle location variations (e.g., "New York City" vs "New York")
                # Remove common location suffixes that don't change meaning
                text = re.sub(r'\s+city\s*$', '', text)
                text = re.sub(r'\s+state\s*$', '', text)
                text = re.sub(r'\s+country\s*$', '', text)
                
                return text.strip()
            
            def are_semantically_equivalent(text1, text2):
                """Check if two texts are semantically equivalent."""
                norm1 = normalize_for_comparison(text1)
                norm2 = normalize_for_comparison(text2)
                
                # Exact match
                if norm1 == norm2:
                    return True
                
                # One contains the other (handles "Greenwich Village, New York City" vs "New York City")
                if norm1 in norm2 or norm2 in norm1:
                    return True
                
                # Check for common semantic relationships
                # Both refer to same country/nationality
                if (norm1 in ['united states', 'usa', 'us', 'american'] and 
                    norm2 in ['united states', 'usa', 'us', 'american']):
                    return True
                if (norm1 in ['united kingdom', 'uk', 'british', 'england'] and 
                    norm2 in ['united kingdom', 'uk', 'british', 'england']):
                    return True
                
                return False
            
            step1_normalized = normalize_for_comparison(step1_answer)
            step2_normalized = normalize_for_comparison(step2_answer)
            
            # If criteria is specified, check if both answers match it
            if criteria:
                criteria_normalized = normalize_for_comparison(criteria)
                # Check if both answers semantically match the criteria
                step1_matches = are_semantically_equivalent(step1_answer, criteria)
                step2_matches = are_semantically_equivalent(step2_answer, criteria)
                
                if step1_matches and step2_matches:
                    return "Yes"
                else:
                    return "No"
            else:
                # No explicit criteria - check if both answers are semantically equivalent
                # This handles questions like "Are X and Y both from the same place?"
                if step1_normalized and step2_normalized:
                    # Use semantic equivalence check
                    if are_semantically_equivalent(step1_answer, step2_answer):
                        return "Yes"
                    else:
                        return "No"
                else:
                    # Fallback: if we can't determine, check if last step has yes/no
                    best_answer = await self._select_best_answer_for_question(query, step_answers)
                    answer_lower = best_answer["answer"].lower()
                    yes_no_match = re.search(r'\b(yes|no)\b', answer_lower)
                    if yes_no_match:
                        return yes_no_match.group(1).capitalize()
        
        # For single answer, use it directly
        if len(step_answers) == 1:
            answer = step_answers[0]["answer"]
            
            # If original question asks for entity name but answer is yes/no, extract from evidence
            if is_entity_name_question and answer.lower().strip() in ["yes", "no"]:
                # Try to extract entity name from evidence or reasoning
                import re
                evidence_text = ""
                if step_answers[0].get("evidence"):
                    evidence_list = step_answers[0]["evidence"]
                    if isinstance(evidence_list, list):
                        evidence_text = " ".join([str(e) for e in evidence_list if e])
                    else:
                        evidence_text = str(evidence_list)
                if step_answers[0].get("reasoning"):
                    evidence_text += " " + str(step_answers[0].get("reasoning", ""))
                
                if evidence_text:
                    extracted = self._extract_concise_answer(query, evidence_text)
                    if extracted and extracted.lower().strip() not in ["yes", "no"]:
                        return extracted
            
            if is_yes_no_question:
                # Extract yes/no if present
                import re
                answer_lower = answer.lower()
                yes_no_match = re.search(r'\b(yes|no)\s*\.?\s*$', answer_lower)
                if yes_no_match:
                    return yes_no_match.group(1).capitalize()
                yes_no_match = re.search(r'\b(yes|no)\b', answer_lower)
                if yes_no_match:
                    return yes_no_match.group(1).capitalize()
            answer = self._extract_concise_answer(query, answer)
            return answer
        
        # For multiple answers (non-both questions), intelligently select the one that best matches the question
        best_answer = await self._select_best_answer_for_question(query, step_answers)
        
        # If original question asks for entity name but best answer is yes/no, extract from evidence
        if is_entity_name_question and best_answer.get("answer", "").lower().strip() in ["yes", "no"]:
            # Try to extract entity name from evidence or reasoning of best answer or all steps
            import re
            evidence_text = ""
            if best_answer.get("evidence"):
                evidence_list = best_answer["evidence"]
                if isinstance(evidence_list, list):
                    evidence_text = " ".join([str(e) for e in evidence_list if e])
                else:
                    evidence_text = str(evidence_list)
            if best_answer.get("reasoning"):
                evidence_text += " " + str(best_answer.get("reasoning", ""))
            
            # If no evidence in best answer, try all step answers
            if not evidence_text:
                for step_answer in step_answers:
                    if step_answer.get("evidence"):
                        evidence_list = step_answer["evidence"]
                        if isinstance(evidence_list, list):
                            evidence_text += " ".join([str(e) for e in evidence_list if e])
                        else:
                            evidence_text += " " + str(evidence_list)
                    if step_answer.get("reasoning"):
                        evidence_text += " " + str(step_answer.get("reasoning", ""))
            
            if evidence_text:
                extracted = self._extract_concise_answer(query, evidence_text)
                if extracted and extracted.lower().strip() not in ["yes", "no"]:
                    return extracted
        
        # For yes/no questions, extract just yes/no
        if is_yes_no_question:
            import re
            answer_lower = best_answer["answer"].lower()
            yes_no_match = re.search(r'\b(yes|no)\s*\.?\s*$', answer_lower)
            if yes_no_match:
                return yes_no_match.group(1).capitalize()
            yes_no_match = re.search(r'\b(yes|no)\b', answer_lower)
            if yes_no_match:
                return yes_no_match.group(1).capitalize()
        
        # Extract concise answer based on question type
        answer = self._extract_concise_answer(query, best_answer["answer"])
        
        # Minimal safety net: if still too long, use first sentence
        if len(answer) > 200:
            sentences = answer.split('.')
            if sentences and len(sentences[0]) < 150:
                return sentences[0].strip() + "."
        
        return answer
    
    async def _select_best_answer_for_question(self, query: str, step_answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select which step's answer best matches the original question.
        Original approach: use the last step answer (was getting 0.77 F1).
        """
        # Simple: use last step answer (original behavior)
        return step_answers[-1]
    
    def _classify_question_type(self, query: str) -> str:
        """Classify what type of answer the question is asking for."""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ["how many", "how much", "what number", "population", "capacity", "inhabitants"]):
            return "number"
        elif any(keyword in query_lower for keyword in ["who", "what is the name", "formed by", "created by"]):
            return "entity"
        elif any(keyword in query_lower for keyword in ["in what", "located in", "based in", "where"]):
            return "location"
        elif any(keyword in query_lower for keyword in ["what position", "what role", "what title"]):
            return "position"
        elif any(keyword in query_lower for keyword in ["during what years", "what timeframe", "what year"]):
            return "time"
        elif any(keyword in query_lower for keyword in ["is", "are", "was", "were", "do", "does", "did"]):
            return "yes_no"
        else:
            return "general"
    
    def _classify_answer_type(self, answer: str, query: str) -> str:
        """Classify the type of answer provided."""
        answer_lower = answer.lower()
        
        # Check for numbers
        import re
        if re.search(r'\d+(?:,\d+)*(?:\.\d+)?', answer):
            return "number"
        
        # Check for yes/no
        if re.search(r'\b(yes|no)\b', answer_lower):
            return "yes_no"
        
        # Check for location patterns
        if re.search(r'[A-Z][^,\.]+,\s*[A-Z][^,\.]+', answer):
            return "location"
        
        # Check for time patterns
        if re.search(r'(from\s+\d+|during\s+\d+|\d+\s+to\s+\d+)', answer_lower):
            return "time"
        
        # Default to general
        return "general"
    
    def _extract_concise_answer(self, query: str, answer: str) -> str:
        """
        Extract concise answer from potentially verbose QA response.
        
        Args:
            query: The original question
            answer: The answer from QA agent (might be a sentence)
            
        Returns:
            Concise answer (entity name, number, yes/no, etc.)
        """
        import re
        
        query_lower = query.lower()
        answer = answer.strip()
        answer_lower = answer.lower()
        
        # Yes/No questions - already handled above, but add as fallback
        if any(keyword in query_lower for keyword in ["yes or no", "is it", "are they", "do they", "does", "did", "was", "were"]):
            yes_no_match = re.search(r'\b(yes|no)\b', answer_lower)
            if yes_no_match:
                return yes_no_match.group(1).capitalize()
        
        # Position/Title questions (What position, what role, what title) - handle multi-word titles with connecting words
        if any(keyword in query_lower for keyword in [
            "what position", "what role", "what title", "held what position", 
            "served as", "position did", "role did", "title did", "government position"
        ]):
            # Pattern for multi-word titles with connecting words (e.g., "Chief of Protocol", "Secretary of State")
            # Allow lowercase connecting words: of, the, and, for, in, at, to, from, etc.
            title_match = re.search(r'\b([A-Z][a-z]+(?:\s+(?:of|the|and|for|in|at|to|from)\s+[A-Z][a-z]+)+)\b', answer)
            if title_match:
                title = title_match.group(1)
                # Limit to reasonable length (e.g., max 6 words for titles)
                if len(title.split()) <= 6:
                    return title
            
            # Fallback: extract first capitalized phrase (handles simple titles without connecting words)
            simple_title = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', answer)
            if simple_title:
                title = simple_title.group(1)
                if len(title.split()) <= 5:
                    return title
        
        # Entity name questions (Who, What is the name of, formed by, created by, etc.)
        if any(keyword in query_lower for keyword in [
            "who", "what is the name", "what is called", "formed by", "created by", 
            "directed by", "written by", "produced by", "performed by"
        ]):
            # Try to extract proper noun (capitalized words)
            # Look for patterns like "X is Y" or "Y, which is X" or just "Y"
            # Extract the most likely entity name
            
            # Pattern 1: "X is based in Y" -> extract Y
            if " is based in " in answer_lower or " based in " in answer_lower:
                match = re.search(r'based in\s+([A-Z][^,\.]+(?:,\s*[A-Z][^,\.]+)?)', answer)
                if match:
                    return match.group(1).strip()
            
            # Pattern 2: "X, which was formed by Y" -> extract Y
            if " formed by " in answer_lower or "which was formed by" in answer_lower:
                match = re.search(r'formed by\s+([A-Z][^,\.]+(?:,\s*[A-Z][^,\.]+)?)', answer)
                if match:
                    return match.group(1).strip()
            
            # Pattern 3: "X managed Y from Z to W" -> extract "from Z to W"
            if " managed " in answer_lower and ("from" in answer_lower or "during" in answer_lower):
                match = re.search(r'(from\s+\d+\s+(?:until|to)\s+\d+)', answer_lower)
                if match:
                    return match.group(1)
                match = re.search(r'(\d+\s+until\s+\d+)', answer_lower)
                if match:
                    return match.group(1)
            
            # Pattern 4: Extract first proper noun phrase (capitalized words)
            # Look for sequences of capitalized words
            entity_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)*)\b', answer)
            if entity_match:
                entity = entity_match.group(1)
                # Don't return if it's too long (likely a sentence start)
                if len(entity.split()) <= 5:
                    return entity
        
        # Location questions (in what city, located in, based in what)
        if any(keyword in query_lower for keyword in ["in what", "located in what", "based in what", "in what city"]):
            # Extract location pattern: "City, State" or "Neighborhood, City"
            location_match = re.search(r'([A-Z][^,\.]+,\s*[A-Z][^,\.]+)', answer)
            if location_match:
                return location_match.group(1).strip()
            # Fallback: extract after "in" or "based in"
            if " based in " in answer_lower:
                match = re.search(r'based in\s+([A-Z][^,\.]+(?:,\s*[A-Z][^,\.]+)?)', answer)
                if match:
                    return match.group(1).strip()
        
        # Number questions (how many, how much, what capacity, population)
        if any(keyword in query_lower for keyword in [
            "how many", "how much", "what number", "what capacity", "population"
        ]):
            # Extract number with optional unit
            number_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*([a-z]+)?', answer_lower)
            if number_match:
                num = number_match.group(1)
                unit = number_match.group(2) if number_match.group(2) else ""
                result = f"{num} {unit}".strip()
                # Also check for "inhabitants", "seated", etc. that might be elsewhere
                if "inhabitants" in answer_lower and "inhabitants" not in result:
                    result = f"{num} inhabitants"
                elif "seated" in answer_lower and "seated" not in result:
                    result = f"{num} seated"
                return result
        
        # Time period questions (during what years, what timeframe, served during)
        if any(keyword in query_lower for keyword in ["during what years", "what timeframe", "served during", "during what timeframe"]):
            # Extract time period with connector
            time_match = re.search(r'(from\s+\d+\s+(?:until|to)\s+\d+)', answer_lower)
            if time_match:
                return time_match.group(1)
            time_match = re.search(r'(\d+\s+until\s+\d+)', answer_lower)
            if time_match:
                return time_match.group(1)
            time_match = re.search(r'(\d+\s+to\s+\d+)', answer_lower)
            if time_match:
                return time_match.group(1)
        
        # Default: return first sentence if answer is too long, otherwise return as-is
        if len(answer) > 100:
            first_sentence = answer.split('.')[0].strip()
            if len(first_sentence) < 150 and len(first_sentence) > 0:
                return first_sentence
            # If first sentence is still long, try to extract key phrase
            return answer[:80].strip()
        
        return answer
    
    async def _calculate_overall_confidence(self, processed_steps: List[Dict[str, Any]]) -> float:
        """
        Calculate overall confidence score.
        
        Args:
            processed_steps: Processed step results
            
        Returns:
            Overall confidence score
        """
        successful_steps = [s for s in processed_steps if s.get("success", False)]
        
        if not successful_steps:
            return 0.0
        
        # Weight by step importance and confidence
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        for step in successful_steps:
            confidence = step.get("confidence", 0.0)
            # Weight by step order (later steps might be more important)
            weight = 1.0 + (step.get("execution_order", 0) * 0.1)
            
            total_weighted_confidence += confidence * weight
            total_weight += weight
        
        return total_weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    async def _collect_all_sources(self, processed_steps: List[Dict[str, Any]]) -> List[str]:
        """
        Collect all unique sources from all steps.
        
        Args:
            processed_steps: Processed step results
            
        Returns:
            List of unique source IDs
        """
        all_sources = set()
        
        for step in processed_steps:
            sources = step.get("sources", [])
            if isinstance(sources, list):
                all_sources.update(sources)
            elif isinstance(sources, str):
                all_sources.add(sources)
        
        return list(all_sources)
    
    async def _assess_evidence_quality(self, processed_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess the quality of evidence across all steps.
        
        Args:
            processed_steps: Processed step results
            
        Returns:
            Evidence quality metrics
        """
        successful_steps = [s for s in processed_steps if s.get("success", False)]
        
        if not successful_steps:
            return {"error": "no_successful_steps"}
        
        # Calculate metrics
        total_confidence = sum(s.get("confidence", 0.0) for s in successful_steps)
        avg_confidence = total_confidence / len(successful_steps)
        
        total_sources = sum(len(s.get("sources", [])) for s in successful_steps)
        avg_sources_per_step = total_sources / len(successful_steps)
        
        # Assess evidence diversity
        all_sources = await self._collect_all_sources(processed_steps)
        source_diversity = len(all_sources) / total_sources if total_sources > 0 else 0
        
        return {
            "avg_confidence": avg_confidence,
            "total_sources": len(all_sources),
            "avg_sources_per_step": avg_sources_per_step,
            "source_diversity": source_diversity,
            "successful_steps": len(successful_steps),
            "total_steps": len(processed_steps),
            "success_rate": len(successful_steps) / len(processed_steps)
        }
    
    async def _create_step_summaries(self, processed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create summaries for each step.
        
        Args:
            processed_steps: Processed step results
            
        Returns:
            List of step summaries
        """
        summaries = []
        
        for step in processed_steps:
            summary = {
                "step_id": step.get("step_id", ""),
                "description": step.get("step_description", ""),
                "success": step.get("success", False),
                "confidence": step.get("confidence", 0.0),
                "sources_count": len(step.get("sources", [])),
                "answer_preview": step.get("answer", "")[:100] + "..." if len(step.get("answer", "")) > 100 else step.get("answer", ""),
                "execution_order": step.get("execution_order", 0)
            }
            
            if not step.get("success", False):
                summary["error"] = step.get("error", "Unknown error")
            
            summaries.append(summary)
        
        return summaries


# Global final assembler instance for easy access
final_assembler = FinalAssembler()


