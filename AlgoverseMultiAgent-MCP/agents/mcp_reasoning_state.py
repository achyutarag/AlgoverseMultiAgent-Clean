"""
MCP Reasoning State

Maintains a consistent memory of what the system is trying to accomplish
across multiple reasoning hops. Prevents reasoning drift by storing the
original reasoning intent and updating it as steps progress.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MCPReasoningState(BaseModel):
    """
    Shared reasoning state that maintains consistency across hops.
    
    Stores the original question and reasoning intent, preventing
    the system from drifting away from the original goal.
    """
    
    # Core reasoning intent
    main_question: str = Field(..., description="The original user question")
    disambiguated_query: str = Field(..., description="Clarified version of the question")
    reasoning_intent: str = Field(..., description="What we're trying to accomplish (from planner)")
    query_type: str = Field(..., description="Type of query (simple, multi-hop, comparative, etc.)")
    
    # Execution tracking
    execution_id: str = Field(..., description="Unique execution identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="When state was created")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update time")
    
    # Step tracking
    completed_steps: List[str] = Field(default_factory=list, description="IDs of completed steps")
    current_step_id: Optional[str] = Field(None, description="Current step being executed")
    
    # Context preservation
    original_plan: Optional[Dict[str, Any]] = Field(None, description="Original plan from planner")
    accumulated_context: Dict[str, Any] = Field(default_factory=dict, description="Context accumulated across steps")
    
    def update_step(self, step_id: str, step_result: Dict[str, Any]) -> None:
        """
        Update state after a step completes.
        
        Args:
            step_id: ID of the completed step
            step_result: Result from the step execution
        """
        if step_id not in self.completed_steps:
            self.completed_steps.append(step_id)
        
        self.current_step_id = step_id
        self.last_updated = datetime.now()
        
        # Preserve key information from step result
        if "answer" in step_result:
            self.accumulated_context[f"step_{step_id}_answer"] = step_result["answer"]
        if "sources" in step_result:
            self.accumulated_context[f"step_{step_id}_sources"] = step_result["sources"]
        
        logger.debug(f"MCP state updated after step {step_id}")
    
    def get_reasoning_summary(self) -> str:
        """
        Get a summary of the reasoning intent for agents to reference.
        
        Returns:
            String summary of what we're trying to accomplish
        """
        summary = f"Main Question: {self.main_question}\n"
        summary += f"Reasoning Intent: {self.reasoning_intent}\n"
        summary += f"Query Type: {self.query_type}\n"
        
        if self.completed_steps:
            summary += f"Completed Steps: {', '.join(self.completed_steps)}\n"
        
        return summary
    
    def check_drift(self, current_step_description: str) -> bool:
        """
        Check if current step might be drifting from original intent.
        Simple heuristic: if step description doesn't mention key terms
        from main question, might be drifting.
        
        Args:
            current_step_description: Description of current step
            
        Returns:
            True if potential drift detected, False otherwise
        """
        # Simple keyword-based drift detection
        main_keywords = set(self.main_question.lower().split())
        step_keywords = set(current_step_description.lower().split())
        
        # Check if step has any connection to main question
        overlap = main_keywords.intersection(step_keywords)
        
        # If less than 20% keyword overlap, might be drifting
        if len(main_keywords) > 0:
            overlap_ratio = len(overlap) / len(main_keywords)
            if overlap_ratio < 0.2:
                logger.warning(f"Potential drift detected: step '{current_step_description}' has low overlap with main question")
                return True
        
        return False


class MCPReasoningStateManager:
    """
    Manager for MCP reasoning states across multiple executions.
    """
    
    def __init__(self):
        """Initialize the MCP state manager."""
        self.states: Dict[str, MCPReasoningState] = {}
        self.current_execution_id: Optional[str] = None
        logger.info("MCP Reasoning State Manager initialized")
    
    def create_state(
        self,
        execution_id: str,
        main_question: str,
        disambiguated_query: str,
        reasoning_intent: str,
        query_type: str,
        original_plan: Optional[Dict[str, Any]] = None
    ) -> MCPReasoningState:
        """
        Create a new MCP reasoning state.
        
        Args:
            execution_id: Unique execution identifier
            main_question: Original user question
            disambiguated_query: Clarified version
            reasoning_intent: What we're trying to accomplish
            query_type: Type of query
            original_plan: Original plan from planner
            
        Returns:
            Created MCPReasoningState
        """
        state = MCPReasoningState(
            execution_id=execution_id,
            main_question=main_question,
            disambiguated_query=disambiguated_query,
            reasoning_intent=reasoning_intent,
            query_type=query_type,
            original_plan=original_plan
        )
        
        self.states[execution_id] = state
        self.current_execution_id = execution_id
        
        logger.info(f"MCP reasoning state created for execution: {execution_id}")
        logger.debug(f"Reasoning intent: {reasoning_intent[:100]}...")
        
        return state
    
    def get_state(self, execution_id: Optional[str] = None) -> Optional[MCPReasoningState]:
        """
        Get the current or specified MCP reasoning state.
        
        Args:
            execution_id: Optional specific execution ID, defaults to current
            
        Returns:
            MCPReasoningState or None if not found
        """
        target_id = execution_id or self.current_execution_id
        if not target_id:
            return None
        
        return self.states.get(target_id)
    
    def update_state(self, step_id: str, step_result: Dict[str, Any], execution_id: Optional[str] = None) -> None:
        """
        Update MCP reasoning state after a step completes.
        
        Args:
            step_id: ID of completed step
            step_result: Result from step execution
            execution_id: Optional specific execution ID
        """
        state = self.get_state(execution_id)
        if not state:
            logger.warning(f"No MCP state found for execution {execution_id}")
            return
        
        state.update_step(step_id, step_result)
        logger.debug(f"MCP state updated for step {step_id}")
    
    def clear_state(self, execution_id: Optional[str] = None) -> None:
        """
        Clear MCP reasoning state.
        
        Args:
            execution_id: Optional specific execution ID, defaults to current
        """
        target_id = execution_id or self.current_execution_id
        if target_id and target_id in self.states:
            del self.states[target_id]
            if target_id == self.current_execution_id:
                self.current_execution_id = None
            logger.info(f"MCP state cleared for execution: {target_id}")


# Global instance for easy access
mcp_state_manager = MCPReasoningStateManager()