from typing import Dict, Any, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
import json
import logging
from datetime import datetime
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ExecutionState(BaseModel):
    """State of a single pipeline execution."""
    execution_id: str = Field(..., description="Unique execution identifier")
    main_query: str = Field(..., description="Original user query")
    disambiguated_query: str = Field("", description="Disambiguated version of query")
    query_type: str = Field("unknown", description="Type of query")
    plan: Optional[Dict[str, Any]] = Field(None, description="Generated plan")
    completed_steps: List[str] = Field(default_factory=list, description="IDs of completed steps")
    step_results: Dict[str, Any] = Field(default_factory=dict, description="Results from each step")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Accumulated history Hi")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    start_time: Optional[datetime] = Field(None, description="Execution start time")
    last_update: Optional[datetime] = Field(None, description="Last state update time")

class StateManager:
    """
    Manages the evolving context and state throughout the reasoning trajectory.
    Handles step dependencies and maintains history Hi = {(s1, a1), ..., (si, ai)}.
    """
    
    def __init__(self, max_history_size: int = 100):
        """
        Initialize the State Manager.
        
        Args:
            max_history_size: Maximum number of history items to keep
        """
        self.max_history_size = max_history_size
        self.executions: Dict[str, ExecutionState] = {}
        self.current_execution_id: Optional[str] = None
        
        logger.info("State Manager initialized")
    
    async def initialize_execution(
        self, 
        execution_id: str, 
        main_query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize a new pipeline execution.
        
        Args:
            execution_id: Unique identifier for this execution
            main_query: The original user query
            context: Optional additional context
        """
        logger.info(f"Initializing execution state: {execution_id}")
        
        self.current_execution_id = execution_id
        
        # Create new execution state
        execution_state = ExecutionState(
            execution_id=execution_id,
            main_query=main_query,
            context=context or {},
            start_time=datetime.now(),
            last_update=datetime.now()
        )
        
        self.executions[execution_id] = execution_state
        
        logger.info(f"Execution state initialized for query: {main_query[:100]}...")
    
    async def update_plan(self, plan: Dict[str, Any]) -> None:
        """
        Update the execution state with the generated plan.
        
        Args:
            plan: The generated plan from the planner agent
        """
        if not self.current_execution_id:
            raise Exception("No active execution to update")
        
        execution_state = self.executions[self.current_execution_id]
        execution_state.plan = plan
        execution_state.disambiguated_query = plan.get("disambiguated_query", execution_state.main_query)
        execution_state.query_type = plan.get("query_type", "unknown")
        execution_state.last_update = datetime.now()
        
        logger.info(f"Plan updated with {len(plan.get('steps', []))} steps")
    
    async def resolve_step_dependencies(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve step dependencies and return executable order.
        Uses topological sort to ensure steps are executed when dependencies are met.
        
        Args:
            steps: List of step dictionaries with dependencies
            
        Returns:
            Ordered list of steps ready for execution
        """
        if not steps:
            return []
        
        logger.info(f"Resolving dependencies for {len(steps)} steps")
        
        # Build dependency graph
        step_map = {step["id"]: step for step in steps}
        dependencies = {step["id"]: set(step.get("dependencies", [])) for step in steps}
        in_degree = {step["id"]: len(step.get("dependencies", [])) for step in steps}
        
        # Topological sort using Kahn's algorithm
        queue = deque([step_id for step_id, degree in in_degree.items() if degree == 0])
        ordered_steps = []
        
        while queue:
            current_step_id = queue.popleft()
            ordered_steps.append(step_map[current_step_id])
            
            # Update in-degrees of dependent steps
            for step_id, deps in dependencies.items():
                if current_step_id in deps:
                    in_degree[step_id] -= 1
                    if in_degree[step_id] == 0:
                        queue.append(step_id)
        
        # Check for circular dependencies
        if len(ordered_steps) != len(steps):
            remaining_steps = [step for step in steps if step not in ordered_steps]
            logger.warning(f"Circular dependencies detected in steps: {[s['id'] for s in remaining_steps]}")
            
            # Add remaining steps at the end (may cause issues but allows execution)
            ordered_steps.extend(remaining_steps)
        
        logger.info(f"Step execution order: {[s['id'] for s in ordered_steps]}")
        return ordered_steps
    
    async def add_step_result(self, step_id: str, result: Dict[str, Any]) -> None:
        """
        Add step result to the execution state and update history.
        
        Args:
            step_id: ID of the completed step
            result: Result from the step execution
        """
        if not self.current_execution_id:
            raise Exception("No active execution to update")
        
        execution_state = self.executions[self.current_execution_id]
        
        # Add to completed steps
        if step_id not in execution_state.completed_steps:
            execution_state.completed_steps.append(step_id)
        
        # Store step result
        execution_state.step_results[step_id] = result
        
        # Update history Hi = {(s1, a1), ..., (si, ai)}
        history_entry = {
            "step_id": step_id,
            "step_description": result.get("step_description", ""),
            "answer": result.get("qa_result", {}).get("answer", ""),
            "confidence": result.get("qa_result", {}).get("confidence", 0.0),
            "sources": result.get("qa_result", {}).get("sources", []),
            "timestamp": datetime.now().isoformat()
        }
        
        execution_state.history.append(history_entry)
        
        # Trim history if too large
        if len(execution_state.history) > self.max_history_size:
            execution_state.history = execution_state.history[-self.max_history_size:]
        
        execution_state.last_update = datetime.now()
        
        logger.info(f"Step result added: {step_id}")
    
    async def get_accumulated_history(self) -> List[Dict[str, Any]]:
        """
        Get the accumulated history Hi-1 = {(s1, a1), ..., (si-1, ai-1)}.
        This provides context for the current step.
        
        Returns:
            List of previous step results
        """
        if not self.current_execution_id:
            return []
        
        execution_state = self.executions[self.current_execution_id]
        return execution_state.history.copy()
    
    async def get_previous_answers(self) -> Dict[str, Any]:
        """
        Get previous answers in a format suitable for agent context.
        
        Returns:
            Dictionary mapping step_id to answer
        """
        if not self.current_execution_id:
            return {}
        
        execution_state = self.executions[self.current_execution_id]
        
        previous_answers = {}
        for step_id, result in execution_state.step_results.items():
            qa_result = result.get("qa_result", {})
            if qa_result:
                previous_answers[step_id] = {
                    "answer": qa_result.get("answer", ""),
                    "confidence": qa_result.get("confidence", 0.0),
                    "sources": qa_result.get("sources", [])
                }
        
        return previous_answers
    
    async def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current execution state.
        
        Returns:
            Current state information
        """
        if not self.current_execution_id:
            return {"status": "no_active_execution"}
        
        execution_state = self.executions[self.current_execution_id]
        
        return {
            "execution_id": execution_state.execution_id,
            "main_query": execution_state.main_query,
            "disambiguated_query": execution_state.disambiguated_query,
            "query_type": execution_state.query_type,
            "plan_available": execution_state.plan is not None,
            "steps_completed": len(execution_state.completed_steps),
            "total_steps": len(execution_state.plan.get("steps", [])) if execution_state.plan else 0,
            "history_size": len(execution_state.history),
            "start_time": execution_state.start_time.isoformat() if execution_state.start_time else None,
            "last_update": execution_state.last_update.isoformat() if execution_state.last_update else None
        }
    
    async def get_execution_snapshots(self) -> List[Dict[str, Any]]:
        """
        Get snapshots of the execution state over time.
        
        Returns:
            List of state snapshots
        """
        if not self.current_execution_id:
            return []
        
        execution_state = self.executions[self.current_execution_id]
        
        snapshots = []
        for i, history_entry in enumerate(execution_state.history):
            snapshot = {
                "step_number": i + 1,
                "step_id": history_entry["step_id"],
                "timestamp": history_entry["timestamp"],
                "completed_steps": execution_state.completed_steps[:i+1],
                "history_size": i + 1
            }
            snapshots.append(snapshot)
        
        return snapshots
    
    async def get_step_dependencies_status(self, step_id: str) -> Dict[str, Any]:
        """
        Check the dependency status for a specific step.
        
        Args:
            step_id: ID of the step to check
            
        Returns:
            Dependency status information
        """
        if not self.current_execution_id or not self.executions[self.current_execution_id].plan:
            return {"ready": False, "reason": "no_plan"}
        
        execution_state = self.executions[self.current_execution_id]
        steps = execution_state.plan.get("steps", [])
        
        # Find the step
        step = next((s for s in steps if s["id"] == step_id), None)
        if not step:
            return {"ready": False, "reason": "step_not_found"}
        
        dependencies = step.get("dependencies", [])
        completed_steps = set(execution_state.completed_steps)
        
        # Check if all dependencies are met
        unmet_dependencies = [dep for dep in dependencies if dep not in completed_steps]
        
        return {
            "ready": len(unmet_dependencies) == 0,
            "dependencies": dependencies,
            "completed_dependencies": [dep for dep in dependencies if dep in completed_steps],
            "unmet_dependencies": unmet_dependencies,
            "step_already_completed": step_id in completed_steps
        }
    
    async def get_execution_summary(self, execution_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of the execution.
        
        Args:
            execution_id: Optional specific execution ID, defaults to current
            
        Returns:
            Execution summary
        """
        target_id = execution_id or self.current_execution_id
        if not target_id or target_id not in self.executions:
            return {"error": "execution_not_found"}
        
        execution_state = self.executions[target_id]
        
        # Calculate metrics
        total_time = None
        if execution_state.start_time:
            end_time = execution_state.last_update or datetime.now()
            total_time = (end_time - execution_state.start_time).total_seconds()
        
        # Analyze step results
        successful_steps = sum(1 for result in execution_state.step_results.values() 
                            if result.get("success", False))
        
        avg_confidence = 0.0
        if execution_state.history:
            confidences = [entry.get("confidence", 0.0) for entry in execution_state.history]
            avg_confidence = sum(confidences) / len(confidences)
        
        return {
            "execution_id": execution_state.execution_id,
            "main_query": execution_state.main_query,
            "query_type": execution_state.query_type,
            "total_steps": len(execution_state.plan.get("steps", [])) if execution_state.plan else 0,
            "completed_steps": len(execution_state.completed_steps),
            "successful_steps": successful_steps,
            "success_rate": successful_steps / len(execution_state.step_results) if execution_state.step_results else 0,
            "avg_confidence": avg_confidence,
            "total_time": total_time,
            "start_time": execution_state.start_time.isoformat() if execution_state.start_time else None,
            "last_update": execution_state.last_update.isoformat() if execution_state.last_update else None
        }
    
    async def cleanup_execution(self, execution_id: Optional[str] = None) -> bool:
        """
        Clean up execution state.
        
        Args:
            execution_id: Optional specific execution ID, defaults to current
            
        Returns:
            True if cleanup successful
        """
        target_id = execution_id or self.current_execution_id
        if not target_id:
            return False
        
        try:
            if target_id in self.executions:
                del self.executions[target_id]
                logger.info(f"Cleaned up execution: {target_id}")
            
            if target_id == self.current_execution_id:
                self.current_execution_id = None
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup execution {target_id}: {str(e)}")
            return False
    
    async def get_all_executions(self) -> List[Dict[str, Any]]:
        """
        Get information about all executions.
        
        Returns:
            List of execution summaries
        """
        summaries = []
        for execution_id in self.executions:
            summary = await self.get_execution_summary(execution_id)
            summaries.append(summary)
        
        return summaries
    
    async def clear_all_executions(self) -> int:
        """
        Clear all execution states.
        
        Returns:
            Number of executions cleared
        """
        count = len(self.executions)
        self.executions.clear()
        self.current_execution_id = None
        
        logger.info(f"Cleared {count} executions")
        return count


# Global state manager instance for easy access
state_manager = StateManager()
