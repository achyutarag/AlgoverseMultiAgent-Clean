# agents/state_manager/metrics.py
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

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

async def get_execution_snapshots(self) -> List[Dict[str, Any]]:
    """Get snapshots of the execution state over time."""
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
    """Check the dependency status for a specific step."""
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

