"""Agent interfaces for LangGraph Evolution System workflows.

This module provides simple agent interfaces that serve as nodes in the workflow.
"""

from typing import Any, Dict, List, Optional

from .state import TaskState


class BaseAgent:
    """Base class for all workflow agents.
    
    Agents are the processing units that operate as nodes in the workflow graph.
    Each agent takes the workflow state, performs some operations, and returns 
    an updated state.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent with optional configuration."""
        self.config = config or {}
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state and return an updated state.
        
        Args:
            state: The current workflow state
            
        Returns:
            An updated workflow state
        """
        # Default implementation does nothing
        return state


class InputProcessorAgent(BaseAgent):
    """Agent for processing input data.
    
    This agent is responsible for validating and preparing the input data
    for further processing in the workflow.
    """
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input data.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated state with processed input
        """
        # Get the task input
        task_input = state.get("task_input", {})
        
        # Simulate input processing (in a real implementation, we'd do actual validation)
        processed_input = {
            "query": task_input.get("query", "No query provided"),
            "parameters": task_input.get("options", {}),
            "metadata": {
                "time_received": "2024-05-08T00:00:00Z",  # In production, use real timestamp
                "source": "api"
            }
        }
        
        # Update the state with processed input
        state["task_input"] = processed_input
        
        # Add a step for tracking
        if "steps" not in state:
            state["steps"] = []
            
        state["steps"].append({
            "agent": "InputProcessor",
            "action": "process_input",
            "input_size": len(str(task_input)),
            "timestamp": "2024-05-08T00:00:00Z"  # In production, use real timestamp
        })
        
        return state


class TaskAnalyzerAgent(BaseAgent):
    """Agent for analyzing the task.
    
    This agent is responsible for understanding the task requirements
    and providing analysis that can guide the task execution.
    """
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the task.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated state with task analysis
        """
        # Simulate task analysis (in a real implementation, we'd use LLMs or other logic)
        analysis = {
            "task_type": "text_processing",
            "complexity": "medium",
            "estimated_time": "5 minutes"
        }
        
        # Update the state with the analysis
        if "context" not in state:
            state["context"] = {}
            
        state["context"]["analysis"] = analysis
        
        # Add a step for tracking
        if "steps" not in state:
            state["steps"] = []
            
        state["steps"].append({
            "agent": "TaskAnalyzer",
            "action": "analyze_task",
            "analysis": analysis,
            "timestamp": "2024-05-08T00:00:01Z"  # In production, use real timestamp
        })
        
        return state


class TaskExecutorAgent(BaseAgent):
    """Agent for executing the task.
    
    This agent is responsible for the actual execution of the task
    based on the input and analysis provided by previous agents.
    """
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the task.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated state with task execution results
        """
        # Get task details
        task_id = state.get("task_id", "unknown")
        task_input = state.get("task_input", {})
        
        # Simulate task execution (in a real implementation, we'd do actual work)
        task_output = {
            "output": f"Processed task {task_id} successfully",
            "metadata": {
                "task_id": task_id,
                "processing_steps": 2,
                "task_type": state.get("context", {}).get("analysis", {}).get("task_type", "unknown")
            }
        }
        
        # Update the state with the execution results
        state["task_output"] = task_output
        
        # Add a step for tracking
        if "steps" not in state:
            state["steps"] = []
            
        state["steps"].append({
            "agent": "TaskExecutor",
            "action": "execute_task",
            "execution_time": "1.2s",  # In production, measure real time
            "timestamp": "2024-05-08T00:00:02Z"  # In production, use real timestamp
        })
        
        return state


class OutputFormatterAgent(BaseAgent):
    """Agent for formatting the output.
    
    This agent is responsible for formatting the task execution results
    into the final output format.
    """
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Format the output.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated state with formatted output
        """
        # Check if we're in an error state - either via error or error_present flag
        if state.get("error") is not None or state.get("error_present", False):
            # If there's an error, don't try to format the output
            # Just add a step to record that we were here
            if "steps" not in state:
                state["steps"] = []
                
            state["steps"].append({
                "agent": "OutputFormatter",
                "action": "format_output",
                "status": "skipped_due_to_error",
                "timestamp": "2024-05-08T00:00:03Z"  # In production, use real timestamp
            })
            
            return state
            
        # Get task output
        task_output = state.get("task_output")
        
        # If no task output is available, create a placeholder
        if task_output is None:
            task_output = {
                "output": "No output available",
                "metadata": {
                    "task_id": state.get("task_id", "unknown"),
                    "status": "no_output"
                }
            }
            
        # Format the output (in a real implementation, we might transform it)
        formatted_output = {
            "result": task_output.get("output", "No output generated"),
            "metadata": task_output.get("metadata", {})
        }
        
        # Update the state with the formatted output
        state["task_output"] = formatted_output
        
        # Add a step for tracking
        if "steps" not in state:
            state["steps"] = []
            
        state["steps"].append({
            "agent": "OutputFormatter",
            "action": "format_output",
            "timestamp": "2024-05-08T00:00:03Z"  # In production, use real timestamp
        })
        
        return state 