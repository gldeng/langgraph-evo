"""Workflow implementation for LangGraph Evolution System.

This module provides the main workflow implementation with fixed configuration
that can be used to process tasks with a defined sequence of steps.
"""

from typing import Any, Dict, List, Optional, cast, Callable, Annotated, TypeVar
import operator
import traceback
import logging
import copy
import inspect
import asyncio

from langgraph.graph import StateGraph
from langgraph.graph.message import MessageGraph

from .state import TaskState, merge_state
from .agents import (
    BaseAgent, 
    InputProcessorAgent,
    TaskAnalyzerAgent,
    TaskExecutorAgent,
    OutputFormatterAgent
)

# Set up logging
logger = logging.getLogger(__name__)


class WorkflowError(Exception):
    """Exception raised for workflow-specific errors."""
    pass


class FixedWorkflow:
    """A workflow with fixed configuration for processing tasks.
    
    This class implements a simple LangGraph workflow that processes tasks
    through a fixed sequence of steps:
    1. Input processing
    2. Task analysis
    3. Task execution
    4. Output formatting
    
    The workflow uses a state object to track information between steps.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the workflow with optional configuration.
        
        Args:
            config: A dictionary of configuration options
        """
        self.config = config or {}
        
        # Initialize agents with configuration
        self.input_processor = InputProcessorAgent(self.config.get("input_processor_config"))
        self.task_analyzer = TaskAnalyzerAgent(self.config.get("task_analyzer_config"))
        self.task_executor = TaskExecutorAgent(self.config.get("task_executor_config"))
        self.output_formatter = OutputFormatterAgent(self.config.get("output_formatter_config"))
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Compile the graph into a runnable
        self.runnable = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph.
        
        Returns:
            A compiled StateGraph that can be run with the workflow
        """
        # Create a new graph
        graph = StateGraph(TaskState)
        
        # Add nodes with wrapped processing functions
        graph.add_node("input_processor", self._create_safe_node(self.input_processor, "InputProcessor"))
        graph.add_node("task_analyzer", self._create_safe_node(self.task_analyzer, "TaskAnalyzer"))
        graph.add_node("task_executor", self._create_safe_node(self.task_executor, "TaskExecutor"))
        graph.add_node("output_formatter", self._create_safe_node(self.output_formatter, "OutputFormatter"))
        
        # Create a simple linear flow - always go through all nodes
        # Error handling happens inside each node
        graph.add_edge("input_processor", "task_analyzer")
        graph.add_edge("task_analyzer", "task_executor")
        graph.add_edge("task_executor", "output_formatter")
        
        # Set the entry point
        graph.set_entry_point("input_processor")
        
        # Set the exit point
        graph.set_finish_point("output_formatter")
        
        return graph
    
    def _create_safe_node(self, agent: BaseAgent, node_name: str) -> Callable:
        """Create a node function that safely calls an agent and handles errors.
        
        Args:
            agent: The agent to wrap
            node_name: The name of the node
            
        Returns:
            A function that can be used as a node in the graph
        """
        def safe_node(state: Dict[str, Any]) -> Dict[str, Any]:
            # Create a deep copy of the state to avoid modifying the original
            current_state = copy.deepcopy(state) if state else {}
            
            # Initialize steps list if it doesn't exist
            if "steps" not in current_state:
                current_state["steps"] = []
            
            # Check if we already have an error - if so, don't process further
            if current_state.get("error") is not None:
                logger.debug(f"Node {node_name} skipping processing due to existing error")
                
                # Add a step for this node with skip action
                step = {
                    "agent": node_name,
                    "action": "skip_due_to_error",
                    "timestamp": "2024-05-08T00:00:00Z"  # In production, use a real timestamp
                }
                current_state["steps"].append(step)
                
                # Special case for OutputFormatter - it should still process the error
                if node_name == "OutputFormatter":
                    return agent.process(current_state)
                
                # For other nodes, just pass through the state
                return current_state
            
            # If no error, add a normal processing step
            step = {
                "agent": node_name,
                "action": "process",
                "timestamp": "2024-05-08T00:00:00Z"  # In production, use a real timestamp
            }
            current_state["steps"].append(step)
            
            try:
                # Process the state using the agent
                updated_state = agent.process(current_state)
                return updated_state if updated_state else current_state
                
            except Exception as e:
                # Log the error
                error_msg = f"Error in {node_name}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                
                # Prepare error state
                error_state = {
                    "task_id": current_state.get("task_id", "unknown"),
                    "error": error_msg,
                    "error_present": True,  # Explicitly set for conditional routing
                    "success": False,
                    "steps": current_state.get("steps", []),
                    "metadata": {**current_state.get("metadata", {}), "error": True}
                }
                
                # Preserve other fields from the original state
                for key, value in current_state.items():
                    if key not in error_state:
                        error_state[key] = value
                
                return error_state
        
        return safe_node
    
    def run(self, input_data: Dict[str, Any], task_id: Optional[str] = None) -> Dict[str, Any]:
        """Run the workflow with the provided input data.
        
        Args:
            input_data: The input data for the task
            task_id: Optional task ID, will be generated if not provided
            
        Returns:
            The result of the workflow execution
        """
        # Generate a task ID if not provided
        if task_id is None:
            task_id = f"task_{id(input_data)}"  # Simple ID generation
        
        # Create initial state
        initial_state: TaskState = {
            "task_id": task_id,
            "task_input": input_data,
            "task_output": None,
            "messages": [],
            "steps": [],
            "metadata": {"workflow_type": "fixed"},
            "context": {},
            "error": None
        }
        
        try:
            # Execute the workflow
            final_state = cast(TaskState, self.runnable.invoke(initial_state))
            
            # Check if there was an error
            if final_state.get("error") is not None:
                # Return error result
                return {
                    "task_id": task_id,
                    "error": final_state.get("error"),
                    "success": False,
                    "steps": final_state.get("steps", []),
                    "metadata": {
                        **final_state.get("metadata", {}),
                        "error": True
                    }
                }
            
            # Return the task output
            return {
                "task_id": task_id,
                "result": final_state.get("task_output", {}),
                "steps": final_state.get("steps", []),
                "metadata": final_state.get("metadata", {}),
                "success": True
            }
        except Exception as e:
            # Log the error
            logger.error(f"Workflow execution failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return error result
            return {
                "task_id": task_id,
                "error": str(e),
                "success": False,
                "steps": initial_state.get("steps", []),
                "metadata": {
                    **initial_state.get("metadata", {}),
                    "error": True
                }
            }


def create_workflow(config: Optional[Dict[str, Any]] = None) -> FixedWorkflow:
    """Create a new workflow with the provided configuration.
    
    This is a convenience function for creating a workflow with a fixed 
    configuration without directly instantiating the FixedWorkflow class.
    
    Args:
        config: A dictionary of configuration options
        
    Returns:
        A new workflow instance
    """
    return FixedWorkflow(config) 