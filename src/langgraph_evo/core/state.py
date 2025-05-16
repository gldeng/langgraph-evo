"""State management for LangGraph Evolution System workflows.

This module provides state classes and utilities for managing state across workflow nodes.
"""

from typing import Any, Dict, List, Optional, TypedDict, Annotated
import operator


class BaseState(TypedDict):
    """Base state for all workflows.
    
    This represents the fundamental state object that gets passed between nodes
    in the workflow graph.
    """
    
    metadata: Dict[str, Any]


class TaskState(BaseState):
    """State for task processing workflows.
    
    Contains information about the task being processed and its current state.
    """
    
    task_id: str
    task_input: Dict[str, Any]
    task_output: Optional[Dict[str, Any]]
    # Messages are appended to the list, creating a history of communication
    messages: List[Dict[str, Any]]
    # Steps track each agent's processing in sequence
    steps: List[Dict[str, Any]]
    # Context holds intermediate data needed between processing steps
    context: Dict[str, Any]
    # Error field for tracking any errors during processing
    error: Optional[str]


def merge_state(old_state: TaskState, new_partial_state: Dict[str, Any]) -> TaskState:
    """Merge a partial state update into the full state.
    
    This utility function helps update the state with partial updates from nodes,
    handling nested merges and list appends appropriately.
    
    Args:
        old_state: The current complete state
        new_partial_state: Partial updates to apply
        
    Returns:
        The merged state with updates applied
    """
    # Start with a copy of the old state
    merged_state = dict(old_state)
    
    # Process each key in the partial update
    for key, value in new_partial_state.items():
        if key not in merged_state:
            # Simply add new keys
            merged_state[key] = value
        elif isinstance(value, dict) and isinstance(merged_state[key], dict):
            # Merge dictionaries recursively
            merged_state[key] = {**merged_state[key], **value}
        elif isinstance(value, list) and isinstance(merged_state[key], list):
            # Append to lists
            merged_state[key] = merged_state[key] + value
        else:
            # Override other values
            merged_state[key] = value
    
    return merged_state


# State selection functions for message passing
def get_messages(state: TaskState) -> List[Dict[str, Any]]:
    """Extract messages from the state for passing to LLM agents.
    
    Args:
        state: The current workflow state
        
    Returns:
        The messages list from the state
    """
    return state.get("messages", [])


def get_context(state: TaskState) -> Dict[str, Any]:
    """Extract the context dictionary from the state.
    
    Args:
        state: The current workflow state
        
    Returns:
        The context dictionary from the state
    """
    return state.get("context", {}) 