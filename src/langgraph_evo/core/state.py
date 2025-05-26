"""State definitions for LangGraph Evolution."""
from __future__ import annotations
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict, get_type_hints
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages, AnyMessage

from langgraph_evo.core.config import ConfigRecord

# Module-level cache for GraphState type hints to avoid repeated computation
_GRAPHSTATE_TYPE_HINTS_CACHE: Optional[Dict[str, Any]] = None

def use_last(left: Any, right: Any) -> Any:
    """Use the last non-None value."""
    return right if right is not None else left

def merge_if_theres_value(left: Optional[Dict[tuple[str, ...], GraphState]], right: Optional[Dict[tuple[str, ...], GraphState]]) -> Dict[tuple[str, ...], GraphState]:
    """Merge if there is a value."""
    if left is None or right is None:
        return left if left is not None else right
    for key, value in right.items():
        if key not in left:
            left[key] = value
        else:
            left[key] = GraphState.reducer(left[key], value)
    return left

def use_last_int(left: int, right: int) -> int:
    """Use the last non-None integer value."""
    return right if right is not None else left

def use_last_bool(left: bool, right: bool) -> bool:
    """Use the last non-None boolean value."""
    return right if right is not None else left

def use_last_str(left: str, right: str) -> str:
    """Use the last non-None string value."""
    return right if right is not None else left

def union_sets(left: Optional[set], right: Optional[set]) -> set:
    """Union two sets, handling None values."""
    if left is None:
        return right if right is not None else set()
    if right is None:
        return left
    return left.union(right)

class GraphState(TypedDict):
    """State for the graph execution."""
    lineage: Annotated[List[str], use_last]
    agent_id: Annotated[str, use_last]
    messages: Annotated[list[AnyMessage], add_messages]
    config: Annotated[ConfigRecord, use_last]
    children_states: Annotated[Dict[tuple[str, ...], GraphState], merge_if_theres_value]
    initialized_node_ids: Annotated[set[str], union_sets]
    # Added fields for attempt tracking and workflow control
    attempt_count: Annotated[int, use_last_int]
    supervisor_success: Annotated[bool, use_last_bool]
    planner_success: Annotated[bool, use_last_bool]
    planner_node_id: Annotated[str, use_last_str]

    @staticmethod
    def reducer(left: Optional[GraphState], right: Optional[GraphState], **kwargs: Any) -> GraphState:
        if left is None or right is None:
            msg = (
                f"Must specify non-null arguments for both 'left' and 'right'. Only "
                f"received: '{'left' if left else 'right'}'."
            )
            raise ValueError(msg)
        
        # Use cached type hints or compute and cache them
        global _GRAPHSTATE_TYPE_HINTS_CACHE
        if _GRAPHSTATE_TYPE_HINTS_CACHE is None:
            _GRAPHSTATE_TYPE_HINTS_CACHE = get_type_hints(GraphState, include_extras=True)
        
        type_hints = _GRAPHSTATE_TYPE_HINTS_CACHE
        
        # Start with a copy of the left state
        merged = left.copy()
        
        # Iterate through each field and apply its annotated reducer function
        for field_name, annotated_type in type_hints.items():
            # Extract the reducer function from the Annotated type
            if hasattr(annotated_type, '__metadata__') and annotated_type.__metadata__:
                reducer_func = annotated_type.__metadata__[0]
                
                # Get the values from left and right states
                left_value = left.get(field_name)
                right_value = right.get(field_name)
                # only apply reducer if  right has values
                if right_value is not None:
                    merged[field_name] = reducer_func(left_value, right_value)
        
        return merged
