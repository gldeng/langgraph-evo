"""State definitions for LangGraph Evolution."""
from __future__ import annotations
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict, get_type_hints
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages, AnyMessage

from langgraph_evo.core.config import ConfigRecord

def use_last(left: Any, right: Any) -> Any:
    """Use the last non-None value."""
    return right if right is not None else left

def merge_if_theres_value(left: Optional[Dict[str, GraphState]], right: Optional[Dict[str, GraphState]]) -> Dict[str, GraphState]:
    """Merge if there is a value."""
    if left is None or right is None:
        return left if left is not None else right
    for key, value in right.items():
        if key not in left:
            left[key] = value
        else:
            left[key] = GraphState.reducer(left[key], value)
    return left

class GraphState(TypedDict):
    """State for the graph execution."""
    lineage: Annotated[List[str], use_last]
    agent_id: Annotated[str, use_last]
    messages: Annotated[list[AnyMessage], add_messages]
    config: Annotated[ConfigRecord, use_last]
    children_states: Annotated[Dict[str, GraphState], merge_if_theres_value]
    initialized_node_ids: Annotated[set[str], set.union]

    @staticmethod
    def reducer(left: Optional[GraphState], right: Optional[GraphState], **kwargs: Any) -> GraphState:
        if left is None or right is None:
            msg = (
                f"Must specify non-null arguments for both 'left' and 'right'. Only "
                f"received: '{'left' if left else 'right'}'."
            )
            raise ValueError(msg)
        
        # Get the type hints with annotations
        type_hints = get_type_hints(GraphState, include_extras=True)
        
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
