"""State definitions for LangGraph Evolution."""
from typing import Annotated, Any, Dict, List, Tuple, TypedDict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages, AnyMessage

class GraphState(TypedDict):
    """State for the graph execution."""
    messages: Annotated[list[AnyMessage], add_messages]
    config: Any  # GraphConfig (forward reference)
    initialized_node_ids: Dict[str, str]  # Maps node names to registry IDs

class PsiState(TypedDict):
    """State for PSI system execution."""
    messages: List[Any]  # Simple list of any message type
    planner_node_id: str  # Store ID instead of actual graph object
    initialized_node_ids: Dict[Tuple[str, str], str]  # Store IDs, not objects 