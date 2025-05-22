"""State definitions for LangGraph Evolution."""
from __future__ import annotations
from typing import Annotated, Any, Dict, List, Tuple, TypedDict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages, AnyMessage

from langgraph_evo.core.config import ConfigRecord

class GraphState(TypedDict):
    """State for the graph execution."""
    lineage: List[str]
    agent_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    config: ConfigRecord
    children_states: Dict[str, GraphState]  # Maps node names to registry IDs
