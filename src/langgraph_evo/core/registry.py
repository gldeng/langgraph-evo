"""Registry for graph nodes."""
from typing import Dict, Any

# Global registry for nodes
node_registry: Dict[str, Any] = {}
PLANNER_NODE_ID = "__planner_node__"
SUPERVISOR_NODE_ID = "__supervisor_node__"
AGENT_CONFIGS_NAMESPACE = ("agent", "configs")

def _get_or_create_node(store) -> str:
    """Get or create a node and return its registry ID.
    
    Args:
        store: The store to use for node creation
        
    Returns:
        str: The registry ID for the node
    """
    # Import here to avoid circular imports
    from langgraph_evo.components.planner import create_planner_node
    
    node_id = PLANNER_NODE_ID
    if node_id not in node_registry:
        node_registry[node_id] = ("planner", create_planner_node(store))
    return node_id 