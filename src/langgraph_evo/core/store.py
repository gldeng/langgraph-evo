"""Store interaction helpers."""
from langgraph.store.base import BaseStore

from langgraph_evo.core.config import ConfigRecord
from langgraph_evo.core.registry import AGENT_CONFIGS_NAMESPACE

def initialize_configs(store: BaseStore, description: str, graph_config: str):
    """Initialize the default configurations in the store.
    
    Args:
        store: The store to initialize
        graph_config: The graph configuration string to store
    """
    # Store the default graph configuration
    config_record = ConfigRecord(
        name="default", 
        version="v1", 
        description=description, 
        config=graph_config
    )
    
    # Store the config
    store.put(
        AGENT_CONFIGS_NAMESPACE,  # namespace
        "default:v1",             # key
        config_record             # value
    ) 