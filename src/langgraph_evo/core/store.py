"""Store interaction helpers."""
from langgraph.store.base import BaseStore

from langgraph_evo.core.config import ConfigRecord
from langgraph_evo.core.registry import AGENT_CONFIGS_NAMESPACE


def add_config(store: BaseStore, config_record: ConfigRecord):
    """Add a configuration to the store.
    
    Args:
        store: The store to add the configuration to
        config_record: The configuration record to add
    """
    store.put(
        AGENT_CONFIGS_NAMESPACE,
        f"{config_record.name}__{config_record.version}",
        config_record
    )
