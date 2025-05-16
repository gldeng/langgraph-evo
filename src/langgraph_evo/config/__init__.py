"""Configuration module for LangGraph Evolution System.

This module provides schema definition, validation, serialization, and storage
for workflow configurations in the LangGraph Evolution System.
"""

from .schema import (
    WorkflowConfig,
    Node,
    Edge,
    NodeType
)

from .validation import (
    validate_workflow_integrity,
    validate_graph_connectivity,
    detect_cycles
)

from .serialization import (
    to_json,
    from_json,
    save_to_file,
    load_from_file,
    dict_to_config,
    config_to_dict,
    SerializationError
)

from .storage import (
    ConfigStorage,
    StorageError
)

from .manager import (
    ConfigManager,
    ConfigError
)

__all__ = [
    # Schema
    'WorkflowConfig',
    'Node',
    'Edge',
    'NodeType',
    
    # Validation
    'validate_workflow_integrity',
    'validate_graph_connectivity',
    'detect_cycles',
    
    # Serialization
    'to_json',
    'from_json',
    'save_to_file',
    'load_from_file',
    'dict_to_config',
    'config_to_dict',
    'SerializationError',
    
    # Storage
    'ConfigStorage',
    'StorageError',
    
    # Manager
    'ConfigManager',
    'ConfigError'
]
