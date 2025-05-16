"""LangGraph Evolution System.

A framework for evolving multi-agent workflows using LangGraph.
"""

__version__ = "0.0.1"

# Export core functionality
from .core import (
    FixedWorkflow,
    create_workflow,
    BaseState,
    TaskState,
    BaseAgent
)

# Define simplified API for quick access
create_fixed_workflow = create_workflow
