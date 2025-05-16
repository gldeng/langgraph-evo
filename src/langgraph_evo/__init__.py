"""LangGraph Evolution System.

This package provides tools for creating, evaluating, and evolving LangGraph configurations
to optimize task performance.
"""

from typing import Dict, Any, List, Optional, Union, Type

# Core imports
from .core.state import TaskState, merge_state
from .core.agents import BaseAgent, ReactAgent, ToolUsingAgent, AgentOutput

# Component imports
from .components.planner import Planner, PlannerConfig, AgentType
from .components.factory import Factory
from .components.evaluator import Evaluator
from .components.evolution import Evolution, MutationType

__version__ = "0.1.0"
__all__ = [
    # Core
    "TaskState",
    "merge_state",
    "BaseAgent",
    "ReactAgent",
    "ToolUsingAgent",
    "AgentOutput",
    
    # Components
    "Planner",
    "PlannerConfig",
    "AgentType",
    "Factory",
    "Evaluator",
    "Evolution",
    "MutationType",
]
