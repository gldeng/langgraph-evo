"""Planner component for LangGraph Evolution."""
from typing import Annotated, List, Dict, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, InjectedStore, create_react_agent
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.store.base import BaseStore

from langgraph_evo.core.config import ConfigRecord
from langgraph_evo.core.registry import AGENT_CONFIGS_NAMESPACE

@tool
def get_configs(
    store: Annotated[BaseStore, InjectedStore]
) -> List[ConfigRecord]:
    """Retrieve the agent configuration.
    
    This tool retrieves all agent configurations from the store.
    Output is a list of ConfigRecord objects.
    """
    configs = store.search(AGENT_CONFIGS_NAMESPACE)
    if configs:
        return list(map(lambda c: c.value, configs))
    return []

def create_react_planner():
    """Create a React agent that functions as a planner."""
    system_prompt = """You are an AI agent configurator that helps set up and run AI agent systems.
    
    When a user asks a question:
    1. Retrieve all the agent configurations using the get_configs tool
    2. For each configuration, analyze whether this configuration can solve the user's query
    3. If it can, return the configuration string

    Output format:
    - If it can, return the configuration string in a <agent_config> tag
    - If it cannot, return "No configuration found"

    Example output:
    <agent_config>
    Config string here
    </agent_config>

    Only do the following but not more than that:
    - Retrieve the agent configurations
    - Analyze whether each configuration can solve the user's query
    - Return the configuration string if it can, otherwise return "No configuration found"
    """
    
    # Create the React agent
    return create_react_agent(
        model="openai:gpt-4",
        tools=[get_configs],
        prompt=system_prompt
    )

def create_planner_node(store: BaseStore):
    """Create a new planner node with the React agent."""
    react_agent = create_react_planner()
    
    # Create the node using the injected store and React agent
    node = (
        StateGraph(MessagesState)
        .add_node('react_planner', react_agent)
        .add_edge(START, "react_planner")
        .add_edge("react_planner", END)
        .compile(store=store)
    )
    return node 