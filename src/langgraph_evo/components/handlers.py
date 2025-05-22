"""Node handler functions."""
import re
import sys
from typing import Annotated, Any, Dict, List

from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore
from langchain_core.messages import AIMessage, HumanMessage
from langgraph_evo.core.state import PsiState
from langgraph_evo.core.registry import _get_or_create_node, node_registry, PLANNER_NODE_ID, AGENT_CONFIGS_NAMESPACE
from langgraph_evo.core.config import parse_graph_config
from langgraph_evo.core.builder import create_graph

def planner_node_handler(
    state: Annotated[Dict[str, Any], InjectedState],
    store: Annotated[BaseStore, InjectedStore]
):
    """A simple handler that echoes the last message and returns any stored configuration.
    
    Args:
        state: The current state
        store: The store to use
        
    Returns:
        Updated state with configuration info
    """
    messages = state["messages"]
    
    # Get the config from the store
    configs = store.search(AGENT_CONFIGS_NAMESPACE)
    response = None
    if configs:
        config_str = configs[0].value.config
        response = AIMessage(content=f"<agent_config>{config_str}</agent_config>")
    else:
        response = AIMessage(content="No configuration found.")
    return {"messages": messages + [response]}

def task_handler(
    state: Annotated[PsiState, InjectedState],
    store: Annotated[BaseStore, InjectedStore],
    config: Dict[str, Any] = None
):
    """A node handler that processes messages using the agent system.
    
    Args:
        state: The current state
        store: The store to use
        config: Optional configuration parameters
        
    Returns:
        Updated state with task results
    """
    messages = state["messages"]
    
    # Extract metadata from config if available
    metadata = {}
    if config and "metadata" in config:
        metadata = config["metadata"]
        print(f"Received metadata in config: {metadata.keys()}")
    
    # Get or create planner node if not already in state
    if "planner_node_id" not in state or not state["planner_node_id"]:
        # Initialize the node registry dictionary if not present
        planner_node_id = _get_or_create_node(store)
        
        initialized_node_ids = state.get("initialized_node_ids", {})
        initialized_node_ids[PLANNER_NODE_ID] = planner_node_id
        
        updated_state = {
            "messages": messages,
            "planner_node_id": planner_node_id,
            "initialized_node_ids": initialized_node_ids
        }
    else:
        updated_state = state
    
    # Get the planner node from registry
    planner_node_id = updated_state["planner_node_id"]
    planner = node_registry[planner_node_id]
    
    try:
        # Find the last user message for fallback processing if needed
        last_user_message = None
        for msg in reversed(messages):
            # Handle both dict and Message object types
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user_message = msg
                break
            elif hasattr(msg, "type") and msg.type == "human":
                # Convert to dict format if it's a Message object
                last_user_message = HumanMessage(content=msg.content)
                break
        
        if not last_user_message and messages:
            # If no user message found, use the last message regardless of type
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                last_user_message = last_msg
            else:
                last_user_message = HumanMessage(content=last_msg.content)
        
        # Process with the planner, passing the full conversation history
        # This allows the planner to understand context from previous interactions
        # which is essential for follow-up questions
        planner_result = planner.invoke({"messages": messages})

        planner_response = planner_result["messages"][-1]

        content = planner_response.content if hasattr(planner_response, "content") else planner_response["content"]
    except (AttributeError, TypeError, KeyError) as e:
        print(f"Warning: Could not extract content from planner response: {e}")
        content = ""

    # Search for agent configuration
    match = re.search(r'<agent_config>(.*?)</agent_config>', content, re.DOTALL)
    if match:
        config = match.group(1)
    else:
        print(f"Warning: Could not find agent_config in response: {content[:100]}...")
        response = AIMessage(content="Sorry, I'm unable to create an agent to perform the task.")
        updated_state["messages"] = messages + [response]
        return updated_state

    # Add a response to the message history
    response = AIMessage(content="I'm analyzing your request...")
    
    try:
        parsed_config = parse_graph_config(config)
        print("Successfully parsed the agent configuration")

        # Create the graph using the registry (no need to pass tools explicitly)
        graph = create_graph(parsed_config)
        
        # Pass the full message history to the graph for proper context in follow-up questions
        result = graph.invoke({"messages": messages})
        response = AIMessage(content=result["messages"][-1].content)

        print("Result:", response)
    except Exception as e:
        print(f"Error creating/running agent: {e}")
        response = AIMessage(content="I processed your request but encountered issues with my agent configuration system.")
    
    # Return updated state with our response
    updated_state["messages"] = messages + [response]
    return updated_state

# Simple wrapper for task_handler that directly invokes it with the store
def task_handler_wrapped(state, store, config=None):
    """Simple wrapper for task_handler that passes the injected store and config."""
    return task_handler(state, store, config)

def planner_node_handler_wrapped(state, store):
    """Simple wrapper for planner_node_handler that passes the injected store."""
    return planner_node_handler(state, store) 