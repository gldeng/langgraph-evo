"""Builder functions for creating graphs from configurations."""
from typing import Any, Dict, List
import sys
import importlib
import builtins

from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore

from langgraph_evo.components.tools import name_of_transfer_tool
from langgraph_evo.core.state import GraphState
from langgraph_evo.core.config import GraphConfig
from langgraph_evo.core.tool_registry import get_tool, has_tool, resolve_tool

def create_graph(config: GraphConfig):
    """Create a graph based on the configuration.
    
    Args:
        config: The graph configuration to use
        
    Returns:
        A compiled graph that can be executed
    """
    # Import here to avoid circular imports
    from langgraph_evo.components.tools import create_handoff_tool
    from langgraph.prebuilt import create_react_agent
    
    # Create a mapping of tool names to actual tool objects
    tool_map = {}
    
    # Process tool configurations
    if config.tools:
        for tool_config in config.tools:
            if hasattr(tool_config, 'name') and hasattr(tool_config, 'script'):
                # Skip if already in map
                if tool_config.name in tool_map:
                    continue
                    
                # Try to get from registry or resolve
                tool_obj = None
                
                # First try the registry
                if has_tool(tool_config.name):
                    tool_obj = get_tool(tool_config.name)
                    
                # Then try to resolve by script name
                if tool_obj is None and tool_config.script:
                    tool_obj = resolve_tool(tool_config.script)
                    
                if tool_obj is not None:
                    tool_map[tool_config.name] = tool_obj
                else:
                    print(f"Warning: Could not resolve tool '{tool_config.name}' with script '{tool_config.script}'")
    
    # Create handoff tools for node transitions
    handoff_tools = {}
    for edge in config.edges:
        tool_name = name_of_transfer_tool(edge.to)
        if tool_name not in handoff_tools:
            handoff_tools[tool_name] = create_handoff_tool(
                agent_name=edge.to, 
                description=f"Transfer to {edge.to}"
            )
    
    # Create nodes
    nodes = {}
    for node in config.nodes:
        if node.type == "react":
            # Collect tools for this node
            node_tools = []
            
            # Add any tools specified in the node config
            if "tools" in node.config and node.config["tools"]:
                for tool_name in node.config["tools"]:
                    if tool_name in tool_map:
                        node_tools.append(tool_map[tool_name])
                    else:
                        print(f"Warning: Tool '{tool_name}' referenced in node '{node.name}' not found in tool map")
            
            # Add handoff tools for edges originating from this node
            for edge in config.edges:
                if edge.from_ == node.name and name_of_transfer_tool(edge.to) in handoff_tools:
                    node_tools.append(handoff_tools[name_of_transfer_tool(edge.to)])
            
            # Get prompt from config or use default
            prompt = node.config.get("prompt", f"You are the {node.name} agent.")
            
            # Create the react agent
            nodes[node.name] = create_react_agent(
                model=node.config["model"],
                tools=node_tools,
                prompt=prompt,
                name=node.name
            )
    
    # Create the graph structure
    graph = StateGraph(GraphState)
    
    # Add all nodes to the graph
    for node_name, node_handler in nodes.items():
        graph.add_node(node_name, node_handler)
    
    # Connect the entry point node to START and END
    try:
        entry_node = next(node for node in config.nodes if node.is_entry_point)
        graph.add_edge(START, entry_node.name)
        # Entry point is also connected to END
        graph.add_edge(entry_node.name, END)
    except StopIteration:
        print("Warning: No entry point node specified in configuration")
    except Exception as e:
        print(f"Error adding entry point edges: {e}")
    
    # Add all other edges defined in the configuration
    for edge in config.edges:
        try:
            graph.add_edge(edge.from_, edge.to)
        except Exception as e:
            print(f"Error adding edge from '{edge.from_}' to '{edge.to}': {e}")
    
    # Compile and return the graph
    return graph.compile(store=InMemoryStore()) 