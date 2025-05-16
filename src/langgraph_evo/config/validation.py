"""Extended validation for LangGraph workflow configurations.

This module provides additional validation functions beyond the basic
Pydantic validations in the schema.py module.
"""

from typing import Dict, List, Set, Tuple
import networkx as nx

from .schema import WorkflowConfig, Node, Edge


def validate_graph_connectivity(config: WorkflowConfig) -> Tuple[bool, List[str]]:
    """Check if the workflow graph is properly connected.
    
    Args:
        config: The workflow configuration to validate
        
    Returns:
        A tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add all nodes
    for node in config.nodes:
        G.add_node(node.id)
    
    # Add all edges
    for edge in config.edges:
        G.add_edge(edge.source, edge.target)
    
    # Check if all nodes are reachable from the entry point
    if not nx.is_weakly_connected(G):
        # Find disconnected components
        components = list(nx.weakly_connected_components(G))
        disconnected = []
        
        # If entry point is in the first component, other components are disconnected
        entry_component = None
        for i, component in enumerate(components):
            if config.entry_point in component:
                entry_component = i
                break
        
        # Collect nodes that are not in the entry point's component
        if entry_component is not None:
            for i, component in enumerate(components):
                if i != entry_component:
                    disconnected.extend(list(component))
        
        errors.append(f"Workflow graph is not connected. Disconnected nodes: {disconnected}")
    
    # Check if there's a path from entry point to at least one exit point
    if config.exit_points:
        reachable_exits = False
        for exit_point in config.exit_points:
            if nx.has_path(G, config.entry_point, exit_point):
                reachable_exits = True
                break
        
        if not reachable_exits:
            errors.append("No exit point is reachable from the entry point")
    
    return (len(errors) == 0, errors)


def detect_cycles(config: WorkflowConfig) -> Tuple[bool, List[str]]:
    """Check for cycles in the workflow graph.
    
    Args:
        config: The workflow configuration to validate
        
    Returns:
        A tuple of (no_cycles, cycle_paths)
    """
    errors = []
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add all nodes
    for node in config.nodes:
        G.add_node(node.id)
    
    # Add all edges
    for edge in config.edges:
        G.add_edge(edge.source, edge.target)
    
    # Check for cycles
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            for cycle in cycles:
                cycle_str = " -> ".join(cycle) + " -> " + cycle[0]
                errors.append(f"Circular dependency detected: {cycle_str}")
    except nx.NetworkXNoCycle:
        # No cycles found
        pass
    
    return (len(errors) == 0, errors)


def validate_workflow_integrity(config: WorkflowConfig) -> Tuple[bool, List[str]]:
    """Perform comprehensive validation on a workflow configuration.
    
    Args:
        config: The workflow configuration to validate
        
    Returns:
        A tuple of (is_valid, error_messages)
    """
    all_errors = []
    
    # Check for connectivity issues
    connectivity_valid, connectivity_errors = validate_graph_connectivity(config)
    if not connectivity_valid:
        all_errors.extend(connectivity_errors)
    
    # Check for cycles
    cycles_valid, cycle_errors = detect_cycles(config)
    if not cycles_valid:
        all_errors.extend(cycle_errors)
    
    # Additional checks can be added here
    
    return (len(all_errors) == 0, all_errors) 