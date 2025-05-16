"""Schema definition for LangGraph workflow configurations.

This module provides Pydantic models for defining, validating, and
serializing workflow configurations in the LangGraph Evolution system.
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator


class NodeType(str, Enum):
    """Types of nodes available in a workflow."""
    INPUT_PROCESSOR = "input_processor"
    TASK_ANALYZER = "task_analyzer"
    TASK_EXECUTOR = "task_executor" 
    OUTPUT_FORMATTER = "output_formatter"
    CUSTOM = "custom"


class Node(BaseModel):
    """Definition of a workflow node."""
    id: str = Field(..., description="Unique identifier for the node")
    type: NodeType = Field(..., description="Type of the node")
    parameters: Dict[str, Any] = Field(default_factory=dict, 
                                     description="Configuration parameters for this node")
    description: Optional[str] = Field(None, description="Human-readable description of the node")

    class Config:
        """Pydantic config for the Node model."""
        extra = "forbid"  # No extra fields allowed


class Edge(BaseModel):
    """Connection between two nodes in a workflow."""
    source: str = Field(..., description="ID of the source node")
    target: str = Field(..., description="ID of the target node") 
    condition: Optional[str] = Field(None, 
                                    description="Optional condition for following this edge")

    class Config:
        """Pydantic config for the Edge model."""
        extra = "forbid"  # No extra fields allowed


class WorkflowConfig(BaseModel):
    """Configuration for a LangGraph workflow."""
    id: str = Field(..., description="Unique identifier for this configuration")
    name: str = Field(..., description="Human-readable name for the workflow")
    description: Optional[str] = Field(None, description="Detailed description of the workflow")
    version: str = Field("1.0.0", description="Version of this configuration")
    nodes: List[Node] = Field(..., description="List of nodes in the workflow")
    edges: List[Edge] = Field(..., description="Connections between nodes")
    entry_point: str = Field(..., description="ID of the entry point node")
    exit_points: List[str] = Field(default_factory=list, 
                                 description="IDs of potential exit point nodes")
    metadata: Dict[str, Any] = Field(default_factory=dict, 
                                   description="Additional metadata for the workflow")

    class Config:
        """Pydantic config for the WorkflowConfig model."""
        extra = "forbid"  # No extra fields allowed

    @validator('nodes')
    def validate_node_ids_unique(cls, nodes):
        """Ensure all node IDs are unique."""
        node_ids = [node.id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Node IDs must be unique")
        return nodes

    @validator('exit_points')
    def validate_exit_points_exist(cls, exit_points, values):
        """Ensure exit points reference existing nodes."""
        if 'nodes' not in values:
            return exit_points  # Skip validation if nodes aren't available
            
        node_ids = [node.id for node in values['nodes']]
        for exit_point in exit_points:
            if exit_point not in node_ids:
                raise ValueError(f"Exit point '{exit_point}' references non-existent node")
        return exit_points

    @validator('entry_point')
    def validate_entry_point_exists(cls, entry_point, values):
        """Ensure entry point references an existing node."""
        if 'nodes' not in values:
            return entry_point  # Skip validation if nodes aren't available
            
        node_ids = [node.id for node in values['nodes']]
        if entry_point not in node_ids:
            raise ValueError(f"Entry point '{entry_point}' references non-existent node")
        return entry_point

    @validator('edges')
    def validate_edge_references(cls, edges, values):
        """Ensure edges reference existing nodes."""
        if 'nodes' not in values:
            return edges  # Skip validation if nodes aren't available
            
        node_ids = [node.id for node in values['nodes']]
        for edge in edges:
            if edge.source not in node_ids:
                raise ValueError(f"Edge source '{edge.source}' references non-existent node")
            if edge.target not in node_ids:
                raise ValueError(f"Edge target '{edge.target}' references non-existent node")
        return edges

    def to_json(self, **kwargs):
        """Convert the configuration to JSON."""
        return self.json(**kwargs)

    @classmethod
    def from_json(cls, json_str: str):
        """Create a configuration instance from JSON."""
        return cls.parse_raw(json_str) 