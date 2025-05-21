"""Configuration models for LangGraph Evolution."""
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field

class Tool(BaseModel):
    """Tool definition with script reference and description."""
    name: str
    script: str
    description: str

# Node configuration models
class NodeConfigBase(BaseModel):
    """Base configuration for all node types."""
    pass

class ReactNodeConfig(NodeConfigBase):
    """Configuration specific to 'react' type nodes."""
    model: str
    tools: Optional[List[str]] = None  # Tool names referenced from top-level tools
    prompt: Optional[str] = None  # Can be multi-line string in YAML using | or > syntax

class NodeConfig(BaseModel):
    """Configuration for a single node."""
    type: Literal["react", "custom", "task_handler"]
    config: Union[ReactNodeConfig, Dict[str, Any]]

class Node(BaseModel):
    """Complete node definition."""
    name: str
    type: str
    is_entry_point: Optional[bool] = None
    config: Dict[str, Any]

# Edge configuration model
class Edge(BaseModel):
    """Connection between nodes."""
    from_: str = Field(..., alias="from")
    to: str

# Complete graph configuration model
class GraphConfig(BaseModel):
    """Complete graph configuration."""
    tools: Optional[List[Tool]] = None
    nodes: List[Node]
    edges: List[Edge]
    
    class Config:
        """Pydantic configuration."""
        populate_by_name = True  # Allow populating from_/from aliases

class ConfigRecord(BaseModel):
    """Record for storing configurations."""
    name: str
    version: str
    description: str
    config: str

def parse_graph_config(config_str: str) -> GraphConfig:
    """Parse graph configuration from YAML string."""
    import yaml
    config_dict = yaml.safe_load(config_str)
    
    # Process the config to match our model expectations
    processed_config = config_dict.copy()
    
    # Process nodes to ensure tool references are properly handled
    if "nodes" in processed_config:
        for node in processed_config["nodes"]:
            if "config" in node and "tools" in node["config"]:
                # Ensure tools is a list of strings (tool names)
                if isinstance(node["config"]["tools"], list):
                    # Convert any non-string entries to strings if needed
                    node["config"]["tools"] = [
                        tool if isinstance(tool, str) else tool["name"] 
                        for tool in node["config"]["tools"]
                    ]
            
            # Ensure prompt is properly handled, especially for multi-line strings
            if "config" in node and "prompt" in node["config"]:
                # YAML loader should already handle |, >, etc. notation properly,
                # but we can add additional processing if needed
                pass
    
    return GraphConfig(**processed_config) 