"""Tool registry for managing tool objects."""
from typing import Dict, Any, Callable, Optional, List, Union
import importlib
import sys

# Global registry to store tool objects by name
_tool_registry: Dict[str, Any] = {}

def register_tool(name: str, tool_object: Any) -> None:
    """Register a tool in the global registry.
    
    Args:
        name: The name of the tool
        tool_object: The tool object/function
    """
    _tool_registry[name] = tool_object
    
def get_tool(name: str) -> Optional[Any]:
    """Get a tool from the registry by name.
    
    Args:
        name: The name of the tool to retrieve
        
    Returns:
        The tool object if found, None otherwise
    """
    return _tool_registry.get(name)

def has_tool(name: str) -> bool:
    """Check if a tool exists in the registry.
    
    Args:
        name: The name of the tool to check
        
    Returns:
        True if the tool exists, False otherwise
    """
    return name in _tool_registry

def list_tools() -> List[str]:
    """List all registered tool names.
    
    Returns:
        A list of registered tool names
    """
    return list(_tool_registry.keys())

def clear_registry() -> None:
    """Clear all tools from the registry."""
    _tool_registry.clear()

def resolve_tool(name_or_path: str) -> Optional[Any]:
    """Resolve a tool by name or import path.
    
    This function tries multiple strategies:
    1. Look in the registry
    2. Try to import from module path
    3. Look in caller's globals
    
    Args:
        name_or_path: Tool name or import path
        
    Returns:
        The tool object if found, None otherwise
    """
    # First check the registry
    if has_tool(name_or_path):
        return get_tool(name_or_path)
    
    # Try to import if it has dot notation
    if '.' in name_or_path:
        try:
            module_parts = name_or_path.split('.')
            module_name = '.'.join(module_parts[:-1])
            function_name = module_parts[-1]
            module = importlib.import_module(module_name)
            return getattr(module, function_name)
        except (ImportError, AttributeError):
            pass
    
    # Try caller's globals
    try:
        frame = sys._getframe(1)
        if name_or_path in frame.f_globals:
            return frame.f_globals[name_or_path]
    except (AttributeError, ValueError):
        pass
    
    # Nothing found
    return None

def register_standard_tools() -> None:
    """Register the standard built-in tools."""
    from langgraph_evo.components.tools import add, multiply, divide
    
    register_tool("add", add)
    register_tool("multiply", multiply)
    register_tool("divide", divide) 