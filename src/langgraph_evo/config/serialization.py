"""Serialization utilities for LangGraph workflow configurations.

This module provides functions for serializing and deserializing workflow
configurations to various formats, with a focus on JSON.
"""

import json
import os
from typing import Dict, Any, Optional, Union, TextIO, BinaryIO
from pathlib import Path
import traceback

from .schema import WorkflowConfig


class SerializationError(Exception):
    """Exception raised for serialization-related errors."""
    pass


def to_json(config: WorkflowConfig, pretty: bool = False) -> str:
    """Serialize a workflow configuration to JSON.
    
    Args:
        config: The workflow configuration to serialize
        pretty: Whether to format the JSON with indentation for readability
        
    Returns:
        A JSON string representation of the configuration
        
    Raises:
        SerializationError: If serialization fails
    """
    try:
        if pretty:
            return config.json(indent=2, exclude_unset=True)
        return config.json(exclude_unset=True)
    except Exception as e:
        raise SerializationError(f"Failed to serialize configuration: {str(e)}") from e


def from_json(json_str: str) -> WorkflowConfig:
    """Deserialize a JSON string to a workflow configuration.
    
    Args:
        json_str: The JSON string to deserialize
        
    Returns:
        A WorkflowConfig instance
        
    Raises:
        SerializationError: If deserialization fails
    """
    try:
        return WorkflowConfig.parse_raw(json_str)
    except Exception as e:
        # Provide more helpful error message
        error_msg = f"Failed to deserialize JSON: {str(e)}"
        try:
            # Try to parse the JSON to see if it's valid JSON but invalid schema
            parsed = json.loads(json_str)
            error_msg = f"JSON is valid but does not match schema: {str(e)}"
        except json.JSONDecodeError:
            error_msg = f"Invalid JSON format: {str(e)}"
        
        raise SerializationError(error_msg) from e


def save_to_file(config: WorkflowConfig, path: Union[str, Path], pretty: bool = True) -> None:
    """Save a workflow configuration to a file.
    
    Args:
        config: The workflow configuration to save
        path: The file path to save to
        pretty: Whether to format the JSON with indentation
        
    Raises:
        SerializationError: If saving fails
    """
    try:
        # Ensure directory exists
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(to_json(config, pretty=pretty))
    except Exception as e:
        raise SerializationError(f"Failed to save configuration to {path}: {str(e)}") from e


def load_from_file(path: Union[str, Path]) -> WorkflowConfig:
    """Load a workflow configuration from a file.
    
    Args:
        path: The file path to load from
        
    Returns:
        A WorkflowConfig instance
        
    Raises:
        SerializationError: If loading fails
        FileNotFoundError: If the file does not exist
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            json_str = f.read()
        return from_json(json_str)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise SerializationError(f"Failed to load configuration from {path}: {str(e)}") from e


def dict_to_config(data: Dict[str, Any]) -> WorkflowConfig:
    """Convert a dictionary to a workflow configuration.
    
    Args:
        data: The dictionary to convert
        
    Returns:
        A WorkflowConfig instance
        
    Raises:
        SerializationError: If conversion fails
    """
    try:
        return WorkflowConfig.parse_obj(data)
    except Exception as e:
        raise SerializationError(f"Failed to convert dictionary to WorkflowConfig: {str(e)}") from e


def config_to_dict(config: WorkflowConfig, exclude_defaults: bool = False) -> Dict[str, Any]:
    """Convert a workflow configuration to a dictionary.
    
    Args:
        config: The workflow configuration to convert
        exclude_defaults: Whether to exclude fields with default values
        
    Returns:
        A dictionary representation of the configuration
        
    Raises:
        SerializationError: If conversion fails
    """
    try:
        if exclude_defaults:
            return json.loads(config.json(exclude_unset=True))
        return json.loads(config.json())
    except Exception as e:
        raise SerializationError(f"Failed to convert WorkflowConfig to dictionary: {str(e)}") from e 