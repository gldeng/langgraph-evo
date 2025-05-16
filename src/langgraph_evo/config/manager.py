"""Configuration management interface for LangGraph workflow configurations.

This module provides a unified interface for working with workflow configurations,
combining schema validation, serialization, and storage operations.
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging

from .schema import WorkflowConfig, Node, Edge, NodeType
from .validation import validate_workflow_integrity
from .serialization import (
    to_json,
    from_json,
    dict_to_config,
    config_to_dict,
    SerializationError
)
from .storage import ConfigStorage, StorageError


# Set up logging
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class ConfigManager:
    """Unified interface for managing workflow configurations.
    
    This class provides a single entry point for all operations related to
    workflow configurations, including creation, validation, serialization,
    and storage.
    """
    
    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        """Initialize the configuration manager.
        
        Args:
            storage_path: The base path for configuration storage.
                          If None, uses the default path.
        """
        self.storage = ConfigStorage(storage_path)
        logger.debug(f"Initialized ConfigManager with storage path: {self.storage.base_path}")
    
    def create_config(self, config_data: Dict[str, Any]) -> WorkflowConfig:
        """Create a new workflow configuration.
        
        Args:
            config_data: Dictionary containing the configuration data
            
        Returns:
            The created WorkflowConfig instance
            
        Raises:
            ConfigError: If the configuration is invalid
        """
        try:
            # Convert to a WorkflowConfig
            config = dict_to_config(config_data)
            
            # Validate the configuration integrity
            is_valid, errors = validate_workflow_integrity(config)
            if not is_valid:
                error_msg = "Configuration validation failed:\n" + "\n".join(errors)
                raise ConfigError(error_msg)
            
            return config
        except Exception as e:
            # Wrap all exceptions in ConfigError
            if isinstance(e, ConfigError):
                raise
            raise ConfigError(f"Failed to create configuration: {str(e)}") from e
    
    def create_empty_config(self, config_id: str, name: str) -> WorkflowConfig:
        """Create a minimal empty configuration with just ID and name.
        
        Args:
            config_id: The ID for the new configuration
            name: The name for the new configuration
            
        Returns:
            A minimal WorkflowConfig instance
        """
        # Create a single input processor node
        input_node = Node(
            id="input_processor",
            type=NodeType.INPUT_PROCESSOR,
            description="Input processor node"
        )
        
        # Create a basic config with minimum required fields
        config_data = {
            "id": config_id,
            "name": name,
            "version": "1.0.0",
            "nodes": [input_node],
            "edges": [],
            "entry_point": "input_processor"
        }
        
        return self.create_config(config_data)
    
    def validate_config(self, config: WorkflowConfig) -> List[str]:
        """Validate a workflow configuration.
        
        Args:
            config: The configuration to validate
            
        Returns:
            A list of validation error messages, empty if valid
        """
        is_valid, errors = validate_workflow_integrity(config)
        return errors
    
    def save_config(self, config: WorkflowConfig) -> Path:
        """Save a workflow configuration.
        
        Args:
            config: The configuration to save
            
        Returns:
            The path where the configuration was saved
            
        Raises:
            ConfigError: If saving fails
        """
        try:
            # Validate before saving
            errors = self.validate_config(config)
            if errors:
                error_msg = "Cannot save invalid configuration:\n" + "\n".join(errors)
                raise ConfigError(error_msg)
            
            # Save using the storage system
            return self.storage.save_config(config)
        except Exception as e:
            if isinstance(e, ConfigError):
                raise
            raise ConfigError(f"Failed to save configuration: {str(e)}") from e
    
    def load_config(self, config_id: str) -> WorkflowConfig:
        """Load a workflow configuration by ID.
        
        Args:
            config_id: The ID of the configuration to load
            
        Returns:
            The loaded configuration
            
        Raises:
            ConfigError: If loading fails
        """
        try:
            return self.storage.load_config(config_id)
        except Exception as e:
            raise ConfigError(f"Failed to load configuration '{config_id}': {str(e)}") from e
    
    def list_configs(self) -> List[Dict[str, Any]]:
        """List all available configurations.
        
        Returns:
            A list of dictionaries containing basic information about each configuration
        """
        return self.storage.list_configs()
    
    def delete_config(self, config_id: str) -> bool:
        """Delete a configuration by ID.
        
        Args:
            config_id: The ID of the configuration to delete
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            ConfigError: If deletion fails
        """
        try:
            return self.storage.delete_config(config_id)
        except Exception as e:
            raise ConfigError(f"Failed to delete configuration '{config_id}': {str(e)}") from e
    
    def duplicate_config(self, config_id: str, new_id: Optional[str] = None) -> WorkflowConfig:
        """Create a duplicate of a configuration.
        
        Args:
            config_id: The ID of the configuration to duplicate
            new_id: Optional new ID for the duplicate
            
        Returns:
            The duplicated configuration
            
        Raises:
            ConfigError: If duplication fails
        """
        try:
            return self.storage.duplicate_config(config_id, new_id)
        except Exception as e:
            raise ConfigError(f"Failed to duplicate configuration '{config_id}': {str(e)}") from e
    
    def export_config_to_json(self, config: WorkflowConfig, pretty: bool = True) -> str:
        """Export a configuration to JSON string.
        
        Args:
            config: The configuration to export
            pretty: Whether to format the JSON with indentation
            
        Returns:
            A JSON string representation of the configuration
            
        Raises:
            ConfigError: If export fails
        """
        try:
            return to_json(config, pretty=pretty)
        except Exception as e:
            raise ConfigError(f"Failed to export configuration to JSON: {str(e)}") from e
    
    def import_config_from_json(self, json_str: str) -> WorkflowConfig:
        """Import a configuration from a JSON string.
        
        Args:
            json_str: The JSON string to import
            
        Returns:
            The imported configuration
            
        Raises:
            ConfigError: If import fails or validation fails
        """
        try:
            # Parse the JSON
            config = from_json(json_str)
            
            # Validate the configuration
            errors = self.validate_config(config)
            if errors:
                error_msg = "Imported configuration is invalid:\n" + "\n".join(errors)
                raise ConfigError(error_msg)
            
            return config
        except Exception as e:
            if isinstance(e, ConfigError):
                raise
            raise ConfigError(f"Failed to import configuration from JSON: {str(e)}") from e 