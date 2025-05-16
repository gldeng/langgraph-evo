"""Storage system for LangGraph workflow configurations.

This module provides a simple file-based storage system for saving
and loading workflow configurations.
"""

import os
import shutil
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import logging

from .schema import WorkflowConfig
from .serialization import (
    save_to_file,
    load_from_file,
    dict_to_config,
    SerializationError
)

# Set up logging
logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Exception raised for storage-related errors."""
    pass


class ConfigStorage:
    """File-based storage for workflow configurations.
    
    This class manages the storage and retrieval of workflow configurations
    using a simple file-based approach.
    """
    
    def __init__(self, base_path: Union[str, Path] = None):
        """Initialize the storage system.
        
        Args:
            base_path: The base directory for storing configurations. 
                       If None, uses a default location.
        """
        if base_path is None:
            # Default to a 'configs' directory in the current working directory
            base_path = Path(os.getcwd()) / "configs"
        self.base_path = Path(base_path)
        
        # Ensure the base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized ConfigStorage with base path: {self.base_path}")
    
    def save_config(self, config: WorkflowConfig, path: Optional[Union[str, Path]] = None) -> Path:
        """Save a workflow configuration to storage.
        
        Args:
            config: The workflow configuration to save
            path: Optional specific path to save to. If None, generates a path
                  based on the configuration ID.
                  
        Returns:
            The path where the configuration was saved
            
        Raises:
            StorageError: If saving fails
        """
        if path is None:
            # Generate filename based on config ID
            filename = f"{config.id}.json"
            path = self.base_path / filename
        else:
            path = Path(path)
            # Ensure the path is within the base path or is absolute
            if not path.is_absolute() and not str(path).startswith(str(self.base_path)):
                path = self.base_path / path
        
        try:
            save_to_file(config, path)
            logger.debug(f"Saved configuration '{config.id}' to {path}")
            return path
        except Exception as e:
            raise StorageError(f"Failed to save configuration: {str(e)}") from e
    
    def load_config(self, config_id: str) -> WorkflowConfig:
        """Load a workflow configuration by its ID.
        
        Args:
            config_id: The ID of the configuration to load
            
        Returns:
            The loaded workflow configuration
            
        Raises:
            StorageError: If loading fails or the configuration doesn't exist
        """
        path = self.base_path / f"{config_id}.json"
        
        if not path.exists():
            raise StorageError(f"Configuration '{config_id}' not found")
        
        try:
            config = load_from_file(path)
            logger.debug(f"Loaded configuration '{config_id}' from {path}")
            return config
        except Exception as e:
            raise StorageError(f"Failed to load configuration '{config_id}': {str(e)}") from e
    
    def load_from_path(self, path: Union[str, Path]) -> WorkflowConfig:
        """Load a workflow configuration from a specific path.
        
        Args:
            path: The path to load from
            
        Returns:
            The loaded workflow configuration
            
        Raises:
            StorageError: If loading fails
        """
        path = Path(path)
        
        try:
            config = load_from_file(path)
            logger.debug(f"Loaded configuration from {path}")
            return config
        except Exception as e:
            raise StorageError(f"Failed to load configuration from {path}: {str(e)}") from e
    
    def list_configs(self) -> List[Dict[str, Any]]:
        """List all available configurations with basic metadata.
        
        Returns:
            A list of dictionaries containing basic information about each configuration
        """
        configs = []
        
        for path in self.base_path.glob("*.json"):
            try:
                # Get file info
                stat = path.stat()
                modified_time = datetime.fromtimestamp(stat.st_mtime)
                
                # Try to read just enough to get the ID and name
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                config_info = {
                    "id": data.get("id", path.stem),
                    "name": data.get("name", path.stem),
                    "path": str(path),
                    "modified": modified_time.isoformat(),
                    "size": stat.st_size
                }
                
                # Add version if available
                if "version" in data:
                    config_info["version"] = data["version"]
                
                configs.append(config_info)
            except Exception as e:
                # Just log the error and continue
                logger.warning(f"Failed to read config metadata from {path}: {str(e)}")
        
        return configs
    
    def delete_config(self, config_id: str) -> bool:
        """Delete a configuration by its ID.
        
        Args:
            config_id: The ID of the configuration to delete
            
        Returns:
            True if the configuration was deleted, False if it didn't exist
            
        Raises:
            StorageError: If deletion fails
        """
        path = self.base_path / f"{config_id}.json"
        
        if not path.exists():
            return False
        
        try:
            path.unlink()
            logger.debug(f"Deleted configuration '{config_id}' from {path}")
            return True
        except Exception as e:
            raise StorageError(f"Failed to delete configuration '{config_id}': {str(e)}") from e
    
    def duplicate_config(self, config_id: str, new_id: Optional[str] = None) -> WorkflowConfig:
        """Create a duplicate of a configuration with a new ID.
        
        Args:
            config_id: The ID of the configuration to duplicate
            new_id: The ID for the duplicate. If None, generates one based on the original
            
        Returns:
            The duplicated configuration
            
        Raises:
            StorageError: If duplication fails
        """
        # Load the original
        config = self.load_config(config_id)
        
        # Generate a new ID if not provided
        if new_id is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            new_id = f"{config_id}_copy_{timestamp}"
        
        # Create a duplicate with the new ID
        config_dict = json.loads(config.json())
        config_dict["id"] = new_id
        config_dict["name"] = f"{config_dict.get('name', config_id)} (Copy)"
        
        # Convert back to a WorkflowConfig
        new_config = dict_to_config(config_dict)
        
        # Save the duplicate
        self.save_config(new_config)
        logger.debug(f"Duplicated configuration '{config_id}' to '{new_id}'")
        
        return new_config 