import os
import yaml
import logging
from typing import Dict, Any, List, Optional

class ConfigManager:
    """
    Manages the configuration for the trading bot.
    Handles loading, validation, and access to configuration parameters.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config = {}
        
        # Load default configuration
        default_config_path = os.path.join(os.path.dirname(__file__), 'default_config.yaml')
        self._load_config(default_config_path)
        
        # Override with custom configuration if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path, override=True)
    
    def _load_config(self, config_path: str, override: bool = False) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to configuration file
            override: Whether to override existing configuration
        """
        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            if override:
                # Deep merge with existing configuration
                self._deep_merge(self.config, loaded_config)
            else:
                self.config = loaded_config
            
            self.logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            self.logger.error(f"Error loading configuration from {config_path}: {e}")
            if not override:
                raise
    
    def _deep_merge(self, dest: Dict, src: Dict) -> None:
        """
        Deep merge two dictionaries, modifying dest in-place.
        
        Args:
            dest: Destination dictionary to merge into
            src: Source dictionary to merge from
        """
        for key, value in src.items():
            if key in dest and isinstance(dest[key], dict) and isinstance(value, dict):
                self._deep_merge(dest[key], value)
            else:
                dest[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using a dot-separated path.
        
        Args:
            key_path: Dot-separated path to the configuration value
            default: Default value to return if the key doesn't exist
            
        Returns:
            The configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using a dot-separated path.
        
        Args:
            key_path: Dot-separated path to the configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for i, key in enumerate(keys[:-1]):
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, config_path: str) -> None:
        """
        Save the current configuration to a YAML file.
        
        Args:
            config_path: Path to save the configuration to
        """
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            self.logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration to {config_path}: {e}")
            raise
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if the configuration is valid, False otherwise
        """
        # TODO: Implement validation logic
        return True