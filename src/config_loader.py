"""
Configuration loader for research paper tagger.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


class ConfigLoader:
    """Load and manage configuration settings."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_default_config()
        
        if config_path and config_path.exists():
            self._load_custom_config(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'processing': {
                'confidence_threshold': 0.3,
                'max_tags': 5,
                'extraction_method': 'auto'
            },
            'output': {
                'default_format': 'csv',
                'create_summary': True,
                'organize_by_tags': False,
                'rename_with_tags': False,
                'max_filename_tags': 2
            },
            'categories': {},
            'advanced': {
                'remove_stopwords': True,
                'use_lemmatization': True,
                'min_word_length': 3,
                'use_tfidf': True,
                'tfidf_max_features': 10000,
                'tfidf_ngram_range': [1, 3],
                'tfidf_min_df': 2,
                'tfidf_max_df': 0.95,
                'tfidf_weight': 0.7,
                'keyword_weight': 0.3,
                'save_model': False,
                'model_path': 'models/classifier.pkl'
            }
        }
    
    def _load_custom_config(self, config_path: Path):
        """Load custom configuration from file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                custom_config = yaml.safe_load(f)
            
            # Merge configurations (custom overrides default)
            self._merge_configs(self.config, custom_config)
            self.logger.info(f"Loaded custom configuration from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load custom config: {e}")
    
    def _merge_configs(self, default: Dict[str, Any], custom: Dict[str, Any]):
        """Recursively merge custom config into default config."""
        for key, value in custom.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'processing.confidence_threshold')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_categories(self) -> Dict[str, list]:
        """Get custom categories configuration."""
        return self.config.get('categories', {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration."""
        return self.config.get('processing', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.config.get('output', {})
    
    def get_advanced_config(self) -> Dict[str, Any]:
        """Get advanced configuration."""
        return self.config.get('advanced', {})
    
    def save_config(self, output_path: Path):
        """Save current configuration to file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            self.logger.info(f"Saved configuration to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
    
    def validate_config(self) -> bool:
        """
        Validate configuration values.
        
        Returns:
            True if config is valid, False otherwise
        """
        try:
            # Validate processing config
            processing = self.get_processing_config()
            confidence = processing.get('confidence_threshold', 0.3)
            if not 0.0 <= confidence <= 1.0:
                self.logger.error("confidence_threshold must be between 0.0 and 1.0")
                return False
            
            max_tags = processing.get('max_tags', 5)
            if not isinstance(max_tags, int) or max_tags <= 0:
                self.logger.error("max_tags must be a positive integer")
                return False
            
            extraction_method = processing.get('extraction_method', 'auto')
            valid_methods = ['auto', 'pymupdf', 'pdfplumber', 'pypdf2']
            if extraction_method not in valid_methods:
                self.logger.error(f"extraction_method must be one of {valid_methods}")
                return False
            
            # Validate output config
            output = self.get_output_config()
            output_format = output.get('default_format', 'csv')
            valid_formats = ['csv', 'json']
            if output_format not in valid_formats:
                self.logger.error(f"default_format must be one of {valid_formats}")
                return False
            
            # Validate advanced config
            advanced = self.get_advanced_config()
            tfidf_weight = advanced.get('tfidf_weight', 0.7)
            keyword_weight = advanced.get('keyword_weight', 0.3)
            
            if not 0.0 <= tfidf_weight <= 1.0:
                self.logger.error("tfidf_weight must be between 0.0 and 1.0")
                return False
            
            if not 0.0 <= keyword_weight <= 1.0:
                self.logger.error("keyword_weight must be between 0.0 and 1.0")
                return False
            
            if abs(tfidf_weight + keyword_weight - 1.0) > 0.01:
                self.logger.warning("tfidf_weight and keyword_weight should sum to 1.0")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Config validation failed: {e}")
            return False
