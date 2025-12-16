"""Configuration management for Weak Labels agent."""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from loguru import logger


class Config:
    """
    Configuration loader with validation and defaults.
    
    Loads from YAML and provides convenient attribute access.
    """
    
    def __init__(self, config_path: str = "configs/base.yaml"):
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        logger.info(f"Loading config from {config_path}")
        
        with open(config_path, 'r') as f:
            self.raw = yaml.safe_load(f)
        
        # Validate structure
        self._validate()
        
        # Create convenient accessors
        self.dataset = self.raw['dataset']
        self.paths = self.raw['paths']
        self.corpus = self.raw['corpus']
        self.bm25 = self.raw['bm25']
        self.dense = self.raw['dense']
        self.llm = self.raw['llm']
        self.agent = self.raw['agent']
        self.output = self.raw['output']
        
        logger.info(f"✓ Config loaded (dataset: {self.dataset['name']})")
    
    def _validate(self):
        """Validate required config sections exist."""
        required = [
            'dataset',
            'paths',
            'corpus',
            'bm25',
            'dense',
            'llm',
            'agent',
            'output'
        ]
        
        missing = [key for key in required if key not in self.raw]
        
        if missing:
            raise ValueError(
                f"Missing required config sections: {missing}\n"
                f"Check your configs/base.yaml file"
            )
        
        # Validate dataset structure
        if 'corpus' not in self.raw['dataset']:
            raise ValueError("dataset.corpus section missing")
        
        if 'queries' not in self.raw['dataset']:
            raise ValueError("dataset.queries section missing")
        
        # Validate required nested keys
        dataset_corpus_required = ['name', 'split']
        for key in dataset_corpus_required:
            if key not in self.raw['dataset']['corpus']:
                raise ValueError(f"dataset.corpus.{key} missing")
        
        dataset_queries_required = ['name', 'split']
        for key in dataset_queries_required:
            if key not in self.raw['dataset']['queries']:
                raise ValueError(f"dataset.queries.{key} missing")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value with dot notation support.
        
        Examples:
            cfg.get('llm.timeout', 120.0)
            cfg.get('agent.checkpoint_interval', 100)
        """
        keys = key.split('.')
        value = self.raw
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getitem__(self, key: str) -> Any:
        """Support dict-like access: cfg['dataset']"""
        return self._data[key]
    
    def __repr__(self) -> str:
        return f"Config(dataset={self.dataset['name']})"


def validate_paths(config: Config) -> bool:
    """
    Validate and create necessary directories.
    
    Args:
        config: Loaded configuration
    
    Returns:
        True if validation passes
    """
    paths_to_create = [
        config.paths['prepared_dir'],
        config.paths['hf_cache_dir'],
        config.agent['checkpoint_dir'],
        Path(config.output['qrels_path']).parent,
        Path(config.output['triples_path']).parent,
    ]
    
    for path in paths_to_create:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    return True


def load_config(config_path: str = "configs/base.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Config object
    """
    return Config(config_path)
