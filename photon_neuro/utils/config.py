"""
Configuration Management System
==============================
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict

@dataclass
class PhotonConfig:
    """Main configuration class for Photon Neuromorphics."""
    
    # Core settings
    log_level: str = "INFO"
    cache_size: int = 1000
    max_workers: int = 4
    timeout_seconds: float = 30.0
    
    # Performance settings
    enable_simd: bool = True
    optimize_memory: bool = True
    parallel_processing: bool = True
    
    # Hardware settings
    hardware_interface_enabled: bool = False
    calibration_interval: float = 3600.0  # 1 hour
    
    # Quality settings
    quality_gates_enabled: bool = True
    monitoring_enabled: bool = True
    alert_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "error_rate": 0.05
            }

class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config = PhotonConfig()
        self.load_config()
    
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in standard locations."""
        possible_paths = [
            "photon_config.yaml",
            "photon_config.json", 
            "config/photon.yaml",
            os.path.expanduser("~/.photon_neuro/config.yaml")
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        return None
    
    def load_config(self):
        """Load configuration from file."""
        if not self.config_path or not Path(self.config_path).exists():
            return
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    # Would use yaml.safe_load(f) if yaml module available
                    data = {}
                else:
                    data = json.load(f)
            
            # Update config with loaded data
            for key, value in data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
    
    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = path or self.config_path or "photon_config.json"
        
        try:
            config_dict = asdict(self.config)
            
            with open(save_path, 'w') as f:
                if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                    # Would use yaml.dump if yaml module available
                    f.write("# Photon Neuro Configuration\n")
                    for key, value in config_dict.items():
                        f.write(f"{key}: {value}\n")
                else:
                    json.dump(config_dict, f, indent=2)
                    
        except Exception as e:
            print(f"Warning: Could not save config to {save_path}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return getattr(self.config, key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        if hasattr(self.config, key):
            setattr(self.config, key, value)

# Global config instance
global_config = ConfigManager()