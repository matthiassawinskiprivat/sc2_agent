import os
import yaml
from typing import List, Dict, Any

class ConfigManager:
    """
    Manages loading and merging of level configurations, specifically for additive properties.
    """
    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        self.pipeline_config = self._load_yaml("pipeline.yaml")
        self.levels = self.pipeline_config.get("levels", [])

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.config_dir, filename)
        if not os.path.exists(path):
            return {}
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def get_cumulative_active_layers(self, target_level: str) -> List[str]:
        """
        Returns the cumulative list of active layers up to the target level.
        Iterates through the levels defined in pipeline.yaml.
        """
        accumulated_layers = []
        
        # Determine if we should process levels strictly from pipeline list
        # or just try to load the target level if not in pipeline.
        
        found_target = False
        
        for level in self.levels:
            level_conf = self._load_yaml(f"{level}.yaml")
            new_layers = level_conf.get("env", {}).get("active_layers", [])
            
            if new_layers:
                for layer in new_layers:
                    if layer not in accumulated_layers:
                        accumulated_layers.append(layer)
            
            if level == target_level:
                found_target = True
                break
        
        if not found_target:
            # If target level is not in pipeline list, try to load it directly
            # but we can't do accumulation effectively without knowing its predecessors.
            # We assume it has no predecessors in this case or is standalone.
            level_conf = self._load_yaml(f"{target_level}.yaml")
            accumulated_layers = level_conf.get("env", {}).get("active_layers", [])

        return accumulated_layers

    def get_cumulative_categorical_heads(self, target_level: str) -> List[str]:
        """
        Returns the cumulative list of categorical heads up to the target level.
        Iterates through the levels defined in pipeline.yaml.
        """
        accumulated_heads = []
        found_target = False
        
        for level in self.levels:
            level_conf = self._load_yaml(f"{level}.yaml")
            new_heads = level_conf.get("env", {}).get("categorical_heads", [])
            
            if new_heads:
                for head in new_heads:
                    if head not in accumulated_heads:
                        accumulated_heads.append(head)
            
            if level == target_level:
                found_target = True
                break
        
        if not found_target:
            level_conf = self._load_yaml(f"{target_level}.yaml")
            new_heads = level_conf.get("env", {}).get("categorical_heads", [])
            if new_heads:
                for head in new_heads:
                    if head not in accumulated_heads:
                        accumulated_heads.append(head)

        return accumulated_heads

    def get_cumulative_action_heads(self, target_level: str) -> List[str]:
        """
        Returns the cumulative list of action heads up to the target level.
        Iterates through the levels defined in pipeline.yaml.
        If target_level is not in pipeline, it assumes it follows the entire pipeline.
        """
        accumulated_heads = []
        found_target = False
        
        for level in self.levels:
            level_conf = self._load_yaml(f"{level}.yaml")
            new_heads = level_conf.get("env", {}).get("action_heads", [])
            
            if new_heads:
                for head in new_heads:
                    if head not in accumulated_heads:
                        accumulated_heads.append(head)
            
            if level == target_level:
                found_target = True
                break
        
        if not found_target:
            # If target level is not in pipeline list, try to load it directly
            # and append its heads to the accumulated list (assuming it follows pipeline)
            level_conf = self._load_yaml(f"{target_level}.yaml")
            new_heads = level_conf.get("env", {}).get("action_heads", [])
            
            if new_heads:
                for head in new_heads:
                    if head not in accumulated_heads:
                        accumulated_heads.append(head)

        return accumulated_heads

    def get_level_config(self, level_name: str) -> Dict[str, Any]:
        """Loads and returns the configuration for a specific level."""
        return self._load_yaml(f"{level_name}.yaml")
