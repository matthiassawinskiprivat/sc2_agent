import os
import glob
from typing import List, Dict, Any
from src.utils.config_manager import ConfigManager

class ConfigResolver:
    """
    Resolves level and map configurations for the Imitation Learning Workstation.
    """
    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.config_dir = os.path.join(project_root, "config")
        self.maps_root = os.path.join(project_root, "train", "maps")
        self.config_manager = ConfigManager(self.config_dir)

    def list_levels(self) -> List[str]:
        """
        Lists all available levels by scanning the config directory.
        Excludes pipeline.yaml and non-yaml files.
        """
        level_files = glob.glob(os.path.join(self.config_dir, "*.yaml"))
        levels = []
        for f in level_files:
            name = os.path.splitext(os.path.basename(f))[0]
            if name != "pipeline":
                levels.append(name)
        return sorted(levels)

    def list_maps(self, level_name: str) -> List[str]:
        """
        Lists all available maps for a specific level.
        Maps are expected to be in train/maps/<level_name>/.
        """
        level_maps_dir = os.path.join(self.maps_root, level_name)
        if not os.path.exists(level_maps_dir):
            return []
        
        map_files = glob.glob(os.path.join(level_maps_dir, "*.SC2Map"))
        maps = [os.path.splitext(os.path.basename(f))[0] for f in map_files]
        return sorted(maps)

    def get_cumulative_config(self, level_name: str) -> Dict[str, Any]:
        """
        Returns the merged configuration for the level, including additive layers and heads.
        """
        layers = self.config_manager.get_cumulative_active_layers(level_name)
        action_heads = self.config_manager.get_cumulative_action_heads(level_name)
        categorical_heads = self.config_manager.get_cumulative_categorical_heads(level_name)
        
        # Get base config for other parameters (rewards, etc.)
        base_config = self.config_manager.get_level_config(level_name)
        
        # Merge cumulative properties into the config
        if "env" not in base_config:
            base_config["env"] = {}
            
        base_config["env"]["active_layers"] = layers
        base_config["env"]["action_heads"] = action_heads
        base_config["env"]["categorical_heads"] = categorical_heads
        
        # Also expose at top level for convenience if needed by some components
        base_config["active_layers"] = layers
        base_config["action_heads"] = action_heads
        base_config["categorical_heads"] = categorical_heads
        
        return base_config
