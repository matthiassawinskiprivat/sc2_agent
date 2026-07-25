import os
import yaml
from pathlib import Path
from typing import List, Dict, Any


def _project_root() -> Path:
    """Repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def _normalize_config_dir(config_dir: str) -> str:
    """
    SubprocVecEnv workers often have a different cwd than the main process. Relative
    ``config_dir`` must not depend on cwd or ``pipeline.yaml`` is missed, ``levels`` is
    empty, and cumulative ``action_heads`` collapses to a single level yaml (e.g. only
    ``attack_local`` → attack-only policy + degenerate spatial).
    """
    if os.path.isabs(config_dir):
        return os.path.normpath(config_dir)
    return os.path.normpath(str(_project_root() / config_dir))


class ConfigManager:
    """
    Manages loading and merging of level configurations, specifically for additive properties.
    """
    def __init__(self, config_dir: str = "config"):
        self.config_dir = _normalize_config_dir(config_dir)
        self._yaml_cache: Dict[str, Dict[str, Any]] = {}
        self.pipeline_config = self._load_yaml("pipeline.yaml")
        self.levels = self.pipeline_config.get("levels", [])

    def get_agent_architecture(self, level_name: str | None = None) -> str:
        """
        Returns agent architecture for a level.

        Per-level YAML may set ``agent_architecture`` at the root or under ``env``;
        otherwise falls back to ``pipeline.yaml``.
        """
        if level_name:
            level_conf = self.get_level_config(level_name)
            arch = level_conf.get("agent_architecture")
            if not arch:
                arch = level_conf.get("env", {}).get("agent_architecture")
            if arch:
                return str(arch)
        return self.pipeline_config.get("agent_architecture", "original")

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        if filename in self._yaml_cache:
            return self._yaml_cache[filename]
        path = os.path.join(self.config_dir, filename)
        if not os.path.exists(path):
            self._yaml_cache[filename] = {}
            return {}
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        parsed = data if isinstance(data, dict) else {}
        self._yaml_cache[filename] = parsed
        return parsed

    def get_cumulative_active_layers(self, target_level: str) -> List[str]:
        """
        Returns the cumulative list of active layers up to the target level.
        Iterates through the levels defined in pipeline.yaml.
        """
        accumulated_layers = []
        
        # Start with layers from pipeline root
        root_layers = self.pipeline_config.get("active_layers", [])
        for layer in root_layers:
            ln = layer.lower()
            if ln not in accumulated_layers:
                accumulated_layers.append(ln)

        found_target = False
        for level in self.levels:
            level_conf = self._load_yaml(f"{level}.yaml")
            new_layers = level_conf.get("env", {}).get("active_layers", [])
            
            if new_layers:
                for layer in new_layers:
                    ln = layer.lower()
                    if ln not in accumulated_layers:
                        accumulated_layers.append(ln)
            
            if level == target_level:
                found_target = True
                break
        
        if not found_target:
            level_conf = self._load_yaml(f"{target_level}.yaml")
            new_layers = level_conf.get("env", {}).get("active_layers", [])
            for layer in new_layers:
                ln = layer.lower()
                if ln not in accumulated_layers:
                    accumulated_layers.append(ln)

        return accumulated_layers

    def get_cumulative_active_tactical_layers(self, target_level: str) -> List[str]:
        """
        Returns the cumulative list of active tactical layers up to the target level.
        Iterates through the levels defined in pipeline.yaml.
        """
        accumulated_layers = []
        
        # Start with tactical layers from pipeline root
        root_layers = self.pipeline_config.get("active_tactical_layers", [])
        for layer in root_layers:
            ln = layer.lower()
            if ln not in accumulated_layers:
                accumulated_layers.append(ln)

        found_target = False
        for level in self.levels:
            level_conf = self._load_yaml(f"{level}.yaml")
            new_layers = level_conf.get("env", {}).get("active_tactical_layers", [])
            
            if new_layers:
                for layer in new_layers:
                    ln = layer.lower()
                    if ln not in accumulated_layers:
                        accumulated_layers.append(ln)
            
            if level == target_level:
                found_target = True
                break
        
        if not found_target:
            level_conf = self._load_yaml(f"{target_level}.yaml")
            new_layers = level_conf.get("env", {}).get("active_tactical_layers", [])
            for layer in new_layers:
                ln = layer.lower()
                if ln not in accumulated_layers:
                    accumulated_layers.append(ln)

        return accumulated_layers

    def get_cumulative_categorical_heads(self, target_level: str) -> List[str]:
        """
        Returns the cumulative list of categorical heads up to the target level.
        Iterates through the levels defined in pipeline.yaml.
        """
        accumulated_heads = []
        
        # Start with heads from pipeline root
        root_heads = self.pipeline_config.get("categorical_heads", [])
        for head in root_heads:
            hn = head.lower()
            if hn not in accumulated_heads:
                accumulated_heads.append(hn)

        found_target = False
        for level in self.levels:
            level_conf = self._load_yaml(f"{level}.yaml")
            new_heads = level_conf.get("env", {}).get("categorical_heads", [])
            
            if new_heads:
                for head in new_heads:
                    hn = head.lower()
                    if hn not in accumulated_heads:
                        accumulated_heads.append(hn)
            
            if level == target_level:
                found_target = True
                break
        
        if not found_target:
            level_conf = self._load_yaml(f"{target_level}.yaml")
            new_heads = level_conf.get("env", {}).get("categorical_heads", [])
            if new_heads:
                for head in new_heads:
                    hn = head.lower()
                    if hn not in accumulated_heads:
                        accumulated_heads.append(hn)

        return accumulated_heads

    def get_cumulative_action_heads(self, target_level: str) -> List[str]:
        """
        Returns the cumulative list of action heads up to the target level.
        Iterates through the levels defined in pipeline.yaml.
        If target_level is not in pipeline, it assumes it follows the entire pipeline.
        """
        accumulated_heads = []
        
        # Start with heads from pipeline root
        root_heads = self.pipeline_config.get("action_heads", [])
        for head in root_heads:
            hn = head.lower()
            if hn not in accumulated_heads:
                accumulated_heads.append(hn)

        found_target = False
        for level in self.levels:
            level_conf = self._load_yaml(f"{level}.yaml")
            new_heads = level_conf.get("env", {}).get("action_heads", [])
            
            if new_heads:
                for head in new_heads:
                    hn = head.lower()
                    if hn not in accumulated_heads:
                        accumulated_heads.append(hn)
            
            if level == target_level:
                found_target = True
                break
        
        if not found_target:
            level_conf = self._load_yaml(f"{target_level}.yaml")
            new_heads = level_conf.get("env", {}).get("action_heads", [])
            
            if new_heads:
                for head in new_heads:
                    hn = head.lower()
                    if hn not in accumulated_heads:
                        accumulated_heads.append(hn)

        return accumulated_heads

    def get_cumulative_build_map_layers(self, target_level: str) -> List[str]:
        """Cumulative build-agent local map layers (separate from micro active_layers)."""
        accumulated: List[str] = []
        root_layers = self.pipeline_config.get("build_map_layers", [])
        for layer in root_layers:
            ln = layer.lower()
            if ln not in accumulated:
                accumulated.append(ln)

        found_target = False
        for level in self.levels:
            level_conf = self._load_yaml(f"{level}.yaml")
            new_layers = level_conf.get("env", {}).get("build_map_layers", [])
            for layer in new_layers:
                ln = layer.lower()
                if ln not in accumulated:
                    accumulated.append(ln)
            if level == target_level:
                found_target = True
                break

        if not found_target:
            level_conf = self._load_yaml(f"{target_level}.yaml")
            for layer in level_conf.get("env", {}).get("build_map_layers", []):
                ln = layer.lower()
                if ln not in accumulated:
                    accumulated.append(ln)
        return accumulated

    def get_cumulative_build_commands(self, target_level: str) -> List[str]:
        """Allowed structure/unit types for scripted manager per curriculum level."""
        accumulated: List[str] = []
        root_cmds = self.pipeline_config.get("build_commands", [])
        for cmd in root_cmds:
            cn = cmd.lower()
            if cn not in accumulated:
                accumulated.append(cn)

        found_target = False
        for level in self.levels:
            level_conf = self._load_yaml(f"{level}.yaml")
            new_cmds = level_conf.get("env", {}).get("build_commands", [])
            for cmd in new_cmds:
                cn = cmd.lower()
                if cn not in accumulated:
                    accumulated.append(cn)
            if level == target_level:
                found_target = True
                break

        if not found_target:
            level_conf = self._load_yaml(f"{target_level}.yaml")
            for cmd in level_conf.get("env", {}).get("build_commands", []):
                cn = cmd.lower()
                if cn not in accumulated:
                    accumulated.append(cn)
        return accumulated

    def get_cumulative_active_unit_scalars(self, target_level: str) -> List[str]:
        """
        Cumulative per-unit scalar feature names for the squad pointer key.

        Same contract as ``active_layers``: pipeline-root list first, then each
        level's ``env.active_unit_scalars`` appended in order; duplicates skipped
        (case-insensitive). Config order IS observation feature order.
        """
        accumulated: List[str] = []

        root = self.pipeline_config.get("active_unit_scalars", [])
        for name in root:
            ln = str(name).lower()
            if ln not in accumulated:
                accumulated.append(ln)

        found_target = False
        for level in self.levels:
            level_conf = self._load_yaml(f"{level}.yaml")
            new_feats = level_conf.get("env", {}).get("active_unit_scalars", [])
            if new_feats:
                for name in new_feats:
                    ln = str(name).lower()
                    if ln not in accumulated:
                        accumulated.append(ln)
            if level == target_level:
                found_target = True
                break

        if not found_target:
            level_conf = self._load_yaml(f"{target_level}.yaml")
            for name in level_conf.get("env", {}).get("active_unit_scalars", []) or []:
                ln = str(name).lower()
                if ln not in accumulated:
                    accumulated.append(ln)

        return accumulated

    def get_level_config(self, level_name: str) -> Dict[str, Any]:
        """Loads and returns the configuration for a specific level."""
        return self._load_yaml(f"{level_name}.yaml")
