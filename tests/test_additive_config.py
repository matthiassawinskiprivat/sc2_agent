import pytest
import yaml
import numpy as np
from unittest.mock import MagicMock, patch
from src.utils.config_manager import ConfigManager
from train.envs.level_0_env import Level0Env


# ---------------------------------------------------------------------------
# Helper: create a minimal pipeline on disk under tmp_path
# ---------------------------------------------------------------------------
def _write_yaml(tmp_path, filename, data):
    (tmp_path / filename).write_text(yaml.dump(data))


def _make_pipeline(tmp_path, levels, **root_keys):
    """Write pipeline.yaml + per-level files.

    ``levels`` is a dict mapping level name → level YAML dict.
    Extra kwargs are written as pipeline-root keys (e.g. active_layers).
    """
    pipeline = {"levels": list(levels.keys()), **root_keys}
    _write_yaml(tmp_path, "pipeline.yaml", pipeline)
    for name, conf in levels.items():
        _write_yaml(tmp_path, f"{name}.yaml", conf)


# ===========================================================================
# get_cumulative_active_layers
# ===========================================================================

class TestCumulativeActiveLayers:
    def test_basic_accumulation(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"active_layers": ["A", "B"]}},
            "lvl1": {"env": {"active_layers": ["C"]}},
            "lvl2": {"env": {}},
        })
        cm = ConfigManager(str(tmp_path))

        assert cm.get_cumulative_active_layers("lvl0") == ["a", "b"]
        assert cm.get_cumulative_active_layers("lvl1") == ["a", "b", "c"]
        assert cm.get_cumulative_active_layers("lvl2") == ["a", "b", "c"]

    def test_duplicates_skipped(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"active_layers": ["A", "B"]}},
            "lvl1": {"env": {"active_layers": ["B", "C"]}},
        })
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_active_layers("lvl1") == ["a", "b", "c"]

    def test_case_insensitive_dedup(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"active_layers": ["Foo", "BAR"]}},
            "lvl1": {"env": {"active_layers": ["foo", "baz"]}},
        })
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_active_layers("lvl1") == ["foo", "bar", "baz"]

    def test_off_pipeline_level(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"active_layers": ["A", "B"]}},
        })
        _write_yaml(tmp_path, "extra.yaml", {"env": {"active_layers": ["C", "D"]}})
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_active_layers("extra") == ["a", "b", "c", "d"]

    def test_pipeline_root_seeds(self, tmp_path):
        _make_pipeline(tmp_path,
            {"lvl0": {"env": {"active_layers": ["C"]}}},
            active_layers=["A", "B"],
        )
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_active_layers("lvl0") == ["a", "b", "c"]

    def test_order_preserved_across_levels(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"active_layers": ["X", "Y"]}},
            "lvl1": {"env": {"active_layers": ["Z"]}},
            "lvl2": {"env": {"active_layers": ["W"]}},
        })
        cm = ConfigManager(str(tmp_path))
        result = cm.get_cumulative_active_layers("lvl2")
        assert result == ["x", "y", "z", "w"]
        assert result[:2] == cm.get_cumulative_active_layers("lvl0")

    def test_empty_pipeline_no_levels(self, tmp_path):
        _write_yaml(tmp_path, "pipeline.yaml", {"levels": []})
        _write_yaml(tmp_path, "solo.yaml", {"env": {"active_layers": ["A"]}})
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_active_layers("solo") == ["a"]

    def test_level_with_no_env_section(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"active_layers": ["A"]}},
            "lvl1": {"training": {"steps": 100}},
        })
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_active_layers("lvl1") == ["a"]


class TestConfigManagerYamlCache:
    def test_load_yaml_is_cached_per_instance(self, tmp_path):
        _make_pipeline(tmp_path, {"lvl0": {"env": {"active_layers": ["A"]}}})
        cm = ConfigManager(str(tmp_path))
        first = cm._load_yaml("lvl0.yaml")
        first["env"]["active_layers"].append("mutated")
        second = cm._load_yaml("lvl0.yaml")
        assert second is first
        assert "mutated" in second["env"]["active_layers"]


# ===========================================================================
# get_cumulative_active_tactical_layers
# ===========================================================================

class TestCumulativeActiveTacticalLayers:
    def test_basic_accumulation(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"active_tactical_layers": ["T1", "T2"]}},
            "lvl1": {"env": {"active_tactical_layers": ["T3"]}},
        })
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_active_tactical_layers("lvl0") == ["t1", "t2"]
        assert cm.get_cumulative_active_tactical_layers("lvl1") == ["t1", "t2", "t3"]

    def test_duplicates_skipped(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"active_tactical_layers": ["A", "B"]}},
            "lvl1": {"env": {"active_tactical_layers": ["B", "C"]}},
        })
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_active_tactical_layers("lvl1") == ["a", "b", "c"]

    def test_independent_from_active_layers(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {
                "active_layers": ["G1", "G2"],
                "active_tactical_layers": ["T1"],
            }},
        })
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_active_layers("lvl0") == ["g1", "g2"]
        assert cm.get_cumulative_active_tactical_layers("lvl0") == ["t1"]

    def test_off_pipeline_level(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"active_tactical_layers": ["T1"]}},
        })
        _write_yaml(tmp_path, "extra.yaml", {"env": {"active_tactical_layers": ["T2"]}})
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_active_tactical_layers("extra") == ["t1", "t2"]

    def test_pipeline_root_seeds(self, tmp_path):
        _make_pipeline(tmp_path,
            {"lvl0": {"env": {"active_tactical_layers": ["T2"]}}},
            active_tactical_layers=["T1"],
        )
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_active_tactical_layers("lvl0") == ["t1", "t2"]

    def test_inheritance_no_new_layers(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"active_tactical_layers": ["T1", "T2"]}},
            "lvl1": {"env": {}},
            "lvl2": {"env": {}},
        })
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_active_tactical_layers("lvl2") == ["t1", "t2"]


# ===========================================================================
# get_cumulative_active_unit_scalars
# ===========================================================================

class TestCumulativeActiveUnitScalars:
    def test_basic_accumulation(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"active_unit_scalars": ["health_pct", "x_norm"]}},
            "lvl1": {"env": {"active_unit_scalars": ["role_attack"]}},
            "lvl2": {"env": {}},
        })
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_active_unit_scalars("lvl0") == ["health_pct", "x_norm"]
        assert cm.get_cumulative_active_unit_scalars("lvl1") == [
            "health_pct", "x_norm", "role_attack"
        ]
        assert cm.get_cumulative_active_unit_scalars("lvl2") == [
            "health_pct", "x_norm", "role_attack"
        ]

    def test_duplicates_skipped(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"active_unit_scalars": ["health_pct", "x_norm"]}},
            "lvl1": {"env": {"active_unit_scalars": ["x_norm", "role_defend"]}},
        })
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_active_unit_scalars("lvl1") == [
            "health_pct", "x_norm", "role_defend"
        ]

    def test_case_insensitive_dedup(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"active_unit_scalars": ["Health_Pct", "X_NORM"]}},
            "lvl1": {"env": {"active_unit_scalars": ["health_pct", "role_attack"]}},
        })
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_active_unit_scalars("lvl1") == [
            "health_pct", "x_norm", "role_attack"
        ]

    def test_pipeline_root_seeds(self, tmp_path):
        _make_pipeline(
            tmp_path,
            {"lvl0": {"env": {"active_unit_scalars": ["role_attack"]}}},
            active_unit_scalars=["health_pct", "x_norm"],
        )
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_active_unit_scalars("lvl0") == [
            "health_pct", "x_norm", "role_attack"
        ]

    def test_order_preserved_across_levels(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"active_unit_scalars": ["health_pct", "shield_pct"]}},
            "lvl1": {"env": {"active_unit_scalars": ["is_melee"]}},
            "lvl2": {"env": {"active_unit_scalars": ["role_defend"]}},
        })
        cm = ConfigManager(str(tmp_path))
        result = cm.get_cumulative_active_unit_scalars("lvl2")
        assert result == ["health_pct", "shield_pct", "is_melee", "role_defend"]
        assert result[:2] == cm.get_cumulative_active_unit_scalars("lvl0")

    def test_independent_from_layers(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {
                "active_layers": ["g1"],
                "active_unit_scalars": ["health_pct"],
            }},
        })
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_active_layers("lvl0") == ["g1"]
        assert cm.get_cumulative_active_unit_scalars("lvl0") == ["health_pct"]


# ===========================================================================
# get_cumulative_action_heads
# ===========================================================================

class TestCumulativeActionHeads:
    def test_basic_accumulation(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"action_heads": ["move"]}},
            "lvl1": {"env": {"action_heads": ["attack"]}},
            "lvl2": {"env": {"action_heads": ["patrol"]}},
        })
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_action_heads("lvl0") == ["move"]
        assert cm.get_cumulative_action_heads("lvl1") == ["move", "attack"]
        assert cm.get_cumulative_action_heads("lvl2") == ["move", "attack", "patrol"]

    def test_off_pipeline_level(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"action_heads": ["move"]}},
            "lvl1": {"env": {"action_heads": ["attack"]}},
        })
        _write_yaml(tmp_path, "lvl_drop.yaml", {"env": {"action_heads": ["load", "unload"]}})
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_action_heads("lvl_drop") == ["move", "attack", "load", "unload"]

    def test_duplicates_skipped(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"action_heads": ["move", "attack"]}},
            "lvl1": {"env": {"action_heads": ["attack", "patrol"]}},
        })
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_action_heads("lvl1") == ["move", "attack", "patrol"]

    def test_pipeline_root_seeds(self, tmp_path):
        _make_pipeline(tmp_path,
            {"lvl0": {"env": {"action_heads": ["attack"]}}},
            action_heads=["move"],
        )
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_action_heads("lvl0") == ["move", "attack"]


# ===========================================================================
# get_cumulative_categorical_heads
# ===========================================================================

class TestCumulativeCategoricalHeads:
    def test_basic_accumulation(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"categorical_heads": ["stop"]}},
            "lvl1": {"env": {"categorical_heads": ["hold"]}},
        })
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_categorical_heads("lvl0") == ["stop"]
        assert cm.get_cumulative_categorical_heads("lvl1") == ["stop", "hold"]

    def test_duplicates_skipped(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"categorical_heads": ["stop"]}},
            "lvl1": {"env": {"categorical_heads": ["stop", "hold"]}},
        })
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_categorical_heads("lvl1") == ["stop", "hold"]

    def test_off_pipeline_level(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"categorical_heads": ["stop"]}},
        })
        _write_yaml(tmp_path, "extra.yaml", {"env": {"categorical_heads": ["hold"]}})
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_categorical_heads("extra") == ["stop", "hold"]

    def test_pipeline_root_seeds(self, tmp_path):
        _make_pipeline(tmp_path,
            {"lvl0": {"env": {}}},
            categorical_heads=["stop"],
        )
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_categorical_heads("lvl0") == ["stop"]

    def test_empty_when_none_defined(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {}},
        })
        cm = ConfigManager(str(tmp_path))
        assert cm.get_cumulative_categorical_heads("lvl0") == []


# ===========================================================================
# get_agent_architecture
# ===========================================================================

class TestGetAgentArchitecture:
    def test_default_is_original(self, tmp_path):
        _write_yaml(tmp_path, "pipeline.yaml", {"levels": []})
        cm = ConfigManager(str(tmp_path))
        assert cm.get_agent_architecture() == "original"

    def test_explicit_universal(self, tmp_path):
        _write_yaml(tmp_path, "pipeline.yaml", {"levels": [], "agent_architecture": "universal"})
        cm = ConfigManager(str(tmp_path))
        assert cm.get_agent_architecture() == "universal"

    def test_level_override_beats_pipeline(self, tmp_path):
        _write_yaml(tmp_path, "pipeline.yaml", {"levels": ["ctf0"], "agent_architecture": "universal"})
        _write_yaml(tmp_path, "ctf0.yaml", {"agent_architecture": "squad", "env": {}})
        cm = ConfigManager(str(tmp_path))
        assert cm.get_agent_architecture("ctf0") == "squad"
        assert cm.get_agent_architecture() == "universal"

    def test_level_env_nested_architecture(self, tmp_path):
        _write_yaml(tmp_path, "pipeline.yaml", {"levels": ["ctf1"], "agent_architecture": "universal"})
        _write_yaml(tmp_path, "ctf1.yaml", {"env": {"agent_architecture": "squad"}})
        cm = ConfigManager(str(tmp_path))
        assert cm.get_agent_architecture("ctf1") == "squad"


# ===========================================================================
# get_level_config
# ===========================================================================

class TestGetLevelConfig:
    def test_returns_level_dict(self, tmp_path):
        _make_pipeline(tmp_path, {
            "lvl0": {"env": {"targets": 3}, "training": {"steps": 1000}},
        })
        cm = ConfigManager(str(tmp_path))
        cfg = cm.get_level_config("lvl0")
        assert cfg["env"]["targets"] == 3
        assert cfg["training"]["steps"] == 1000

    def test_missing_level_returns_empty(self, tmp_path):
        _write_yaml(tmp_path, "pipeline.yaml", {"levels": []})
        cm = ConfigManager(str(tmp_path))
        assert cm.get_level_config("nonexistent") == {}


# ===========================================================================
# _load_yaml edge cases
# ===========================================================================

class TestLoadYaml:
    def test_missing_file_returns_empty_dict(self, tmp_path):
        _write_yaml(tmp_path, "pipeline.yaml", {"levels": []})
        cm = ConfigManager(str(tmp_path))
        assert cm._load_yaml("does_not_exist.yaml") == {}


# ===========================================================================
# Level0Env integration (mocked ConfigManager)
# ===========================================================================

def _mock_config_manager(**overrides):
    """Create a fully-specified mock ConfigManager instance."""
    mock = MagicMock()
    mock.get_cumulative_active_layers.return_value = overrides.get("layers", ["a", "b", "c"])
    mock.get_cumulative_active_tactical_layers.return_value = overrides.get("tactical_layers", ["a", "b"])
    mock.get_cumulative_action_heads.return_value = overrides.get("action_heads", ["move"])
    mock.get_cumulative_categorical_heads.return_value = overrides.get("categorical_heads", [])
    mock.get_agent_architecture.return_value = overrides.get("architecture", "original")
    mock.get_level_config.return_value = overrides.get("level_config", {"env": {"rewards": {}}})
    return mock


@patch("train.envs.level_0_env.ConfigManager")
@patch("train.envs.level_0_env.SC2Process")
@patch("train.envs.level_0_env.Client")
def test_level0_env_dynamic_observation_space(mock_client, mock_sc2, mock_cm_cls):
    mock_cm_cls.return_value = _mock_config_manager(layers=["a", "b", "c"])
    with patch.object(Level0Env, "_discover_maps", return_value=["map1"]):
        env = Level0Env(level_name="test_level", visualize=False)

    assert env.observation_space["map"].shape == (3, 128, 128)
    assert env.training_conf["active_layers"] == ["a", "b", "c"]


@patch("train.envs.level_0_env.ConfigManager")
@patch("train.envs.level_0_env.SC2Process")
@patch("train.envs.level_0_env.Client")
def test_level0_env_dynamic_action_space(mock_client, mock_sc2, mock_cm_cls):
    mock_cm_cls.return_value = _mock_config_manager(action_heads=["move", "extra"])
    with patch.object(Level0Env, "_discover_maps", return_value=["map1"]):
        env = Level0Env(level_name="test_level", visualize=False, grid_size=32)

    assert env.action_space.n == 2048
    assert env.action_heads == ["move", "extra"]


@patch("train.envs.level_0_env.ConfigManager")
@patch("train.envs.level_0_env.SC2Process")
@patch("train.envs.level_0_env.Client")
def test_level7_env_dynamic_action_space(mock_client, mock_sc2, mock_cm_cls):
    from train.envs.level_7_env import Level7Env

    mock_cm_cls.return_value = _mock_config_manager(action_heads=["move", "attack"])
    with patch.object(Level7Env, "_discover_maps", return_value=["map1"]):
        env = Level7Env(level_name="level_7", visualize=False, grid_size=10)

    assert env.action_space.n == 200
    assert env.action_heads == ["move", "attack"]


@patch("train.envs.level_0_env.ConfigManager")
@patch("train.envs.level_0_env.SC2Process")
@patch("train.envs.level_0_env.Client")
def test_level_drop_env_accumulation(mock_client, mock_sc2, mock_cm_cls):
    from train.envs.level_drop_env import LevelDropEnv

    mock_cm_cls.return_value = _mock_config_manager(
        action_heads=["move", "attack", "load", "unload"]
    )
    with patch.object(LevelDropEnv, "_discover_maps", return_value=["map1"]):
        env = LevelDropEnv(level_name="level_drop", visualize=False, grid_size=10)

    assert env.action_space.n == 400
    assert env.action_heads == ["move", "attack", "load", "unload"]
