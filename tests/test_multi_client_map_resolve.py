import pytest
from train.envs.capture_the_flag_env import CaptureTheFlagEnv
from train.envs.multi_client_env import MultiClientEnv
from unittest.mock import MagicMock, patch
import os
from pathlib import Path


def test_ctf_env_map_resolution():
    """Test that MultiClientEnv can resolve maps in the project directory."""
    level_name = "capture_the_flag_terran"
    env = CaptureTheFlagEnv(level_name=level_name)

    # We expect the map '00_128x128_Full' to be found in train/maps/capture_the_flag_terran/
    assert env.map_name == "00_128x128_Full"

    resolved_map = env._resolve_map(env.map_name)
    assert resolved_map is not None
    assert "00_128x128_Full.SC2Map" in str(resolved_map.path)
    assert Path(resolved_map.path).exists()


def test_ctf_env_attributes():
    """Test that observation and action spaces are correctly initialized."""
    from gymnasium import spaces
    env = CaptureTheFlagEnv(level_name="capture_the_flag_terran")
    assert env.observation_space is not None
    assert env.action_space is not None
    # Universal arch uses a Dict observation space; original uses a flat Box.
    if isinstance(env.observation_space, spaces.Dict):
        assert env.observation_space["map"].shape[1:] == (128, 128)
        assert isinstance(env.action_space, spaces.MultiDiscrete)
    else:
        assert env.observation_space.shape[1:] == (128, 128)


def test_multi_client_discover_maps_uses_maps_dir(tmp_path):
    """_discover_maps() reads from training_conf['maps_dir'] when set."""
    (tmp_path / "map_a.SC2Map").touch()
    (tmp_path / "map_b.SC2Map").touch()

    env = MultiClientEnv(
        map_name="fallback",
        training_conf={"maps_dir": str(tmp_path)},
    )

    assert env.map_list == ["map_a", "map_b"]
    assert env.map_name in env.map_list


def test_multi_client_resolve_map_uses_maps_dir(tmp_path):
    """_resolve_map() builds the path from training_conf['maps_dir']."""
    (tmp_path / "my_map.SC2Map").touch()

    env = MultiClientEnv(
        map_name="my_map",
        training_conf={"maps_dir": str(tmp_path)},
    )

    from sc2.maps import Map
    resolved = env._resolve_map("my_map")
    assert isinstance(resolved, Map)
    assert "my_map.SC2Map" in str(resolved.path)


def test_multi_client_resolve_map_raises_when_maps_dir_map_missing(tmp_path):
    """_resolve_map() raises FileNotFoundError if the map file does not exist."""
    env = MultiClientEnv(
        map_name="fallback",
        training_conf={"maps_dir": str(tmp_path)},
    )

    with pytest.raises(FileNotFoundError):
        env._resolve_map("nonexistent_map")


def test_multi_client_map_rotates_on_reset(tmp_path):
    """map_name cycles through map_list on successive resets."""
    for name in ["alpha", "beta", "gamma"]:
        (tmp_path / f"{name}.SC2Map").touch()

    env = MultiClientEnv(
        map_name="fallback",
        training_conf={"maps_dir": str(tmp_path)},
        worker_index=0,
        num_workers=1,
    )
    assert env.map_list == ["alpha", "beta", "gamma"]

    seen = []
    with patch.object(env, "_launch_sc2_processes"), \
         patch.object(env, "_setup_game_session"), \
         patch.object(env, "_get_joint_observation", return_value=[0]):
        import asyncio
        import unittest.mock as mock

        async def noop(*a, **kw): return [0]
        env._launch_sc2_processes = mock.AsyncMock()
        env._setup_game_session = mock.AsyncMock()
        env._get_joint_observation = mock.AsyncMock(return_value=[0])

        for _ in range(3):
            env.reset()
            seen.append(env.map_name)

    # All three maps must appear across three resets (round-robin)
    assert set(seen) == {"alpha", "beta", "gamma"}


def test_ctf_env_passes_training_conf_to_super(tmp_path):
    """CaptureTheFlagEnv passes training_conf to MultiClientEnv so maps_dir is used."""
    (tmp_path / "ladder_map.SC2Map").touch()

    training_conf = {
        "maps_dir": str(tmp_path),
        "mineral_budget": 350,
        "vespene_budget": 0,
        "unit_pool": ["MARINE"],
        "active_layers": [],
        "active_tactical_layers": [],
        "action_heads": ["move"],
        "categorical_heads": [],
        "agent_architecture": "universal",
        "rewards": {},
        "env": {},
    }

    from unittest.mock import patch, MagicMock
    mock_cm = MagicMock()
    mock_cm.get_cumulative_active_layers.return_value = ["allied_unit_density"]
    mock_cm.get_cumulative_action_heads.return_value = ["move"]

    with patch("train.envs.multi_client_env.ConfigManager", return_value=mock_cm), \
         patch("train.envs.capture_the_flag_env.CaptureTheFlagEnv._load_unit_data", return_value={}):
        env = CaptureTheFlagEnv(
            level_name="test_ctf",
            training_conf=training_conf,
        )

    assert env.map_list == ["ladder_map"]
    assert env.map_name == "ladder_map"
