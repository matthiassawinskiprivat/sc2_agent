from unittest.mock import patch

from src.ui.level_preview import (
    build_preview_training_conf,
    create_preview_env,
    discover_level_maps,
    load_pipeline_levels,
    resolve_env_class_name,
)


def test_load_pipeline_levels_reads_universal_pipeline():
    levels = load_pipeline_levels("config_universal")
    assert "level_move_single_0" in levels
    assert "ctf_level_01" in levels


def test_build_preview_training_conf_accumulates_layers():
    conf = build_preview_training_conf("level_move_single_0", config_dir="config_universal")
    assert conf["preview_mode"] is True
    assert conf["preview_computer_opponent"] is True
    assert "command_objective" in conf["active_layers"]
    assert conf["agent_architecture"] == "universal"


def test_discover_level_maps_uses_maps_dir_from_level_yaml():
    conf = build_preview_training_conf("level_move_single_0", config_dir="config_universal")
    maps = discover_level_maps("level_move_single_0", conf)
    assert isinstance(maps, list)


def test_resolve_env_class_name_level_attack():
    assert resolve_env_class_name("level_attack_0", "config_universal") == "Level7Env"


def test_resolve_env_class_name_ctf():
    assert resolve_env_class_name("ctf_level_01", "config_universal") == "CaptureTheFlagEnv"


def test_create_preview_env_ctf_uses_step_based_simulation():
    """CTF preview must not force realtime=True (breaks obs refresh / tactical fovea)."""
    with patch("train.train_pipeline.ENV_REGISTRY", {"CaptureTheFlagEnv": _PreviewRealtimeProbe}):
        env = create_preview_env("ctf_level_01", "InterloperAIE")
        assert env.realtime is False


def test_create_preview_env_move_multi_passes_step_based_realtime():
    """Movement preview envs must accept realtime=False from create_preview_env."""
    with patch("train.train_pipeline.ENV_REGISTRY", {"Level3Env": _PreviewRealtimeProbe}):
        env = create_preview_env("level_move_multi_0", "00_128x128_No_Middle")
        assert env.realtime is False


def test_level0_env_accepts_realtime_kwarg():
    from unittest.mock import patch

    from train.envs.level_0_env import Level0Env

    conf = {"config_dir": "config_universal", "env": {}}
    with patch.object(Level0Env, "_discover_maps", return_value=["test.SC2Map"]):
        env = Level0Env(
            level_name="level_move_multi_0",
            training_conf=conf,
            realtime=False,
        )
    assert env.realtime is False
    assert env.training_conf["env"]["realtime"] is False


def test_sync_preview_debug_click_overlay_records_move_order():
    from unittest.mock import MagicMock
    from sc2.position import Point2

    from src.agent.universal_unit_control_agent import UniversalUnitControlAgent
    from src.sc2_bot import sync_agent_debug_click_overlay
    from src.ui.level_preview import sync_preview_debug_click_overlay

    ui_agent = UniversalUnitControlAgent(grid_size=32)
    gi = MagicMock()
    gi.map_size.width = gi.map_size.height = 128
    gi.playable_area = MagicMock(x=0, y=0, width=128, height=128)

    order = MagicMock()
    order.HasField.return_value = True
    order.target_world_space_pos = Point2((80.0, 70.0))

    active = MagicMock()
    active.tag = 42
    active.position = Point2((64.0, 64.0))
    active._proto.orders = [order]

    bot = MagicMock()
    bot.game_info = gi
    bot.units.selected.exists = True
    bot.units.selected.first = active
    bot.units.find_by_tag.return_value = active
    bot.structures.selected.exists = False

    env = MagicMock()
    env._bot = bot

    sync_preview_debug_click_overlay(env, ui_agent)

    assert ui_agent.last_game_info is gi
    assert ui_agent.debug_active_unit is active
    assert ui_agent.last_debug_click is not None
    assert ui_agent.last_debug_click["world"] == (80.0, 70.0)


class _PreviewRealtimeProbe:
    """Minimal stand-in to capture kwargs passed by create_preview_env."""

    def __init__(self, **kwargs):
        self.realtime = kwargs.get("realtime", False)
