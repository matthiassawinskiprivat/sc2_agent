"""Regression: universal command axis is fixed at 64; padded slots stay masked."""

from unittest.mock import MagicMock, patch

from src.agent.universal_constants import (
    FIXED_UNIVERSAL_COMMAND_DIM,
    UNUSED_COMMAND_HEAD,
    build_action_head_manifest,
)
from src.sc2_bot_universal import SC2BotUniversal


def test_build_action_head_manifest_pads_to_64():
    m = build_action_head_manifest({}, ["move_local", "attack_local"])
    assert len(m) == FIXED_UNIVERSAL_COMMAND_DIM
    assert m[0] == "move_local"
    assert m[1] == "attack_local"
    assert all(h == UNUSED_COMMAND_HEAD for h in m[2:])


def _minimal_universal_bot(training_conf):
    with patch.object(SC2BotUniversal, "__init__", lambda self, *a, **kw: None):
        bot = SC2BotUniversal.__new__(SC2BotUniversal)
    bot.conn = MagicMock()
    bot.training_conf = training_conf
    bot.agent_system = None
    bot.zergling = None
    bot.beacon = None
    bot.previous_distance = None
    bot.last_target_pos = None
    bot._game_state_initialized = False
    bot._debug_show_map_enabled = False
    bot.action_heads = training_conf.get("action_heads", ["move_global"])
    bot.categorical_heads = training_conf.get("categorical_heads", [])
    bot.spatial_res = 32
    bot.distance_calculation_method = 2
    bot.unit_command_uses_self_do = False
    bot.actions = []
    bot.unit_tags_received_action = set()
    bot.action_head_manifest = build_action_head_manifest(
        bot.training_conf, list(bot.action_heads)
    )
    bot.training_conf["action_head_manifest"] = bot.action_head_manifest
    return bot


def test_sc2_bot_universal_masks_unused_command_indices():
    bot = _minimal_universal_bot(
        {
            "agent_architecture": "universal",
            "action_heads": ["move_global"],
            "categorical_heads": [],
            "active_layers": ["allied_unit_density"],
            "rewards": {},
        }
    )
    mask = bot.action_masks()
    assert mask.shape == (FIXED_UNIVERSAL_COMMAND_DIM + 1024,)
    assert mask[0]
    assert not mask[1:FIXED_UNIVERSAL_COMMAND_DIM].any()
    assert mask[FIXED_UNIVERSAL_COMMAND_DIM:].all()
