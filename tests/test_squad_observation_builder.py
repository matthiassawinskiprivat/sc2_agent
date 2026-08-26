"""Tests for squad observation builder shapes."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sc2.ids.unit_typeid import UnitTypeId

from src.transformers.squad_observation_builder import SquadObservationBuilder
from src.agent.squad_constants import (
    MAX_SQUAD_UNITS,
    SQUAD_COMMAND_DIM,
    SQUAD_RACE_DIM,
    SQUAD_UNIT_FEAT_DIM,
)
from src.agent.unit_scalar_constants import UnitScalarConstants as USC


@pytest.fixture
def training_conf():
    return {
        "active_layers": ["allied_unit_biological", "enemy_unit_biological"],
        "active_tactical_layers": ["allied_unit_biological", "enemy_unit_biological"],
        "agent_architecture": "squad",
    }


def test_zero_observation_shapes(training_conf):
    builder = SquadObservationBuilder(training_conf)
    obs = builder.zero_observation()
    assert obs["map"].shape == (2, 128, 128)
    assert obs["command"].shape == (SQUAD_COMMAND_DIM,)
    assert obs["anchor_tactical"].shape == (2, 32, 32)
    assert "unit_tactical" not in obs
    assert "race" not in obs
    assert obs["unit_scalars"].shape == (MAX_SQUAD_UNITS, builder.unit_feat_dim)
    assert builder.unit_feat_dim == SQUAD_UNIT_FEAT_DIM  # default = full catalog
    assert obs["unit_mask"].shape == (MAX_SQUAD_UNITS,)


def test_zero_observation_includes_race_when_enabled(training_conf):
    training_conf = {**training_conf, "include_race": True}
    builder = SquadObservationBuilder(training_conf)
    obs = builder.zero_observation()
    assert obs["race"].shape == (SQUAD_RACE_DIM,)
    assert obs["race"].dtype == np.float32


def test_resolve_anchor_tag_prefers_alive_preferred():
    assert SquadObservationBuilder.resolve_anchor_tag([10, 20], [10, 20], preferred_tag=20) == 20
    assert SquadObservationBuilder.resolve_anchor_tag([10, 20], [10], preferred_tag=20) == 10
    assert SquadObservationBuilder.resolve_anchor_tag([10, 20], [], preferred_tag=10) is None


def test_build_masks_and_scalars(training_conf):
    builder = SquadObservationBuilder(training_conf)

    unit_a = MagicMock()
    unit_a.tag = 10
    unit_a.position.x = 64.0
    unit_a.position.y = 32.0

    unit_b = MagicMock()
    unit_b.tag = 20
    unit_b.position.x = 80.0
    unit_b.position.y = 40.0

    bot = MagicMock()
    bot.player_id = 1
    bot.units = MagicMock()
    bot.units.exists = True
    bot.units.first = unit_a
    bot.enemy_units = MagicMock()
    bot.structures = MagicMock()
    bot.enemy_structures = MagicMock()
    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None
    bot.game_info = MagicMock()
    bot.game_data = MagicMock()
    if hasattr(bot.game_info, "pathing_grid"):
        del bot.game_info.pathing_grid
    if hasattr(bot.game_info, "terrain_height"):
        del bot.game_info.terrain_height
    bot.unit_target_map = {10: MagicMock(x=100.0, y=100.0)}
    bot.unit_command_map = {10: "attack", 20: "attack"}
    bot.enemy_pylon_pos = {1: MagicMock(x=120.0, y=120.0)}
    # Provide an empty upgrades set so build() doesn't misinterpret a MagicMock
    bot.state.upgrades = frozenset()

    def find_by_tag(tag):
        return {10: unit_a, 20: unit_b}.get(tag)

    bot.all_units.find_by_tag = find_by_tag

    fake_tactical = np.ones((2, 32, 32), dtype=np.float32)
    with patch.object(
        builder._transformer, "transform_tactical", return_value=fake_tactical
    ) as mock_tac:
        obs = builder.build(
            bot,
            perspective_player_id=1,
            squad_unit_tags=[20, 10],
            pool_high_level_command="attack",
            playable_size=(128.0, 128.0),
            anchor_unit_tag=20,
        )
        assert mock_tac.call_count == 1
        assert mock_tac.call_args[0][1] is unit_b

    assert obs["unit_mask"][0] == 1.0
    assert obs["unit_mask"][1] == 1.0
    assert obs["anchor_tactical"].shape == (2, 32, 32)
    assert np.allclose(obs["anchor_tactical"], fake_tactical)
    # ROLE_ATTACK lives at USC.FEATURES[USC.ROLE_ATTACK] in the new layout
    assert obs["unit_scalars"][0, USC.FEATURES[USC.ROLE_ATTACK]] == 1.0
    assert obs["map"].shape[0] == 2
    assert "race" not in obs


def test_build_encodes_race_when_enabled(training_conf):
    training_conf = {**training_conf, "include_race": True}
    builder = SquadObservationBuilder(training_conf)

    unit_a = MagicMock()
    unit_a.tag = 10
    unit_a.position.x = 64.0
    unit_a.position.y = 32.0

    bot = MagicMock()
    bot.player_id = 1
    bot.units = MagicMock()
    bot.units.exists = True
    bot.units.first = unit_a
    bot.enemy_units = MagicMock()
    bot.structures = MagicMock()
    bot.enemy_structures = MagicMock()
    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None
    bot.state.upgrades = frozenset()
    bot.game_info = MagicMock()
    bot.game_data = MagicMock()
    if hasattr(bot.game_info, "pathing_grid"):
        del bot.game_info.pathing_grid
    if hasattr(bot.game_info, "terrain_height"):
        del bot.game_info.terrain_height
    bot.unit_target_map = {}
    bot.unit_command_map = {10: "attack"}
    bot.enemy_pylon_pos = {}
    bot.all_units.find_by_tag = lambda tag: unit_a if tag == 10 else None

    fake_tactical = np.ones((2, 32, 32), dtype=np.float32)
    with patch.object(builder._transformer, "transform_tactical", return_value=fake_tactical):
        obs = builder.build(
            bot,
            perspective_player_id=1,
            squad_unit_tags=[10],
            pool_high_level_command="attack",
            playable_size=(128.0, 128.0),
            own_race="terran",
            enemy_race="zerg",
        )
    assert obs["race"].shape == (SQUAD_RACE_DIM,)
    assert np.allclose(obs["race"], [1, 0, 0, 0, 1, 0])


def test_build_raw_obs_selected_units_include_all_squad_members(training_conf):
    """Global COMMAND_ASSIGNMENT uses selected_units; squad mode must list the whole pool."""
    builder = SquadObservationBuilder(training_conf)

    unit_a = MagicMock()
    unit_a.tag = 10
    unit_a.position.x = 40.0
    unit_a.position.y = 40.0

    unit_b = MagicMock()
    unit_b.tag = 20
    unit_b.position.x = 60.0
    unit_b.position.y = 60.0

    bot = MagicMock()
    bot.player_id = 1
    bot.units = MagicMock()
    bot.units.exists = True
    bot.enemy_units = MagicMock()
    bot.structures = MagicMock()
    bot.structures.return_value.exists = False
    bot.enemy_structures = MagicMock()
    bot.state = MagicMock()
    bot.state.effects = []
    bot.game_info = MagicMock()
    bot.game_data = MagicMock()
    bot.unit_target_map = {}
    bot.unit_command_map = {10: "attack", 20: "attack"}
    bot.enemy_pylon_pos = {1: MagicMock(x=100.0, y=100.0)}
    bot.all_units.find_by_tag = lambda tag: {10: unit_a, 20: unit_b}.get(tag)

    raw_obs = builder._build_raw_obs(
        bot,
        perspective_player_id=1,
        sorted_tags=[20, 10],
        pool_high_level_command="attack",
        anchor_unit_tag=20,
    )

    selected_tags = {u.tag for u in raw_obs["selected_units"]}
    assert selected_tags == {10, 20}
    # Anchor 20 has no unit_target_map entry → attack fallback uses enemy_pylon_pos.
    assert raw_obs["command_target"] is bot.enemy_pylon_pos[1]
    # active_unit for command_target should be the hard anchor when provided
    assert bot.all_units.find_by_tag(20) is unit_b


def test_p2_trainee_uses_bot_state_upgrades_for_range_scalars(training_conf):
    """debug_upgrade is global; P2 trainee must not drop state.upgrades (bot is P1)."""
    from sc2.ids.upgrade_id import UpgradeId

    training_conf = {
        **training_conf,
        "active_unit_scalars": ["ground_range_norm", "air_range_norm"],
    }
    builder = SquadObservationBuilder(training_conf)

    hydra = MagicMock()
    hydra.tag = 10
    hydra.owner_id = 2
    hydra.type_id = UnitTypeId.HYDRALISK
    hydra.position.x = 40.0
    hydra.position.y = 40.0
    hydra.ground_range = 5.0
    hydra.air_range = 5.0
    hydra.ground_dps = 10.0
    hydra.air_dps = 10.0
    hydra.health = 90.0
    hydra.health_max = 90.0
    hydra.shield = 0.0
    hydra.shield_max = 0.0
    hydra.energy = 0.0
    hydra.energy_max = 0.0
    hydra.weapon_cooldown = 0.0
    hydra.weapon_ready = True
    hydra.calculate_speed = MagicMock(return_value=3.15)
    hydra.distance_to_weapon_ready = 0.0
    hydra.movement_speed = 3.15

    bot = MagicMock()
    bot.player_id = 1
    bot.units = MagicMock()
    bot.units.exists = True
    bot.units.first = hydra
    bot.enemy_units = MagicMock()
    bot.structures = MagicMock()
    bot.enemy_structures = MagicMock()
    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None
    bot.state.upgrades = frozenset({UpgradeId.EVOLVEGROOVEDSPINES})
    bot.game_info = MagicMock()
    bot.game_data = MagicMock()
    if hasattr(bot.game_info, "pathing_grid"):
        del bot.game_info.pathing_grid
    if hasattr(bot.game_info, "terrain_height"):
        del bot.game_info.terrain_height
    bot.unit_target_map = {}
    bot.unit_command_map = {10: "attack"}
    bot.enemy_pylon_pos = {}
    bot.all_units.find_by_tag = lambda tag: hydra if tag == 10 else None

    fake_tactical = np.zeros((2, 32, 32), dtype=np.float32)
    with patch.object(builder._transformer, "transform_tactical", return_value=fake_tactical):
        obs_p2 = builder.build(
            bot,
            perspective_player_id=2,
            squad_unit_tags=[10],
            pool_high_level_command="attack",
            playable_size=(128.0, 128.0),
        )
        builder.clear_global_cache()
        obs_p1 = builder.build(
            bot,
            perspective_player_id=1,
            squad_unit_tags=[10],
            pool_high_level_command="attack",
            playable_size=(128.0, 128.0),
        )

    # Grooved spines: +1 ground and air on a 5.0 catalog range → 6/10.
    expected = 6.0 / 10.0
    assert abs(float(obs_p2["unit_scalars"][0, 0]) - expected) < 1e-5
    assert abs(float(obs_p2["unit_scalars"][0, 1]) - expected) < 1e-5
    assert np.allclose(obs_p2["unit_scalars"][0], obs_p1["unit_scalars"][0])
