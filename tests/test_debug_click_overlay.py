import math

import pytest
from unittest.mock import MagicMock
from sc2.position import Point2

from src.agent.universal_unit_control_agent import UniversalUnitControlAgent
from src.transformers.observation_transformer import ObservationTransformer
from src.utils.obs_geometry import (
    obs_pixel_center_to_display,
    tactical_crop_origin,
    world_to_tactical_obs_pixel,
)


def _game_info(width=128, height=128):
    gi = MagicMock()
    gi.map_size.width = width
    gi.map_size.height = height
    gi.playable_area = MagicMock()
    gi.playable_area.x = 0
    gi.playable_area.y = 0
    gi.playable_area.width = width
    gi.playable_area.height = height
    return gi


def test_cnn_pixel_to_display_tactical_uses_cell_center():
    agent = UniversalUnitControlAgent(grid_size=128)
    cx, cy = agent.cnn_pixel_to_display(16, 15, (512, 512), obs_size=(32, 32))
    assert cx == obs_pixel_center_to_display(16, 15, (512, 512), (32, 32))[0]
    assert cy == obs_pixel_center_to_display(16, 15, (512, 512), (32, 32))[1]
    assert cx == 264
    assert cy == 248


def test_tactical_pixel_tracks_world_when_focal_unit_moves():
    """Same world target must map to different tactical pixels as the fovea recenters."""
    active_start = MagicMock()
    active_start.position = Point2((64.0, 64.0))
    active_moved = MagicMock()
    active_moved.position = Point2((72.0, 66.0))
    target = (80.0, 70.0)

    t0 = world_to_tactical_obs_pixel(*target, 64.0, 64.0)
    t1 = world_to_tactical_obs_pixel(*target, 72.0, 66.0)
    assert t0 != t1


def test_tactical_unit_pixel_aligns_with_terrain_row_for_fractional_y():
    """Fractional world y must use the terrain row that samples floor(world_y)."""
    focal = Point2((50.7, 50.7))
    world = (50.7, 50.7)
    crop_x, crop_y = tactical_crop_origin(focal.x, focal.y)

    transformer = ObservationTransformer(map_size=(128, 128))
    transformer.scale = 1.0
    transformer.offset_x = 0.0
    transformer.offset_y = 0.0
    transformer.feature_map_width = 32
    transformer.feature_map_height = 32
    transformer.is_tactical_mode = True
    transformer.crop_x, transformer.crop_y = crop_x, crop_y

    px, py = transformer._game_to_obs_pixel(*world)
    terrain_sc2_y = (31 - py) + crop_y
    assert terrain_sc2_y == int(math.floor(world[1]))
    assert py == world_to_tactical_obs_pixel(*world, focal.x, focal.y)[1]
    assert py == 15


def test_tactical_click_pixel_matches_observation_transformer():
    """Agent/UI tactical mapping must match ObservationTransformer pixel placement."""
    active = MagicMock()
    active.position = Point2((64.0, 64.0))
    wx, wy = 80.0, 70.0

    agent = UniversalUnitControlAgent(grid_size=128)
    agent_px, agent_py = agent.world_to_tactical_cnn_pixel(wx, wy, active)

    transformer = ObservationTransformer(map_size=(128, 128))
    transformer.scale = 1.0
    transformer.offset_x = 0.0
    transformer.offset_y = 0.0
    transformer.feature_map_width = 32
    transformer.feature_map_height = 32
    transformer.is_tactical_mode = True
    transformer.crop_x, transformer.crop_y = tactical_crop_origin(
        active.position.x, active.position.y
    )

    expected = world_to_tactical_obs_pixel(
        wx, wy, active.position.x, active.position.y
    )
    tx, ty = transformer._game_to_obs_pixel(wx, wy)
    assert (agent_px, agent_py) == expected
    assert (tx, ty) == expected


def test_extract_order_target_unknown_ability_id_reads_proto():
    """Probe gather (and similar) must not crash when ability_id is missing from GameData."""
    from src.sc2_bot import _extract_order_target

    order_proto = MagicMock()
    order_proto.ability_id = 4132
    order_proto.target_unit_tag = 0
    order_proto.HasField = lambda name: name == "target_world_space_pos"
    order_proto.target_world_space_pos = MagicMock(x=150.25, y=136.46484375)

    unit = MagicMock()
    unit._proto.orders = [order_proto]

    target = _extract_order_target(unit)
    assert target == Point2((150.25, 136.46484375))


def test_extract_order_target_resolves_target_unit_tag():
    from src.sc2_bot import _extract_order_target

    order_proto = MagicMock()
    order_proto.ability_id = 4132
    order_proto.target_unit_tag = 4358406145
    order_proto.HasField = lambda name: False

    mineral = MagicMock()
    mineral.position = Point2((42.0, 55.0))

    bot = MagicMock()
    bot.units.find_by_tag.return_value = None
    bot.structures.find_by_tag.return_value = None
    bot.all_units.find_by_tag.return_value = mineral

    unit = MagicMock()
    unit._proto.orders = [order_proto]
    unit._bot_object = bot

    target = _extract_order_target(unit)
    assert target == Point2((42.0, 55.0))


def test_record_debug_click_stores_global_and_tactical_pixels():
    agent = UniversalUnitControlAgent(grid_size=128)
    agent.last_game_info = _game_info()
    active = MagicMock()
    active.position = Point2((64.0, 64.0))

    agent.record_debug_click(Point2((80.0, 70.0)), active)

    assert agent.last_debug_click is not None
    assert "global_cnn" in agent.last_debug_click
    assert "tactical_cnn" in agent.last_debug_click
    assert agent.last_debug_click["world"] == (80.0, 70.0)
    gx, gy = agent.last_debug_click["global_cnn"]
    tx, ty = agent.last_debug_click["tactical_cnn"]
    assert 0 <= gx < 128
    assert 0 <= gy < 128
    assert 0 <= tx < 32
    assert 0 <= ty < 32
    assert (tx, ty) == world_to_tactical_obs_pixel(80.0, 70.0, 64.0, 64.0)
