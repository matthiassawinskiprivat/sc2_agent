import numpy as np
from unittest.mock import MagicMock

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.unit_control_agent import UnitControlAgent
from src.agent.universal_unit_control_agent import UniversalUnitControlAgent
from src.transformers.observation_transformer import ObservationTransformer
from src.utils.obs_geometry import compute_global_obs_geometry, world_to_tactical_obs_pixel
from tests.test_observation_transformer import _make_game_info, create_mock_unit


def _precomputed_move_global_targets(agent, focal):
    """All 1024 decode targets once — avoids O(spawns × 1024) actuation calls."""
    return [
        agent.convert_spatial_action_to_world("move_global", idx, active_unit=focal)
        for idx in range(1024)
    ]


def _nearest_target_distance(targets, x: float, y: float) -> float:
    return min(((wx - x) ** 2 + (wy - y) ** 2) ** 0.5 for wx, wy in targets)


def _sync_transformer_global_geometry(transformer, game_info, path_data=None):
    """Mirror ObservationTransformer global crop/scale after transform()."""
    cx, cy, mw, mh, sx, sy, ox, oy = compute_global_obs_geometry(
        game_info, transformer.feature_map_width, transformer.feature_map_height, path_data
    )
    transformer.crop_x = cx
    transformer.crop_y = cy
    transformer.scale_x = sx
    transformer.scale_y = sy
    transformer.scale = min(sx, sy)
    transformer.offset_x = ox
    transformer.offset_y = oy
    transformer._obs_span_w = float(mw)
    transformer._obs_span_h = float(mh)


def _mock_game_info(map_w: int, map_h: int, area_x: int, area_y: int, area_w: int, area_h: int):
    gi = MagicMock()
    gi.map_size.width = map_w
    gi.map_size.height = map_h
    gi.playable_area.x = area_x
    gi.playable_area.y = area_y
    gi.playable_area.width = area_w
    gi.playable_area.height = area_h
    return gi


def test_global_decode_clamped_to_playable_area():
    agent = UnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    agent.last_game_info = _mock_game_info(map_w=200, map_h=180, area_x=10, area_y=20, area_w=128, area_h=96)

    # Sample a few edge indices; they should always land inside playable_area
    indices = [0, 31, 32 * 31, 32 * 32 - 1, 511, 512]
    for idx in indices:
        x, y = agent.convert_spatial_action_to_world(idx)
        assert 10.0 <= x < 10.0 + 128.0
        assert 20.0 <= y < 20.0 + 96.0


def test_global_roundtrip_is_stable_within_cell():
    agent = UnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    agent.last_game_info = _mock_game_info(map_w=200, map_h=180, area_x=7, area_y=9, area_w=150, area_h=150)

    rng = np.random.default_rng(0)
    for _ in range(100):
        x = float(rng.uniform(7.0, 7.0 + 150.0 - 1e-3))
        y = float(rng.uniform(9.0, 9.0 + 150.0 - 1e-3))
        pos = MagicMock()
        pos.x = x
        pos.y = y

        idx = agent.get_grid_index_at_world_pos(pos)
        x2, y2 = agent.convert_spatial_action_to_world(idx)

        # Returned point should always be inside playable area.
        assert 7.0 <= x2 < 7.0 + 150.0
        assert 9.0 <= y2 < 9.0 + 150.0


def test_local_decode_clamped_to_playable_area():
    agent = UniversalUnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    agent.last_game_info = _mock_game_info(map_w=200, map_h=180, area_x=10, area_y=20, area_w=128, area_h=96)

    active_unit = MagicMock()
    active_unit.position.x = 10.0  # at min corner of playable area
    active_unit.position.y = 20.0

    # Extreme local offsets should clamp back into playable.
    indices = [0, 32 * 32 - 1]
    for idx in indices:
        x, y = agent.convert_spatial_action_to_world("move_local", idx, active_unit=active_unit)
        assert 10.0 <= x < 10.0 + 128.0
        assert 20.0 <= y < 20.0 + 96.0


# --- get_action_mask shape tests ---

def test_universal_get_action_mask_no_heads_returns_spatial_only():
    """head_names=None → flat mask of size spatial_resolution^2 (1024)."""
    agent = UniversalUnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    mask = agent.get_action_mask(head_names=None)
    assert mask.shape == (1024,)
    assert mask.dtype == bool
    assert mask.all()


def test_universal_get_action_mask_with_two_heads():
    """head_names with 2 entries → command prefix 64 + spatial 1024."""
    from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM

    agent = UniversalUnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    mask = agent.get_action_mask(head_names=["move_global", "attack_global"])
    assert mask.shape == (FIXED_UNIVERSAL_COMMAND_DIM + 1024,)
    assert mask[:2].all()
    assert not mask[2:FIXED_UNIVERSAL_COMMAND_DIM].any()
    assert mask[FIXED_UNIVERSAL_COMMAND_DIM:].all()


def test_universal_get_action_mask_with_three_heads():
    from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM

    agent = UniversalUnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    mask = agent.get_action_mask(head_names=["move_global", "move_local", "attack_global"])
    assert mask.shape == (FIXED_UNIVERSAL_COMMAND_DIM + 1024,)
    assert mask[:3].all()
    assert not mask[3:FIXED_UNIVERSAL_COMMAND_DIM].any()
    assert mask[FIXED_UNIVERSAL_COMMAND_DIM:].all()


def test_global_spatial_decode_matches_observation_transformer():
    """
    Training actuation must invert the same geometry ObservationTransformer uses.

    Policy spatial indices are on a 32×32 grid; global observations are 128×128.
    """
    for map_size in (128, 512):
        gi = _make_game_info(map_size, map_size, with_grids=True)
        gi.playable_area = MagicMock(x=0, y=0, width=map_size, height=map_size)

        transformer = ObservationTransformer(
            map_size=(128, 128),
            training_conf={"agent_architecture": "universal", "active_layers": ["terrain_pathable"]},
        )
        path = gi.pathing_grid.data_numpy if hasattr(gi, "pathing_grid") else None
        _sync_transformer_global_geometry(transformer, gi, path)
        agent = UniversalUnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
        agent.last_game_info = gi
        focal = create_mock_unit(UnitTypeId.MARINE, Point2((map_size / 2, map_size / 2)), radius=0.375)

        for idx in range(1024):
            wx, wy = agent.convert_spatial_action_to_world(
                "move_global", idx, active_unit=focal
            )
            tx, ty = transformer._game_to_obs_pixel(wx, wy)
            px, py = agent.world_to_cnn_pixel(wx, wy)
            assert (px, py) == (tx, ty), f"map={map_size} idx={idx}"
            gy, gx = divmod(idx, 32)
            step = 128 // 32
            assert gx * step <= px < (gx + 1) * step
            assert gy * step <= py < (gy + 1) * step


def test_edge_spawn_points_within_collection_radius():
    """Level3 spawn grid must be reachable by some spatial cell on 128×128 maps."""
    import numpy as np
    from unittest.mock import MagicMock
    from src.agent.universal_unit_control_agent import UniversalUnitControlAgent

    gi = MagicMock()
    gi.map_size.width = gi.map_size.height = 128
    gi.playable_area = MagicMock(x=0, y=0, width=128, height=128)
    gi.pathing_grid = MagicMock(data_numpy=np.ones((128, 128), dtype=np.uint8))

    agent = UniversalUnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    agent.last_game_info = gi
    focal = MagicMock()
    focal.position.x = 64.0
    focal.position.y = 64.0

    margin = 5.0
    spawns = [
        (x, y)
        for x in range(int(margin), int(128 - margin), 4)
        for y in range(int(margin), int(128 - margin), 4)
    ]
    targets = _precomputed_move_global_targets(agent, focal)
    for x, y in spawns:
        best = _nearest_target_distance(targets, x, y)
        assert best < 2.0, f"spawn ({x},{y}) nearest decode dist {best:.3f} >= 2.0"


def test_ladder_map_beacon_roundtrip_within_cell():
    """512 maps: world→idx→decode must stay within one spatial cell (~16 gu)."""
    import numpy as np
    from unittest.mock import MagicMock
    from src.agent.universal_unit_control_agent import UniversalUnitControlAgent
    from src.transformers.observation_transformer import ObservationTransformer

    gi = MagicMock()
    gi.map_size.width = gi.map_size.height = 512
    gi.playable_area = MagicMock(x=0, y=0, width=512, height=512)
    gi.pathing_grid = MagicMock(data_numpy=np.ones((512, 512), dtype=np.uint8))

    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={"agent_architecture": "universal", "active_layers": ["terrain_pathable"]},
    )
    _sync_transformer_global_geometry(transformer, gi, gi.pathing_grid.data_numpy)
    agent = UniversalUnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    agent.last_game_info = gi
    focal = MagicMock()
    focal.position.x = 256.0
    focal.position.y = 256.0

    rng = np.random.default_rng(0)
    dists = []
    for _ in range(100):
        wx = float(rng.uniform(5, 507))
        wy = float(rng.uniform(5, 507))
        pos = MagicMock()
        pos.x = wx
        pos.y = wy
        idx = agent.get_grid_index_at_world_pos(pos)
        dx, dy = agent.convert_spatial_action_to_world("move_global", idx, active_unit=focal)
        dists.append(((dx - wx) ** 2 + (dy - wy) ** 2) ** 0.5)
        # Decode must land in the same obs cell as the beacon marker.
        px0, py0 = transformer._game_to_obs_pixel(wx, wy)
        px1, py1 = transformer._game_to_obs_pixel(dx, dy)
        step = 128 // 32
        assert px0 // step == px1 // step
        assert py0 // step == py1 // step

    assert np.mean(dists) < 8.0
    assert np.max(dists) < 14.0


def test_ladder_map_edge_spawns_within_move_local_range():
    """
    512×512 ladder maps letterbox into 128×128 obs; edge spawns must be within
    move_local (±16) of the best move_global target — not 8+ units away.
    """
    import numpy as np
    from unittest.mock import MagicMock
    from src.agent.universal_unit_control_agent import UniversalUnitControlAgent

    gi = MagicMock()
    gi.map_size.width = gi.map_size.height = 512
    gi.playable_area = MagicMock(x=0, y=0, width=512, height=512)
    gi.pathing_grid = MagicMock(data_numpy=np.ones((512, 512), dtype=np.uint8))

    agent = UniversalUnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    agent.last_game_info = gi
    focal = MagicMock()
    focal.position.x = 256.0
    focal.position.y = 256.0

    margin = 5.0
    edge_pts = [
        (int(margin), int(margin)),
        (int(512 - margin), int(margin)),
        (int(margin), int(512 - margin)),
        (int(512 - margin), int(512 - margin)),
        (256, int(margin)),
        (256, int(512 - margin)),
    ]
    targets = _precomputed_move_global_targets(agent, focal)
    for x, y in edge_pts:
        best = _nearest_target_distance(targets, x, y)
        assert best < 14.0, f"ladder edge spawn ({x},{y}) nearest decode dist {best:.3f} >= 14.0"


def test_move_local_spatial_matches_tactical_obs_pixels():
    """move_local / attack_local decode must match world_to_tactical_obs_pixel (32×32 fovea)."""
    for map_size in (128, 512):
        gi = _make_game_info(map_size, map_size, with_grids=True)
        gi.playable_area = MagicMock(x=0, y=0, width=map_size, height=map_size)
        agent = UniversalUnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
        agent.last_game_info = gi
        focal = MagicMock()
        focal.position = Point2((map_size / 2 + 0.37, map_size / 2 - 0.21))

        for head in ("move_local", "attack_local"):
            for idx in (0, 16, 496, 771, 1023):
                wx, wy = agent.convert_spatial_action_to_world(head, idx, active_unit=focal)
                px, py = world_to_tactical_obs_pixel(
                    wx, wy, float(focal.position.x), float(focal.position.y)
                )
                gy, gx = divmod(idx, 32)
                assert (px, py) == (gx, gy), f"map={map_size} head={head} idx={idx}"

