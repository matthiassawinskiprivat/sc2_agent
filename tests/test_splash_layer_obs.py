"""Splash layer rendering via ObservationTransformer._draw_unit_splash."""
from unittest.mock import MagicMock

import numpy as np
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

# Older/partial burnysc2 installs may omit detector constants used by ObservationTransformer.
import sc2.constants as _sc2_constants

if not hasattr(_sc2_constants, "IS_DETECTOR"):
    _sc2_constants.IS_DETECTOR = frozenset(
        {
            UnitTypeId.OBSERVER,
            UnitTypeId.OBSERVERSIEGEMODE,
            UnitTypeId.OVERSEER,
            UnitTypeId.OVERSEERSIEGEMODE,
            UnitTypeId.RAVEN,
            UnitTypeId.MISSILETURRET,
            UnitTypeId.SPORECRAWLER,
        }
    )
if not hasattr(_sc2_constants, "UNIT_PHOTONCANNON"):
    _sc2_constants.UNIT_PHOTONCANNON = UnitTypeId.PHOTONCANNON

from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants
from tests.test_observation_transformer import _make_game_info, create_mock_unit


def _splash_transformer(active_layers):
    return ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_layers": active_layers,
            "active_tactical_layers": active_layers,
        },
    )


def _full_layer_buffer():
    return np.zeros((LayerConstants.get_num_layers(), 128, 128), dtype=np.float32)


def test_baneling_paints_allied_splash_disc():
    transformer = _splash_transformer(
        [LayerConstants.ALLIED_SPLASH, LayerConstants.ENEMY_SPLASH]
    )
    layers = _full_layer_buffer()
    baneling = create_mock_unit(UnitTypeId.BANELING, Point2((64.0, 64.0)))
    transformer._draw_unit_splash(baneling, layers, False, 64, 63)
    idx = LayerConstants.LAYERS[LayerConstants.ALLIED_SPLASH]
    enemy_idx = LayerConstants.LAYERS[LayerConstants.ENEMY_SPLASH]
    assert layers[idx].max() > 0.0
    assert layers[idx, 63, 64] > 0.0
    assert layers[enemy_idx].max() == 0.0


def test_baneling_dps_is_melee_not_explosion_radius():
    """Splash owns the 2.2 blast; DPS must follow melee contact, not explode radius."""
    transformer = _dps_draw_setup(
        [LayerConstants.ALLIED_G_DPS_BASE, LayerConstants.ALLIED_SPLASH]
    )
    layers = _full_layer_buffer()
    baneling = create_mock_unit(
        UnitTypeId.BANELING,
        Point2((64.0, 64.0)),
        ground_range=0.25,
        air_range=0.0,
        radius=0.375,
    )
    transformer._draw_unit_dps(baneling, layers, False, 64, 63, MagicMock())
    transformer._draw_unit_splash(baneling, layers, False, 64, 63)
    dps_idx = LayerConstants.LAYERS[LayerConstants.ALLIED_G_DPS_BASE]
    splash_idx = LayerConstants.LAYERS[LayerConstants.ALLIED_SPLASH]
    assert layers[dps_idx, 63, 64] > 0.0
    assert layers[splash_idx, 63, 64] > 0.0
    # Pixel ~1.6 gu from the unit: inside splash r=2.2, outside melee DPS.
    assert layers[splash_idx, 63, 65] > 0.0
    assert layers[dps_idx, 63, 65] == 0.0
    assert transformer._unit_max_weapon_reach(baneling) < 1.5


def test_sieged_tank_splash_is_donut():
    transformer = _splash_transformer([LayerConstants.ALLIED_SPLASH])
    layers = _full_layer_buffer()
    tank = create_mock_unit(
        UnitTypeId.SIEGETANKSIEGED,
        Point2((64.0, 64.0)),
        radius=0.875,
        ground_range=13.0,
    )
    transformer._draw_unit_splash(tank, layers, False, 64, 63)
    idx = LayerConstants.LAYERS[LayerConstants.ALLIED_SPLASH]
    assert layers[idx, 63, 64] == 0.0
    assert layers[idx, 63 + 5, 64] > 0.0


def test_non_splash_unit_leaves_splash_layer_zero():
    transformer = _splash_transformer(
        [LayerConstants.ALLIED_SPLASH, LayerConstants.ENEMY_SPLASH]
    )
    layers = _full_layer_buffer()
    marine = create_mock_unit(UnitTypeId.MARINE, Point2((64.0, 64.0)))
    transformer._draw_unit_splash(marine, layers, False, 64, 63)
    allied = LayerConstants.LAYERS[LayerConstants.ALLIED_SPLASH]
    enemy = LayerConstants.LAYERS[LayerConstants.ENEMY_SPLASH]
    assert float(layers[allied].max()) == 0.0
    assert float(layers[enemy].max()) == 0.0


def test_enemy_splash_uses_enemy_channel():
    transformer = _splash_transformer(
        [LayerConstants.ALLIED_SPLASH, LayerConstants.ENEMY_SPLASH]
    )
    layers = _full_layer_buffer()
    archon = create_mock_unit(UnitTypeId.ARCHON, Point2((64.0, 64.0)))
    transformer._draw_unit_splash(archon, layers, True, 64, 63)
    allied = LayerConstants.LAYERS[LayerConstants.ALLIED_SPLASH]
    enemy = LayerConstants.LAYERS[LayerConstants.ENEMY_SPLASH]
    assert layers[enemy].max() > 0.0
    assert layers[allied].max() == 0.0


def test_lurker_splash_only_when_burrowed():
    transformer = _splash_transformer([LayerConstants.ALLIED_SPLASH])
    idx = LayerConstants.LAYERS[LayerConstants.ALLIED_SPLASH]

    layers_up = _full_layer_buffer()
    transformer._draw_unit_splash(
        create_mock_unit(UnitTypeId.LURKERMP, Point2((64.0, 64.0))),
        layers_up,
        False,
        64,
        63,
    )
    assert float(layers_up[idx].max()) == 0.0

    layers_down = _full_layer_buffer()
    transformer._draw_unit_splash(
        create_mock_unit(UnitTypeId.LURKERMPBURROWED, Point2((64.0, 64.0))),
        layers_down,
        False,
        64,
        63,
    )
    assert float(layers_down[idx].max()) > 0.0


def test_widowmine_splash_only_when_burrowed():
    transformer = _splash_transformer([LayerConstants.ALLIED_SPLASH])
    idx = LayerConstants.LAYERS[LayerConstants.ALLIED_SPLASH]

    layers_up = _full_layer_buffer()
    transformer._draw_unit_splash(
        create_mock_unit(UnitTypeId.WIDOWMINE, Point2((64.0, 64.0))),
        layers_up,
        False,
        64,
        63,
    )
    assert float(layers_up[idx].max()) == 0.0

    layers_down = _full_layer_buffer()
    transformer._draw_unit_splash(
        create_mock_unit(UnitTypeId.WIDOWMINEBURROWED, Point2((64.0, 64.0))),
        layers_down,
        False,
        64,
        63,
    )
    assert float(layers_down[idx].max()) > 0.0


def _dps_draw_setup(active_layers):
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_layers": active_layers,
            "active_tactical_layers": active_layers,
        },
    )
    transformer.current_layer_map = LayerConstants.LAYERS
    transformer.is_tactical_mode = False
    transformer._update_active_indices()
    transformer.scale = transformer.scale_x = transformer.scale_y = 1.0
    transformer.offset_x = transformer.offset_y = 0.0
    transformer.crop_x = transformer.crop_y = 0.0
    transformer._obs_span_w = transformer._obs_span_h = 128.0
    transformer._grid_x = np.arange(128).reshape(1, -1)
    transformer._grid_y = np.arange(128).reshape(-1, 1)
    return transformer


def test_lurker_and_widowmine_dps_only_when_burrowed():
    """Unburrowed morphs must not paint DPS; burrowed forms must."""
    layers_name = [LayerConstants.ALLIED_G_DPS_BASE]
    transformer = _dps_draw_setup(layers_name)
    idx = LayerConstants.LAYERS[LayerConstants.ALLIED_G_DPS_BASE]
    game_data = MagicMock()
    px, py = 64, 63

    for unburrowed in (UnitTypeId.LURKERMP, UnitTypeId.WIDOWMINE):
        layers = _full_layer_buffer()
        transformer._draw_unit_dps(
            create_mock_unit(
                unburrowed,
                Point2((64.0, 64.0)),
                ground_range=0.0,
                air_range=0.0,
                radius=0.75,
            ),
            layers,
            False,
            px,
            py,
            game_data,
        )
        assert float(layers[idx].max()) == 0.0, unburrowed

    for burrowed in (UnitTypeId.LURKERMPBURROWED, UnitTypeId.WIDOWMINEBURROWED):
        layers = _full_layer_buffer()
        transformer._draw_unit_dps(
            create_mock_unit(
                burrowed,
                Point2((64.0, 64.0)),
                weapon_cooldown=0,
                ground_range=0.0,
                air_range=0.0,
                radius=0.75,
            ),
            layers,
            False,
            px,
            py,
            game_data,
        )
        assert float(layers[idx].max()) > 0.0, burrowed


def test_liberator_ag_has_no_splash_hull_or_zone():
    """Sieged Liberator Concord Cannon is single-target; zone is DPS-only, not splash."""
    from sc2.ids.effect_id import EffectId

    transformer = _splash_transformer(
        [LayerConstants.ALLIED_SPLASH, LayerConstants.ENEMY_SPLASH]
    )
    idx = LayerConstants.LAYERS[LayerConstants.ALLIED_SPLASH]

    layers = _full_layer_buffer()
    transformer._draw_unit_splash(
        create_mock_unit(UnitTypeId.LIBERATORAG, Point2((64.0, 64.0)), is_flying=True),
        layers,
        False,
        64,
        63,
    )
    assert float(layers[idx].max()) == 0.0

    class _LiberatorEffect:
        def __init__(self, eid, x, y):
            self.id = eid
            self.positions = [Point2((x, y))]

    lib = create_mock_unit(
        UnitTypeId.LIBERATORAG,
        Point2((58.0, 50.0)),
        radius=0.75,
        is_flying=True,
    )
    eff = _LiberatorEffect(EffectId.LIBERATORTARGETMORPHDELAYPERSISTENT, 50.0, 50.0)
    gi = _make_game_info(128, 128, with_grids=True)
    raw_obs = {
        "units": [lib],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [eff],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    global_map = transformer.transform(raw_obs)["map"]
    assert float(global_map[0].max()) == 0.0
    assert float(global_map[1].max()) == 0.0


def test_fighter_liberator_paints_hull_splash():
    """Unsieged Liberator AA splash is hull-centered."""
    transformer = _splash_transformer([LayerConstants.ALLIED_SPLASH])
    idx = LayerConstants.LAYERS[LayerConstants.ALLIED_SPLASH]
    layers = _full_layer_buffer()
    transformer._draw_unit_splash(
        create_mock_unit(UnitTypeId.LIBERATOR, Point2((64.0, 64.0)), is_flying=True, radius=0.75),
        layers,
        False,
        64,
        63,
    )
    assert float(layers[idx].max()) > 0.0


def test_thor_explosive_payload_paints_splash_high_impact_does_not():
    transformer = _splash_transformer([LayerConstants.ALLIED_SPLASH])
    idx = LayerConstants.LAYERS[LayerConstants.ALLIED_SPLASH]

    layers_thor = _full_layer_buffer()
    transformer._draw_unit_splash(
        create_mock_unit(UnitTypeId.THOR, Point2((64.0, 64.0)), radius=1.0),
        layers_thor,
        False,
        64,
        63,
    )
    assert float(layers_thor[idx].max()) > 0.0

    layers_ap = _full_layer_buffer()
    transformer._draw_unit_splash(
        create_mock_unit(UnitTypeId.THORAP, Point2((64.0, 64.0)), radius=1.0),
        layers_ap,
        False,
        64,
        63,
    )
    assert float(layers_ap[idx].max()) == 0.0


def test_fighter_liberator_paints_splash_on_transform():
    """Unsieged Liberator AA splash appears on the splash layer without a zone effect."""
    transformer = _splash_transformer(
        [LayerConstants.ALLIED_SPLASH, LayerConstants.ENEMY_SPLASH]
    )
    lib = create_mock_unit(
        UnitTypeId.LIBERATOR,
        Point2((50.0, 50.0)),
        radius=0.75,
        is_flying=True,
    )
    gi = _make_game_info(128, 128, with_grids=True)
    raw_obs = {
        "units": [lib],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    global_map = transformer.transform(raw_obs)["map"]
    assert float(global_map[0].max()) > 0.0
    assert float(global_map[1].max()) == 0.0
