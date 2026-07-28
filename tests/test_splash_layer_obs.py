"""Splash layer rendering via ObservationTransformer._draw_unit_splash."""
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
from tests.test_observation_transformer import create_mock_unit


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
