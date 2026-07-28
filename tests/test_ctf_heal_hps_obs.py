"""Queen and SCV appear on allied_hps_field; SCV uses small radius."""
from unittest.mock import MagicMock, patch

from sc2.ids.unit_typeid import UnitTypeId

from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants


def _transformer():
    conf = {
        "agent_architecture": "universal",
        "active_layers": [
            LayerConstants.ALLIED_HPS_FIELD,
            LayerConstants.ENEMY_HPS_FIELD,
        ],
        "active_tactical_layers": [
            LayerConstants.ALLIED_HPS_FIELD,
            LayerConstants.ENEMY_HPS_FIELD,
        ],
    }
    t = ObservationTransformer(map_size=(128, 128), training_conf=conf)
    t.current_layer_map = {
        "allied_hps_field": 0,
        "enemy_hps_field": 1,
    }
    t.active_indices = {0, 1}
    return t


def test_scv_hps_field_small_radius():
    t = _transformer()
    unit = MagicMock()
    unit.type_id = UnitTypeId.SCV
    unit.energy = 0
    with patch.object(t, "_draw_circle_vectorized") as draw:
        t._draw_unit_healing(unit, MagicMock(), False, 64, 64, MagicMock())
    draw.assert_called_once()
    # radius_game is positional arg index 4
    assert draw.call_args.args[4] == 1.0
    assert draw.call_args.args[6] == 9.3


def test_queen_hps_field_when_energy_available():
    t = _transformer()
    unit = MagicMock()
    unit.type_id = UnitTypeId.QUEEN
    unit.energy = 50
    with patch.object(t, "_draw_circle_vectorized") as draw:
        t._draw_unit_healing(unit, MagicMock(), False, 64, 64, MagicMock())
    draw.assert_called_once()
    assert draw.call_args.args[4] == 7.0
    assert draw.call_args.args[6] == 125.0


def test_queen_hps_field_skipped_without_energy():
    t = _transformer()
    unit = MagicMock()
    unit.type_id = UnitTypeId.QUEEN
    unit.energy = 10
    with patch.object(t, "_draw_circle_vectorized") as draw:
        t._draw_unit_healing(unit, MagicMock(), False, 64, 64, MagicMock())
    draw.assert_not_called()
