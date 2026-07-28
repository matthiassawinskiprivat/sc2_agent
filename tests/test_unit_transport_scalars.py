"""Transport / cargo unit scalar helpers."""
from unittest.mock import MagicMock

from sc2.ids.unit_typeid import UnitTypeId

from src.utils.unit_feature_helpers import (
    unit_can_load,
    unit_can_unload,
    unit_cargo_free_norm,
    unit_cargo_size_norm,
    unit_cargo_used_norm,
    unit_is_transport,
)


def _u(**kwargs):
    u = MagicMock()
    u.type_id = kwargs.get("type_id", UnitTypeId.MARINE)
    u.cargo_max = kwargs.get("cargo_max", 0)
    u.cargo_used = kwargs.get("cargo_used", 0)
    u.cargo_size = kwargs.get("cargo_size", 1)
    return u


def test_transport_scalars():
    med = _u(type_id=UnitTypeId.MEDIVAC, cargo_max=8, cargo_used=2, cargo_size=0)
    assert unit_is_transport(med) is True
    assert unit_cargo_used_norm(med) == 0.25
    assert unit_cargo_free_norm(med) == 0.75
    assert unit_cargo_size_norm(med) == 0.0
    assert unit_can_load(med) is True
    assert unit_can_unload(med) is True

    full = _u(type_id=UnitTypeId.MEDIVAC, cargo_max=8, cargo_used=8)
    assert unit_can_load(full) is False


def test_passenger_cargo_size_norm():
    marine = _u(cargo_max=0, cargo_size=1)
    assert unit_is_transport(marine) is False
    assert unit_cargo_size_norm(marine) == 1.0 / 8.0
    assert unit_can_load(marine) is True
    assert unit_can_unload(marine) is False
    assert unit_cargo_used_norm(marine) == 0.0
