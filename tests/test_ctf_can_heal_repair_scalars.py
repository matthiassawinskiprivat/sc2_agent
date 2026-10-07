"""can_heal / can_repair accumulate from ctf_heal_queen_vs_queen."""
from src.agent.unit_scalar_constants import UnitScalarConstants as USC
from src.utils.config_manager import ConfigManager


def test_ctf_level_01_declares_neural_and_structure_scalars():
    cm = ConfigManager("config_universal")
    scalars = cm.get_cumulative_active_unit_scalars("ctf_level_01")
    assert scalars.count(USC.IS_NEURALED) == 1
    assert scalars.count(USC.IS_STRUCTURE) == 1
    assert scalars.count(USC.ECONOMIC_VALUE_NORM) == 1
    assert scalars.count(USC.HITPOINTS_TOTAL_NORM) == 1
    assert scalars.index(USC.IS_NEURALED) < scalars.index(USC.IS_STRUCTURE)
    assert scalars.index(USC.IS_STRUCTURE) < scalars.index(USC.ECONOMIC_VALUE_NORM)
    assert scalars.index(USC.ECONOMIC_VALUE_NORM) < scalars.index(USC.HITPOINTS_TOTAL_NORM)


def test_heal_intro_adds_can_heal_and_can_repair():
    cm = ConfigManager("config_universal")
    scalars = cm.get_cumulative_active_unit_scalars("ctf_heal_queen_vs_queen")
    assert "can_heal" in scalars
    assert "can_repair" in scalars
    assert scalars.index("can_heal") < scalars.index("can_repair")


def test_later_levels_inherit_heal_repair_scalars():
    cm = ConfigManager("config_universal")
    for level in (
        "ctf_heal_thor_scv_vs_thor_scv",
        "ctf_heal_queen_vs_battlecruiser_scv",
        "ctf_level_18",
        "ctf_level_19",
        "ctf_level_20",
        "ctf_level_21",
    ):
        scalars = cm.get_cumulative_active_unit_scalars(level)
        assert "can_heal" in scalars
        assert "can_repair" in scalars
