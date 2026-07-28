"""can_heal / can_repair accumulate from ctf_level_18."""
from src.utils.config_manager import ConfigManager


def test_ctf_level_18_adds_can_heal_and_can_repair():
    cm = ConfigManager("config_universal")
    scalars = cm.get_cumulative_active_unit_scalars("ctf_level_18")
    assert "can_heal" in scalars
    assert "can_repair" in scalars
    assert scalars.index("can_heal") < scalars.index("can_repair")


def test_later_levels_inherit_heal_repair_scalars():
    cm = ConfigManager("config_universal")
    for level in ("ctf_level_19", "ctf_level_20", "ctf_level_21"):
        scalars = cm.get_cumulative_active_unit_scalars(level)
        assert "can_heal" in scalars
        assert "can_repair" in scalars
