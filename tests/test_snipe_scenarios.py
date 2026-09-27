"""Snipe scenario levels sit after the heal revisit in phase 1."""
from pathlib import Path

from src.utils.config_manager import ConfigManager

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"

_PHASE1_SNIPE = (
    "snipe_ghost_marauder_vs_roach_hydra",
    "snipe_infestor_roach_vs_battlecruiser_scv",
    "snipe_high_templar_zealot_vs_ghost_medivac",
    "snipe_battlecruiser_marine_vs_corruptor_hydra",
)


def test_snipe_levels_follow_heal_revisit_in_pipeline():
    cm = ConfigManager(str(_CONFIG_DIR))
    idx_heal_revisit = cm.levels.index("scenario_queen_vs_battlecruiser")
    for offset, level in enumerate(_PHASE1_SNIPE, start=1):
        assert cm.levels.index(level) == idx_heal_revisit + offset, level
    # Phase 2 starts immediately after the last snipe intro.
    assert cm.levels.index("ctf_level_02") == idx_heal_revisit + len(_PHASE1_SNIPE) + 1


def test_first_snipe_level_adds_snipe_head_and_inherits_can_snipe():
    cm = ConfigManager(str(_CONFIG_DIR))
    first = _PHASE1_SNIPE[0]
    heads = cm.get_cumulative_action_heads(first)
    assert "snipe" in [h.lower() for h in heads]
    # can_snipe is declared once at ctf_level_01 (full scalar catalog).
    assert "can_snipe" in cm.get_cumulative_active_unit_scalars("ctf_level_01")
    assert "can_snipe" in cm.get_cumulative_active_unit_scalars(first)


def test_later_snipe_levels_do_not_relist_snipe_head():
    import yaml

    for level in _PHASE1_SNIPE[1:]:
        raw = yaml.safe_load((_CONFIG_DIR / f"{level}.yaml").read_text(encoding="utf-8"))
        heads = (raw.get("env") or {}).get("action_heads") or []
        assert raw.get("agent_architecture") == "squad"
        assert "snipe" in [str(h).lower() for h in heads] or heads == []
