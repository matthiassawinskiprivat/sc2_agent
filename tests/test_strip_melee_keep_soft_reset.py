"""SoftReset-only strip must not false-fail on the 'MeleeInit removed' comment."""

from pathlib import Path

import pytest

from tools.strip_melee_keep_soft_reset import SOFT_RESET_ONLY_MAPSCRIPT


def _has_melee_init_trigger(script: str) -> bool:
    return "gt_MeleeInitialization" in script


def test_canonical_soft_reset_script_has_no_melee_trigger():
    assert _has_melee_init_trigger(SOFT_RESET_ONLY_MAPSCRIPT) is False
    # Comment may mention MeleeInit; trigger id must stay absent.
    assert "MeleeInit" in SOFT_RESET_ONLY_MAPSCRIPT


def test_build_agent_maps_have_no_melee_init_trigger():
    maps_dir = Path("train/maps/ladder/build_agent")
    if not maps_dir.is_dir():
        pytest.skip("build_agent maps dir missing")
    try:
        from tools.inject_soft_reset_maps import _read_archive_text
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"map archive helpers unavailable: {exc}")

    maps = sorted(maps_dir.glob("*.SC2Map"))
    assert maps, "expected build_agent maps"
    bad = []
    for path in maps:
        try:
            script = _read_archive_text(path, "MapScript.galaxy") or ""
        except Exception as exc:
            bad.append(f"{path.name}: read failed ({exc})")
            continue
        if _has_melee_init_trigger(script):
            bad.append(f"{path.name}: still has gt_MeleeInitialization")
        elif "SOFT_RESET" not in script:
            bad.append(f"{path.name}: missing SOFT_RESET")
    assert not bad, "solo build maps must be SoftReset-only:\n" + "\n".join(bad)
