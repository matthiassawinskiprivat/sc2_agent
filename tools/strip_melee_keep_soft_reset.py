"""Make build_agent maps SoftReset-only (no MeleeInit).

Melee Initialization + use_computer:false ends the game immediately
(\"Game has already ended\") before debug spawn can finish.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse StormLib writer / SoftReset TriggerData from the inject tool.
from tools.inject_soft_reset_maps import (
    SOFT_RESET_CHAT,
    SOFT_RESET_TRIGGER_ID,
    TRIGGERSTRINGS_LINE,
    SOFT_RESET_TRIGGERS_FRAGMENT,
    _find_storm_dll,
    _read_archive_text,
    _write_files_with_stormlib,
    _list_triggerstrings_keys,
)

SOFT_RESET_ONLY_MAPSCRIPT = '''//==================================================================================================
//
// SoftReset-only map script (build-agent training)
// MeleeInit removed: single-player sessions must not run melee victory/AI init.
//
//==================================================================================================
include "TriggerLibs/NativeLib"

//--------------------------------------------------------------------------------------------------
// Library Initialization
//--------------------------------------------------------------------------------------------------
void InitLibs () {
    libNtve_InitLib();
}

//--------------------------------------------------------------------------------------------------
// Trigger Variables
//--------------------------------------------------------------------------------------------------
trigger gt_SoftReset;

//--------------------------------------------------------------------------------------------------
// Trigger: SoftReset
//--------------------------------------------------------------------------------------------------
bool gt_SoftReset_Func (bool testConds, bool runActions) {
    // Automatic Variable Declarations
    // Actions
    if (!runActions) {
        return true;
    }

    PlayerModifyPropertyInt(1, c_playerPropVespene, c_playerPropOperSetTo, 0);
    PlayerModifyPropertyInt(1, c_playerPropMinerals, c_playerPropOperSetTo, 0);
    return true;
}

//--------------------------------------------------------------------------------------------------
void gt_SoftReset_Init () {
    gt_SoftReset = TriggerCreate("gt_SoftReset_Func");
    TriggerAddEventChatMessage(gt_SoftReset, c_playerAny, "SOFT_RESET", true);
}

//--------------------------------------------------------------------------------------------------
// Trigger Initialization
//--------------------------------------------------------------------------------------------------
void InitTriggers () {
    gt_SoftReset_Init();
}

//--------------------------------------------------------------------------------------------------
// Map Initialization
//--------------------------------------------------------------------------------------------------
void InitMap () {
    InitLibs();
    InitTriggers();
}
'''

SOFT_RESET_ONLY_TRIGGERS = f'''<?xml version="1.0" encoding="utf-8"?>
<TriggerData>
    <Root>
        <Item Type="Trigger" Id="{SOFT_RESET_TRIGGER_ID}"/>
    </Root>
{SOFT_RESET_TRIGGERS_FRAGMENT}</TriggerData>
'''


def _soft_reset_only_triggerstrings(existing: str | None) -> bytes:
    bom = "\ufeff"
    body = TRIGGERSTRINGS_LINE + "\n"
    if existing and existing.startswith("\ufeff"):
        return (bom + body).encode("utf-8")
    return body.encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=Path("train/maps/ladder/build_agent"),
    )
    parser.add_argument("--storm-dll", type=Path, default=None)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    maps_dir = args.maps_dir
    storm = _find_storm_dll(args.storm_dll)
    do_write = bool(args.write) and not args.dry_run
    if do_write and storm is None:
        print("ERROR: StormLib.dll required for --write", file=sys.stderr)
        return 1

    maps = sorted(maps_dir.glob("*.SC2Map"))
    if not maps:
        print(f"No maps in {maps_dir}", file=sys.stderr)
        return 1

    ok = failed = skipped = 0
    for map_path in maps:
        try:
            script = _read_archive_text(map_path, "MapScript.galaxy") or ""
            triggers = _read_archive_text(map_path, "Triggers") or ""
            needs = (
                "MeleeInit" in script
                or "gt_MeleeInitialization" in script
                or "MeleeInit" in triggers
                or SOFT_RESET_CHAT not in script
                or SOFT_RESET_CHAT not in triggers
                or script.count("gt_SoftReset") == 0
            )
            # Always rewrite to the canonical SoftReset-only pair for consistency.
            patches = {
                "MapScript.galaxy": SOFT_RESET_ONLY_MAPSCRIPT.encode("utf-8"),
                "Triggers": SOFT_RESET_ONLY_TRIGGERS.encode("utf-8"),
            }
            for key in _list_triggerstrings_keys(map_path):
                existing = _read_archive_text(map_path, key)
                patches[key] = _soft_reset_only_triggerstrings(existing)

            already_clean = (
                "MeleeInit" not in script
                and "gt_MeleeInitialization" not in script
                and SOFT_RESET_CHAT in script
                and SOFT_RESET_CHAT in triggers
                and "gt_MeleeInitialization" not in triggers
            )
            if already_clean and script.strip() == SOFT_RESET_ONLY_MAPSCRIPT.strip():
                print(f"[ok]     {map_path.name}")
                skipped += 1
                continue

            if args.dry_run or not do_write:
                print(f"[would]  {map_path.name} (strip MeleeInit, SoftReset-only)")
                ok += 1
                continue

            # Restore pristine ladder copy if we still have the first backup.
            bak = map_path.with_suffix(".SC2Map.bak")
            if bak.exists():
                shutil.copy2(bak, map_path)

            _write_files_with_stormlib(map_path, patches, storm)

            verify_s = _read_archive_text(map_path, "MapScript.galaxy") or ""
            verify_t = _read_archive_text(map_path, "Triggers") or ""
            if "MeleeInit" in verify_s or "gt_MeleeInitialization" in verify_s:
                raise RuntimeError("MeleeInit still present in MapScript")
            if SOFT_RESET_CHAT not in verify_s or SOFT_RESET_CHAT not in verify_t:
                raise RuntimeError("SoftReset missing after write")
            print(f"[wrote]  {map_path.name}")
            ok += 1
            _ = needs
        except Exception as exc:
            print(f"[FAIL]   {map_path.name}: {exc}", file=sys.stderr)
            failed += 1

    print(f"Done: {ok} changed/pending, {skipped} clean, {failed} failed")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
