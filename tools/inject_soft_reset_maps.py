"""Inject SoftReset into build_agent SC2Map files (MapScript + Editor Triggers).

The SC2 Editor UI reads ``Triggers`` / TriggerStrings, while the game runs
``MapScript.galaxy``. Both must contain SoftReset or the Editor will look empty
and saving from the Editor can wipe a MapScript-only patch.

Requires StormLib.dll for writing.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SOFT_RESET_MARKER = "gt_SoftReset"
SOFT_RESET_CHAT = "SOFT_RESET"
SOFT_RESET_TRIGGER_ID = "A897E218"
TRIGGERSTRINGS_LINE = f"Trigger/Name/{SOFT_RESET_TRIGGER_ID}=SoftReset"

SOFT_RESET_TRIGGER_BLOCK = """
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
""".lstrip(
    "\n"
)

# Editor TriggerData fragment copied from AbyssalReefAIE SoftReset (IDs fixed).
SOFT_RESET_TRIGGERS_FRAGMENT = """
    <Element Type="Trigger" Id="A897E218">
        <Event Type="FunctionCall" Id="CD0188F3"/>
        <Action Type="FunctionCall" Id="CC245E2B"/>
        <Action Type="FunctionCall" Id="8BC7B11B"/>
    </Element>
    <Element Type="FunctionCall" Id="CD0188F3">
        <FunctionDef Type="FunctionDef" Library="Ntve" Id="00000121"/>
        <Parameter Type="Param" Id="8C15D66D"/>
        <Parameter Type="Param" Id="D629B670"/>
        <Parameter Type="Param" Id="F256137A"/>
    </Element>
    <Element Type="Param" Id="8C15D66D">
        <ParameterDef Type="ParamDef" Library="Ntve" Id="00000191"/>
        <Preset Type="PresetValue" Library="Ntve" Id="2999701E"/>
    </Element>
    <Element Type="Param" Id="D629B670">
        <ParameterDef Type="ParamDef" Library="Ntve" Id="00000193"/>
        <Preset Type="PresetValue" Library="Ntve" Id="00000073"/>
    </Element>
    <Element Type="Param" Id="F256137A">
        <ParameterDef Type="ParamDef" Library="Ntve" Id="00000192"/>
        <Value>SOFT_RESET</Value>
        <ValueType Type="string"/>
    </Element>
    <Element Type="FunctionCall" Id="CC245E2B">
        <FunctionDef Type="FunctionDef" Library="Ntve" Id="943DF2F7"/>
        <Parameter Type="Param" Id="8A1CDDA7"/>
        <Parameter Type="Param" Id="03567DF5"/>
        <Parameter Type="Param" Id="FB4837DF"/>
        <Parameter Type="Param" Id="05A0D565"/>
    </Element>
    <Element Type="Param" Id="8A1CDDA7">
        <ParameterDef Type="ParamDef" Library="Ntve" Id="75FC31CB"/>
        <Value>1</Value>
        <ValueType Type="int"/>
    </Element>
    <Element Type="Param" Id="03567DF5">
        <ParameterDef Type="ParamDef" Library="Ntve" Id="4AA9487D"/>
        <Preset Type="PresetValue" Library="Ntve" Id="D41E85DA"/>
    </Element>
    <Element Type="Param" Id="FB4837DF">
        <ParameterDef Type="ParamDef" Library="Ntve" Id="C93A9DCB"/>
        <Preset Type="PresetValue" Library="Ntve" Id="D6715A64"/>
    </Element>
    <Element Type="Param" Id="05A0D565">
        <ParameterDef Type="ParamDef" Library="Ntve" Id="EEF303AC"/>
        <Value>0</Value>
        <ValueType Type="int"/>
    </Element>
    <Element Type="FunctionCall" Id="8BC7B11B">
        <FunctionDef Type="FunctionDef" Library="Ntve" Id="943DF2F7"/>
        <Parameter Type="Param" Id="ECF76B12"/>
        <Parameter Type="Param" Id="63E2DA00"/>
        <Parameter Type="Param" Id="06A3CF98"/>
        <Parameter Type="Param" Id="6879074E"/>
    </Element>
    <Element Type="Param" Id="ECF76B12">
        <ParameterDef Type="ParamDef" Library="Ntve" Id="75FC31CB"/>
        <Value>1</Value>
        <ValueType Type="int"/>
    </Element>
    <Element Type="Param" Id="63E2DA00">
        <ParameterDef Type="ParamDef" Library="Ntve" Id="4AA9487D"/>
        <Preset Type="PresetValue" Library="Ntve" Id="0794D450"/>
    </Element>
    <Element Type="Param" Id="06A3CF98">
        <ParameterDef Type="ParamDef" Library="Ntve" Id="C93A9DCB"/>
        <Preset Type="PresetValue" Library="Ntve" Id="D6715A64"/>
    </Element>
    <Element Type="Param" Id="6879074E">
        <ParameterDef Type="ParamDef" Library="Ntve" Id="EEF303AC"/>
        <Value>0</Value>
        <ValueType Type="int"/>
    </Element>
""".rstrip() + "\n"


def _inject_mapscript(script: str) -> Optional[str]:
    """Return patched MapScript.galaxy, or None if already present."""
    if SOFT_RESET_MARKER in script:
        return None

    if "trigger gt_MeleeInitialization;" in script and "trigger gt_SoftReset;" not in script:
        script = script.replace(
            "trigger gt_MeleeInitialization;",
            "trigger gt_MeleeInitialization;\ntrigger gt_SoftReset;",
            1,
        )
    elif "trigger gt_SoftReset;" not in script:
        if "void InitMap" in script:
            script = script.replace(
                "void InitMap",
                "trigger gt_SoftReset;\n\nvoid InitMap",
                1,
            )
        else:
            script = "trigger gt_SoftReset;\n" + script

    if "void InitTriggers" in script:
        script = script.replace(
            "void InitTriggers",
            SOFT_RESET_TRIGGER_BLOCK + "\nvoid InitTriggers",
            1,
        )
    elif "void InitMap" in script:
        script = script.replace(
            "void InitMap",
            SOFT_RESET_TRIGGER_BLOCK
            + "\nvoid InitTriggers () {\n    gt_SoftReset_Init();\n}\n\nvoid InitMap",
            1,
        )
    else:
        raise ValueError("MapScript has no InitTriggers/InitMap")

    if "gt_SoftReset_Init();" not in script:
        if "gt_MeleeInitialization_Init();" in script:
            script = script.replace(
                "gt_MeleeInitialization_Init();",
                "gt_MeleeInitialization_Init();\n    gt_SoftReset_Init();",
                1,
            )
        else:
            script = re.sub(
                r"(void InitTriggers\s*\(\s*\)\s*\{)",
                r"\1\n    gt_SoftReset_Init();",
                script,
                count=1,
            )

    if re.search(r"void InitMap\s*\(\s*\)\s*\{\s*\}", script):
        script = re.sub(
            r"void InitMap\s*\(\s*\)\s*\{\s*\}",
            "void InitMap () {\n    InitTriggers();\n}",
            script,
            count=1,
        )
    elif "void InitMap" in script and "InitTriggers();" not in script.split("void InitMap", 1)[1]:
        script = re.sub(
            r"(void InitMap\s*\(\s*\)\s*\{)",
            r"\1\n    InitTriggers();",
            script,
            count=1,
        )

    return script


def _inject_triggers_xml(xml: str) -> Optional[str]:
    """Merge SoftReset TriggerData elements so the Editor UI shows the trigger."""
    if SOFT_RESET_CHAT in xml or SOFT_RESET_TRIGGER_ID in xml:
        return None
    if "</Root>" not in xml or "</TriggerData>" not in xml:
        raise ValueError("Triggers XML missing Root/TriggerData")
    xml = xml.replace(
        "</Root>",
        f'        <Item Type="Trigger" Id="{SOFT_RESET_TRIGGER_ID}"/>\n    </Root>',
        1,
    )
    xml = xml.replace("</TriggerData>", SOFT_RESET_TRIGGERS_FRAGMENT + "</TriggerData>", 1)
    return xml


def _inject_triggerstrings(text: str) -> Optional[str]:
    if SOFT_RESET_TRIGGER_ID in text and "SoftReset" in text:
        return None
    # Preserve UTF-8 BOM if present.
    bom = ""
    body = text
    if body.startswith("\ufeff"):
        bom = "\ufeff"
        body = body[1:]
    body = body.rstrip("\r\n") + "\n" + TRIGGERSTRINGS_LINE + "\n"
    return bom + body


def _read_archive_text(map_path: Path, name: str) -> Optional[str]:
    from mpyq import MPQArchive

    archive = MPQArchive(str(map_path))
    raw = archive.read_file(name)
    if raw is None and "\\" not in name and "/" in name:
        raw = archive.read_file(name.replace("/", "\\"))
    if raw is None:
        return None
    return raw.decode("utf-8", errors="replace")


def _list_triggerstrings_keys(map_path: Path) -> List[str]:
    from mpyq import MPQArchive

    archive = MPQArchive(str(map_path))
    out = []
    for f in archive.files:
        n = f.decode() if isinstance(f, bytes) else f
        if n.endswith("TriggerStrings.txt"):
            out.append(n)
    return out


def _read_mapscript(map_path: Path) -> str:
    text = _read_archive_text(map_path, "MapScript.galaxy")
    if text is None:
        raise RuntimeError(f"{map_path.name}: missing MapScript.galaxy")
    return text


def _plan_patches(map_path: Path) -> Dict[str, bytes]:
    """Compute archive files that need updating for SoftReset (Editor + runtime)."""
    patches: Dict[str, bytes] = {}

    script = _read_mapscript(map_path)
    new_script = _inject_mapscript(script)
    if new_script is None:
        new_script = script  # keep existing SoftReset MapScript
    elif SOFT_RESET_MARKER not in new_script:
        raise RuntimeError("MapScript patch missing SoftReset")
    if new_script != script:
        patches["MapScript.galaxy"] = new_script.encode("utf-8")

    triggers = _read_archive_text(map_path, "Triggers")
    if triggers is None:
        raise RuntimeError(f"{map_path.name}: missing Triggers")
    new_triggers = _inject_triggers_xml(triggers)
    if new_triggers is not None:
        patches["Triggers"] = new_triggers.encode("utf-8")

    for key in _list_triggerstrings_keys(map_path):
        text = _read_archive_text(map_path, key)
        if text is None:
            continue
        new_text = _inject_triggerstrings(text)
        if new_text is not None:
            patches[key] = new_text.encode("utf-8")

    # Ensure at least enUS strings exist conceptually; if map has no TriggerStrings, skip.
    return patches


def _write_files_with_stormlib(
    map_path: Path, files: Dict[str, bytes], storm_dll: Path
) -> None:
    """Replace/add archived files inside an SC2Map using StormLib CreateFile API."""
    import ctypes
    from ctypes import wintypes

    dll = ctypes.WinDLL(str(storm_dll))

    def bind(name, argtypes, restype=wintypes.BOOL):
        fn = getattr(dll, name)
        fn.argtypes = argtypes
        fn.restype = restype
        return fn

    SFileOpenArchive = bind(
        "SFileOpenArchive",
        [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)],
    )
    SFileCloseArchive = bind("SFileCloseArchive", [ctypes.c_void_p])
    SFileRemoveFile = bind(
        "SFileRemoveFile", [ctypes.c_void_p, wintypes.LPCSTR, wintypes.DWORD]
    )
    SFileCreateFile = bind(
        "SFileCreateFile",
        [
            ctypes.c_void_p,
            wintypes.LPCSTR,
            ctypes.c_ulonglong,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
        ],
    )
    SFileWriteFile = bind(
        "SFileWriteFile",
        [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD],
    )
    SFileFinishFile = bind("SFileFinishFile", [ctypes.c_void_p])
    SFileFlushArchive = bind("SFileFlushArchive", [ctypes.c_void_p])
    SFileCompactArchive = bind(
        "SFileCompactArchive", [ctypes.c_void_p, wintypes.LPCWSTR, wintypes.BOOL]
    )

    MPQ_FILE_COMPRESS = 0x00000200
    MPQ_FILE_REPLACEEXISTING = 0x80000000
    MPQ_FILE_SINGLE_UNIT = 0x01000000
    MPQ_COMPRESSION_ZLIB = 0x02

    h_mpq = ctypes.c_void_p()
    if not SFileOpenArchive(str(map_path), 0, 0, ctypes.byref(h_mpq)):
        err = ctypes.GetLastError()
        raise RuntimeError(f"SFileOpenArchive failed for {map_path} (err={err})")

    try:
        for archived_name, data in files.items():
            archived_bytes = archived_name.encode("utf-8")
            # Best-effort remove prior entry.
            SFileRemoveFile(h_mpq, archived_bytes, 0)
            h_file = ctypes.c_void_p()
            flags = MPQ_FILE_COMPRESS | MPQ_FILE_REPLACEEXISTING | MPQ_FILE_SINGLE_UNIT
            if not SFileCreateFile(
                h_mpq,
                archived_bytes,
                0,
                len(data),
                0,
                flags,
                ctypes.byref(h_file),
            ):
                err = ctypes.GetLastError()
                raise RuntimeError(
                    f"SFileCreateFile failed for {archived_name} (err={err})"
                )
            buf = (ctypes.c_char * len(data)).from_buffer_copy(data)
            if not SFileWriteFile(h_file, buf, len(data), MPQ_COMPRESSION_ZLIB):
                err = ctypes.GetLastError()
                raise RuntimeError(
                    f"SFileWriteFile failed for {archived_name} (err={err})"
                )
            if not SFileFinishFile(h_file):
                err = ctypes.GetLastError()
                raise RuntimeError(
                    f"SFileFinishFile failed for {archived_name} (err={err})"
                )
        SFileFlushArchive(h_mpq)
        SFileCompactArchive(h_mpq, None, False)
        SFileFlushArchive(h_mpq)
    finally:
        SFileCloseArchive(h_mpq)


def _find_storm_dll(explicit: Optional[Path]) -> Optional[Path]:
    candidates = []
    if explicit:
        candidates.append(explicit)
    root = Path(__file__).resolve().parent
    candidates += [
        root / "StormLib.dll",
        root / "lib" / "StormLib.dll",
        root / "stormlib" / "StormLib.dll",
        root / "stormlib_dll" / "x64" / "StormLib.dll",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _has_editor_soft_reset(map_path: Path) -> bool:
    triggers = _read_archive_text(map_path, "Triggers") or ""
    return SOFT_RESET_CHAT in triggers and SOFT_RESET_TRIGGER_ID in triggers


def _has_runtime_soft_reset(map_path: Path) -> bool:
    script = _read_mapscript(map_path)
    return SOFT_RESET_MARKER in script and SOFT_RESET_CHAT in script


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=Path("train/maps/ladder/build_agent"),
    )
    parser.add_argument("--storm-dll", type=Path, default=None)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write patched files back into maps (requires StormLib.dll)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would change",
    )
    args = parser.parse_args()

    maps_dir = args.maps_dir
    if not maps_dir.is_dir():
        print(f"Maps dir not found: {maps_dir}", file=sys.stderr)
        return 1

    storm = _find_storm_dll(args.storm_dll)
    do_write = bool(args.write) and not args.dry_run
    if do_write and storm is None:
        print(
            "ERROR: --write needs StormLib.dll next to this script or via --storm-dll.",
            file=sys.stderr,
        )
        do_write = False

    maps = sorted(maps_dir.glob("*.SC2Map"))
    if not maps:
        print(f"No SC2Map files in {maps_dir}", file=sys.stderr)
        return 1

    ok = already = failed = 0
    for map_path in maps:
        try:
            patches = _plan_patches(map_path)
            runtime_ok = _has_runtime_soft_reset(map_path) or "MapScript.galaxy" in patches
            editor_ok = _has_editor_soft_reset(map_path) or "Triggers" in patches

            if not patches and _has_runtime_soft_reset(map_path) and _has_editor_soft_reset(map_path):
                print(f"[ok]     {map_path.name} (runtime+Editor SoftReset)")
                already += 1
                continue

            # If MapScript already good but Triggers missing, patches should include Triggers.
            if not patches:
                # Force re-check: maybe MapScript missing SoftReset after Editor save.
                script = _read_mapscript(map_path)
                if SOFT_RESET_MARKER not in script:
                    patched = _inject_mapscript(script)
                    if patched:
                        patches["MapScript.galaxy"] = patched.encode("utf-8")
                triggers = _read_archive_text(map_path, "Triggers") or ""
                if SOFT_RESET_CHAT not in triggers:
                    new_t = _inject_triggers_xml(triggers)
                    if new_t:
                        patches["Triggers"] = new_t.encode("utf-8")

            if not patches:
                print(f"[ok]     {map_path.name}")
                already += 1
                continue

            keys = ", ".join(sorted(patches))
            if args.dry_run or not do_write:
                print(f"[would]  {map_path.name} ({keys})")
                ok += 1
                continue

            bak = map_path.with_suffix(".SC2Map.bak")
            if not bak.exists():
                shutil.copy2(map_path, bak)
            _write_files_with_stormlib(map_path, patches, storm)

            if not _has_runtime_soft_reset(map_path):
                raise RuntimeError("runtime SoftReset verification failed")
            if not _has_editor_soft_reset(map_path):
                raise RuntimeError("Editor Triggers SoftReset verification failed")
            print(f"[wrote]  {map_path.name} ({keys})")
            ok += 1
            _ = (runtime_ok, editor_ok)
        except Exception as exc:
            print(f"[FAIL]   {map_path.name}: {exc}", file=sys.stderr)
            failed += 1

    print(
        f"Done: {ok} patched/pending, {already} already ok, {failed} failed "
        f"(write={'on' if do_write else 'off'})"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
