from pathlib import Path

from tools.inject_soft_reset_maps import (
    SOFT_RESET_CHAT,
    SOFT_RESET_TRIGGER_ID,
    _inject_mapscript,
    _inject_triggers_xml,
    _inject_triggerstrings,
)


def test_inject_soft_reset_into_melee_mapscript():
    script = """include \"TriggerLibs/NativeLib\"

trigger gt_MeleeInitialization;

bool gt_MeleeInitialization_Func (bool testConds, bool runActions) {
    if (!runActions) {
        return true;
    }
    MeleeInitResources();
    return true;
}

void gt_MeleeInitialization_Init () {
    gt_MeleeInitialization = TriggerCreate(\"gt_MeleeInitialization_Func\");
    TriggerAddEventMapInit(gt_MeleeInitialization);
}

void InitTriggers () {
    gt_MeleeInitialization_Init();
}

void InitMap () {
    InitLibs();
    InitTriggers();
}
"""
    patched = _inject_mapscript(script)
    assert patched is not None
    assert "trigger gt_SoftReset;" in patched
    assert SOFT_RESET_CHAT in patched
    assert "gt_SoftReset_Init();" in patched
    assert "PlayerModifyPropertyInt(1, c_playerPropMinerals" in patched
    # Idempotent
    assert _inject_mapscript(patched) is None


def test_inject_soft_reset_into_triggers_xml():
    xml = """<?xml version=\"1.0\" encoding=\"utf-8\"?>
<TriggerData>
    <Root>
        <Item Type=\"Trigger\" Id=\"3371C661\"/>
    </Root>
    <Element Type=\"Trigger\" Id=\"3371C661\">
        <Event Type=\"FunctionCall\" Id=\"DF5DE3E2\"/>
    </Element>
</TriggerData>
"""
    patched = _inject_triggers_xml(xml)
    assert patched is not None
    assert SOFT_RESET_TRIGGER_ID in patched
    assert SOFT_RESET_CHAT in patched
    assert _inject_triggers_xml(patched) is None


def test_inject_triggerstrings_adds_softreset_name():
    text = "Trigger/Name/3371C661=Melee Initialization\n"
    patched = _inject_triggerstrings(text)
    assert patched is not None
    assert "SoftReset" in patched
    assert SOFT_RESET_TRIGGER_ID in patched
    assert _inject_triggerstrings(patched) is None


def test_build_agent_maps_all_have_soft_reset():
    maps_dir = Path("train/maps/ladder/build_agent")
    if not maps_dir.is_dir():
        return
    from mpyq import MPQArchive

    maps = sorted(maps_dir.glob("*.SC2Map"))
    assert maps, "expected build_agent maps"
    missing = []
    for path in maps:
        archive = MPQArchive(str(path))
        script = (archive.read_file("MapScript.galaxy") or b"").decode(
            "utf-8", errors="replace"
        )
        triggers = (archive.read_file("Triggers") or b"").decode("utf-8", errors="replace")
        if "gt_SoftReset" not in script or SOFT_RESET_CHAT not in script:
            missing.append(f"{path.name}:MapScript")
        if SOFT_RESET_CHAT not in triggers:
            missing.append(f"{path.name}:Triggers")
    assert missing == [], f"maps missing SoftReset: {missing}"
