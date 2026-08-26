"""Curriculum contract for level_build_zerg_1 manager script."""

from pathlib import Path

import yaml
from sc2.ids.unit_typeid import UnitTypeId

from src.agent.commands import CommandType
from src.runtime.scripted_manager import ScriptedManager
from src.utils.config_manager import ConfigManager


def test_level_build_zerg_1_script_order():
    path = Path("config_build_zerg/level_build_zerg_1.yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    mgr = ScriptedManager.from_level_config(cfg["env"])
    steps = mgr.script_steps
    assert [s.command.command_type for s in steps] == [
        CommandType.BUILD,
        CommandType.TRAIN,
        CommandType.MINE,
        CommandType.MINE_GAS,
        CommandType.BUILD,
        CommandType.BUILD,
        CommandType.TRAIN,
        CommandType.TRAIN,
    ]
    assert steps[0].command.unit_type_id == UnitTypeId.EXTRACTOR
    assert steps[1].command.unit_type_id == UnitTypeId.DRONE
    assert steps[1].command.count == 8
    assert steps[2].command.worker_count == 14
    assert steps[3].command.command_type == CommandType.MINE_GAS
    assert steps[3].command.worker_count == 3
    assert steps[4].command.unit_type_id == UnitTypeId.SPAWNINGPOOL
    assert steps[5].command.unit_type_id == UnitTypeId.ROACHWARREN
    assert steps[6].command.unit_type_id == UnitTypeId.OVERLORD
    assert steps[6].command.count == 3
    assert steps[7].command.unit_type_id == UnitTypeId.ROACH
    assert steps[7].command.count == 20


def test_level_build_zerg_1_cumulative_build_commands():
    """L1 is available off-pipeline; its YAML commands accumulate when targeted."""
    cm = ConfigManager("config_build_zerg")
    cmds = cm.get_cumulative_build_commands("level_build_zerg_1")
    # Off-pipeline: L1 YAML only (plus any pipeline walk that completed before miss).
    # Pipeline currently ends at manager L0; L1 file still defines its append list.
    for name in ("extractor", "drone", "roachwarren", "overlord", "roach"):
        assert name in cmds


def test_pipeline_build_then_manager():
    cfg = yaml.safe_load(
        Path("config_build_zerg/pipeline.yaml").read_text(encoding="utf-8")
    )
    assert "level_build_zerg_0" in cfg["levels"]
    assert "level_manager_zerg_0" in cfg["levels"]
    assert cfg["levels"].index("level_build_zerg_0") < cfg["levels"].index(
        "level_manager_zerg_0"
    )
