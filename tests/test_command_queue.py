from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.commands import Command, CommandType
from src.runtime.command_queue import CommandQueue
from src.runtime.scripted_manager import ManagerScriptStep, ScriptedManager


def test_command_queue_macro_activation():
    q = CommandQueue()
    q.enqueue(
        Command(command_type=CommandType.BUILD, unit_type_id=UnitTypeId.EXTRACTOR)
    )
    assert not q.has_active_macro()
    active = q.activate_next()
    assert active is not None
    assert q.has_active_macro()
    q.complete_active()
    assert not q.has_active_macro()


def test_command_queue_mine_is_standing_not_active_macro():
    q = CommandQueue()
    q.enqueue(
        Command(
            command_type=CommandType.MINE,
            target_pos=Point2((10, 10)),
            worker_count=8,
        )
    )
    q.enqueue(
        Command(
            command_type=CommandType.BUILD,
            unit_type_id=UnitTypeId.SPAWNINGPOOL,
            target_pos=Point2((10, 10)),
        )
    )
    assert q.standing_mine is not None
    assert q.standing_mine.command_type == CommandType.MINE
    assert q.active is None
    active = q.activate_next()
    assert active is not None
    assert active.command_type == CommandType.BUILD
    assert q.standing_mine.worker_count == 8


def test_scripted_manager_parses_mine_worker_count():
    mgr = ScriptedManager.from_level_config(
        {
            "macro_tick_interval": 4,
            "manager_script": [
                {
                    "command_type": "MINE",
                    "target_pos": [-1, -1],
                    "worker_count": 8,
                },
                {
                    "command_type": "BUILD",
                    "unit_type": "SPAWNINGPOOL",
                    "target_pos": [-1, -1],
                },
            ],
        }
    )
    assert len(mgr.script_steps) == 2
    assert mgr.script_steps[0].command.command_type == CommandType.MINE
    assert mgr.script_steps[0].command.worker_count == 8
    mgr._emit_step_if_available()
    assert mgr.command_queue.standing_mine is not None
    assert mgr.command_queue.active is None
    mgr._emit_step_if_available()
    assert mgr.command_queue.active.unit_type_id == UnitTypeId.SPAWNINGPOOL


def test_scripted_manager_parses_train_count():
    mgr = ScriptedManager.from_level_config(
        {
            "manager_script": [
                {
                    "command_type": "TRAIN",
                    "unit_type": "ZERGLING",
                    "count": 6,
                },
            ],
        }
    )
    assert mgr.script_steps[0].command.command_type == CommandType.TRAIN
    assert mgr.script_steps[0].command.count == 6
    assert mgr.script_steps[0].command.unit_type_id == UnitTypeId.ZERGLING


def test_emit_next_if_idle_does_not_emit_while_macro_active():
    mgr = ScriptedManager.from_level_config(
        {
            "manager_script": [
                {
                    "command_type": "BUILD",
                    "unit_type": "SPAWNINGPOOL",
                    "target_pos": [-1, -1],
                },
                {
                    "command_type": "TRAIN",
                    "unit_type": "ZERGLING",
                    "count": 6,
                },
            ],
        }
    )
    assert mgr.emit_next_if_idle() is True
    assert mgr.command_queue.active.command_type == CommandType.BUILD
    assert mgr.emit_next_if_idle() is False
    assert mgr._script_index == 1
    mgr.command_queue.complete_active()
    assert mgr.emit_next_if_idle() is True
    assert mgr.command_queue.active.command_type == CommandType.TRAIN
    assert mgr.command_queue.active.count == 6


def test_scripted_manager_emits_on_reset_and_macro_tick():
    cmd = Command(
        command_type=CommandType.BUILD,
        unit_type_id=UnitTypeId.SPAWNINGPOOL,
        target_pos=Point2((32, 32)),
    )
    mgr = ScriptedManager(
        script_steps=[ManagerScriptStep(command=cmd)],
        macro_tick_interval=4,
    )
    mgr.reset()
    assert mgr.command_queue.active is None
    mgr.on_game_step()
    assert mgr.command_queue.active is not None
    assert mgr.command_queue.active.unit_type_id == UnitTypeId.SPAWNINGPOOL
