from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.commands import CommandType
from src.runtime.command_queue import CommandQueue
from src.runtime.scripted_manager import ScriptedManager, ManagerScriptStep
from src.agent.commands import Command


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
