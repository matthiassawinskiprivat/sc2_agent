"""Tests for INJECT command, standing combat, manager BO penalties, Zerg UC stub."""

import asyncio
from unittest.mock import MagicMock

import numpy as np
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.agent.commands import INJECT_ENERGY_COST, Command, CommandType
from src.agent.universal_unit_control_agent import UniversalUnitControlAgent
from src.agent.zerg_unit_control_agent import ZergUnitControlAgent
from src.runtime.command_queue import CommandQueue
from src.runtime.manager_bo_penalties import evaluate_manager_emission, penalty_amount
from src.runtime.opponent_config import resolve_opponent_config
from src.runtime.scripted_manager import ScriptedManager
from src.runtime.stub_combat_controller import StubCombatController
from src.zerg_build_bot import ZergBuildBot
from train.envs.sc2_session_config import (
    resolve_computer_difficulty,
    use_computer_player,
)


def test_inject_is_macro():
    cmd = Command(command_type=CommandType.INJECT)
    assert cmd.is_macro
    assert not cmd.is_combat


def test_command_queue_standing_combat_attack_defend():
    q = CommandQueue()
    q.enqueue(
        Command(command_type=CommandType.ATTACK, target_pos=Point2((10, 10)))
    )
    assert q.standing_combat is not None
    assert q.standing_combat.command_type == CommandType.ATTACK
    assert q.active is None
    q.enqueue(
        Command(
            command_type=CommandType.BUILD,
            unit_type_id=UnitTypeId.SPAWNINGPOOL,
            target_pos=Point2((10, 10)),
        )
    )
    active = q.activate_next()
    assert active.command_type == CommandType.BUILD
    assert q.standing_combat.command_type == CommandType.ATTACK

    q.enqueue(Command(command_type=CommandType.DEFEND, target_pos=Point2((5, 5))))
    assert q.standing_combat.command_type == CommandType.DEFEND


def test_stub_combat_attack_move():
    bot = MagicMock()
    bot.do = MagicMock()
    ling = MagicMock()
    ling.type_id = UnitTypeId.ZERGLING
    ling.is_structure = False
    ling.is_worker = False
    ling.attack = MagicMock(return_value="atk")
    bot.units = [ling]
    bot.townhalls = MagicMock()
    bot.townhalls.exists = True
    bot.townhalls.first.position = Point2((1, 1))

    ctrl = StubCombatController()
    ctrl.apply(
        bot,
        Command(command_type=CommandType.ATTACK, target_pos=Point2((40, 40))),
    )
    ling.attack.assert_called_once()
    bot.do.assert_called_once()


def test_scripted_manager_gates_inject_on_energy():
    energy = {"ok": False}

    def gate(step):
        if step.command.command_type == CommandType.INJECT:
            return energy["ok"]
        return True

    mgr = ScriptedManager.from_level_config(
        {
            "manager_script": [
                {"command_type": "INJECT", "target_pos": [-1, -1]},
            ]
        },
        can_emit_step=gate,
    )
    assert mgr.emit_next_if_idle() is False
    assert mgr._script_index == 0
    energy["ok"] = True
    assert mgr.emit_next_if_idle() is True
    assert mgr.command_queue.active.command_type == CommandType.INJECT


def test_apply_inject_success_and_reward():
    async def _run():
        bot = ZergBuildBot(
            training_conf={
                "manager_script": [
                    {"command_type": "INJECT", "target_pos": [40.0, 40.0]},
                ],
                "rewards": {
                    "command_complete": 15.0,
                    "inject_reward": 20.0,
                    "step_penalty": 0.0,
                },
            }
        )
        bot.scripted_manager.can_emit_step = None
        bot.scripted_manager._emit_step_if_available()
        assert bot.command_queue.active.command_type == CommandType.INJECT

        queen = MagicMock()
        queen.tag = 9
        queen.type_id = UnitTypeId.QUEEN
        queen.energy = INJECT_ENERGY_COST + 5
        queen.position = Point2((40.0, 40.0))
        hatch = MagicMock()
        hatch.position = Point2((40.0, 40.0))
        hatch.distance_to = lambda p: 0.0
        bot._queens = lambda: [queen]
        bot.townhalls = [hatch]
        bot.do = MagicMock()

        ok = await bot.apply_build_action(np.array([0, 8 * 16 + 8], dtype=np.int64))
        assert ok
        assert bot._inject_succeeded is True
        assert bot._try_complete_active_command() is True

    asyncio.run(_run())


def test_train_queen_requires_pool_and_uses_hatch():
    async def _run():
        bot = ZergBuildBot(
            training_conf={
                "manager_script": [
                    {
                        "command_type": "TRAIN",
                        "unit_type": "QUEEN",
                        "count": 1,
                    }
                ],
            }
        )
        bot.scripted_manager._emit_step_if_available()
        bot._train_baseline = 0
        hatch = MagicMock()
        hatch.train = MagicMock(return_value="q")
        ready = MagicMock()
        ready.exists = True
        ready.random = hatch
        ready.__bool__ = lambda self: True
        ready.__iter__ = lambda self: iter([hatch])
        bot.townhalls = MagicMock()
        bot.townhalls.ready = ready
        pool = MagicMock()
        pool.ready.exists = False
        bot.structures = MagicMock(return_value=pool)
        bot.can_afford = MagicMock(return_value=True)
        bot.do = MagicMock()
        bot.workers = [MagicMock(tag=1, position=Point2((1, 1)))]

        # No pool → invalid
        assert bot._train_tech_ready(bot.command_queue.active) is False

        pool.ready.exists = True
        bot.structures = MagicMock(side_effect=lambda ut: pool if ut == UnitTypeId.SPAWNINGPOOL else MagicMock(ready=MagicMock(exists=False)))
        assert bot._train_tech_ready(bot.command_queue.active) is True
        await bot._apply_train(bot.command_queue.active, bot.workers[0])
        hatch.train.assert_called_once_with(UnitTypeId.QUEEN)

    asyncio.run(_run())


def test_zerg_unit_control_stub():
    agent = ZergUnitControlAgent()
    assert agent.agent_kind == "zerg_unit_control"
    assert isinstance(agent, UniversalUnitControlAgent)


def test_opponent_config_and_difficulty():
    conf = {
        "env": {
            "opponent": {"mode": "computer", "difficulty": "Medium"},
            "use_computer": True,
        }
    }
    opp = resolve_opponent_config(conf)
    assert opp["mode"] == "computer"
    assert opp["difficulty"] == "Medium"
    assert resolve_computer_difficulty(conf) == "Medium"

    assert use_computer_player("level_manager_zerg_0", {}) is True
    assert use_computer_player("level_build_zerg_0", {}) is False
    assert use_computer_player(
        "level_build_zerg_0", {"use_computer": True}
    ) is True


def test_manager_bo_penalties():
    assert (
        evaluate_manager_emission(
            Command(command_type=CommandType.TRAIN, unit_type_id=UnitTypeId.ZERGLING),
            has_pool=False,
            structure_counts={},
        )
        == "wrong_build_order"
    )
    assert (
        evaluate_manager_emission(
            Command(
                command_type=CommandType.BUILD, unit_type_id=UnitTypeId.SPAWNINGPOOL
            ),
            has_pool=True,
            structure_counts={UnitTypeId.SPAWNINGPOOL: 1},
        )
        == "surplus_building"
    )
    assert penalty_amount({"wrong_build_order_penalty": -7.0}, "wrong_build_order") == -7.0


def test_cumulative_build_commands_include_inject(tmp_path):
    (tmp_path / "pipeline.yaml").write_text(
        "levels:\n  - lvl0\nbuild_map_layers: []\n"
    )
    (tmp_path / "lvl0.yaml").write_text(
        "env:\n  build_commands: [spawningpool, zergling, overlord, queen, inject]\n"
    )
    from src.utils.config_manager import ConfigManager

    cm = ConfigManager(str(tmp_path))
    cmds = cm.get_cumulative_build_commands("lvl0")
    assert "inject" in cmds
    assert "queen" in cmds
    assert "overlord" in cmds


def test_manager_active_layers_accumulate(tmp_path):
    (tmp_path / "pipeline.yaml").write_text(
        "levels:\n  - build0\n  - mgr0\n"
    )
    (tmp_path / "build0.yaml").write_text("env:\n  build_commands: [spawningpool]\n")
    (tmp_path / "mgr0.yaml").write_text(
        "env:\n  active_layers: [terrain_pathable, terrain_creep]\n"
    )
    from src.utils.config_manager import ConfigManager

    cm = ConfigManager(str(tmp_path))
    layers = cm.get_cumulative_active_layers("mgr0")
    assert layers == ["terrain_pathable", "terrain_creep"]
