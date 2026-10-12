"""CTF mineral/vespene grants for agent and opponent (all races)."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

_HELPER = Path(__file__).resolve().parents[1] / "train" / "envs" / "ctf" / "heal_resources.py"
_spec = importlib.util.spec_from_file_location("heal_resources", _HELPER)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
apply_ctf_resources = _mod.apply_ctf_resources
terran_player_ids = _mod.terran_player_ids
CTF_RESOURCE_FLOOR = _mod.CTF_RESOURCE_FLOOR
HEAL_TERRAN_RESOURCE_FLOOR = _mod.HEAL_TERRAN_RESOURCE_FLOOR


def test_resource_floor_is_100():
    assert CTF_RESOURCE_FLOOR == 100
    assert HEAL_TERRAN_RESOURCE_FLOOR == CTF_RESOURCE_FLOOR


def test_terran_player_ids_from_pools():
    pools = [
        {"race": "terran", "agent": True},
        {"race": "zerg"},
        {"race": "Terran"},
    ]
    pids = terran_player_ids(pools, lambda p: 1 if p.get("agent") else 2)
    assert pids == {1, 2}


def _bot(*, minerals=0, vespene=0):
    bot = MagicMock()
    bot.minerals = minerals
    bot.vespene = vespene
    bot.client.debug_all_resources = AsyncMock()
    bot.client.debug_minerals = AsyncMock()
    bot.client.debug_gas = AsyncMock()
    bot.client.debug_free = AsyncMock()
    bot.client.step = AsyncMock()
    return bot


def test_grants_all_resources_when_either_bank_low():
    """debug_gas is broken — use debug_all_resources for minerals + vespene."""
    bot = _bot(minerals=5000, vespene=0)

    free = asyncio.run(apply_ctf_resources(bot=bot, free_enabled=True))

    bot.client.debug_all_resources.assert_awaited_once()
    bot.client.debug_minerals.assert_not_awaited()
    bot.client.debug_gas.assert_not_awaited()
    bot.client.debug_free.assert_not_awaited()
    assert free is True


def test_grants_all_resources_when_both_empty():
    bot = _bot()
    asyncio.run(apply_ctf_resources(bot=bot, free_enabled=True))
    bot.client.debug_all_resources.assert_awaited_once()


def test_does_not_enable_debug_free():
    """debug_free also waives ability energy, so Transfusion would not spend it."""
    bot = _bot()

    free = asyncio.run(apply_ctf_resources(bot=bot, free_enabled=False))

    assert free is False
    bot.client.debug_free.assert_not_awaited()
    bot.client.debug_all_resources.assert_awaited_once()


def test_grants_without_heal_mode_and_for_any_race():
    """Resources are unconditional — not gated on heal mode or Terran pools."""
    bot = _bot()
    free = asyncio.run(apply_ctf_resources(bot=bot, free_enabled=False))
    assert free is False
    bot.client.debug_all_resources.assert_awaited_once()
    bot.client.debug_free.assert_not_awaited()


def test_skips_topup_when_both_banks_at_or_above_floor():
    bot = _bot(minerals=CTF_RESOURCE_FLOOR, vespene=CTF_RESOURCE_FLOOR)
    asyncio.run(apply_ctf_resources(bot=bot, free_enabled=True))
    bot.client.debug_all_resources.assert_not_awaited()
    bot.client.debug_free.assert_not_awaited()


def test_env_grants_resources_without_heal_mode():
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.training_conf = {"env": {}}
    env.bot = _bot()
    env._refresh_bot_state = AsyncMock()
    env._ctf_resource_free_enabled = False

    asyncio.run(env._apply_ctf_resources())

    env.bot.client.debug_all_resources.assert_awaited_once()
    env.bot.client.debug_free.assert_not_awaited()
    assert env._ctf_resource_free_enabled is False
    env._refresh_bot_state.assert_awaited_once()
