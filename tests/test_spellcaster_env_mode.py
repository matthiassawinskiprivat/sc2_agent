"""CTF env_mode=spellcaster: max energy and spell-tech debug_upgrade ladder."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from train.envs.ctf.spellcaster import (
    SPELLCASTER_UPGRADE_CALLS,
    apply_max_energy,
    apply_spell_upgrades,
    energy_units,
)


def _unit(tag, energy_max, energy=0.0):
    u = MagicMock()
    u.tag = tag
    u.energy_max = energy_max
    u.energy = energy
    return u


def _bot(units):
    bot = MagicMock()
    bot.all_units = units
    bot.client.debug_set_unit_value = AsyncMock()
    bot.client.debug_upgrade = AsyncMock()
    bot.client.step = AsyncMock()
    return bot


def _make_env(env_mode=None):
    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    conf = {"env": {}}
    if env_mode is not None:
        conf["env"]["env_mode"] = env_mode
        conf["env_mode"] = env_mode
    env.training_conf = conf
    env.bot = _bot([])
    env._refresh_bot_state = AsyncMock()
    return env


def test_energy_units_skips_no_bar_and_bad_tags():
    ht = _unit(1, 200.0)
    marine = _unit(2, 0.0)
    ghost = _unit(0, 200.0)
    assert energy_units(MagicMock(all_units=[ht, marine, ghost])) == [(1, 200.0)]


def test_apply_max_energy_groups_by_energy_max():
    bot = _bot([_unit(1, 200.0), _unit(2, 200.0), _unit(3, 50.0), _unit(4, 0.0)])
    filled = asyncio.run(apply_max_energy(bot))
    assert filled == 3
    calls = bot.client.debug_set_unit_value.await_args_list
    by_value = {call.args[2]: set(call.args[0]) for call in calls}
    assert by_value[200.0] == {1, 2}
    assert by_value[50.0] == {3}
    for call in calls:
        assert call.args[1] == 1


def test_apply_spell_upgrades_runs_ladder():
    bot = _bot([])
    n = asyncio.run(
        apply_spell_upgrades(
            bot=bot, already_full_ladder=False, refresh_state=AsyncMock()
        )
    )
    assert n == SPELLCASTER_UPGRADE_CALLS
    assert bot.client.debug_upgrade.await_count == SPELLCASTER_UPGRADE_CALLS


def test_apply_spell_upgrades_skips_when_ladder_already_done():
    bot = _bot([])
    n = asyncio.run(
        apply_spell_upgrades(
            bot=bot, already_full_ladder=True, refresh_state=AsyncMock()
        )
    )
    assert n == 0
    bot.client.debug_upgrade.assert_not_called()


def test_spellcaster_env_runs_upgrades_and_energy():
    env = _make_env("heal,cloak,upgrades,spellcaster")
    env.bot.all_units = [_unit(11, 200.0)]
    asyncio.run(env._apply_spellcaster_env_mode_upgrades())
    asyncio.run(env._apply_spellcaster_energy())
    assert env.bot.client.debug_upgrade.await_count == SPELLCASTER_UPGRADE_CALLS
    env.bot.client.debug_set_unit_value.assert_awaited()


def test_spellcaster_skips_upgrade_ladder_when_cloak_already_ran_it():
    env = _make_env("heal,cloak,spellcaster")
    asyncio.run(env._apply_spellcaster_env_mode_upgrades())
    env.bot.client.debug_upgrade.assert_not_called()


def test_upgrades_mode_skipped_when_spellcaster_present():
    env = _make_env("upgrades,spellcaster")
    env.episode_counter = 50
    asyncio.run(env._apply_upgrades_env_mode())
    env.bot.client.debug_upgrade.assert_not_called()


def test_without_spellcaster_energy_and_upgrades_not_applied():
    env = _make_env("heal,cloak,upgrades")
    env.bot.all_units = [_unit(11, 200.0)]
    asyncio.run(env._apply_spellcaster_env_mode_upgrades())
    asyncio.run(env._apply_spellcaster_energy())
    env.bot.client.debug_upgrade.assert_not_called()
    env.bot.client.debug_set_unit_value.assert_not_called()
