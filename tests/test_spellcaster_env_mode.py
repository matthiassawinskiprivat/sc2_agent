"""CTF env_mode=spellcaster: max energy and spell-tech debug_upgrade ladder."""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import yaml

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from train.envs.ctf.pool_assignment import static_spawn_player_id
from train.envs.ctf.spellcaster import (
    RACE_SPELLCASTER_UNIT,
    SPELLCASTER_CAST_UPGRADES,
    SPELLCASTER_UPGRADE_CALLS,
    apply_max_energy,
    apply_spell_upgrades,
    energy_units,
)

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"
_CASTERS = ("HIGHTEMPLAR", "GHOST", "INFESTOR")


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


def test_heal_mode_applies_max_energy_without_spell_upgrades():
    """Queens spawn at 25 energy; Transfuse costs 50, so heal fills bars."""
    env = _make_env("heal")
    env.bot.all_units = [_unit(11, 200.0)]
    asyncio.run(env._apply_spellcaster_env_mode_upgrades())
    asyncio.run(env._apply_spellcaster_energy())
    env.bot.client.debug_upgrade.assert_not_called()
    env.bot.client.debug_set_unit_value.assert_awaited()


def test_phase1_cloak_ladder_runs_for_player_one_and_spellcaster_skips():
    """Phase-1 AoE relies on the cloak ladder. That ladder is the player-1 cheat."""
    env = _make_env("heal,cloak,spellcaster")
    asyncio.run(env._apply_cloak_env_mode_upgrades())
    asyncio.run(env._apply_spellcaster_env_mode_upgrades())
    assert env.bot.client.debug_upgrade.await_count == SPELLCASTER_UPGRADE_CALLS


def test_phase2_spellcaster_ladder_runs_when_cloak_is_skipped():
    """Upgrades mode skips the cloak ladder, so spellcaster itself researches Psi Storm."""
    env = _make_env("heal,cloak,spellcaster,drop,upgrades")
    asyncio.run(env._apply_cloak_env_mode_upgrades())
    env.bot.client.debug_upgrade.assert_not_called()
    asyncio.run(env._apply_spellcaster_env_mode_upgrades())
    assert env.bot.client.debug_upgrade.await_count == SPELLCASTER_UPGRADE_CALLS


def _spellcaster_levels():
    levels = []
    for path in sorted(_CONFIG_DIR.glob("*.yaml")):
        if path.name == "pipeline.yaml":
            continue
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        env = cfg.get("env") or {}
        modes = {
            part.strip().lower()
            for part in str(env.get("env_mode") or "").split(",")
            if part.strip()
        }
        if "spellcaster" in modes:
            levels.append((path.stem, cfg))
    return levels


def _caster_owners(pools):
    owners = {}
    for pool in pools or []:
        pid = static_spawn_player_id(pool, pools)
        for name in pool.get("unit_pool") or []:
            key = str(name).upper()
            if key in _CASTERS:
                owners.setdefault(key, set()).add(pid)
    return owners


def _pipeline_levels():
    pipe = yaml.safe_load((_CONFIG_DIR / "pipeline.yaml").read_text(encoding="utf-8"))
    return list(pipe["levels"])


def test_spellcaster_levels_keep_protoss_as_player_one():
    """Psi Storm is researched by the player-1 cheat. Protoss must be that player.

    Ghost EMP and Infestor Fungal Growth do not need research.
    """
    assert SPELLCASTER_CAST_UPGRADES["HIGHTEMPLAR"] == "PSISTORMTECH"
    assert SPELLCASTER_CAST_UPGRADES["GHOST"] is None
    assert SPELLCASTER_CAST_UPGRADES["INFESTOR"] is None

    levels = _spellcaster_levels()
    assert levels, "expected spellcaster configs"
    by_name = {name: cfg for name, cfg in levels}
    pipeline = _pipeline_levels()
    after_aoe = pipeline[pipeline.index("aoe_zergling_roach_vs_high_templar_immortal") + 1 :]

    for name in after_aoe:
        cfg = by_name[name]
        env = cfg["env"]
        modes = {part.strip().lower() for part in str(env.get("env_mode") or "").split(",")}
        assert "spellcaster" in modes, name
        pools = env["unit_pools"]
        races = []
        for pool in pools:
            race = str(pool.get("race", "")).lower()
            if race and race not in races:
                races.append(race)
            caster = RACE_SPELLCASTER_UNIT[race]
            assert caster in [str(unit).upper() for unit in pool["unit_pool"]], (name, race)
        if "protoss" in races:
            assert races[0] == "protoss", name

    later_pools = by_name["aoe_zergling_infestor_vs_stalker_templar"]["env"]["unit_pools"]
    assert _caster_owners(later_pools)["HIGHTEMPLAR"] == {1}
    assert _caster_owners(later_pools)["INFESTOR"] == {2}
    first_pools = by_name["aoe_high_templar_zealot_vs_marine_marauder"]["env"]["unit_pools"]
    assert _caster_owners(first_pools)["HIGHTEMPLAR"] == {1}
    assert _caster_owners(first_pools)["GHOST"] == {2}


def test_cloak_upgrades_without_heal_or_spellcaster_skips_energy():
    env = _make_env("cloak,upgrades")
    env.bot.all_units = [_unit(11, 200.0)]
    asyncio.run(env._apply_spellcaster_env_mode_upgrades())
    asyncio.run(env._apply_spellcaster_energy())
    env.bot.client.debug_upgrade.assert_not_called()
    env.bot.client.debug_set_unit_value.assert_not_called()
