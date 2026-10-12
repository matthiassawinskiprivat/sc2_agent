"""CTF env_mode=spellcaster: max energy and spell-tech debug_upgrade ladder."""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import yaml

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from train.envs.ctf.pool_assignment import static_spawn_player_id
from sc2.ids.unit_typeid import UnitTypeId

from train.envs.ctf.spellcaster import (
    QUEEN_HEAL_START_ENERGY,
    RACE_SPELLCASTER_UNIT,
    SPELLCASTER_CAST_UPGRADES,
    SPELLCASTER_UPGRADE_CALLS,
    apply_caster_start_energy,
    apply_spell_upgrades,
    caster_start_energy_table,
    energy_units,
)

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"
_CASTERS = ("HIGHTEMPLAR", "GHOST", "INFESTOR")


def _unit(tag, energy_max, energy=0.0, type_id=None):
    u = MagicMock()
    u.tag = tag
    u.energy_max = energy_max
    u.energy = energy
    u.type_id = type_id
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


def test_apply_caster_start_energy_is_two_casts_not_a_full_bar():
    """Known casters get a limited budget. Unknown energy units are left alone."""
    bot = _bot(
        [
            _unit(1, 200.0, type_id=UnitTypeId.QUEEN),
            _unit(2, 200.0, type_id=UnitTypeId.HIGHTEMPLAR),
            _unit(3, 200.0, type_id=UnitTypeId.VIPER),
            _unit(4, 200.0, type_id=UnitTypeId.BATTLECRUISER),
            _unit(5, 100.0, type_id=UnitTypeId.SHIELDBATTERY),
            _unit(6, 200.0, type_id=UnitTypeId.MEDIVAC),
            _unit(7, 200.0),  # energy bar, no caster type
            _unit(8, 0.0, type_id=UnitTypeId.MARINE),
        ]
    )
    filled = asyncio.run(apply_caster_start_energy(bot))
    assert filled == 6
    by_value = {call.args[2]: set(call.args[0]) for call in bot.client.debug_set_unit_value.await_args_list}
    table = caster_start_energy_table()
    assert 1 in by_value[table[UnitTypeId.QUEEN]]
    assert 2 in by_value[table[UnitTypeId.HIGHTEMPLAR]]
    assert 3 in by_value[table[UnitTypeId.VIPER]]
    assert 4 in by_value[table[UnitTypeId.BATTLECRUISER]]
    assert 5 in by_value[table[UnitTypeId.SHIELDBATTERY]]
    assert 6 in by_value[table[UnitTypeId.MEDIVAC]]
    assert 7 not in {tag for tags in by_value.values() for tag in tags}
    assert table[UnitTypeId.QUEEN] == 100.0
    assert table[UnitTypeId.HIGHTEMPLAR] == 150.0
    assert table[UnitTypeId.VIPER] == 125.0
    assert table[UnitTypeId.BATTLECRUISER] == 125.0
    assert table[UnitTypeId.SHIELDBATTERY] == 50.0
    assert max(table.values()) < 200.0
    for call in bot.client.debug_set_unit_value.await_args_list:
        assert call.args[1] == 1
        assert call.args[2] < 200.0


def test_caster_start_energy_caps_at_energy_max():
    bot = _bot([_unit(1, 80.0, type_id=UnitTypeId.HIGHTEMPLAR)])
    asyncio.run(apply_caster_start_energy(bot))
    bot.client.debug_set_unit_value.assert_awaited_once_with([1], 1, 80.0)


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
    env.bot.all_units = [_unit(11, 200.0, type_id=UnitTypeId.HIGHTEMPLAR)]
    asyncio.run(env._apply_spellcaster_env_mode_upgrades())
    asyncio.run(env._apply_spellcaster_energy())
    assert env.bot.client.debug_upgrade.await_count == SPELLCASTER_UPGRADE_CALLS
    env.bot.client.debug_set_unit_value.assert_awaited_once_with(
        [11], 1, caster_start_energy_table()[UnitTypeId.HIGHTEMPLAR]
    )


def test_spellcaster_skips_upgrade_ladder_when_cloak_already_ran_it():
    env = _make_env("heal,cloak,spellcaster")
    asyncio.run(env._apply_spellcaster_env_mode_upgrades())
    env.bot.client.debug_upgrade.assert_not_called()


def test_upgrades_mode_skipped_when_spellcaster_present():
    env = _make_env("upgrades,spellcaster")
    env.episode_counter = 50
    asyncio.run(env._apply_upgrades_env_mode())
    env.bot.client.debug_upgrade.assert_not_called()


def test_heal_mode_caps_every_energy_caster():
    """Heal and spellcaster share one budget. No caster is filled to energy_max."""
    env = _make_env("heal")
    env.bot.all_units = [
        _unit(11, 200.0, type_id=UnitTypeId.QUEEN),
        _unit(12, 100.0, type_id=UnitTypeId.SHIELDBATTERY),
        _unit(13, 200.0, type_id=UnitTypeId.MEDIVAC),
    ]
    asyncio.run(env._apply_spellcaster_env_mode_upgrades())
    asyncio.run(env._apply_spellcaster_energy())
    env.bot.client.debug_upgrade.assert_not_called()
    by_value = {
        call.args[2]: set(call.args[0])
        for call in env.bot.client.debug_set_unit_value.await_args_list
    }
    table = caster_start_energy_table()
    assert 11 in by_value[table[UnitTypeId.QUEEN]]
    assert 12 in by_value[table[UnitTypeId.SHIELDBATTERY]]
    assert 13 in by_value[table[UnitTypeId.MEDIVAC]]
    assert QUEEN_HEAL_START_ENERGY == 100.0


def test_burrowed_casters_use_the_same_budget():
    queen_burrowed = getattr(UnitTypeId, "QUEENBURROWED", None)
    infestor_burrowed = getattr(UnitTypeId, "INFESTORBURROWED", None)
    units = [
        _unit(1, 200.0, type_id=UnitTypeId.QUEEN),
        _unit(3, 200.0, type_id=UnitTypeId.MEDIVAC),
        _unit(0, 200.0, type_id=UnitTypeId.QUEEN),
    ]
    if queen_burrowed is not None:
        units.append(_unit(2, 200.0, type_id=queen_burrowed))
    if infestor_burrowed is not None:
        units.append(_unit(4, 200.0, type_id=infestor_burrowed))
    bot = _bot(units)
    n = asyncio.run(apply_caster_start_energy(bot))
    expected = 2 + (queen_burrowed is not None) + (infestor_burrowed is not None)
    assert n == expected
    by_value = {call.args[2]: set(call.args[0]) for call in bot.client.debug_set_unit_value.await_args_list}
    queen_tags = {1}
    if queen_burrowed is not None:
        queen_tags.add(2)
    assert by_value[QUEEN_HEAL_START_ENERGY] == queen_tags
    assert 3 in by_value[50.0]


def test_spellcaster_mode_uses_the_same_limited_budget():
    env = _make_env("heal,spellcaster")
    env.bot.all_units = [
        _unit(1, 200.0, type_id=UnitTypeId.QUEEN),
        _unit(2, 200.0, type_id=UnitTypeId.HIGHTEMPLAR),
        _unit(3, 200.0, type_id=UnitTypeId.GHOST),
    ]
    asyncio.run(env._apply_spellcaster_energy())
    by_value = {
        call.args[2]: set(call.args[0])
        for call in env.bot.client.debug_set_unit_value.await_args_list
    }
    table = caster_start_energy_table()
    assert 1 in by_value[table[UnitTypeId.QUEEN]]
    assert 2 in by_value[table[UnitTypeId.HIGHTEMPLAR]]
    assert 3 in by_value[table[UnitTypeId.GHOST]]
    assert 200.0 not in by_value


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

    Exception: ``*_reversed`` siblings intentionally put the other race first so
    Neural / Yamato unlock on the alternate half of the split.
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
        if name.endswith("_reversed"):
            continue
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
        if "protoss" in races:
            assert races[0] == "protoss", name
            for pool in pools:
                if str(pool.get("race", "")).lower() != "protoss":
                    continue
                assert "HIGHTEMPLAR" in [str(u).upper() for u in pool["unit_pool"]], name

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
