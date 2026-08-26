"""CTF soft-reset map cleanup retries until killable units are gone."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sc2.data import Alliance
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId

from src.sc2_bot import MapCleanupFailedError
from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from train.envs.ctf.map_cleanup import queued_unit_command, reveal_ability_for_cleanup


class _KillableUnits:
    def __init__(self, tags=None, units=None):
        if units is not None:
            self._units = list(units)
            return
        self._units = []
        for tag in tags or []:
            u = MagicMock()
            u.tag = tag
            self._units.append(u)

    def filter(self, pred):
        kept = [u for u in self._units if pred(u)]
        out = _KillableUnits(units=kept)
        return out

    @property
    def amount(self):
        return len(self._units)

    def __iter__(self):
        return iter(self._units)

    def __bool__(self):
        return bool(self._units)


def _make_env(all_units_provider):
    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env._log = MagicMock()
    env.steps_count = 42
    env.bot = SimpleNamespace()
    env.bot.client = SimpleNamespace()
    env.bot.client.debug_kill_unit = AsyncMock()
    env.bot.client.step = AsyncMock()
    env.bot.client.actions = AsyncMock()
    env.bot.all_units = _KillableUnits([])

    async def _refresh():
        env.bot.all_units = all_units_provider()

    env._refresh_bot_state = AsyncMock(side_effect=_refresh)
    return env


def _cleanup_unit(*, tag, type_id, is_burrowed=False, is_cloaked=False):
    u = MagicMock()
    u.tag = tag
    u.type_id = type_id
    u.alliance = Alliance.Self
    u.is_mineral_field = False
    u.is_vespene_geyser = False
    u.is_burrowed = is_burrowed
    u.is_cloaked = is_cloaked

    def _issue(ability, *args, **kwargs):
        return SimpleNamespace(combining_tuple=(ability, None, False, False))

    u.side_effect = _issue
    return u


def test_queued_unit_command_rejects_bool_returns():
    assert queued_unit_command(False) is None
    assert queued_unit_command(True) is None
    assert queued_unit_command(None) is None
    cmd = SimpleNamespace(combining_tuple=(AbilityId.STOP, None, False, False))
    assert queued_unit_command(cmd) is cmd


def test_reveal_ability_unburrows_burrowed_morph():
    u = _cleanup_unit(tag=1, type_id=UnitTypeId.ZERGLINGBURROWED, is_burrowed=True)
    assert reveal_ability_for_cleanup(u) == AbilityId.BURROWUP_ZERGLING


def test_reveal_ability_uncloaks_ghost():
    u = _cleanup_unit(tag=1, type_id=UnitTypeId.GHOST, is_cloaked=True)
    assert reveal_ability_for_cleanup(u) == AbilityId.BEHAVIOR_CLOAKOFF_GHOST


def test_reveal_ability_skips_visible_marine():
    u = _cleanup_unit(tag=1, type_id=UnitTypeId.MARINE)
    assert reveal_ability_for_cleanup(u) is None


def test_reveal_ability_stops_mothership_cloak_field():
    u = _cleanup_unit(tag=1, type_id=UnitTypeId.MOTHERSHIP)
    assert reveal_ability_for_cleanup(u) == AbilityId.STOP


def test_reveal_ability_stops_oracle_cloak_field():
    u = _cleanup_unit(tag=1, type_id=UnitTypeId.ORACLE)
    assert reveal_ability_for_cleanup(u) == AbilityId.STOP


def test_clear_ctf_map_units_retries_until_empty():
    remaining = [2]

    def provider():
        n = remaining[0]
        return _KillableUnits(list(range(1, n + 1)))

    env = _make_env(provider)

    async def _kill(tags):
        remaining[0] = max(0, remaining[0] - 1)

    env.bot.client.debug_kill_unit.side_effect = _kill

    with patch("src.sc2_bot.unit_is_subject_to_map_cleanup", return_value=True):
        asyncio.run(env._clear_ctf_map_units())

    assert remaining[0] == 0
    assert env.bot.client.debug_kill_unit.await_count == 2
    assert env.steps_count == 42
    env.bot.client.actions.assert_not_awaited()


def test_clear_ctf_map_units_raises_when_stuck():
    env = _make_env(lambda: _KillableUnits([11, 22]))

    with patch("src.sc2_bot.unit_is_subject_to_map_cleanup", return_value=True):
        try:
            asyncio.run(env._clear_ctf_map_units())
            raised = False
        except MapCleanupFailedError as exc:
            raised = True
            assert exc.remaining_count == 2
    assert raised
    assert env.steps_count == 42


def test_clear_ctf_map_units_skips_when_map_already_empty():
    env = _make_env(lambda: _KillableUnits([]))
    asyncio.run(env._clear_ctf_map_units())
    env.bot.client.debug_kill_unit.assert_not_awaited()
    env.bot.client.actions.assert_not_awaited()


def test_clear_reveals_burrowed_units_before_kill():
    unit = _cleanup_unit(tag=7, type_id=UnitTypeId.ZERGLINGBURROWED, is_burrowed=True)
    remaining = [True]
    order = []

    def provider():
        if remaining[0]:
            return _KillableUnits(units=[unit])
        return _KillableUnits([])

    env = _make_env(provider)

    async def _actions(cmds):
        order.append("reveal")

    async def _kill(tags):
        order.append("kill")
        remaining[0] = False

    env.bot.client.actions.side_effect = _actions
    env.bot.client.debug_kill_unit.side_effect = _kill

    asyncio.run(env._clear_ctf_map_units())

    unit.assert_called_once_with(AbilityId.BURROWUP_ZERGLING)
    assert order == ["reveal", "kill"]
    assert env.bot.client.actions.await_count == 1
    assert env.bot.client.debug_kill_unit.await_count == 1
    assert env.steps_count == 42


def test_clear_reveals_cloaked_ghost_before_kill():
    unit = _cleanup_unit(tag=8, type_id=UnitTypeId.GHOST, is_cloaked=True)
    remaining = [True]

    def provider():
        if remaining[0]:
            return _KillableUnits(units=[unit])
        return _KillableUnits([])

    env = _make_env(provider)

    async def _kill(tags):
        remaining[0] = False

    env.bot.client.debug_kill_unit.side_effect = _kill
    asyncio.run(env._clear_ctf_map_units())

    unit.assert_called_once_with(AbilityId.BEHAVIOR_CLOAKOFF_GHOST)
    env.bot.client.actions.assert_awaited()
    assert env.steps_count == 42


def test_clear_stops_mothership_cloak_field_before_kill():
    mothership = _cleanup_unit(tag=9, type_id=UnitTypeId.MOTHERSHIP)
    remaining = [True]

    def provider():
        if remaining[0]:
            return _KillableUnits(units=[mothership])
        return _KillableUnits([])

    env = _make_env(provider)

    async def _kill(tags):
        remaining[0] = False

    env.bot.client.debug_kill_unit.side_effect = _kill
    asyncio.run(env._clear_ctf_map_units())

    mothership.assert_called_once_with(AbilityId.STOP)
    env.bot.client.actions.assert_awaited()
    assert env.steps_count == 42


def test_reveal_skips_rejected_bool_commands_and_still_kills():
    """python-sc2 unit(ability) returns False when the order is rejected."""
    unit = _cleanup_unit(tag=7, type_id=UnitTypeId.ZERGLINGBURROWED, is_burrowed=True)
    unit.side_effect = None
    unit.return_value = False
    remaining = [True]

    def provider():
        if remaining[0]:
            return _KillableUnits(units=[unit])
        return _KillableUnits([])

    env = _make_env(provider)

    async def _kill(tags):
        remaining[0] = False

    env.bot.client.debug_kill_unit.side_effect = _kill
    asyncio.run(env._clear_ctf_map_units())

    env.bot.client.actions.assert_not_awaited()
    env.bot.client.debug_kill_unit.assert_awaited()
    assert env.steps_count == 42


def test_reveal_flushes_commands_already_queued_by_bot_do():
    unit = _cleanup_unit(tag=7, type_id=UnitTypeId.ZERGLINGBURROWED, is_burrowed=True)
    unit.side_effect = None
    unit.return_value = True
    remaining = [True]
    queued = SimpleNamespace(combining_tuple=(AbilityId.BURROWUP_ZERGLING, None, False, False))
    order = []

    def provider():
        if remaining[0]:
            return _KillableUnits(units=[unit])
        return _KillableUnits([])

    env = _make_env(provider)
    env.bot.actions = [queued]

    async def _actions(cmds):
        order.append("reveal")
        assert cmds is env.bot.actions or list(cmds) == [queued]

    async def _kill(tags):
        order.append("kill")
        remaining[0] = False

    env.bot.client.actions.side_effect = _actions
    env.bot.client.debug_kill_unit.side_effect = _kill
    asyncio.run(env._clear_ctf_map_units())

    assert order == ["reveal", "kill"]
    assert env.bot.actions == []
    assert env.steps_count == 42
