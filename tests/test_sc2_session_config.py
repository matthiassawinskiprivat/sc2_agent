"""Regression: Level3Env and Level0Env must agree on Computer player for movement levels."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from train.envs.sc2_session_config import (
    SC2_STATUS_IN_GAME,
    session_realtime,
    use_computer_player,
    wait_for_sc2_in_game,
)


def test_move_multi_level_defaults_without_computer():
    assert use_computer_player("level_move_multi_3", {}) is False


def test_legacy_level_3_prefix_defaults_without_computer():
    assert use_computer_player("level_3", {}) is False


def test_attack_level_defaults_with_computer():
    assert use_computer_player("level_attack_0", {}) is True


def test_env_use_computer_overrides_movement_default():
    conf = {"env": {"use_computer": True}}
    assert use_computer_player("level_move_multi_0", conf) is True


def test_build_level_defaults_without_computer():
    assert use_computer_player("level_build_zerg_0", {}) is False
    assert use_computer_player("level_build_zerg_1", {}) is False


def test_flattened_use_computer_can_override_build_default():
    assert use_computer_player("level_build_zerg_0", {"use_computer": True}) is True
    assert use_computer_player(
        "level_build_zerg_0", {"env": {"use_computer": False}}
    ) is False


def test_session_realtime_defaults_false_even_for_preview():
    """Visualize must not imply realtime — breaks debug spawn on multi-unit levels."""
    assert session_realtime({"preview_mode": True}) is False
    assert session_realtime({}) is False


def test_session_realtime_honors_env_override():
    assert session_realtime({"env": {"realtime": True}}) is True
    assert session_realtime({"env": {"realtime": False}}) is False


def test_session_realtime_honors_flat_training_conf():
    assert session_realtime({"realtime": True}) is True
    assert session_realtime({"realtime": False}) is False


def test_wait_for_sc2_in_game_returns_when_ready():
    async def _run():
        client = MagicMock()
        init = MagicMock(status=2)
        ready = MagicMock(status=SC2_STATUS_IN_GAME)
        client._execute = AsyncMock(side_effect=[init, ready])
        logs = []
        await wait_for_sc2_in_game(
            client, log=logs.append, timeout_sec=5.0, poll_sec=0.0
        )
        assert client._execute.await_count == 2
        assert any("in_game" in m for m in logs)

    asyncio.run(_run())


def test_wait_for_sc2_in_game_times_out():
    async def _run():
        client = MagicMock()
        client._execute = AsyncMock(return_value=MagicMock(status=2))
        with pytest.raises(RuntimeError, match="did not reach in_game"):
            await wait_for_sc2_in_game(
                client, timeout_sec=0.05, poll_sec=0.02, launched_fail_sec=10.0
            )

    asyncio.run(_run())
