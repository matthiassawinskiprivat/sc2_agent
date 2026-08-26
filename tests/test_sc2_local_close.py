"""Worker-local SC2 teardown must not call burnysc2 kill_all."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils.sc2_cleanup import (
    close_sc2_process_local,
    sc2_process_pid,
    unregister_sc2_process,
)
from train.envs.sc2_session_config import (
    SC2_STATUS_IN_GAME,
    SC2_STATUS_LAUNCHED,
    check_create_game_result,
    wait_for_sc2_in_game,
)


def test_unregister_sc2_process_removes_from_kill_switch():
    proc = object()
    other = object()
    fake_switch = MagicMock()
    fake_switch._to_kill = [other, proc, proc]

    with patch("sc2.sc2process.kill_switch", fake_switch):
        assert unregister_sc2_process(proc) is True
        assert proc not in fake_switch._to_kill
        assert other in fake_switch._to_kill


def test_sc2_process_pid_reads_inner_popen():
    proc = MagicMock()
    proc._process = MagicMock(pid=4321)
    assert sc2_process_pid(proc) == 4321


def test_close_sc2_process_local_skips_aexit_kill_all():
    async def _run():
        proc = MagicMock()
        proc._process = MagicMock(pid=99)
        proc._close_connection = AsyncMock()
        proc._clean = MagicMock()
        proc.__aexit__ = AsyncMock()

        with patch("src.utils.sc2_cleanup.unregister_sc2_process") as mock_unreg, patch(
            "src.utils.sc2_cleanup.kill_sc2_process_tree"
        ) as mock_tree:
            await close_sc2_process_local(proc, wait_sec=0.0)
            mock_unreg.assert_called_once_with(proc)
            proc._close_connection.assert_awaited_once()
            proc._clean.assert_called_once()
            mock_tree.assert_called_once_with(99, wait_sec=0.0)
            proc.__aexit__.assert_not_called()

    asyncio.run(_run())


def test_check_create_game_result_raises_on_nested_error():
    result = MagicMock()
    result.create_game.HasField.side_effect = lambda name: name in {
        "error",
        "error_details",
    }
    result.create_game.error = 1
    result.create_game.error_details = "MissingMap"
    with pytest.raises(RuntimeError, match="create_game failed"):
        check_create_game_result(result)


def test_wait_for_sc2_in_game_fails_fast_when_stuck_launched():
    async def _run():
        client = MagicMock()
        client._execute = AsyncMock(return_value=MagicMock(status=SC2_STATUS_LAUNCHED))
        with pytest.raises(RuntimeError, match="stuck in launched"):
            await wait_for_sc2_in_game(
                client,
                timeout_sec=30.0,
                poll_sec=0.05,
                launched_fail_sec=0.1,
            )

    asyncio.run(_run())


def test_wait_for_sc2_in_game_still_accepts_in_game():
    async def _run():
        client = MagicMock()
        client._execute = AsyncMock(
            side_effect=[
                MagicMock(status=2),
                MagicMock(status=SC2_STATUS_IN_GAME),
            ]
        )
        await wait_for_sc2_in_game(
            client, timeout_sec=5.0, poll_sec=0.0, launched_fail_sec=5.0
        )

    asyncio.run(_run())


def test_level0_close_uses_local_teardown_not_aexit():
    from train.envs.level_0_env import Level0Env

    env = Level0Env.__new__(Level0Env)
    env.worker_id = 1
    env.worker_index = 0
    env.log_file = MagicMock()
    env._log = MagicMock()
    env._loop = MagicMock()
    env._loop.is_closed.return_value = False
    env._loop.run_until_complete = MagicMock()
    env._loop.close = MagicMock()
    proc = MagicMock()
    proc.__aexit__ = AsyncMock()
    env._sc2_process = proc
    env._client = MagicMock()
    env._controller = MagicMock()
    env._bot = MagicMock()

    loop = env._loop
    with patch(
        "train.envs.level_0_env.close_sc2_process_local", new_callable=AsyncMock
    ) as mock_local:
        env.close()
        assert loop.run_until_complete.called
        mock_local.assert_called_once_with(proc)
        proc.__aexit__.assert_not_called()
    assert env._sc2_process is None
    assert env._loop is None
