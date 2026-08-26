"""create_game → fresh Client → join host flow."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sc2.data import Race
from sc2.player import Bot, Computer
from sc2.data import Difficulty

from train.envs.sc2_session_config import (
    SC2_STATUS_IN_GAME,
    SC2_STATUS_INIT_GAME,
    SC2_STATUS_LAUNCHED,
    check_join_game_result,
    create_sc2_game,
    host_local_sc2_game,
    join_sc2_game,
)


def test_check_join_game_result_raises():
    result = MagicMock()
    result.join_game.HasField.side_effect = lambda name: name in {
        "error",
        "error_details",
    }
    result.join_game.error = 9
    result.join_game.error_details = "MapDoesNotExist"
    with pytest.raises(RuntimeError, match="join_game failed"):
        check_join_game_result(result)


def test_create_sc2_game_rejects_launched_status():
    async def _run():
        controller = MagicMock()
        controller._execute = AsyncMock(
            return_value=MagicMock(
                status=SC2_STATUS_LAUNCHED,
                create_game=MagicMock(HasField=MagicMock(return_value=False)),
            )
        )
        game_map = MagicMock()
        game_map.relative_path = "x.SC2Map"
        game_map.path = Path("x.SC2Map")
        game_map.name = "x"
        with pytest.raises(RuntimeError, match="did not enter init_game"):
            await create_sc2_game(
                controller,
                game_map,
                [Bot(Race.Zerg, None)],
                realtime=False,
            )

    asyncio.run(_run())


def test_create_sc2_game_embeds_map_data_with_filename_only_path(tmp_path):
    """Absolute repo map paths must not be sent to SC2 (join LaunchError=6)."""

    async def _run():
        map_file = tmp_path / "Foo.SC2Map"
        map_file.write_bytes(b"MAPBYTES")
        game_map = MagicMock()
        game_map.relative_path = map_file
        game_map.path = map_file
        game_map.name = "Foo"
        game_map.data = b"MAPBYTES"

        captured = {}

        async def _execute(**kwargs):
            captured["req"] = kwargs["create_game"]
            resp = MagicMock()
            resp.status = SC2_STATUS_INIT_GAME
            resp.create_game = MagicMock(HasField=MagicMock(return_value=False))
            return resp

        controller = MagicMock()
        controller._execute = _execute
        await create_sc2_game(
            controller,
            game_map,
            [Bot(Race.Zerg, None), Computer(Race.Zerg, Difficulty.Easy)],
            realtime=False,
        )
        local = captured["req"].local_map
        assert local.map_data == b"MAPBYTES"
        assert local.map_path == "Foo.SC2Map"
        assert ":" not in local.map_path  # no Windows drive / absolute path

    asyncio.run(_run())


def test_local_map_proto_filename_only_when_outside_sc2_maps(tmp_path):
    from train.envs.sc2_session_config import _local_map_proto

    map_file = tmp_path / "AcropolisAIE.SC2Map"
    map_file.write_bytes(b"MAPBYTES")
    game_map = MagicMock()
    game_map.path = map_file
    game_map.relative_path = map_file
    game_map.name = "AcropolisAIE"
    game_map.data = b"MAPBYTES"

    proto = _local_map_proto(game_map)
    assert proto.map_path == "AcropolisAIE.SC2Map"
    assert proto.map_data == b"MAPBYTES"


def test_join_sc2_game_rejects_launched_status():
    async def _run():
        client = MagicMock()
        client.raw_affects_selection = False
        resp = MagicMock()
        resp.status = SC2_STATUS_LAUNCHED
        resp.join_game = MagicMock(
            HasField=MagicMock(return_value=False), player_id=1
        )
        client._execute = AsyncMock(return_value=resp)
        with pytest.raises(RuntimeError, match="join_game returned status=launched"):
            await join_sc2_game(client, Race.Zerg, name="Bot")

    asyncio.run(_run())


def test_host_local_sc2_game_creates_fresh_client_after_create():
    async def _run():
        ws = object()
        controller = MagicMock()
        controller._ws = ws
        controller.ping = AsyncMock()

        create_resp = MagicMock()
        create_resp.status = SC2_STATUS_INIT_GAME
        create_resp.create_game = MagicMock(HasField=MagicMock(return_value=False))

        join_resp = MagicMock()
        join_resp.status = SC2_STATUS_IN_GAME
        join_resp.join_game = MagicMock(
            HasField=MagicMock(return_value=False), player_id=1
        )

        ping_resp = MagicMock(status=SC2_STATUS_IN_GAME)

        # controller._execute for create; client._execute for join + wait ping
        controller._execute = AsyncMock(return_value=create_resp)

        client = MagicMock()
        client.raw_affects_selection = False
        client._execute = AsyncMock(side_effect=[join_resp, ping_resp])
        client_cls = MagicMock(return_value=client)

        game_map = MagicMock()
        game_map.relative_path = "rel.SC2Map"
        game_map.path = Path("rel.SC2Map")
        game_map.name = "rel"

        logs = []
        out = await host_local_sc2_game(
            controller=controller,
            client_cls=client_cls,
            game_map=game_map,
            players=[Bot(Race.Zerg, None)],
            realtime=False,
            join_race=Race.Zerg,
            log=logs.append,
            post_create_delay_sec=0.0,
            player_name="Trainee",
        )
        assert out is client
        client_cls.assert_called_once_with(ws)
        controller.ping.assert_awaited_once()
        assert any("create_game ok" in m for m in logs)
        assert any("join_game ok" in m for m in logs)

    asyncio.run(_run())
