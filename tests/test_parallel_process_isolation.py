"""SingleClientSC2Manager.close must stay worker-local (no global SC2 kill)."""
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from src.utils.single_client_manager import SingleClientSC2Manager


def test_manager_close_isolation():
    """manager.close() must not call kill_switch_sc2 (would nuke sibling workers)."""

    async def _run():
        with patch("src.utils.sc2_cleanup.kill_switch_sc2") as mock_kill_switch:
            with patch("src.utils.sc2_cleanup.kill_sc2_process_tree") as mock_kill_tree:
                manager = SingleClientSC2Manager()

                mock_process = MagicMock()
                mock_process.__aexit__ = AsyncMock()
                mock_popen = MagicMock()
                mock_popen.pid = 1234
                mock_process._process = mock_popen
                manager.process = mock_process

                await manager.close()

                mock_kill_switch.assert_not_called()
                mock_kill_tree.assert_called_once()
                assert mock_kill_tree.call_args.args[0] == 1234

    asyncio.run(_run())
