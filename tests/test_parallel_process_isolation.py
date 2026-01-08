import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from src.utils.single_client_manager import SingleClientSC2Manager

@pytest.mark.asyncio
async def test_manager_close_isolation():
    """
    Verify that manager.close() does NOT call kill_switch_sc2 (global nuke).
    """
    # We patch the actual function in its home module
    with patch("src.utils.sc2_cleanup.kill_switch_sc2") as mock_kill_switch:
        with patch("psutil.Process") as mock_psutil_proc:
            manager = SingleClientSC2Manager()
            
            # Mock a running process
            mock_process = MagicMock()
            mock_process.__aexit__ = AsyncMock()
            
            # Setup PID mocks to match implementation logic
            # getattr(self.process, "_process", None) -> obj with .pid
            mock_popen = MagicMock()
            mock_popen.pid = 1234
            mock_process._process = mock_popen
            
            manager.process = mock_process
            
            await manager.close()
            
            # 1. Verify global kill switch was NOT called
            mock_kill_switch.assert_not_called()
            
            # 2. Verify psutil was used for targeted kill
            mock_psutil_proc.assert_called_with(1234)

if __name__ == "__main__":
    asyncio.run(test_manager_close_isolation())
