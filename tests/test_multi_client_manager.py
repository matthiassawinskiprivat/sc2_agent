import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from src.utils.multi_client_manager import MultiClientSC2Manager

@pytest.mark.asyncio
async def test_launch_two_processes():
    """Test that the manager launches two SC2 processes."""
    mock_process_cls = MagicMock()
    mock_process_instance = AsyncMock()
    # Mock context manager behavior
    mock_process_instance.__aenter__.return_value = AsyncMock() 
    mock_process_cls.return_value = mock_process_instance

    mock_client_cls = MagicMock()
    mock_client_instance = AsyncMock()
    mock_client_cls.return_value = mock_client_instance

    manager = MultiClientSC2Manager(
        sc2_process_cls=mock_process_cls,
        client_cls=mock_client_cls
    )

    await manager.launch_processes()

    assert mock_process_cls.call_count == 2
    assert len(manager.processes) == 2
    assert len(manager.clients) == 2
    assert manager.host_client is not None
    assert manager.join_client is not None

@pytest.mark.asyncio
async def test_close_processes():
    """Test that the manager closes all processes."""
    mock_process_cls = MagicMock()
    mock_process_instance = AsyncMock()
    mock_process_instance.__aenter__.return_value = AsyncMock()
    mock_process_cls.return_value = mock_process_instance

    manager = MultiClientSC2Manager(sc2_process_cls=mock_process_cls)
    await manager.launch_processes()
    
    await manager.close()

    # Verify clients are closed (if they exist)
    # And processes are cleaned up
    assert len(manager.processes) == 0
    assert len(manager.clients) == 0

@pytest.mark.asyncio
async def test_create_and_join_game():
    """Test the flow of creating and joining a game."""
    mock_process_cls = MagicMock()
    # Mock controller
    mock_controller = AsyncMock()
    mock_process_instance = AsyncMock()
    mock_process_instance.__aenter__.return_value = mock_controller
    mock_process_cls.return_value = mock_process_instance

    mock_client_cls = MagicMock()
    mock_client_instance = AsyncMock()
    mock_client_cls.return_value = mock_client_instance

    manager = MultiClientSC2Manager(
        sc2_process_cls=mock_process_cls,
        client_cls=mock_client_cls
    )
    await manager.launch_processes()

    # Mock map
    mock_map = MagicMock()
    mock_players = [MagicMock(), MagicMock()]

    await manager.create_join_game(mock_map, mock_players, realtime=False)

    # Host creates game
    mock_controller.create_game.assert_called_once()
    
    # Both join
    assert mock_client_instance.join_game.call_count == 2
