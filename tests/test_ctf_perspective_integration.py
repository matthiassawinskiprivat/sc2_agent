import asyncio
import numpy as np
from unittest.mock import MagicMock
from src.agent.ctf_bot import CaptureTheFlagBot


def test_ctf_perspective_integration():
    """Player 2 keeps world-frame maps and spatial indices (ownership swap only)."""
    training_conf = {
        "grid_size": 4,
        "env": {"active_layers_count": 1},
    }

    bot = CaptureTheFlagBot(player_id=2, training_conf=training_conf)
    asyncio.run(bot.on_start())

    bot.game_info = MagicMock()
    bot.game_info.map_size = MagicMock(width=100, height=100)
    bot.game_info.pathing_grid = MagicMock(width=100, height=100)
    bot.game_data = MagicMock()
    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None

    bot.agent_system = MagicMock()
    unit_agent = MagicMock()
    bot.agent_system.unit_agents = {"active_agent": unit_agent}

    mock_obs = np.zeros((1, 4, 4), dtype=np.uint8)
    mock_obs[0, 0, 0] = 1
    mock_obs[0, 3, 3] = 2
    bot.agent_system.process_observation.return_value = {"active_agent": mock_obs}

    obs = bot.get_observation()
    assert obs[0, 0, 0] == 1
    assert obs[0, 3, 3] == 2

    bot.all_units = MagicMock()
    unit = MagicMock()
    bot.all_units.find_by_tag.return_value = unit
    unit.owner_id = 2

    unit_agent.convert_spatial_action_to_world.side_effect = lambda idx: (idx, idx)

    bot.do = MagicMock()
    asyncio.run(bot.apply_action(123, 0))

    unit_agent.convert_spatial_action_to_world.assert_called_with(0)
