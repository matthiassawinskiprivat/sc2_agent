import asyncio
import numpy as np
from unittest.mock import MagicMock
from src.agent.ctf_bot import CaptureTheFlagBot


def test_bot_player2_spatial_stays_world_frame():
    """Player 2 issues the policy's spatial index in world coordinates."""
    grid_size = 128
    bot2 = CaptureTheFlagBot(player_id=2, training_conf={"grid_size": grid_size})
    bot2.is_trainee = False

    mock_model = MagicMock()
    mock_model.predict.return_value = (np.array(10), None)
    bot2.model = mock_model

    raw_obs = np.zeros((23, grid_size, grid_size))
    raw_obs[0, 0, 0] = 1.0
    bot2.get_observation = MagicMock(return_value=raw_obs)
    bot2.action_masks = MagicMock(return_value=np.ones(2 * grid_size * grid_size, dtype=bool))

    action = bot2.predict_action()
    assert action == 10

    bot2.all_units = MagicMock()
    mock_unit = MagicMock()
    mock_unit.owner_id = 2
    bot2.all_units.find_by_tag.return_value = mock_unit
    bot2.do = MagicMock()
    bot2.agent_system = MagicMock()
    bot2.agent_system.unit_agents = {"active_agent": MagicMock()}
    bot2.agent_system.unit_agents["active_agent"].convert_spatial_action_to_world.return_value = (5, 5)

    asyncio.run(bot2.apply_action(201, action))

    args, _ = bot2.agent_system.unit_agents["active_agent"].convert_spatial_action_to_world.call_args
    assert args[0] == 10
