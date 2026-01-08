import pytest
import numpy as np
from unittest.mock import MagicMock
from src.agent.ctf_bot import CaptureTheFlagBot

def test_bot_trainee_flag():
    """Verify that the is_trainee flag can be set and defaults to True."""
    bot = CaptureTheFlagBot(player_id=1)
    # Default should be True for backward compatibility or safety
    assert hasattr(bot, "is_trainee")
    assert bot.is_trainee is True
    
    bot2 = CaptureTheFlagBot(player_id=2)
    bot2.is_trainee = False
    assert bot2.is_trainee is False

def test_bot_model_support():
    """Verify that the bot can store a model and predict actions."""
    bot = CaptureTheFlagBot(player_id=2)
    bot.is_trainee = False
    
    # Mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = (np.array(5), None)
    
    bot.model = mock_model
    
    # Mock observation
    mock_obs = np.zeros((23, 128, 128))
    mock_mask = np.ones(2 * 128 * 128, dtype=bool)
    
    # We need to mock get_observation and action_masks for the bot
    # because they depend on SC2 internal state
    bot.get_observation = MagicMock(return_value=mock_obs)
    bot.action_masks = MagicMock(return_value=mock_mask)
    
    action = bot.predict_action()
    
    assert action == 5
    mock_model.predict.assert_called_once()
    # Check that it passed the observation and mask
    args, kwargs = mock_model.predict.call_args
    assert np.array_equal(args[0], np.expand_dims(mock_obs, axis=0))
    assert np.array_equal(kwargs['action_masks'], np.expand_dims(mock_mask, axis=0))

def test_bot_predict_action_none_if_trainee():
    """A trainee bot should not predict actions (it gets them from the trainer)."""
    bot = CaptureTheFlagBot(player_id=1)
    bot.is_trainee = True
    
    # Even if it has a model, it should probably return None or raise an error
    # to signify it expects external input.
    bot.model = MagicMock()
    
    with pytest.raises(RuntimeError, match="Trainee bot should not predict actions"):
        bot.predict_action()
