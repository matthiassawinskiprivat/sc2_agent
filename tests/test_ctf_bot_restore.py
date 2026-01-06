import pytest
from src.agent.ctf_bot import CaptureTheFlagBot

def test_bot_has_required_attributes():
    """
    Verify that the bot has required library attributes and unshadowed properties.
    """
    bot = CaptureTheFlagBot(player_id=1)
    
    # Check for required library attribute that caused the crash
    assert hasattr(bot, "distance_calculation_method"), "Bot missing distance_calculation_method"
    
    # Check for units attribute (needed by _prepare_step)
    # With the current (buggy) implementation, this should FAIL because it was deleted
    assert hasattr(bot, "units"), "Bot missing units attribute"

if __name__ == "__main__":
    test_bot_has_required_attributes()