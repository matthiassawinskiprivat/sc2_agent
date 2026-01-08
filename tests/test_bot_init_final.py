import pytest
from src.agent.ctf_bot import CaptureTheFlagBot
from sc2.bot_ai import BotAI

def test_bot_initialization_requirements():
    """
    Verify that the bot has required library attributes and unshadowed properties.
    """
    bot = CaptureTheFlagBot(player_id=1)
    
    # Check for required library attribute that caused the crash
    assert hasattr(bot, "distance_calculation_method"), "Bot missing distance_calculation_method"
    
    # Check that 'units' is NOT in __dict__ (which would mean it's shadowed)
    # Note: In some burnysc2 versions, it might be set during _initialize_variables.
    # The goal is that it works as a dynamic property eventually.
    
    # Let's check if it's currently shadowed by an instance attribute
    has_units_shadow = "units" in bot.__dict__
    print(f"Units in __dict__: {has_units_shadow}")
    
    # If we want it unshadowed, it should not be in __dict__
    assert not has_units_shadow, "units is shadowed by an instance attribute"

if __name__ == "__main__":
    test_bot_initialization_requirements()
