import pytest
from unittest.mock import MagicMock
from src.agent.ctf_bot import CaptureTheFlagBot
from sc2.units import Units

def test_bot_property_shadowing_details():
    bot = CaptureTheFlagBot(player_id=1)
    
    has_units_attr = "units" in bot.__dict__
    print(f"Has units in bot.__dict__? {has_units_attr}")
    
    prop = getattr(CaptureTheFlagBot, "units", None)
    print(f"Class units type: {type(prop)}")
    is_property = isinstance(prop, property)
    print(f"Is class units a property? {is_property}")
    
    # In burnysc2, units is a property. 
    # If it's shadowed, it won't be updated by BotAI._prepare_step
    
    if has_units_attr:
        print(f"Shadowed units value: {bot.__dict__['units']}")
        
    assert is_property, "Expected 'units' to be a property on class CaptureTheFlagBot"
    assert not has_units_attr, "'units' should not be in __dict__ as it shadows the property"