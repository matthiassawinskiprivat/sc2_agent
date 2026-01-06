import pytest
from sc2.bot_ai import BotAI

def test_bot_ai_properties():
    print(f"BotAI attributes: {dir(BotAI)}")
    units_attr = getattr(BotAI, "units", None)
    print(f"BotAI.units: {units_attr}")
    print(f"BotAI.units type: {type(units_attr)}")
    
    # Check for other common burnysc2 properties
    print(f"BotAI.structures type: {type(getattr(BotAI, 'structures', None))}")
    print(f"BotAI.all_units type: {type(getattr(BotAI, 'all_units', None))}")
