import pytest
from unittest.mock import MagicMock
from src.level_3_bot import Level3Bot
from src.level_7_bot import Level7Bot
from sc2.ids.unit_typeid import UnitTypeId

def test_level_3_bot_unit_type_propagation():
    # If config has unit_type nested under 'env'
    conf = {
        "env": {
            "unit_type": "MARAUDER"
        }
    }
    bot = Level3Bot(conn=MagicMock(), training_conf=conf)
    # Current behavior check: does it find it?
    # If not, it defaults to ZERGLING
    print(f"Level3Bot allied_unit_type: {bot.allied_unit_type}")
    
def test_level_7_bot_unit_type_propagation():
    conf = {
        "env": {
            "unit_type": "ROACH"
        }
    }
    bot = Level7Bot(conn=MagicMock(), training_conf=conf)
    print(f"Level7Bot allied_unit_type: {bot.allied_unit_type}")
    assert bot.allied_unit_type == UnitTypeId.ROACH
