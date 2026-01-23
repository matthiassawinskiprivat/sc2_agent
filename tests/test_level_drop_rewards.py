import pytest
from unittest.mock import MagicMock, PropertyMock
from src.level_drop_bot import LevelDropBot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

class MockUnit:
    def __init__(self, tag, type_id, health=100, shield=0, cargo_used=0, position=Point2((0, 0))):
        self.tag = tag
        self.type_id = type_id
        self.health = health
        self.shield = shield
        self.health_max = 100
        self.shield_max = 0
        self.cargo_used = cargo_used
        self.position = position
        self.is_flying = type_id in {UnitTypeId.MEDIVAC, UnitTypeId.OVERLORD, UnitTypeId.WARPPRISM}
        self.passengers = []

    def distance_to(self, target):
        if hasattr(target, "position"):
            target = target.position
        return ((self.position.x - target.x)**2 + (self.position.y - target.y)**2)**0.5

@pytest.fixture
def bot():
    mock_conn = MagicMock()
    bot = LevelDropBot(mock_conn, training_conf={'rewards': {}})
    
    # Create mock units list that supports chaining filters
    def create_mock_filter_result(units_list):
        m = MagicMock()
        m.__iter__.return_value = iter(units_list)
        m.filter.side_effect = lambda f: create_mock_filter_result([u for u in units_list if f(u)])
        m.__bool__.return_value = len(units_list) > 0
        m.__len__.return_value = len(units_list)
        return m

    bot.units = create_mock_filter_result([])
    bot.all_units = create_mock_filter_result([])
    bot.enemy_units = []
    bot.enemy_structures = create_mock_filter_result([])
    
    bot.state = MagicMock()
    bot.state.game_loop = 0
    return bot

@pytest.mark.asyncio
async def test_loading_tracking(bot):
    await bot.reset_game_state()
    marine = MockUnit(tag=101, type_id=UnitTypeId.MARINE)
    medivac = MockUnit(tag=1, type_id=UnitTypeId.MEDIVAC)
    medivac.passengers = [marine]
    medivac.cargo_used = 1
    
    # Mock units: medivac is on map, marine is NOT on map (in cargo)
    def create_mock_filter_result(units_list):
        m = MagicMock()
        m.__iter__.return_value = iter(units_list)
        m.filter.side_effect = lambda f: create_mock_filter_result([u for u in units_list if f(u)])
        m.find_by_tag.side_effect = lambda tag: next((u for u in units_list if u.tag == tag), None)
        m.__bool__.return_value = len(units_list) > 0
        m.__len__.return_value = len(units_list)
        return m

    bot.units = create_mock_filter_result([medivac])
    bot.all_units = create_mock_filter_result([medivac, marine])
    
    # Simulate step with medivac as actor
    bot.calculate_drop_reward(active_tag=1)
    
    assert 101 in bot.initially_loaded_units

@pytest.mark.asyncio
async def test_unit_health_check(bot):
    await bot.reset_game_state()
    # 100% health
    marine_healthy = MockUnit(tag=101, type_id=UnitTypeId.MARINE, health=45, shield=0)
    marine_healthy.health_max = 45
    marine_healthy.shield_max = 0
    
    # 40% health (damaged)
    marine_damaged = MockUnit(tag=102, type_id=UnitTypeId.MARINE, health=10, shield=0)
    marine_damaged.health_max = 45
    marine_damaged.shield_max = 0
    
    # Protoss with shields
    zealot_damaged = MockUnit(tag=103, type_id=UnitTypeId.ZEALOT, health=50, shield=10)
    zealot_damaged.health_max = 100
    zealot_damaged.shield_max = 50
    # Total 60/150 = 40% (damaged)
    
    assert bot.is_unit_damaged(marine_healthy) is False
    assert bot.is_unit_damaged(marine_damaged) is True
    assert bot.is_unit_damaged(zealot_damaged) is True
