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
async def test_target_unloading_tracking(bot):
    await bot.reset_game_state()
    target_pos = Point2((10, 10))
    bot._objective_pos = target_pos
    
    marine = MockUnit(tag=101, type_id=UnitTypeId.MARINE, position=target_pos)
    medivac = MockUnit(tag=1, type_id=UnitTypeId.MEDIVAC, position=target_pos)
    
    # Pre-condition: unit was in cargo
    bot.previous_passenger_tags.add(101)
    
    # Mock units: both on map
    def create_mock_filter_result(units_list):
        m = MagicMock()
        m.__iter__.return_value = iter(units_list)
        m.filter.side_effect = lambda f: create_mock_filter_result([u for u in units_list if f(u)])
        m.find_by_tag.side_effect = lambda tag: next((u for u in units_list if u.tag == tag), None)
        m.__bool__.return_value = len(units_list) > 0
        m.__len__.return_value = len(units_list)
        return m

    bot.units = create_mock_filter_result([medivac, marine])
    bot.all_units = create_mock_filter_result([medivac, marine])
    
    # Simulate step with medivac as actor
    bot.calculate_drop_reward(active_tag=1)
    
    assert 101 in bot.target_unloaded_units
