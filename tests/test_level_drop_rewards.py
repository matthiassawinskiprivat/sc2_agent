import pytest
from unittest.mock import MagicMock
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

def create_mock_filter_result(units_list):
    m = MagicMock()
    m.__iter__.return_value = iter(units_list)
    m.filter.side_effect = lambda f: create_mock_filter_result([u for u in units_list if f(u)])
    m.find_by_tag.side_effect = lambda tag: next((u for u in units_list if u.tag == tag), None)
    m.__bool__.return_value = len(units_list) > 0
    m.__len__.return_value = len(units_list)
    m.first = units_list[0] if units_list else None
    return m

@pytest.fixture
def bot():
    mock_conn = MagicMock()
    bot = LevelDropBot(mock_conn, training_conf={'rewards': {}})
    
    bot.units = create_mock_filter_result([])
    bot.all_units = create_mock_filter_result([])
    bot.enemy_units = []
    bot.enemy_structures = create_mock_filter_result([])
    
    bot.state = MagicMock()
    bot.state.game_loop = 0
    return bot

@pytest.mark.asyncio
async def test_registry_initialization(bot):
    await bot.reset_game_state()
    assert hasattr(bot, 'initially_loaded_units')
    assert isinstance(bot.initially_loaded_units, set)
    assert hasattr(bot, 'target_unloaded_units')
    assert isinstance(bot.target_unloaded_units, set)

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

@pytest.mark.asyncio
async def test_loading_tracking(bot):
    await bot.reset_game_state()
    marine = MockUnit(tag=101, type_id=UnitTypeId.MARINE)
    medivac = MockUnit(tag=1, type_id=UnitTypeId.MEDIVAC)
    medivac.passengers = [marine]
    medivac.cargo_used = 1
    
    bot.units = create_mock_filter_result([medivac])
    bot.all_units = create_mock_filter_result([medivac, marine])
    
    bot.calculate_drop_reward(active_tag=1)
    assert 101 in bot.initially_loaded_units

@pytest.mark.asyncio
async def test_target_unloading_tracking(bot):
    await bot.reset_game_state()
    target_pos = Point2((10, 10))
    bot._objective_pos = target_pos
    
    marine = MockUnit(tag=101, type_id=UnitTypeId.MARINE, position=target_pos)
    medivac = MockUnit(tag=1, type_id=UnitTypeId.MEDIVAC, position=target_pos)
    
    bot.previous_passenger_tags.add(101)
    bot.units = create_mock_filter_result([medivac, marine])
    bot.all_units = create_mock_filter_result([medivac, marine])
    
    bot.calculate_drop_reward(active_tag=1)
    assert 101 in bot.target_unloaded_units

@pytest.mark.asyncio
async def test_refined_unloading_rewards(bot):
    await bot.reset_game_state()
    bot.training_conf['rewards']['unload_reward'] = 1.0
    target_pos = Point2((10, 10))
    bot._objective_pos = target_pos
    
    # 1. Unload from full transport (cargo=4)
    marine1 = MockUnit(tag=101, type_id=UnitTypeId.MARINE, position=target_pos)
    medivac = MockUnit(tag=1, type_id=UnitTypeId.MEDIVAC, position=target_pos, cargo_used=3)
    
    bot.previous_passenger_tags = {101, 102, 103, 104}
    bot.previous_transport_cargo = {1: 4}
    bot.units = create_mock_filter_result([medivac, marine1])
    bot.all_units = create_mock_filter_result([medivac, marine1])
    
    reward = bot.calculate_drop_reward(active_tag=1)
    assert reward == pytest.approx(1.75) # 1.0 * (1 + (4-1)*0.25)
    assert 101 in bot.target_unloaded_units

    # 2. Second unit from same drop (cargo was 3 before this one)
    # Note: bot.previous_units_in_zone will contain marine1 now
    bot.previous_units_in_zone = {101}
    bot.previous_transport_cargo = {1: 3}
    marine2 = MockUnit(tag=102, type_id=UnitTypeId.MARINE, position=target_pos)
    medivac.cargo_used = 2
    bot.units = create_mock_filter_result([medivac, marine1, marine2])
    
    reward = bot.calculate_drop_reward(active_tag=1)
    assert reward == pytest.approx(1.5) # 1.0 * (1 + (3-1)*0.25)
    assert 102 in bot.target_unloaded_units

    # 3. Last unit (cargo was 1 before this one)
    bot.previous_units_in_zone = {101, 102, 103}
    bot.previous_transport_cargo = {1: 1}
    marine4 = MockUnit(tag=104, type_id=UnitTypeId.MARINE, position=target_pos)
    medivac.cargo_used = 0
    bot.units = create_mock_filter_result([medivac, marine1, marine2, marine4])
    
    reward = bot.calculate_drop_reward(active_tag=1)
    assert reward == pytest.approx(1.0) # 1.0 * (1 + (1-1)*0.25)
    
    # 4. Second unload of same unit
    bot.previous_units_in_zone = set() # Force "new arrival"
    bot.target_unloaded_units = {104}
    reward = bot.calculate_drop_reward(active_tag=1)
    assert reward == 0.0 # Already in target_unloaded_units