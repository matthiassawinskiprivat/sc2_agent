import pytest
import yaml
from pathlib import Path
from src.level_drop_bot import LevelDropBot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from unittest.mock import MagicMock

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
    return m

@pytest.fixture
def config():
    config_path = Path("config/level_drop_0.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def bot(config):
    mock_conn = MagicMock()
    bot = LevelDropBot(mock_conn, training_conf=config)
    bot.units = create_mock_filter_result([])
    bot.all_units = create_mock_filter_result([])
    bot.enemy_units = []
    bot.enemy_structures = create_mock_filter_result([])
    bot.state = MagicMock()
    bot.state.game_loop = 0
    return bot

@pytest.mark.asyncio
async def test_integration_reward_flow(bot, config):
    await bot.reset_game_state()
    base_load = config['env']['rewards']['load_reward']
    base_unload = config['env']['rewards']['unload_reward']
    
    # 1. Load Unit 1 into empty transport
    marine1 = MockUnit(tag=101, type_id=UnitTypeId.MARINE)
    medivac = MockUnit(tag=1, type_id=UnitTypeId.MEDIVAC, cargo_used=1)
    medivac.passengers = [marine1]
    
    bot.units = create_mock_filter_result([medivac])
    bot.all_units = create_mock_filter_result([medivac, marine1])
    bot.previous_transport_cargo = {1: 0}
    
    reward = bot.calculate_drop_reward(active_tag=1)
    assert reward == pytest.approx(base_load)
    
    # 2. Load Unit 2 into transport (cargo=1)
    marine2 = MockUnit(tag=102, type_id=UnitTypeId.MARINE)
    medivac.passengers = [marine1, marine2]
    medivac.cargo_used = 2
    bot.previous_transport_cargo = {1: 1}
    bot.previous_passenger_tags = {101}
    bot.all_units = create_mock_filter_result([medivac, marine1, marine2])
    
    reward = bot.calculate_drop_reward(active_tag=1)
    assert reward == pytest.approx(base_load * 1.25)
    
    # 3. Move to Target Zone
    target_pos = Point2((10, 10))
    bot._objective_pos = target_pos
    medivac.position = target_pos
    # No reward for moving (distance reward is for transports with cargo, but we just check arrival)
    
    # 4. Unload Unit 1 (cargo was 2 before)
    bot.previous_transport_cargo = {1: 2}
    bot.previous_passenger_tags = {101, 102}
    bot.previous_units_in_zone = set()
    
    marine1.position = target_pos # Unloaded
    bot.units = create_mock_filter_result([medivac, marine1])
    
    reward = bot.calculate_drop_reward(active_tag=1)
    assert reward == pytest.approx(base_unload * 1.25) # (1 + (2-1)*0.25)
