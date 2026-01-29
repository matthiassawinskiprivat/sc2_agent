
import pytest
from unittest.mock import MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from src.sc2_bot import SC2Bot
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock Classes
class MockUnit:
    def __init__(self, tag, type_id, position, cargo_max=0, cargo_used=0, energy=0):
        self.tag = tag
        self.type_id = type_id
        self._position = position
        self.cargo_max = cargo_max
        self.cargo_used = cargo_used
        self.energy = energy
        self.cmd_mock = MagicMock()
        self.orders = []

    @property
    def position(self):
        return self._position

    def distance_to(self, pos):
        if hasattr(pos, "position"):
            pos = pos.position
        return self.position.distance_to(pos)
        
    def smart(self, target):
        self.cmd_mock("smart", target)
        return "smart_cmd"
        
    def move(self, target):
        self.cmd_mock("move", target)
        return "move_cmd"
        
    def attack(self, target):
        self.cmd_mock("attack", target)
        return "attack_cmd"

    def __call__(self, ability, target=None):
        # Determine if ability is an AbilityId enum or just a value
        ability_name = ability.name if hasattr(ability, "name") else str(ability)
        self.cmd_mock(ability_name, target)
        return f"{ability_name}_cmd"
        
    def __str__(self):
        return f"MockUnit({self.type_id}, tag={self.tag})"

class MockUnits(list):
    def find_by_tag(self, tag):
        for u in self:
            if u.tag == tag: return u
        return None
        
    def filter(self, func):
        return MockUnits([u for u in self if func(u)])
    
    # Support 'unit in units' check if needed, list already does this

class TestableSC2Bot(SC2Bot):
    def __init__(self):
        # Bypass SC2Bot and BotAI init
        self.units = MockUnits()
        self.action_heads = ["move", "attack", "load", "unload"]
        self.grid_size = 32
        self.agent_system = MagicMock()
        self.agent_system.unit_agents = {"agent_0": MagicMock()}
        # Setup coordinate conversion to return the position (10, 10)
        self.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world.return_value = (10, 10)
        self.client = MagicMock()
        self.last_target_pos = None
        
    def do(self, action):
        # Mock do to just accept the action
        pass

@pytest.mark.asyncio
async def test_marine_enters_medivac():
    """Test that a Marine uses SMART to enter a Medivac."""
    bot = TestableSC2Bot()
    marine = MockUnit(tag=1, type_id=UnitTypeId.MARINE, position=Point2((10, 10)))
    medivac = MockUnit(tag=2, type_id=UnitTypeId.MEDIVAC, position=Point2((10, 10)), cargo_max=8)
    
    bot.units.append(marine)
    bot.units.append(medivac)
    
    # Action index for 'load' head (idx 2)
    grid_area = 32*32
    action_idx = 2 * grid_area + 0 
    
    await bot.apply_action(marine.tag, action_idx)
    
    # Should use smart on the transport
    marine.cmd_mock.assert_called_with("smart", medivac)

@pytest.mark.asyncio
async def test_medivac_picks_up_marine():
    """Test that a Medivac uses LOAD_MEDIVAC to pick up a Marine."""
    bot = TestableSC2Bot()
    marine = MockUnit(tag=1, type_id=UnitTypeId.MARINE, position=Point2((10, 10)))
    medivac = MockUnit(tag=2, type_id=UnitTypeId.MEDIVAC, position=Point2((10, 10)), cargo_max=8)
    
    bot.units.append(marine)
    bot.units.append(medivac)
    
    # Action: Medivac tries to LOAD at (10, 10)
    grid_area = 32*32
    action_idx = 2 * grid_area + 0 
    
    await bot.apply_action(medivac.tag, action_idx)
    
    # CURRENT FAIL EXPECTATION: The current code will likely call 'smart' on itself or move
    # The NEW requirement is to call LOAD_MEDIVAC on the marine
    
    # This assertion defines the goal. It fails if the code is not updated.
    medivac.cmd_mock.assert_called_with("LOAD_MEDIVAC", marine)

@pytest.mark.asyncio
async def test_overlord_picks_up_drone():
    """Test that an Overlord uses LOAD_OVERLORD to pick up a Drone."""
    bot = TestableSC2Bot()
    drone = MockUnit(tag=3, type_id=UnitTypeId.DRONE, position=Point2((10, 10)))
    overlord = MockUnit(tag=4, type_id=UnitTypeId.OVERLORD, position=Point2((10, 10)), cargo_max=8)
    
    bot.units.append(drone)
    bot.units.append(overlord)
    
    grid_area = 32*32
    action_idx = 2 * grid_area + 0 
    
    await bot.apply_action(overlord.tag, action_idx)
    overlord.cmd_mock.assert_called_with("LOAD_OVERLORD", drone)

@pytest.mark.asyncio
async def test_warpprism_picks_up_probe():
    """Test that a WarpPrism uses LOAD_WARPPRISM to pick up a Probe."""
    bot = TestableSC2Bot()
    probe = MockUnit(tag=5, type_id=UnitTypeId.PROBE, position=Point2((10, 10)))
    prism = MockUnit(tag=6, type_id=UnitTypeId.WARPPRISM, position=Point2((10, 10)), cargo_max=8)
    
    bot.units.append(probe)
    bot.units.append(prism)
    
    grid_area = 32*32
    action_idx = 2 * grid_area + 0 
    
    await bot.apply_action(prism.tag, action_idx)
    prism.cmd_mock.assert_called_with("LOAD_WARPPRISM", probe)

@pytest.mark.asyncio
async def test_fallback_to_smart_if_not_transport():
    """Test fallback: A Zergling (non-transport) trying to load should just Smart onto target (e.g. Overlord)."""
    bot = TestableSC2Bot()
    zergling = MockUnit(tag=7, type_id=UnitTypeId.ZERGLING, position=Point2((10, 10)))
    overlord = MockUnit(tag=8, type_id=UnitTypeId.OVERLORD, position=Point2((10, 10)), cargo_max=8)
    
    bot.units.append(zergling)
    bot.units.append(overlord)
    
    grid_area = 32*32
    action_idx = 2 * grid_area + 0 
    
    await bot.apply_action(zergling.tag, action_idx)
    # Zergling isn't a transport, so it should look for a transport to enter (Smart)
    zergling.cmd_mock.assert_called_with("smart", overlord)
