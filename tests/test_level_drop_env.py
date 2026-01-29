import pytest
from unittest.mock import MagicMock
from sc2.unit import Unit
from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId

class MockLevelDropEnv:
    def __init__(self):
        self.training_conf = {
            'rewards': {
                'load_reward': 0.1,
                'unload_reward': 1.0,
            }
        }
        self.command_center = None
        self.previous_cargo_sum = 0
        self.previous_units_in_zone = set()
        
        self.units = []
        
    def calculate_drop_reward(self):
        total_reward = 0.0
        
        # 1. Load Reward (Cargo Increase)
        current_cargo_sum = sum(u.cargo_used for u in self.units if hasattr(u, "cargo_used"))
        cargo_diff = current_cargo_sum - self.previous_cargo_sum
        
        if cargo_diff > 0:
            total_reward += cargo_diff * self.training_conf['rewards']['load_reward']
        self.previous_cargo_sum = current_cargo_sum
        
        # 2. Arrival Reward (Unload/Enter Zone)
        if self.command_center:
            # Definition of Zone: 15 range around CC
            units_in_zone = {u.tag for u in self.units 
                             if not u.is_flying 
                             and u.distance_to(self.command_center.position) < 15.0}
            
            new_arrivals = units_in_zone - self.previous_units_in_zone
            if new_arrivals:
                total_reward += len(new_arrivals) * self.training_conf['rewards']['unload_reward']
                
            self.previous_units_in_zone = units_in_zone
            
        return total_reward

def create_mock_unit(unit_type, pos, tag, cargo_used=0):
    u = MagicMock(spec=Unit)
    u.type_id = unit_type
    u.position = pos
    u.tag = tag
    u.cargo_used = cargo_used
    u.is_flying = (unit_type == UnitTypeId.MEDIVAC)
    u.distance_to = lambda p: ((u.position.x - p.x)**2 + (u.position.y - p.y)**2)**0.5
    return u

def test_drop_reward_logic():
    env = MockLevelDropEnv()
    cc_pos = Point2((50, 50))
    env.command_center = create_mock_unit(UnitTypeId.COMMANDCENTER, cc_pos, tag=99)
    
    # 1. Start: Medivac + Marine (Far away)
    medivac = create_mock_unit(UnitTypeId.MEDIVAC, Point2((10, 10)), tag=1)
    marine = create_mock_unit(UnitTypeId.MARINE, Point2((10, 10)), tag=2)
    env.units = [medivac, marine]
    
    # Step 1: No Reward
    assert env.calculate_drop_reward() == 0.0
    
    # Step 2: Load Marine
    medivac.cargo_used = 1
    env.units = [medivac] # Marine disappears
    
    assert env.calculate_drop_reward() == 0.1 # Load Reward
    
    # Step 3: Move to Zone
    medivac.position = Point2((50, 50))
    # Marine still inside
    
    assert env.calculate_drop_reward() == 0.0
    
    # Step 4: Unload
    medivac.cargo_used = 0
    marine.position = Point2((50, 50))
    env.units = [medivac, marine] # Marine reappears in zone
    
    assert env.calculate_drop_reward() == 1.0 # Unload Reward (1 unit arrived)
    
    # Step 5: Stay there
    assert env.calculate_drop_reward() == 0.0 # No double counting