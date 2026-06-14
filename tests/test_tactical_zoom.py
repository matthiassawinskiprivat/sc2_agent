
import numpy as np
import pytest
from unittest.mock import MagicMock
from src.transformers.observation_transformer import ObservationTransformer
from sc2.position import Point2
from src.utils.layer_constants import LayerConstants

def test_observation_transformer_tactical_zoom():
    # Setup transformer
    training_conf = {
        'agent_architecture': 'universal',
        'active_layers': ['allied_unit_density', 'enemy_unit_density', 'command_objective', 'command_assignment', 'terrain_pathable'],
        'active_tactical_layers': ['allied_unit_density', 'command_objective', 'command_assignment', 'terrain_pathable']
    }
    transformer = ObservationTransformer(map_size=(128, 128), training_conf=training_conf)
    
    # Mock SC2 Unit
    active_unit = MagicMock()
    active_unit.position = Point2((50.0, 50.0))
    active_unit.radius = 0.5
    active_unit.alliance = 1 # Allied
    active_unit.type_id.name = "ZERGLING"
    active_unit.is_structure = False
    active_unit.health = 35
    active_unit.health_max = 35
    active_unit.shield = 0
    active_unit.shield_max = 0
    active_unit.energy = 0
    active_unit.energy_max = 0
    active_unit.detect_range = 0
    active_unit.is_cloaked = False
    active_unit.is_biological = True
    active_unit.is_mechanical = False
    active_unit.is_armored = False
    active_unit.is_massive = False
    active_unit.is_flying = False
    active_unit.type_id.value = 105
    active_unit.is_blip = False
    active_unit.build_progress = 1.0
    active_unit.weapon_cooldown = 0
    
    # Target Beacon (for command_objective)
    beacon = MagicMock()
    beacon.position = Point2((52.0, 52.0))
    beacon.radius = 1.5
    
    # Mock Raw Obs
    raw_obs = {
        'units': [active_unit],
        'enemy_units': [],
        'structures': [],
        'enemy_structures': [],
        'game_info': MagicMock(),
        'game_data': MagicMock(),
        'high_level_command': 'scout',
        'command_target': beacon
    }
    raw_obs['game_info'].map_name = "TestMap"
    raw_obs['game_info'].map_size.width = 128
    raw_obs['game_info'].map_size.height = 128
    raw_obs['game_info'].playable_area.x = 0
    raw_obs['game_info'].playable_area.y = 0
    raw_obs['game_info'].playable_area.width = 128
    raw_obs['game_info'].playable_area.height = 128
    
    # Fill pathing grid with 1.0 (all pathable) for the test
    raw_obs['game_info'].pathing_grid = MagicMock()
    raw_obs['game_info'].pathing_grid.data_numpy = np.ones((128, 128))
    # Fill terrain height with 10 (flat ground)
    raw_obs['game_info'].terrain_height = MagicMock()
    raw_obs['game_info'].terrain_height.data_numpy = np.full((128, 128), 10.0)

    # Test Global Transform
    global_obs = transformer.transform(raw_obs)
    assert isinstance(global_obs, dict)
    assert "map" in global_obs
    assert global_obs["map"].shape == (5, 128, 128)
    
    # Test Tactical Transform
    tactical_obs = transformer.transform_tactical(raw_obs, active_unit)
    assert tactical_obs.shape == (4, 32, 32) # Density, Objective, Assignment, Terrain
    
    # Layer 0: Allied Density (Centered at 16, 15)
    center_area = tactical_obs[0, 13:19, 13:19]
    assert np.any(center_area > 0), "Active unit not found in tactical density layer"

    # Layer 1: Command Objective (at 52, 52)
    # Scale is 1.0. Crop is (50-16, 50-16) = (34, 34)
    # Target PX: (52 - 34) = 18
    # Target PY: 31 - (52 - 34) = 31 - 18 = 13
    obj_pixel = tactical_obs[1, 13, 18]
    assert obj_pixel > 0, f"Command objective not found at expected tactical coordinates (13, 18), got {obj_pixel}"

    # Layer 2: Command Assignment (Self) — footprint must match density (not a special-case path).
    assign_layer = tactical_obs[2]
    assert np.count_nonzero(assign_layer > 0) == np.count_nonzero(tactical_obs[0] > 0)
    self_pixel = assign_layer[15, 16]
    assert self_pixel > 0, f"Command assignment (Self) not found at tactical center (15, 16), got {self_pixel}"
    
    # Layer 3: Terrain Pathable
    # Normalization floor is 0.2, so if it's all ones it will be around 0.2
    assert np.mean(tactical_obs[3]) >= 0.15, "Tactical terrain layer is empty"

    # --- Action Head Alignment Verification ---
    from src.agent.universal_unit_control_agent import UniversalUnitControlAgent
    agent = UniversalUnitControlAgent() # Uses hardcoded 32
    spatial_idx = 15 * 32 + 16
    world_x, world_y = agent.convert_spatial_action_to_world("move_local", spatial_idx, active_unit=active_unit)
    
    assert world_x == 50.0
    assert world_y == 50.0

    print("Tactical zoom test passed with Terrain and Command Layers!")

if __name__ == "__main__":
    test_observation_transformer_tactical_zoom()
