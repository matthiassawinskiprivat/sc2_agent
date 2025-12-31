import numpy as np
import pytest
from unittest.mock import MagicMock
from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants
from sc2.position import Point2
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId

class MockUnit:
    def __init__(self, tag, pos, radius=1.0, type_id=UnitTypeId.ZERGLING):
        self.tag = tag
        self.position = pos
        self.radius = radius
        self.type_id = type_id
        self.is_structure = False
        self.is_enemy = False
        self.is_blip = False
        self.is_cloaked = False
        self.is_detected = True
        self.is_ready = True
        self.health = 10
        self.health_max = 10
        self.shield = 0
        self.shield_max = 0
        self.energy = 0
        self.energy_max = 0
        self.weapon_cooldown = 0
        self.is_biological = True
        self.is_mechanical = False
        self.is_armored = False
        self.is_massive = False
        self.is_flying = False
        self.is_hallucination = False
        self.build_progress = 1.0
        self.detect_range = 0

def test_observation_identity_selection():
    """
    Verify that providing 'selected_units' restricts COMMAND_ASSIGNMENT layer
    to only those units.
    """
    transformer = ObservationTransformer(map_size=(64, 64))
    
    # Setup units
    # Unit A at (10, 10)
    # Unit B at (20, 20)
    unit_a = MockUnit(tag=1, pos=Point2((10, 10)))
    unit_b = MockUnit(tag=2, pos=Point2((20, 20)))
    
    raw_obs = {
        'units': [unit_a, unit_b],
        'selected_units': [unit_a], # ONLY Unit A is selected
        'enemy_units': [],
        'effects': [],
        'game_info': MagicMock(map_size=MagicMock(width=64, height=64)),
        'game_data': None,
        'creep': None,
        'command_target': None
    }
    
    # Mock playable area and grids
    raw_obs['game_info'].playable_area = MagicMock(x=0, y=0, width=64, height=64)
    raw_obs['game_info'].map_name = "test_map"
    
    grid = np.zeros((64, 64), dtype=np.uint8)
    raw_obs['game_info'].pathing_grid = MagicMock()
    raw_obs['game_info'].pathing_grid.data_numpy = grid
    
    raw_obs['game_info'].terrain_height = MagicMock()
    raw_obs['game_info'].terrain_height.data_numpy = grid
    
    layers = transformer.transform(raw_obs)
    
    idx_sel = LayerConstants.LAYERS[LayerConstants.COMMAND_ASSIGNMENT]
    sel_layer = layers[idx_sel]
    
    # Get pixel coords
    px_a, py_a = transformer._get_pixel_coords(unit_a.position)
    px_b, py_b = transformer._get_pixel_coords(unit_b.position)
    
    # Verify Unit A is selected
    assert sel_layer[py_a, px_a] > 0, "Unit A should be marked as selected"
    
    # Verify Unit B is NOT selected
    assert sel_layer[py_b, px_b] == 0, "Unit B should NOT be marked as selected"

def test_observation_legacy_fallback():
    """
    Verify that omitting 'selected_units' falls back to selecting ALL units
    (Backward compatibility for Level 0).
    """
    transformer = ObservationTransformer(map_size=(64, 64))
    
    unit_a = MockUnit(tag=1, pos=Point2((10, 10)))
    unit_b = MockUnit(tag=2, pos=Point2((20, 20)))
    
    raw_obs = {
        'units': [unit_a, unit_b],
        # 'selected_units' is MISSING
        'enemy_units': [],
        'effects': [],
        'game_info': MagicMock(map_size=MagicMock(width=64, height=64)),
        'game_data': None,
        'creep': None,
        'command_target': None
    }
    
    raw_obs['game_info'].playable_area = MagicMock(x=0, y=0, width=64, height=64)
    raw_obs['game_info'].map_name = "test_map"

    grid = np.zeros((64, 64), dtype=np.uint8)
    raw_obs['game_info'].pathing_grid = MagicMock()
    raw_obs['game_info'].pathing_grid.data_numpy = grid
    
    raw_obs['game_info'].terrain_height = MagicMock()
    raw_obs['game_info'].terrain_height.data_numpy = grid
    
    layers = transformer.transform(raw_obs)
    
    idx_sel = LayerConstants.LAYERS[LayerConstants.COMMAND_ASSIGNMENT]
    sel_layer = layers[idx_sel]
    
    px_a, py_a = transformer._get_pixel_coords(unit_a.position)
    px_b, py_b = transformer._get_pixel_coords(unit_b.position)
    
    assert sel_layer[py_a, px_a] > 0, "Unit A should be selected in fallback mode"
    assert sel_layer[py_b, px_b] > 0, "Unit B should be selected in fallback mode"
