
import pytest
import numpy as np
from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants

def test_observation_transformer_layer_pruning_config():
    # Test that we can initialize with a subset of layers
    active_layers = [
        LayerConstants.ALLIED_UNIT_DENSITY,
        LayerConstants.ENEMY_UNIT_DENSITY,
        LayerConstants.TERRAIN_PATHABLE,
        LayerConstants.COMMAND_OBJECTIVE
    ]
    
    # We'll use a new config parameter 'active_layers'
    transformer = ObservationTransformer(
        map_size=(32, 32),
        training_conf={'active_layers': active_layers}
    )
    
    assert transformer.active_layers == active_layers
    assert transformer.num_active_layers == 4

def test_observation_transformer_pruned_output_shape():
    active_layers = [
        LayerConstants.ALLIED_UNIT_DENSITY,
        LayerConstants.ENEMY_UNIT_DENSITY,
        LayerConstants.TERRAIN_PATHABLE,
        LayerConstants.COMMAND_OBJECTIVE
    ]
    
    transformer = ObservationTransformer(
        map_size=(32, 32),
        training_conf={'active_layers': active_layers}
    )
    
    # Mock minimal raw_obs
    raw_obs = {
        'units': [],
        'enemy_units': [],
        'effects': [],
        'game_info': None,
        'game_data': None
    }
    
    layers = transformer.transform(raw_obs)
    
    # Should only return 4 layers instead of 42
    assert layers.shape == (4, 32, 32)

def test_observation_transformer_default_full_layers():
    # Test backward compatibility
    transformer = ObservationTransformer(map_size=(32, 32))
    
    raw_obs = {
        'units': [],
        'enemy_units': [],
        'effects': [],
        'game_info': None,
        'game_data': None
    }
    
    layers = transformer.transform(raw_obs)
    assert layers.shape == (LayerConstants.get_num_layers(), 32, 32)
