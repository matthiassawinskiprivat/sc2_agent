import pytest
from src.agent.commands import CommandType
from src.utils.layer_constants import LayerConstants

def test_command_type_drop_exists():
    """Verify DROP command is defined."""
    assert hasattr(CommandType, "DROP"), "CommandType.DROP missing"
    assert CommandType.DROP.name == "DROP"

def test_layer_constants_transport_layers():
    """Verify Transport layers are defined and have unique indices."""
    assert hasattr(LayerConstants, "TRANSPORT_SLOTS")
    assert hasattr(LayerConstants, "PASSENGER_DENSITY")
    
    idx_slots = LayerConstants.LAYERS[LayerConstants.TRANSPORT_SLOTS]
    idx_pass = LayerConstants.LAYERS[LayerConstants.PASSENGER_DENSITY]
    
    assert idx_slots == 42
    assert idx_pass == 43
    
    # Check for uniqueness
    all_indices = list(LayerConstants.LAYERS.values())
    assert all_indices.count(idx_slots) == 1
    assert all_indices.count(idx_pass) == 1
    
def test_layer_count():
    """Verify total layer count includes new layers."""
    assert LayerConstants.get_num_layers() == 44
