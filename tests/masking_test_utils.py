"""
Utilities for testing action masking.
Provides mock game_info and units with configurable pathing and footprints.
"""

import numpy as np
from typing import Any, List, Tuple
from dataclasses import dataclass

@dataclass
class MockArea:
    x: float
    y: float
    width: float
    height: float

@dataclass
class MockMapSize:
    width: int
    height: int

class MockPathingGrid:
    def __init__(self, data: np.ndarray):
        self.data_numpy = data
        self.width = data.shape[1]
        self.height = data.shape[0]

class MockGameInfo:
    def __init__(self, pathing_data: np.ndarray, playable_area: MockArea = None):
        self.pathing_grid = MockPathingGrid(pathing_data)
        self.map_size = MockMapSize(pathing_data.shape[1], pathing_data.shape[0])
        self.playable_area = playable_area or MockArea(0, 0, self.map_size.width, self.map_size.height)
        self.terrain_height = np.zeros_like(pathing_data)

class MockUnit:
    def __init__(self, pos: Tuple[float, float], radius: float, is_structure: bool = False, type_id: Any = None):
        from sc2.position import Point2
        self.position = Point2(pos)
        self.radius = radius
        self.footprint_radius = radius if is_structure else 0.0
        self.is_structure = is_structure
        self.is_mineral_field = False
        self.is_vespene_geyser = False
        self.is_flying = False
        self.type_id = type_id

def create_mock_game_state(grid_size: int = 128, pathing_value: int = 1) -> MockGameInfo:
    """Creates a basic mock game info with uniform pathing."""
    pathing = np.full((grid_size, grid_size), pathing_value, dtype=np.uint8)
    return MockGameInfo(pathing)

def add_cliff_to_mock_state(game_info: MockGameInfo, x_range: Tuple[int, int], y_range: Tuple[int, int]):
    """Sets a rectangular area to unpathable (0)."""
    game_info.pathing_grid.data_numpy[y_range[0]:y_range[1], x_range[0]:x_range[1]] = 0
