import random
import math
from typing import Tuple, Optional
from sc2.position import Point2
from loguru import logger

class SpawningUtils:
    @staticmethod
    def calculate_safe_distance(width: float, height: float, requested_dist: Optional[float] = None) -> float:
        """
        Calculates a safe distance between pylons.
        If requested_dist is None, defaults to 40% of the map diagonal.
        Ensures the distance doesn't exceed 80% of the map's smaller dimension minus margins.
        """
        diagonal = math.sqrt(width**2 + height**2)
        margin = 10.0 # Total margin (5 on each side)
        
        # Default to 40% of diagonal if not specified
        target_dist = requested_dist if requested_dist is not None else (diagonal * 0.4)
        
        # Max possible distance is constrained by the smaller dimension to ensure spawnability
        max_possible = (min(width, height) - margin) * 0.8
        
        safe_dist = min(target_dist, max_possible)
        return max(0.0, safe_dist)

    @staticmethod
    def get_random_pylon_positions(
        width: float, 
        height: float, 
        min_distance: float, 
        margin: float = 5.0,
        max_attempts: int = 1000
    ) -> Tuple[Point2, Point2]:
        """
        Generates two random positions for Pylons that respect a minimum distance.
        """
        def get_rand_pos():
            return Point2((
                round(random.uniform(margin, width - margin), 1),
                round(random.uniform(margin, height - margin), 1)
            ))

        pos1 = get_rand_pos()
        pos2 = get_rand_pos()
        
        # Relaxed distance slightly for grid snapping stability
        check_dist = max(0, min_distance - 1.0)

        attempts = 0
        while pos1.distance_to(pos2) < check_dist and attempts < max_attempts:
            pos1 = get_rand_pos()
            pos2 = get_rand_pos()
            attempts += 1
            
        if attempts >= max_attempts:
            logger.warning(f"Failed to find valid random positions after {max_attempts} attempts. Using fallback.")
            # Fallback: Opposite corners
            if random.random() > 0.5:
                pos1 = Point2((margin, margin))
                pos2 = Point2((width - margin, height - margin))
            else:
                pos1 = Point2((margin, height - margin))
                pos2 = Point2((width - margin, margin))
                
        return pos1, pos2
