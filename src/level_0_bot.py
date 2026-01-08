
import random
from typing import Dict, Optional, Tuple
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from loguru import logger
from src.sc2_bot import SC2Bot

class Level0Bot(SC2Bot):
    """
    Bot specifically for Level 0 (Collect the Beacon).
    Handles spawning of a Zergling and a Beacon.
    """
    async def reset_game_state(self, verify: bool = True):
        """
        Clears the map and spawns a Zergling and a Beacon.
        """
        await super().reset_game_state(verify=verify)
        
        # 1. Determine spawn area
        # We use the playable area from game_info
        area = self.game_info.playable_area
        margin = 5.0
        
        # Find valid pathable points
        grid = self.game_info.pathing_grid
        valid_points = []
        # Sample points to avoid huge list
        for x in range(int(area.x + margin), int(area.x + area.width - margin), 2):
            for y in range(int(area.y + margin), int(area.y + area.height - margin), 2):
                if grid[(x, y)] != 0:
                    valid_points.append(Point2((x, y)))

        if len(valid_points) < 2:
            # Fallback to simple random if grid is weird
            def get_rand_pos():
                return Point2((
                    random.uniform(area.x + margin, area.x + area.width - margin),
                    random.uniform(area.y + margin, area.y + area.height - margin)
                ))
            p1 = get_rand_pos()
            p2 = get_rand_pos()
        else:
            p1, p2 = random.sample(valid_points, 2)
        
        # Ensure minimum distance between zergling and beacon
        max_attempts = 10
        while p1.distance_to(p2) < 10.0 and max_attempts > 0:
            if len(valid_points) >= 2:
                p1, p2 = random.sample(valid_points, 2)
            max_attempts -= 1

        # 2. Spawn units
        # Use player_id for both to ensure visibility and simple 'units' access
        await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 1, p1, self.player_id]])
        await self.client.debug_create_unit([[UnitTypeId.BEACON_TERRANSMALL, 1, p2, self.player_id]])
        
        # We need a step to materialize them
        await self.client.step()
        await self._refresh_state()
        
        # Identify them
        self.zergling = self.units(UnitTypeId.ZERGLING).first if self.units(UnitTypeId.ZERGLING).exists else None
        self.beacon = self.units(UnitTypeId.BEACON_TERRANSMALL).first if self.units(UnitTypeId.BEACON_TERRANSMALL).exists else None
        
        if not self.zergling:
             logger.error("Level0Bot: Failed to spawn Zergling!")
        if not self.beacon:
             logger.error("Level0Bot: Failed to spawn Beacon!")
        
        self.previous_distance = None
        if self.zergling and self.beacon:
            self.previous_distance = self.zergling.distance_to(self.beacon)
            logger.info(f"Level0Bot spawned at {self.zergling.position}, target at {self.beacon.position}")
