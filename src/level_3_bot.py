from src.sc2_bot import SC2Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
import random
import numpy as np
from loguru import logger
from typing import Tuple, Dict

class Level3Bot(SC2Bot):
    def __init__(self, conn, training_conf=None):
        super().__init__(conn, training_conf)
        self.targets_count = self.training_conf.get("targets", 1)
        self.active_zerglings = [] # List of unit tags? Or Unit objects (be careful with staleness)
        self.active_beacons = []
        # We need to map zergling -> beacon? 
        # Spec says: "Direct multiple zerglings to multiple beacons". 
        # Usually any zergling can take any beacon.
        # "Sequential Collection: When a zergling touches a beacon, both that specific zergling and the beacon are collected"
        
        # We will use lists and remove as we go.
        self.previous_distances = {} # Map zergling_tag -> distance to nearest beacon

    async def reset_game_state(self):
        self.previous_distances = {}
        
        # Kill existing
        tags_to_kill = []
        if self.units(UnitTypeId.ZERGLING).exists:
             tags_to_kill.extend([u.tag for u in self.units(UnitTypeId.ZERGLING)])
        if self.units(UnitTypeId.BEACON_TERRANSMALL).exists:
             tags_to_kill.extend([u.tag for u in self.units(UnitTypeId.BEACON_TERRANSMALL)])
             
        if tags_to_kill:
            await self.client.debug_kill_unit(tags_to_kill)
        
        # Spawn new ones
        grid = self.game_info.pathing_grid
        valid_spawn_points = []
        for x in range(4, grid.width - 4, 4):
            for y in range(4, grid.height - 4, 4):
                if grid[(x, y)] != 0:
                    valid_spawn_points.append(Point2((x, y)))
        
        # If not enough points, fallback to random
        if len(valid_spawn_points) < self.targets_count * 2:
            w, h = self.game_info.map_size
            for _ in range(self.targets_count):
                start = Point2((random.uniform(w*0.1, w*0.9), random.uniform(h*0.1, h*0.9)))
                end = Point2((random.uniform(w*0.1, w*0.9), random.uniform(h*0.1, h*0.9)))
                await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 1, start, self.player_id]])
                await self.client.debug_create_unit([[UnitTypeId.BEACON_TERRANSMALL, 1, end, self.player_id]])
        else:
            locations = random.sample(valid_spawn_points, self.targets_count * 2)
            for i in range(self.targets_count):
                start = locations[i]
                end = locations[self.targets_count + i]
                await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 1, start, self.player_id]])
                await self.client.debug_create_unit([[UnitTypeId.BEACON_TERRANSMALL, 1, end, self.player_id]])

    def get_active_zerglings(self):
        """Returns sorted list of active zerglings."""
        if not hasattr(self, "units"): return []
        return sorted(self.units(UnitTypeId.ZERGLING), key=lambda u: u.tag)

    async def apply_action(self, unit_tag, action):
        """Applies action to specific unit."""
        if not self.agent_system: return
        
        # Convert action
        world_x, world_y = self.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world(action)
        target_pos = Point2((world_x, world_y))
        
        # Find unit
        unit = self.units.find_by_tag(unit_tag)
        if unit:
            cmd = unit.move(target_pos)
            if cmd:
                await self.do(cmd)

    def get_observation_for_unit(self, unit_tag) -> Tuple[np.ndarray, Dict]:
        """Returns observation centered on specific unit."""
        # We need to hack SC2Bot or AgentSystem to center on specific unit?
        # The ObservationTransformer typically centers on the "agent".
        # SC2Bot._get_observation sets self.zergling/self.beacon.
        # We should set self.zergling to the specific unit before calling agent_system.process_observation.
        
        unit = self.units.find_by_tag(unit_tag)
        if not unit:
            # Fallback if unit missing (should not happen in loop if careful)
            return np.zeros((1, 1, 1)), {} # Should return correct shape
            
        # Temporarily override self.zergling for observation generation
        original_zergling = self.zergling
        self.zergling = unit
        
        # Find nearest beacon for this unit?
        # Level0Env uses self.beacon.
        # We should find the nearest beacon to THIS unit.
        beacons = self.units(UnitTypeId.BEACON_TERRANSMALL)
        if beacons.exists:
            self.beacon = beacons.closest_to(unit)
        else:
            self.beacon = None
            
        # Call original _get_observation logic (but adapted)
        # SC2Bot._get_observation does:
        # self.zergling = ... (overwrites what we just set)
        # So we cannot call super()._get_observation() directly if it overwrites self.zergling.
        
        # We must copy logic from SC2Bot._get_observation but respect our selection.
        
        enemies = [self.beacon] if self.beacon else []
        
        raw_obs = {
            'units': self.units,
            'enemy_units': enemies,
            'effects': getattr(self.state, "effects", []),
            'game_info': getattr(self, "game_info", getattr(self, "_game_info", None)),
            'game_data': getattr(self, "game_data", getattr(self, "_game_data", None)),
            'creep': getattr(self.state, "creep", None),
            'command_target': self.last_target_pos
        }
        
        if self.agent_system:
             processed_obs = self.agent_system.process_observation(observation=raw_obs)["agent_0"]
        else:
             processed_obs = np.zeros((1, 64, 64)) # Fallback
             
        # Restore (optional)
        self.zergling = original_zergling
        
        return processed_obs, {}
        
    def _get_observation(self) -> Tuple[np.ndarray, Dict]:
        # Need to handle observation for multiple units.
        # For now, let's just make it compilable.
        return super()._get_observation()
        
    async def check_collections(self):
        """
        Checks for zergling-beacon collisions.
        Removes collected pairs and returns success reward.
        """
        reward = 0.0
        
        # Get active units
        zerglings = self.units(UnitTypeId.ZERGLING)
        beacons = self.units(UnitTypeId.BEACON_TERRANSMALL)
        
        if not zerglings.exists or not beacons.exists:
            return 0.0
        
        # Load reward constants
        rewards_conf = self.training_conf.get('rewards', {})
        success_reward_total = rewards_conf.get('success', 100.0)
        success_reward_per_event = success_reward_total / max(1, self.targets_count)
        
        tags_to_kill = []
        collected_zerglings = set()
        collected_beacons = set()
        
        for z in zerglings:
            if z.tag in collected_zerglings:
                continue
            for b in beacons:
                if b.tag in collected_beacons:
                    continue
                
                # Check distance
                dist = z.distance_to(b)
                if dist < 2.0:
                    # Collection!
                    logger.info(f"Collection! Zergling {z.tag} reached Beacon {b.tag}")
                    tags_to_kill.append(z.tag)
                    tags_to_kill.append(b.tag)
                    collected_zerglings.add(z.tag)
                    collected_beacons.add(b.tag)
                    reward += success_reward_per_event
                    break # Zergling consumed
        
        if tags_to_kill:
            await self.client.debug_kill_unit(tags_to_kill)
            
        return reward
