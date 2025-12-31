from src.sc2_bot import SC2Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
import random
import numpy as np
from loguru import logger
from typing import Tuple, Dict

class Level3Bot(SC2Bot):
    """
    Level 3 Bot for handling multiple units and multiple targets (beacons).
    It implements a cycle-based control where each unit gets an observation
    and takes an action sequentially within a single game step.
    """
    def __init__(self, conn, training_conf=None):
        super().__init__(conn, training_conf)
        self.targets_count = self.training_conf.get("targets", 1)
        self.active_zerglings = [] 
        self.active_beacons = []
        
        # previous_distances is used to calculate reward based on progress towards beacons
        # for each individual zergling.
        self.previous_distances = {} # Map zergling_tag -> distance to nearest beacon

    async def reset_game_state(self):
        """
        Cleans up the map and spawns new Zerglings and Beacons.
        This is called at the start of each episode.
        """
        self.previous_distances = {}
        
        # Kill existing units to ensure a clean slate
        tags_to_kill = []
        if self.units(UnitTypeId.ZERGLING).exists:
             tags_to_kill.extend([u.tag for u in self.units(UnitTypeId.ZERGLING)])
        if self.units(UnitTypeId.BEACON_TERRANSMALL).exists:
             tags_to_kill.extend([u.tag for u in self.units(UnitTypeId.BEACON_TERRANSMALL)])
             
        if tags_to_kill:
            await self.client.debug_kill_unit(tags_to_kill)
        
        # Spawn new units at valid pathable locations
        grid = self.game_info.pathing_grid
        valid_spawn_points = []
        # Sample points on a grid to avoid spawning units inside walls or on top of each other
        for x in range(4, grid.width - 4, 4):
            for y in range(4, grid.height - 4, 4):
                if grid[(x, y)] != 0:
                    valid_spawn_points.append(Point2((x, y)))
        
        # Fallback to random sampling if the grid approach doesn't provide enough points
        if len(valid_spawn_points) < self.targets_count * 2:
            w, h = self.game_info.map_size
            for _ in range(self.targets_count):
                start = Point2((random.uniform(w*0.1, w*0.9), random.uniform(h*0.1, h*0.9)))
                end = Point2((random.uniform(w*0.1, w*0.9), random.uniform(h*0.1, h*0.9)))
                await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 1, start, self.player_id]])
                await self.client.debug_create_unit([[UnitTypeId.BEACON_TERRANSMALL, 1, end, self.player_id]])
        else:
            # Pair each Zergling with a Beacon by sampling unique locations
            locations = random.sample(valid_spawn_points, self.targets_count * 2)
            for i in range(self.targets_count):
                start = locations[i]
                end = locations[self.targets_count + i]
                await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 1, start, self.player_id]])
                await self.client.debug_create_unit([[UnitTypeId.BEACON_TERRANSMALL, 1, end, self.player_id]])

    def get_active_zerglings(self):
        """Returns sorted list of active zerglings to ensure deterministic iteration order."""
        if not hasattr(self, "units"): return []
        return sorted(self.units(UnitTypeId.ZERGLING), key=lambda u: u.tag)

    async def apply_action(self, unit_tag, action):
        """
        Translates the agent's spatial action (index in grid) to a world-space move command
        for a specific unit identified by its tag.
        """
        if not self.agent_system: return
        
        # Convert the discrete action index to world coordinates (x, y)
        world_x, world_y = self.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world(action)
        target_pos = Point2((world_x, world_y))
        
        # Find the specific unit and issue the move command
        unit = self.units.find_by_tag(unit_tag)
        if unit:
            cmd = unit.move(target_pos)
            if cmd:
                # We use self.do(cmd) which adds the action to the batch 
                # to be sent in the next client.step() call in Level3Env
                self.do(cmd)

    def get_observation_for_unit(self, unit_tag) -> Tuple[np.ndarray, Dict]:
        """
        Generates an observation that is centered on the specified unit.
        This is a 'hack' to allow the single-unit AgentSystem to process 
        observations for multiple units by swapping the focused unit.
        """
        unit = self.units.find_by_tag(unit_tag)
        if not unit:
            # If the unit is missing (e.g., died or collected), return an empty observation
            # The environment should ideally prevent this via proper tag management.
            return np.zeros((1, 1, 1)), {} 
            
        # Temporarily override self.zergling so the ObservationTransformer 
        # treats THIS unit as the primary agent for centering and feature layers.
        original_zergling = self.zergling
        self.zergling = unit
        
        # Find the nearest beacon for THIS unit to include in the observation
        beacons = self.units(UnitTypeId.BEACON_TERRANSMALL)
        if beacons.exists:
            self.beacon = beacons.closest_to(unit)
        else:
            self.beacon = None
            
        # Construct the raw observation dictionary that the transformer expects
        enemies = [self.beacon] if self.beacon else []
        
        raw_obs = {
            'units': self.units,
            'selected_units': [unit],  # Explicitly mark THIS unit as selected
            'enemy_units': enemies,
            'effects': getattr(self.state, "effects", []),
            'game_info': getattr(self, "game_info", getattr(self, "_game_info", None)),
            'game_data': getattr(self, "game_data", getattr(self, "_game_data", None)),
            'creep': getattr(self.state, "creep", None),
            'command_target': self.last_target_pos
        }
        
        # Process the observation through the HierarchicalAgentSystem's transformer
        if self.agent_system:
             processed_obs = self.agent_system.process_observation(observation=raw_obs)["agent_0"]
        else:
             processed_obs = np.zeros((1, 64, 64))
             
        # Restore the original zergling state (though it might not be strictly necessary)
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
