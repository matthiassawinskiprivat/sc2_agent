"""
A simple bot that acts as a wrapper for the HierarchicalAgentSystem,
and handles the communication with the main process.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import multiprocessing
import multiprocessing.connection
import random
from loguru import logger

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from src.hierarchical_agent import HierarchicalAgentSystem


class SC2Bot(BotAI):
    """A simple bot that wraps the HierarchicalAgentSystem."""

    def __init__(self, conn: multiprocessing.connection.Connection, training_conf: Optional[Dict] = None):
        super().__init__()
        self.conn = conn
        self.training_conf = training_conf or {}
        self.agent_system: Optional[HierarchicalAgentSystem] = None
        self.zergling: Optional[Unit] = None
        self.beacon: Optional[Unit] = None
        self.previous_distance = None
        self.last_target_pos: Optional[Point2] = None
        
        from sc2.units import Units
        self.units = Units([], self)
        self.structures = Units([], self)
        self.enemy_units = Units([], self)
        self.enemy_structures = Units([], self)
        self.all_units = Units([], self)
        self.enemy_race = None
        self.unit_command_uses_self_do = False
        self.actions = []
        self.unit_tags_received_action = set()

    async def on_step(self, iteration: int):
        if iteration == 0:
            # Try to get grid_size from parent env if possible
            grid_size = 128
            if hasattr(self, "_bot_object") and hasattr(self._bot_object, "grid_size"):
                grid_size = self._bot_object.grid_size
            elif hasattr(self, "grid_size"):
                grid_size = self.grid_size

            self.agent_system = HierarchicalAgentSystem(
                map_size=(128, 128),
                grid_size=grid_size,
                training_conf=self.training_conf
            )
            self.agent_system.create_unit_agent("agent_0")

            await self.reset_game_state()
            obs, info = self._get_observation()
            self.conn.send((obs, info))

        command, data = self.conn.recv()
        if command == "step":
            action = data
            world_x, world_y = self.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world(action)
            self.last_target_pos = Point2((world_x, world_y))

            if self.zergling:
                cmd = self.zergling.move(self.last_target_pos)
                if cmd and not isinstance(cmd, bool):
                    await self.do(cmd)
            
            obs, reward, done, truncated, info = self._get_step_results()
            self.conn.send((obs, reward, done, truncated, info))

        elif command == "reset":
            await self.reset_game_state()
            obs, info = self._get_observation()
            self.conn.send((obs, info))

        elif command == "close":
            await self.client.leave()

    async def reset_game_state(self):
        self.previous_distance = None
        zergling = self.units(UnitTypeId.ZERGLING).first if self.units(UnitTypeId.ZERGLING).exists else None
        beacon = self.units(UnitTypeId.BEACON_TERRANSMALL).first if self.units(UnitTypeId.BEACON_TERRANSMALL).exists else None
        
        grid = self.game_info.pathing_grid
        valid_spawn_points = []
        for x in range(4, grid.width - 4, 4):
            for y in range(4, grid.height - 4, 4):
                if grid[(x, y)] != 0:
                    valid_spawn_points.append(Point2((x, y)))
        
        if valid_spawn_points:
            start = random.choice(valid_spawn_points)
            end = random.choice(valid_spawn_points)
        else:
            w, h = self.game_info.map_size
            start = Point2((random.uniform(w*0.1, w*0.9), random.uniform(h*0.1, h*0.9)))
            end = Point2((random.uniform(w*0.1, w*0.9), random.uniform(h*0.1, h*0.9)))

        if zergling:
            await self.client.debug_kill_unit([zergling.tag])
        await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 1, start, self.player_id]])

        if beacon:
            await self.client.debug_kill_unit([beacon.tag])
        await self.client.debug_create_unit([[UnitTypeId.BEACON_TERRANSMALL, 1, end, self.player_id]])
        
        await self.client._send_debug()


    def action_masks(self) -> np.ndarray:
        if self.agent_system and "agent_0" in self.agent_system.unit_agents:
            return self.agent_system.unit_agents["agent_0"].get_action_mask()
        
        # Determine fallback grid_size
        gs = 128
        if self.agent_system:
            gs = self.agent_system.grid_size
        return np.ones(gs * gs, dtype=bool)

    def _get_observation(self) -> Tuple[np.ndarray, Dict]:
        self.zergling = self.units(UnitTypeId.ZERGLING).first if self.units(UnitTypeId.ZERGLING).exists else None
        self.beacon = self.units(UnitTypeId.BEACON_TERRANSMALL).first if self.units(UnitTypeId.BEACON_TERRANSMALL).exists else None

        # Initialize previous_distance on first observation if zergling and beacon are present
        if self.previous_distance is None and self.zergling and self.beacon:
            self.previous_distance = ((self.zergling.position.x - self.beacon.position.x)**2 + (self.zergling.position.y - self.beacon.position.y)**2)**0.5

        enemies = [self.beacon] if self.beacon else []
        
        raw_obs = {
            'units': self.units,
            'enemy_units': enemies,
            'effects': self.state.effects,
            'game_info': self.game_info,
            'game_data': self.game_data,
            'creep': self.state.creep,
            'command_target': self.last_target_pos
        }
        
        processed_obs = self.agent_system.process_observation(observation=raw_obs)["agent_0"]
        return processed_obs, {}

    def _calculate_reward(self) -> Tuple[float, bool]:
        if not self.zergling or not self.beacon:
            return 0, False # No reward if zergling or beacon are missing

        # Manual distance calculation to ensure parity with master branch
        current_dist = ((self.zergling.position.x - self.beacon.position.x) ** 2 + (self.zergling.position.y - self.beacon.position.y) ** 2)**0.5
        reward = 0.0
        reached_target = False

        # Win Condition: Agent reached the beacon
        if current_dist < 2.0:  # Slightly larger radius to ensure hit
            logger.info("SUCCESS! Reached Beacon.")
            reward = 100.0
            reached_target = True
        elif self.previous_distance is not None:
            # Reward for getting closer (positive) or penalty for moving away (negative)
            reward = (self.previous_distance - current_dist) * 1.0 # Scale the reward

        # Update previous distance for the next step
        self.previous_distance = current_dist

        return reward, reached_target

    def _get_step_results(self) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, info = self._get_observation()
        reward, reached_target = self._calculate_reward()
        done = False
        truncated = False
        if reached_target:
            done = True
        return obs, reward, done, truncated, info
