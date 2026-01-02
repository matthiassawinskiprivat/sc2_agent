import gymnasium as gym
import asyncio
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import random
from gymnasium import spaces
from loguru import logger

from train.envs.multi_client_env import MultiClientEnv
from src.utils.layer_constants import LayerConstants
from src.agent.ctf_bot import CaptureTheFlagBot

# Standard SC2 Unit Costs (Simplified)
UNIT_COSTS = {
    "ZERGLING": {"minerals": 25, "vespene": 0, "type_id": 105},
    "ROACH": {"minerals": 75, "vespene": 25, "type_id": 110},
    "MARINE": {"minerals": 50, "vespene": 0, "type_id": 48}
}

class CaptureTheFlagEnv(MultiClientEnv):
    def __init__(self, 
                 minerals: int = 500, 
                 vespene: int = 0, 
                 flag_distance_min: float = 30.0,
                 unit_pool: List[str] = None,
                 grid_size: int = 128,
                 map_name: str = "Simple64",
                 max_steps: int = 1000,
                 **kwargs):
        super().__init__(map_name=map_name, **kwargs)
        self.minerals = minerals
        self.vespene = vespene
        self.flag_distance_min = flag_distance_min
        self.unit_pool = unit_pool or ["ZERGLING", "ROACH", "MARINE"]
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_count = 0
        
        self.active_layers = [
            "allied_g_dps_base",
            "enemy_g_dps_base",
            "allied_health",
            "enemy_health",
            "command_assignment",
            "command_objective",
            "terrain_pathable"
        ]
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(7, 128, 128), dtype=np.uint8)
        self.action_space = spaces.Discrete(grid_size * grid_size)
        
        self.bot1: Optional[CaptureTheFlagBot] = None
        self.bot2: Optional[CaptureTheFlagBot] = None
        
        self.active_player_id = 1
        self.current_unit_index = 0
        self.player_units = {1: [], 2: []} # List of tags

    async def _setup_game_session(self):
        from sc2.player import Bot
        from sc2.data import Race
        from sc2 import maps
        
        self.bot1 = CaptureTheFlagBot(player_id=1, training_conf={"active_layers": self.active_layers})
        self.bot2 = CaptureTheFlagBot(player_id=2, training_conf={"active_layers": self.active_layers})
        
        players = [Bot(Race.Zerg, self.bot1), Bot(Race.Zerg, self.bot2)]
        resolved_map = maps.get(self.map_name)
        
        await self._manager.create_join_game(resolved_map, players, realtime=self.realtime)
        
        await self.bot1.on_start()
        await self.bot2.on_start()
        
        await self._spawn_initial_units()
        await self._refresh_unit_lists()

    async def _spawn_initial_units(self):
        from sc2.ids.unit_typeid import UnitTypeId
        w, h = self.bot1.game_info.playable_area.width, self.bot1.game_info.playable_area.height
        
        pos1 = (10, 10) # Placeholder for better random placement logic
        pos2 = (w-10, h-10)
        
        await self.bot1.client.debug_create_unit([[UnitTypeId.PYLON, 1, pos1, 1]])
        await self.bot1.client.debug_create_unit([[UnitTypeId.PYLON, 1, pos2, 2]])
        
        # Resource spawning
        spawns1 = self._calculate_spawns(self.minerals, self.vespene)
        spawns2 = self._calculate_spawns(self.minerals, self.vespene)
        
        def to_id(name): return getattr(UnitTypeId, name)
        
        cmds1 = [[to_id(u), 1, pos1, 1] for u in spawns1]
        cmds2 = [[to_id(u), 1, pos2, 2] for u in spawns2]
        
        await self.bot1.client.debug_create_unit(cmds1)
        await self.bot1.client.debug_create_unit(cmds2)
        
        # Set reduced health for Pylons
        # (Will implement in Task 2.3 logic)
        
        await self.bot1.client.step()

    def _calculate_spawns(self, m, v):
        spawns = []
        affordable = [u for u in self.unit_pool if UNIT_COSTS[u]["minerals"] <= m and UNIT_COSTS[u]["vespene"] <= v]
        while affordable:
            u = random.choice(affordable)
            spawns.append(u)
            m -= UNIT_COSTS[u]["minerals"]; v -= UNIT_COSTS[u]["vespene"]
            affordable = [u for u in self.unit_pool if UNIT_COSTS[u]["minerals"] <= m and UNIT_COSTS[u]["vespene"] <= v]
        return spawns

    async def _refresh_unit_lists(self):
        """Updates the list of alive units for both players."""
        self.player_units[1] = [u.tag for u in self.bot1.units if not u.is_structure]
        self.player_units[2] = [u.tag for u in self.bot2.units if not u.is_structure]

    async def _get_joint_observation(self):
        """Returns observation for the current active unit."""
        active_bot = self.bot1 if self.active_player_id == 1 else self.bot2
        tags = self.player_units[self.active_player_id]
        
        if self.current_unit_index < len(tags):
            active_bot.active_unit_tag = tags[self.current_unit_index]
        else:
            active_bot.active_unit_tag = None
            
        return active_bot.get_observation()

    async def _async_step(self, action):
        """Executes one action for one unit, then advances the cycle."""
        self.steps_count += 1
        active_bot = self.bot1 if self.active_player_id == 1 else self.bot2
        tags = self.player_units[self.active_player_id]
        
        if tags and self.current_unit_index < len(tags):
            tag = tags[self.current_unit_index]
            await active_bot.apply_action(tag, action)
            
            # Flush actions
            if active_bot._actions:
                await active_bot.client.actions(active_bot._actions)
                active_bot._actions = []

        # Advance game physics
        await asyncio.gather(self.bot1.client.step(), self.bot2.client.step())
        
        # Update cycle
        self.current_unit_index += 1
        await self._refresh_unit_lists()
        
        if self.current_unit_index >= len(self.player_units[self.active_player_id]):
            self.current_unit_index = 0
            self.active_player_id = 2 if self.active_player_id == 1 else 1
            
        # Check termination
        done, reward = self._check_termination_and_reward()
        
        obs = await self._get_joint_observation()
        truncated = self.steps_count >= self.max_steps
        
        return obs, reward, done, truncated, {}

    def _check_termination_and_reward(self):
        """Placeholder for reward logic (Phase 3)."""
        # Basic check for pylon death
        p1_lost = self.bot1.structures(UnitTypeId.PYLON).amount == 0
        p2_lost = self.bot2.structures(UnitTypeId.PYLON).amount == 0
        
        done = p1_lost or p2_lost
        reward = 0.0 # To be implemented in Phase 3
        return done, reward
