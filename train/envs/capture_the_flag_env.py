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
from sc2.ids.unit_typeid import UnitTypeId

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
                 max_steps: int = 2000,
                 training_conf: Dict = None,
                 **kwargs):
        super().__init__(map_name=map_name, **kwargs)
        self.minerals = minerals
        self.vespene = vespene
        self.flag_distance_min = flag_distance_min
        self.unit_pool = unit_pool or ["ZERGLING", "ROACH", "MARINE"]
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_count = 0
        self.training_conf = training_conf or {}
        
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
        self.player_units = {1: [], 2: []}
        
        # Self-play model
        self.model = None

    def set_model(self, model):
        """Sets the model for self-play inference."""
        self.model = model

    async def _setup_game_session(self):
        from sc2.player import Bot
        from sc2.data import Race
        from sc2 import maps
        
        self.bot1 = CaptureTheFlagBot(player_id=1, training_conf=self.training_conf)
        self.bot2 = CaptureTheFlagBot(player_id=2, training_conf=self.training_conf)
        
        players = [Bot(Race.Zerg, self.bot1), Bot(Race.Zerg, self.bot2)]
        resolved_map = self._resolve_map(self.map_name)
        
        await self._manager.create_join_game(resolved_map, players, realtime=self.realtime)
        
        # Wire bots to clients and prepare them
        # BotAI._prepare_start(client, player_id, game_info, game_data, realtime)
        # We need game_info and game_data from the client
        async def prepare_bot(bot, client, player_id):
            game_info = await client.get_game_info()
            game_data = await client.get_game_data()
            bot._prepare_start(client, player_id, game_info, game_data, self.realtime)
            await bot.on_start()

        await asyncio.gather(
            prepare_bot(self.bot1, self._manager.clients[0], 1),
            prepare_bot(self.bot2, self._manager.clients[1], 2)
        )
        
        # Initial step to synchronize state
        await self._synchronized_step()
        
        await self._spawn_initial_units()
        await self._refresh_unit_lists()

    async def _synchronized_step(self):
        """Steps both clients simultaneously to advance the game."""
        await asyncio.gather(self.bot1.client.step(), self.bot2.client.step())

    async def _spawn_initial_units(self):
        from sc2.ids.unit_typeid import UnitTypeId
        w, h = self.bot1.game_info.playable_area.width, self.bot1.game_info.playable_area.height
        
        pos1 = (10, 10)
        pos2 = (w-10, h-10)
        
        # Spawn Pylons
        await self.bot1.client.debug_create_unit([[UnitTypeId.PYLON, 1, pos1, 1]])
        await self.bot1.client.debug_create_unit([[UnitTypeId.PYLON, 1, pos2, 2]])
        await self._synchronized_step()
        
        # Set reduced health for Pylons (10 HP + 10 Shield)
        p1_pylon = self.bot1.structures(UnitTypeId.PYLON).first
        p2_pylon = self.bot1.enemy_structures(UnitTypeId.PYLON).first
        
        if p1_pylon and p2_pylon:
            await self.bot1.client.debug_set_unit_value([p1_pylon.tag, p2_pylon.tag], 2, 10.0) # 2 = Health
            await self.bot1.client.debug_set_unit_value([p1_pylon.tag, p2_pylon.tag], 3, 10.0) # 3 = Shield
        
        # Resource spawning
        spawns1 = self._calculate_spawns(self.minerals, self.vespene)
        spawns2 = self._calculate_spawns(self.minerals, self.vespene)
        
        def to_id(name): return getattr(UnitTypeId, name)
        cmds1 = [[to_id(u), 1, pos1, 1] for u in spawns1]
        cmds2 = [[to_id(u), 1, pos2, 2] for u in spawns2]
        
        await self.bot1.client.debug_create_unit(cmds1)
        await self.bot1.client.debug_create_unit(cmds2)
        await self._synchronized_step()

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
        self.player_units[1] = [u.tag for u in self.bot1.units if not u.is_structure]
        self.player_units[2] = [u.tag for u in self.bot2.units if not u.is_structure]

    async def _get_joint_observation(self):
        active_bot = self.bot1 if self.active_player_id == 1 else self.bot2
        tags = self.player_units[self.active_player_id]
        if tags and self.current_unit_index < len(tags):
            active_bot.active_unit_tag = tags[self.current_unit_index]
        return active_bot.get_observation()

    async def _async_step(self, action_p_active):
        """
        Executes action for active unit, then uses self-play model 
        to pick actions for other units if needed before advancing.
        """
        self.steps_count += 1
        
        # 1. Apply action for the LEARNING unit
        active_bot = self.bot1 if self.active_player_id == 1 else self.bot2
        tags = self.player_units[self.active_player_id]
        
        if tags and self.current_unit_index < len(tags):
            await active_bot.apply_action(tags[self.current_unit_index], action_p_active)

        # 2. Self-Play: Pick actions for the OTHER player's current unit if we were in their cycle.
        # However, since we are SERIAL, we only do one unit action per step.
        # To make it 'real' self-play, we should ideally have both sides move 'often'.
        
        # Flush actions for BOTH bots
        for bot in [self.bot1, self.bot2]:
            if hasattr(bot, "actions") and bot.actions:
                await bot.client.actions(bot.actions)
                bot.actions = []
            elif hasattr(bot, "_actions") and bot._actions:
                await bot.client.actions(bot._actions)
                bot._actions = []

        # 3. Physics Step
        await asyncio.gather(self.bot1.client.step(), self.bot2.client.step())
        
        # 4. Calculate Reward for the LEARNING agent
        # The bot.calculate_reward() returns the reward from that bot's perspective.
        reward = active_bot.calculate_reward()
        
        # 5. Advance Cycle
        self.current_unit_index += 1
        await self._refresh_unit_lists()
        
        # Check if we need to switch players
        if self.current_unit_index >= len(self.player_units[self.active_player_id]):
            self.current_unit_index = 0
            self.active_player_id = 2 if self.active_player_id == 1 else 1
            # Refresh again for new player
            await self._refresh_unit_lists()
            
        # 6. Check Termination
        p1_lost = self.bot1.structures(UnitTypeId.PYLON).amount == 0
        p2_lost = self.bot2.structures(UnitTypeId.PYLON).amount == 0
        
        win_reward = self.training_conf.get("rewards", {}).get("win_reward", 100.0)
        
        done = False
        if p1_lost:
            done = True
            if self.active_player_id == 2: reward += win_reward # P1 lost, P2 wins
            else: reward -= win_reward
        elif p2_lost:
            done = True
            if self.active_player_id == 1: reward += win_reward # P2 lost, P1 wins
            else: reward -= win_reward
            
        truncated = self.steps_count >= self.max_steps
        obs = await self._get_joint_observation()
        
        return obs, reward, done, truncated, {}

    def action_masks(self):
        """Returns valid actions for the active unit."""
        # For now, all spatial actions are valid.
        return np.ones(self.action_space.n, dtype=bool)

    def step(self, action):
        # Synchronous wrapper as per MultiClientEnv
        return super().step(action)