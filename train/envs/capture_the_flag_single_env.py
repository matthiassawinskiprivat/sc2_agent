
import gymnasium as gym
import asyncio
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import random
from gymnasium import spaces
from loguru import logger
import logging

from train.envs.single_client_env import SingleClientEnv
from src.agent.ctf_bot import CaptureTheFlagBot
from sc2.ids.unit_typeid import UnitTypeId

# Standard SC2 Unit Costs (Simplified)
UNIT_COSTS = {
    "ZERGLING": {"minerals": 25, "vespene": 0, "type_id": 105},
    "ROACH": {"minerals": 75, "vespene": 25, "type_id": 110},
    "MARINE": {"minerals": 50, "vespene": 0, "type_id": 48}
}

class CaptureTheFlagSingleEnv(SingleClientEnv):
    def __init__(self, 
                 minerals: int = 500, 
                 vespene: int = 0, 
                 unit_pool: List[str] = None,
                 grid_size: int = 128,
                 map_name: str = "BotMicroArena_6",
                 max_steps: int = 2000,
                 training_conf: Dict = None,
                 **kwargs):
        super().__init__(map_name=map_name, **kwargs)
        self.minerals = minerals
        self.vespene = vespene
        self.unit_pool = unit_pool or ["ZERGLING", "ROACH", "MARINE"]
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_count = 0
        self.training_conf = training_conf or {}
        self.client_count = 1 # God-Mode uses only 1 connection
        
        # Observation Layers
        self.base_layers = [
            "allied_unit_density", "enemy_unit_density", "terrain_pathable",
            "command_objective", "command_assignment", "allied_g_dps_base",
            "enemy_g_dps_base", "allied_health", "enemy_health"
        ]
        self.ctf_layers = self.training_conf.get("env", {}).get("active_layers", [
            "allied_unit_biological", "allied_unit_mechanical", "allied_unit_armored", "allied_unit_massive",
            "enemy_unit_biological", "enemy_unit_mechanical", "enemy_unit_armored", "enemy_unit_massive",
            "allied_g_dps_light", "allied_g_dps_armor", "allied_g_dps_massive",
            "enemy_g_dps_light", "enemy_g_dps_armor", "enemy_g_dps_massive"
        ])
        self.active_layers = self.base_layers + self.ctf_layers
        self.training_conf["active_layers"] = self.active_layers
        
        num_layers = len(self.active_layers)
        self.observation_space = spaces.Box(low=0, high=255, shape=(num_layers, 128, 128), dtype=np.uint8)
        self.action_space = spaces.Discrete(2 * grid_size * grid_size)
        
        self.bot: Optional[CaptureTheFlagBot] = None
        self.active_player_id = 1
        self.current_unit_index = 0
        self.player_units = {1: [], 2: []}
        self.model = None

    def set_model(self, model):
        self.model = model

    async def _setup_game_session(self):
        from sc2.player import Bot
        from sc2.data import Race
        from s2clientprotocol import sc2api_pb2 as sc_pb
        
        self.bot = CaptureTheFlagBot(player_id=1, training_conf=self.training_conf)
        # In single-process mode, we only join as the first player
        players = [Bot(Race.Zerg, self.bot), Bot(Race.Zerg, None)]
        resolved_map = self._resolve_map(self.map_name)
        
        self._log("Starting single-process God-Mode game session...")
        await self._manager.create_join_game(resolved_map, players, realtime=self.realtime)
        
        # Prepare bot
        client = self._manager.clients[0]
        raw_game_info = await client._execute(game_info=sc_pb.RequestGameInfo())
        from sc2.game_info import GameInfo
        game_info = GameInfo(raw_game_info.game_info)
        game_data = await client.get_game_data()
        
        self.bot._prepare_start(client, 1, game_info, game_data, self.realtime)
        self.bot._proto_game_info = raw_game_info
        await self.bot.on_start()
        
        # God-Mode: Enable full visibility and enemy control
        await client.debug_show_map()
        await client.debug_control_enemy()
        
        # Initial step and spawn
        await client.step()
        await self._refresh_bot_state()
        await self._spawn_initial_units()
        await self._refresh_unit_lists()
        self._log("Setup complete.")

    async def _refresh_bot_state(self):
        from sc2.game_state import GameState
        obs = await self.bot.client.observation()
        state = GameState(obs.observation)
        self.bot._prepare_step(state, self.bot._proto_game_info)

    async def _spawn_initial_units(self):
        from sc2.ids.unit_typeid import UnitTypeId
        from sc2.position import Point2
        w, h = self.bot.game_info.playable_area.width, self.bot.game_info.playable_area.height
        
        pos1 = Point2((10, 10))
        pos2 = Point2((w-10, h-10))
        
        # Spawn Pylons for P1 and P2
        await self.bot.client.debug_create_unit([[UnitTypeId.PYLON, 1, pos1, 1]])
        await self.bot.client.debug_create_unit([[UnitTypeId.PYLON, 1, pos2, 2]])
        
        # Resource spawning
        spawns1 = self._calculate_spawns(self.minerals, self.vespene)
        spawns2 = self._calculate_spawns(self.minerals, self.vespene)
        
        def to_id(name): return getattr(UnitTypeId, name)
        cmds1 = [[to_id(u), 1, pos1, 1] for u in spawns1]
        cmds2 = [[to_id(u), 1, pos2, 2] for u in spawns2]
        
        await self.bot.client.debug_create_unit(cmds1)
        await self.bot.client.debug_create_unit(cmds2)
        
        await self.bot.client.step()
        await self._refresh_bot_state()
        
        # Set reduced health for Pylons
        p1_pylon = self.bot.structures(UnitTypeId.PYLON).first
        # In God-Mode, P2 units are in enemy_structures
        p2_pylon = self.bot.enemy_structures(UnitTypeId.PYLON).first
        
        if p1_pylon and p2_pylon:
            await self.bot.client.debug_set_unit_value([p1_pylon.tag, p2_pylon.tag], 2, 10.0)
            await self.bot.client.debug_set_unit_value([p1_pylon.tag, p2_pylon.tag], 3, 10.0)
        
        await self.bot.client.step()
        await self._refresh_bot_state()

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
        # Refresh all units from the bot's God-Mode perspective
        all_units = self.bot.all_units
        # Filter for alive and not structure
        self.player_units[1] = [u.tag for u in all_units if u.owner_id == 1 and not u.is_structure]
        self.player_units[2] = [u.tag for u in all_units if u.owner_id == 2 and not u.is_structure]

    async def _get_joint_observation(self):
        tags = self.player_units[self.active_player_id]
        if tags and self.current_unit_index < len(tags):
            self.bot.active_unit_tag = tags[self.current_unit_index]
        return self.bot.get_observation(perspective_player_id=self.active_player_id)

    async def _apply_action(self, action):
        tags = self.player_units[self.active_player_id]
        if tags and self.current_unit_index < len(tags):
            # In God-Mode with debug_control_enemy(), we can command enemy tags
            await self.bot.apply_action(tags[self.current_unit_index], action)

    def reset(self, seed=None, options=None):
        self._log("Environment reset started...")
        self.steps_count = 0
        self.current_unit_index = 0
        self.active_player_id = 1
        return super().reset(seed=seed, options=options)

    async def _async_step(self, action_p_active):
        self.steps_count += 1
        if self.steps_count % 100 == 0:
            self._log(f"Step {self.steps_count} in progress...")
        
        # Save acting player ID for terminal reward attribution
        acting_player_id = self.active_player_id
        
        await self._apply_action(action_p_active)
        
        # Flush actions
        if hasattr(self.bot, "actions") and self.bot.actions:
            await self.bot.client.actions(self.bot.actions)
            self.bot.actions = []

        await self.bot.client.step()
        await self._refresh_bot_state()
        
        # 1. Calculate step reward for the acting player
        reward = self.bot.calculate_reward(perspective_player_id=acting_player_id)
        
        # 2. Check for Win/Loss (Terminal)
        p1_lost = self.bot.structures(UnitTypeId.PYLON).amount == 0
        p2_lost = self.bot.enemy_structures(UnitTypeId.PYLON).amount == 0
        
        done = False
        truncated = False
        r_conf = self.training_conf.get("rewards", {})
        win_reward = r_conf.get("win_reward", 100.0)
        loss_penalty = r_conf.get("loss_penalty", -100.0)
        draw_penalty = r_conf.get("draw_penalty", -10.0)

        if p1_lost:
            done = True
            self._log(f"Termination: Player 1 Pylon destroyed. Winner: Player 2.")
            if acting_player_id == 2: reward += win_reward
            else: reward += loss_penalty
        elif p2_lost:
            done = True
            self._log(f"Termination: Player 2 Pylon destroyed. Winner: Player 1.")
            if acting_player_id == 1: reward += win_reward
            else: reward += loss_penalty
        
        # 3. Check for Draw (Mutual Destruction)
        await self._refresh_unit_lists()
        p1_units_count = len(self.player_units[1])
        p2_units_count = len(self.player_units[2])
        
        if not done and p1_units_count == 0 and p2_units_count == 0:
            done = True
            reward += draw_penalty
            self._log(f"Termination: Mutual Destruction (Draw).")

        # 4. Check for Timeout
        if not done and self.steps_count >= self.max_steps:
            truncated = True
            reward += draw_penalty
            self._log(f"Termination: Timeout reached ({self.max_steps} steps). Draw.")

        # 5. Handle Turn Cycling (only if game not done)
        if not done and not truncated:
            self.current_unit_index += 1
            if self.current_unit_index >= len(self.player_units[self.active_player_id]):
                self.current_unit_index = 0
                self.active_player_id = 2 if self.active_player_id == 1 else 1
                await self._refresh_unit_lists()
            
        obs = await self._get_joint_observation()
        return obs, reward, done, truncated, {}

    def action_masks(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=bool)
