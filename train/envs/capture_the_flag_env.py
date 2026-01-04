import gymnasium as gym
import asyncio
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import random
from gymnasium import spaces
from loguru import logger
import json
import os

from train.envs.multi_client_env import MultiClientEnv
from src.utils.layer_constants import LayerConstants
from src.agent.ctf_bot import CaptureTheFlagBot
from sc2.ids.unit_typeid import UnitTypeId
from src.utils.spawning_utils import SpawningUtils

class CaptureTheFlagEnv(MultiClientEnv):
    def __init__(self, 
                 mineral_budget: int = 500, 
                 vespene_budget: int = 0, 
                 flag_distance_min: float = 30.0,
                 unit_pool: List[str] = None,
                 grid_size: int = 128,
                 map_name: str = "BotMicroArena_6",
                 max_steps: int = 2000,
                 training_conf: Dict = None,
                 **kwargs):
        super().__init__(map_name=map_name, **kwargs)
        self.training_conf = training_conf or {}
        
        # Robustly load parameters from training_conf (handling flattened vs nested)
        env_conf = self.training_conf.get("env", {})
        self.mineral_budget = self.training_conf.get("mineral_budget", 
                                 env_conf.get("mineral_budget", mineral_budget))
        self.vespene_budget = self.training_conf.get("vespene_budget", 
                                 env_conf.get("vespene_budget", vespene_budget))
        self.unit_pool = self.training_conf.get("unit_pool", 
                           env_conf.get("unit_pool", unit_pool or ["ZERGLING", "ROACH", "MARINE"]))
        
        # Load unit data for costs
        self.unit_data = self._load_unit_data()
        
        self.flag_distance_min = flag_distance_min
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_count = 0
        
        # Strictly Additive Observation Layers
        # 1. Base layers from Level 0-10 (ensuring parity with the 9-layer model)
        self.base_layers = [
            "allied_unit_density",
            "enemy_unit_density",
            "terrain_pathable",
            "command_objective",
            "command_assignment",
            "allied_g_dps_base",
            "enemy_g_dps_base",
            "allied_health",
            "enemy_health"
        ]
        
        # 2. CTF Specific layers (as defined in config/capture_the_flag.yaml)
        self.ctf_layers = self.training_conf.get("env", {}).get("active_layers", [
            "allied_unit_biological",
            "allied_unit_mechanical",
            "allied_unit_armored",
            "allied_unit_massive",
            "enemy_unit_biological",
            "enemy_unit_mechanical",
            "enemy_unit_armored",
            "enemy_unit_massive",
            "allied_g_dps_light",
            "allied_g_dps_armor",
            "allied_g_dps_massive",
            "enemy_g_dps_light",
            "enemy_g_dps_armor",
            "enemy_g_dps_massive"
        ])
        
        # Total combined active layers
        self.active_layers = self.base_layers + self.ctf_layers
        
        # Inject combined layers into training_conf for the bots/agent system
        self.training_conf["active_layers"] = self.active_layers
        
        num_layers = len(self.active_layers)
        self.observation_space = spaces.Box(low=0, high=255, shape=(num_layers, 128, 128), dtype=np.uint8)
        
        # Action Space Restoration (Move + Attack grids)
        self.action_space = spaces.Discrete(2 * grid_size * grid_size)
        
        self.bot1: Optional[CaptureTheFlagBot] = None
        self.bot2: Optional[CaptureTheFlagBot] = None
        
        self.active_player_id = 1
        self.current_unit_index = 0
        self.player_units = {1: [], 2: []}
        
        # Self-play model
        self.model = None

    def _load_unit_data(self) -> Dict[str, Any]:
        path = os.path.join(os.getcwd(), "data", "unit_data.json")
        if not os.path.exists(path):
            logger.warning(f"unit_data.json not found at {path}. Using empty dict.")
            return {}
        with open(path, "r") as f:
            return json.load(f)

    def set_model(self, model):
        """Sets the model for self-play inference."""
        self.model = model

    async def _refresh_bot_states(self):
        """Fetches latest observation and updates bot internal state."""
        from sc2.game_state import GameState
        
        async def refresh(bot, client):
            obs = await client.observation()
            state = GameState(obs.observation)
            # BotAIInternal._prepare_step(state, proto_game_info)
            # bot._proto_game_info was stored during prepare_bot
            bot._prepare_step(state, bot._proto_game_info)

        await asyncio.gather(
            refresh(self.bot1, self.bot1.client),
            refresh(self.bot2, self.bot2.client)
        )

    async def _setup_game_session(self):
        from sc2.player import Bot
        from sc2.data import Race
        from sc2 import maps
        from s2clientprotocol import sc2api_pb2 as sc_pb
        
        self.bot1 = CaptureTheFlagBot(player_id=1, training_conf=self.training_conf)
        self.bot2 = CaptureTheFlagBot(player_id=2, training_conf=self.training_conf)
        
        players = [Bot(Race.Zerg, self.bot1), Bot(Race.Zerg, self.bot2)]
        resolved_map = self._resolve_map(self.map_name)
        
        self._log("Starting multi-client game session...")
        await self._manager.create_join_game(resolved_map, players, realtime=self.realtime)
        self._log("Game session established. Preparing bots...")
        
        # Wire bots to clients and prepare them
        async def prepare_bot(bot, client, player_id):
            self._log(f"Preparing Bot {player_id}...")
            # We need both the wrapper and the raw proto
            raw_game_info = await client._execute(game_info=sc_pb.RequestGameInfo())
            from sc2.game_info import GameInfo
            game_info = GameInfo(raw_game_info.game_info)
            game_data = await client.get_game_data()
            
            bot._prepare_start(client, player_id, game_info, game_data, self.realtime)
            bot._proto_game_info = raw_game_info # Store for _prepare_step
            await bot.on_start()
            self._log(f"Bot {player_id} ready.")

        await asyncio.gather(
            prepare_bot(self.bot1, self._manager.clients[0], 1),
            prepare_bot(self.bot2, self._manager.clients[1], 2)
        )
        
        # Initial step to synchronize state
        self._log("Synchronizing initial state...")
        await self._synchronized_step()
        await self._refresh_bot_states()
        
        self._log("Spawning units...")
        await self._spawn_initial_units()
        await self._refresh_unit_lists()
        self._log("Setup complete.")

    async def _synchronized_step(self):
        """Steps both clients simultaneously to advance the game."""
        await asyncio.gather(self.bot1.client.step(), self.bot2.client.step())

    async def _spawn_initial_units(self):
        from sc2.ids.unit_typeid import UnitTypeId
        from sc2.position import Point2
        
        # 1. Clear Map (Optional for MultiClient if fresh session, but good for stability)
        if self.bot1.all_units.exists:
            await self.bot1.client.debug_kill_unit(self.bot1.all_units.tags)
            await self._synchronized_step()
            await self._refresh_bot_states()

        w, h = self.bot1.game_info.playable_area.width, self.bot1.game_info.playable_area.height
        
        # 2. Generate Randomized Positions
        requested_dist = self.training_conf.get("env", {}).get("flag_distance_min", self.flag_distance_min)
        safe_dist = SpawningUtils.calculate_safe_distance(w, h, requested_dist)
        
        pos1, pos2 = SpawningUtils.get_random_pylon_positions(w, h, safe_dist)
        self._log(f"Spawning Pylons at {pos1} and {pos2} (Distance: {pos1.distance_to(pos2):.2f}, SafeTarget: {safe_dist:.2f})")
        
        # 3. Spawn Pylons
        await self.bot1.client.debug_create_unit([[UnitTypeId.PYLON, 1, pos1, 1]])
        await self.bot1.client.debug_create_unit([[UnitTypeId.PYLON, 1, pos2, 2]])
        
        # 4. Resource spawning
        spawns1 = self._calculate_spawns(self.mineral_budget, self.vespene_budget)
        spawns2 = self._calculate_spawns(self.mineral_budget, self.vespene_budget)
        
        def to_id(name): return getattr(UnitTypeId, name)
        cmds1 = [[to_id(u), 1, pos1, 1] for u in spawns1]
        cmds2 = [[to_id(u), 1, pos2, 2] for u in spawns2]
        
        await self.bot1.client.debug_create_unit(cmds1)
        await self.bot1.client.debug_create_unit(cmds2)
        
        # Advance physics to materialize units
        await self._synchronized_step()
        await self._refresh_bot_states()
        
        # 5. Set reduced health for Pylons (10 HP + 10 Shield)
        p1_pylon = self.bot1.structures(UnitTypeId.PYLON).first
        p2_pylon = self.bot1.enemy_structures(UnitTypeId.PYLON).first
        
        if p1_pylon and p2_pylon:
            await self.bot1.client.debug_set_unit_value([p1_pylon.tag, p2_pylon.tag], 2, 10.0) # 2 = Health
            await self.bot1.client.debug_set_unit_value([p1_pylon.tag, p2_pylon.tag], 3, 10.0) # 3 = Shield
            
        # Final sync
        await self._synchronized_step()
        await self._refresh_bot_states()

    def _calculate_spawns(self, m, v):
        spawns = []
        
        def can_afford(u):
            if u not in self.unit_data:
                logger.error(f"Unit {u} not found in unit_data.json. Skipping.")
                return False
            cost = self.unit_data[u]["cost"]
            return cost["minerals"] <= m and cost["vespene"] <= v

        affordable = [u for u in self.unit_pool if can_afford(u)]
        while affordable:
            u = random.choice(affordable)
            spawns.append(u)
            cost = self.unit_data[u]["cost"]
            m -= cost["minerals"]
            v -= cost["vespene"]
            affordable = [u for u in self.unit_pool if can_afford(u)]
            
        if spawns:
            self._log(f"Spawned army: {len(spawns)} units. Remaining budget: {m}m, {v}v")
        return spawns

    async def _refresh_unit_lists(self):
        self.player_units[1] = [u.tag for u in self.bot1.units if not u.is_structure]
        self.player_units[2] = [u.tag for u in self.bot2.units if not u.is_structure]

    @property
    def observation(self):
        """Returns the current observation for the active learning agent."""
        active_bot = self.bot1 if self.active_player_id == 1 else self.bot2
        return active_bot.get_observation()

    async def _get_joint_observation(self):
        active_bot = self.bot1 if self.active_player_id == 1 else self.bot2
        tags = self.player_units[self.active_player_id]
        if tags and self.current_unit_index < len(tags):
            active_bot.active_unit_tag = tags[self.current_unit_index]
        obs = active_bot.get_observation()
        # logger.debug(f"CTF Env: Returning observation for player {self.active_player_id}, unit {self.current_unit_index}. Shape: {obs.shape}")
        return obs

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
        
        # Flush actions for BOTH bots
        for bot in [self.bot1, self.bot2]:
            if hasattr(bot, "actions") and bot.actions:
                await bot.client.actions(bot.actions)
                bot.actions = []
            elif hasattr(bot, "_actions") and bot._actions:
                await bot.client.actions(bot._actions)
                bot._actions = []

        # 3. Physics Step
        try:
            await asyncio.gather(self.bot1.client.step(), self.bot2.client.step())
        except Exception as e:
            # Check if game ended
            p1_lost = self.bot1.structures(UnitTypeId.PYLON).amount == 0
            p2_lost = self.bot2.structures(UnitTypeId.PYLON).amount == 0
            if p1_lost or p2_lost:
                self._log("Game ended during step.")
            else:
                self._log(f"Error during synchronized step: {e}")
                raise e

        await self._refresh_bot_states()
        
        # 4. Calculate Reward for the LEARNING agent
        reward = active_bot.calculate_reward()
        
        # 5. Advance Cycle
        self.current_unit_index += 1
        await self._refresh_unit_lists()
        
        # Check if we need to switch players
        if self.current_unit_index >= len(self.player_units[self.active_player_id]):
            self.current_unit_index = 0
            self.active_player_id = 2 if self.active_player_id == 1 else 1
            await self._refresh_unit_lists()
            
        # 6. Check Termination
        # Refresh structures explicitly if possible
        p1_pylons = self.bot1.structures(UnitTypeId.PYLON)
        p2_pylons = self.bot2.structures(UnitTypeId.PYLON)
        
        p1_lost = p1_pylons.amount == 0
        p2_lost = p2_pylons.amount == 0
        
        win_reward = self.training_conf.get("rewards", {}).get("win_reward", 100.0)
        
        done = False
        if p1_lost:
            done = True
            self._log("Player 1 lost their Pylon.")
            if self.active_player_id == 2: reward += win_reward 
            else: reward -= win_reward
        elif p2_lost:
            done = True
            self._log("Player 2 lost their Pylon.")
            if self.active_player_id == 1: reward += win_reward 
            else: reward -= win_reward
            
        truncated = self.steps_count >= self.max_steps
        
        # IMPORTANT: Fetch observation AFTER state updates/player switches
        obs = await self._get_joint_observation()
        
        return obs, reward, done, truncated, {}

    def action_masks(self):
        """Returns valid actions for the active unit."""
        # Restored parity with Level 7: Move (first half) and Attack (second half)
        # All actions are currently valid for CTF.
        return np.ones(self.action_space.n, dtype=bool)

    def _determine_winner(self) -> Optional[int]:
        """
        Determines the winner based on remaining Pylons.
        Returns 1 if Player 1 has a Pylon, 2 if Player 2 has a Pylon, or None if neither.
        """
        # Defensive checks for bot objects
        if not self.bot1 or not self.bot2:
            return None
            
        try:
            p1_has_pylon = self.bot1.structures(UnitTypeId.PYLON).exists
            p2_has_pylon = self.bot2.structures(UnitTypeId.PYLON).exists
            
            if p1_has_pylon and not p2_has_pylon:
                return 1
            if p2_has_pylon and not p1_has_pylon:
                return 2
        except Exception:
            # SC2 objects might be invalid if connection is lost
            pass
        return None

    def _get_terminal_info(self, error_msg: str) -> Tuple[float, Dict]:
        """
        Calculates final rewards and terminal info when game ends.
        """
        winner = self._determine_winner()
        win_reward = self.training_conf.get("rewards", {}).get("win_reward", 100.0)
        
        final_reward = 0.0
        info = {"winner": winner, "error": error_msg}
        
        if winner is not None:
            if winner == self.active_player_id:
                final_reward = win_reward
            else:
                final_reward = -win_reward
                
        return final_reward, info

    def step(self, action):
        """
        Overriding MultiClientEnv.step() because SB3 provides a single action
        for the learning agent, not a list of actions for both players.
        """
        from sc2.protocol import ProtocolError
        
        if self.game_ended:
             return self._last_obs, 0.0, True, False, {"game_ended": True, "already_ended": True}

        info = {}
        try:
            # self._log(f"step() called with action: {action}")
            results = self._loop.run_until_complete(self._async_step(action))
            obs, reward, terminated, truncated, info = results
            
            self._last_obs = obs
            if terminated or truncated:
                self.game_ended = True
                
            # self._log("step() complete.")
            return results
        except ProtocolError as e:
            if "Game has already ended" in str(e):
                self._log("ProtocolError: Game ended during step. Transitioning to terminal state.")
                self.game_ended = True
                
                # Terminal hook for subclasses to inject rewards/info
                reward, terminal_info = self._get_terminal_info(str(e))
                # Base info might be empty here
                info = {"game_ended": True}
                info.update(terminal_info)
                
                return self._last_obs, reward, True, False, info
            
            self._log(f"CRITICAL: ProtocolError during step: {e}")
            self.close()
            raise e
        except Exception as e:
            self._log(f"CRITICAL: Error during step: {e}")
            import traceback
            self._log(traceback.format_exc())
            self.close()
            raise e