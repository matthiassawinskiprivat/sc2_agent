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
        
        # Dynamic Observation Layers from Curriculum
        self.active_layers = self.config_manager.get_cumulative_active_layers(self.level_name)
        
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
        
        # Asymmetric Self-Play Configuration
        # Player 1 is ALWAYS the trainee
        self.bot1.is_trainee = True
        self.bot1.model = None
        
        # Player 2 is ALWAYS the automated opponent
        self.bot2.is_trainee = False
        self.bot2.model = self.model # Snapshot for the current episode
        
        players = [Bot(Race.Zerg, self.bot1), Bot(Race.Zerg, self.bot2)]
        resolved_map = self._resolve_map(self.map_name)
        
        await self._manager.create_join_game(resolved_map, players, realtime=self.realtime)
        
        # Wire bots to clients and prepare them
        async def prepare_bot(bot, client, player_id):
            # We need both the wrapper and the raw proto
            raw_game_info = await client._execute(game_info=sc_pb.RequestGameInfo())
            from sc2.game_info import GameInfo
            game_info = GameInfo(raw_game_info.game_info)
            game_data = await client.get_game_data()
            
            bot._prepare_start(client, player_id, game_info, game_data, self.realtime)
            bot._proto_game_info = raw_game_info # Store for _prepare_step
            
            # Enable full visibility for self-play fairness and Pylon objective tracking
            await client.debug_show_map()
            
            await bot.on_start()

        await asyncio.gather(
            prepare_bot(self.bot1, self._manager.clients[0], 1),
            prepare_bot(self.bot2, self._manager.clients[1], 2)
        )
        
        # Initial step to synchronize state
        await self._synchronized_step()
        await self._refresh_bot_states()
        
        await self._spawn_initial_units()
        await self._refresh_unit_lists()

    async def _synchronized_step(self):
        """Steps both clients simultaneously to advance the game."""
        await asyncio.gather(self.bot1.client.step(), self.bot2.client.step())

    async def _spawn_initial_units(self):
        from sc2.ids.unit_typeid import UnitTypeId
        from sc2.position import Point2
        import math
        
        # 1. Clear Map (Optional for MultiClient if fresh session, but good for stability)
        if self.bot1.all_units.exists:
            await self.bot1.client.debug_kill_unit(self.bot1.all_units.tags)
            await self._synchronized_step()
            await self._refresh_bot_states()

        grid = self.bot1.game_info.pathing_grid
        
        def is_pathable(pos: Point2):
            gx, gy = int(pos.x), int(pos.y)
            if 0 <= gx < grid.width and 0 <= gy < grid.height:
                return grid[(gx, gy)] != 0
            return False

        def get_valid_pos(base_pos: Point2, max_dist=10.0):
            if is_pathable(base_pos): return base_pos
            for d in range(1, int(max_dist)):
                for angle in range(0, 360, 45):
                    rad = math.radians(angle)
                    p = Point2((base_pos.x + d * math.cos(rad), base_pos.y + d * math.sin(rad)))
                    if is_pathable(p): return p
            return base_pos

        w, h = self.bot1.game_info.playable_area.width, self.bot1.game_info.playable_area.height
        offset_x, offset_y = self.bot1.game_info.playable_area.x, self.bot1.game_info.playable_area.y
        
        # 2. Generate Randomized Positions
        requested_dist = self.training_conf.get("env", {}).get("flag_distance_min", self.flag_distance_min)
        safe_dist = SpawningUtils.calculate_safe_distance(w, h, requested_dist)
        
        rel_pos1, rel_pos2 = SpawningUtils.get_random_pylon_positions(w, h, safe_dist, margin=1.0)
        
        # Convert to absolute and ensure pathable
        pos1 = get_valid_pos(Point2((rel_pos1.x + offset_x, rel_pos1.y + offset_y)))
        pos2 = get_valid_pos(Point2((rel_pos2.x + offset_x, rel_pos2.y + offset_y)))
        
        # 3. Spawn Pylons
        await asyncio.gather(
            self.bot1.client.debug_create_unit([[UnitTypeId.PYLON, 1, pos1, 1]]),
            self.bot2.client.debug_create_unit([[UnitTypeId.PYLON, 1, pos2, 2]])
        )
        
        # 4. Resource spawning
        spawns1 = self._calculate_spawns(self.mineral_budget, self.vespene_budget)
        spawns2 = self._calculate_spawns(self.mineral_budget, self.vespene_budget)
        
        def to_id(name): return getattr(UnitTypeId, name)
        
        def get_offset_pos(base_pos, index, total):
            angle = (2 * math.pi * index) / max(total, 1)
            dist = 3.0
            p = Point2((base_pos.x + dist * math.cos(angle), base_pos.y + dist * math.sin(angle)))
            return get_valid_pos(p)

        cmds1 = [[to_id(u), 1, get_offset_pos(pos1, i, len(spawns1)), 1] for i, u in enumerate(spawns1)]
        cmds2 = [[to_id(u), 1, get_offset_pos(pos2, i, len(spawns2)), 2] for i, u in enumerate(spawns2)]
        
        await asyncio.gather(
            self.bot1.client.debug_create_unit(cmds1),
            self.bot2.client.debug_create_unit(cmds2)
        )
        
        # Advance physics multiple steps to ensure all units materialize
        for _ in range(3):
            await self._synchronized_step()
            await self._refresh_bot_states()
        
        # 5. Set reduced health for Pylons (10 HP + 10 Shield)
        p1_pylons = self.bot1.structures(UnitTypeId.PYLON)
        p2_pylons = self.bot2.structures(UnitTypeId.PYLON)
        
        if p1_pylons.exists and p2_pylons.exists:
            # Sequential calls to avoid 'Concurrent call to receive()' in aiohttp
            tags = [p1_pylons.first.tag, p2_pylons.first.tag]
            await self.bot1.client.debug_set_unit_value(tags, 2, 10.0) # 2 = Health
            await self.bot1.client.debug_set_unit_value(tags, 3, 10.0) # 3 = Shield
            
        # Final sync
        await self._synchronized_step()
        await self._refresh_bot_states()
            
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
            
        return spawns

    async def _refresh_unit_lists(self):
        ignored_types = {UnitTypeId.LARVA, UnitTypeId.EGG, UnitTypeId.OVERLORD, UnitTypeId.PYLON}
        
        # We refresh the tags based on OWNED units for each bot
        for p_id, bot in [(1, self.bot1), (2, self.bot2)]:
            self.player_units[p_id] = [
                u.tag for u in bot.units 
                if not u.is_structure and u.unit_type not in ignored_types
            ]

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
        Executes action for active unit, then automatically handles 
        any non-trainee units before returning.
        """
        self.steps_count += 1
        total_reward = 0.0
        
        # --- 1. LEARNING UNIT STEP ---
        active_bot = self.bot1 if self.active_player_id == 1 else self.bot2
        
        # Safety check: if we somehow end up in _async_step with a non-trainee, 
        # it means the loop logic below failed.
        if not active_bot.is_trainee:
            logger.warning(f"Asymmetric Step: _async_step called for non-trainee player {self.active_player_id}. Automating.")
            action_p_active = active_bot.predict_action()

        tags = self.player_units[self.active_player_id]
        if tags and self.current_unit_index < len(tags):
            await active_bot.apply_action(tags[self.current_unit_index], action_p_active)

        # Flush actions and physics step
        await self._flush_and_step_physics()
        await self._refresh_bot_states()
        
        # Collect reward for the trainee (always bot1 in our current config)
        total_reward += self.bot1.calculate_reward()
        
        # Advance cycle
        await self._advance_unit_cycle()
        
        # --- 2. AUTOMATED TURNS (Asymmetric Loop) ---
        # Continue automating units until we reach a trainee OR the game ends
        while True:
            # Check Termination after the previous physics step
            done, truncated, reward_delta = self._check_termination()
            total_reward += reward_delta
            if done or truncated:
                obs = await self._get_joint_observation()
                return obs, total_reward, done, truncated, {}

            active_bot = self.bot1 if self.active_player_id == 1 else self.bot2
            if active_bot.is_trainee:
                # We reached a trainee unit, time to return control to the trainer
                break
                
            # Automate non-trainee unit
            tags = self.player_units[self.active_player_id]
            if tags and self.current_unit_index < len(tags):
                # Automated inference for opponent
                auto_action = active_bot.predict_action()
                await active_bot.apply_action(tags[self.current_unit_index], auto_action)
                
            # Physics step for the automated unit
            await self._flush_and_step_physics()
            await self._refresh_bot_states()
            
            # Collect reward for trainee
            total_reward += self.bot1.calculate_reward()
            
            # Advance cycle
            await self._advance_unit_cycle()

        # Final observation for the trainee
        obs = await self._get_joint_observation()
        return obs, total_reward, False, False, {}

    async def _flush_and_step_physics(self):
        """Flushes bot actions and advances SC2 game state."""
        for bot in [self.bot1, self.bot2]:
            if hasattr(bot, "actions") and bot.actions:
                await bot.client.actions(bot.actions)
                bot.actions = []
            elif hasattr(bot, "_actions") and bot._actions:
                await bot.client.actions(bot._actions)
                bot._actions = []

        try:
            await self._synchronized_step()
        except Exception as e:
            # Check if game ended
            p1_lost = self.bot1.structures(UnitTypeId.PYLON).amount == 0
            p2_lost = self.bot2.structures(UnitTypeId.PYLON).amount == 0
            if not (p1_lost or p2_lost):
                raise e

    async def _advance_unit_cycle(self):
        """Moves to the next unit and switches players if necessary."""
        self.current_unit_index += 1
        await self._refresh_unit_lists()
        
        if self.current_unit_index >= len(self.player_units[self.active_player_id]):
            self.current_unit_index = 0
            self.active_player_id = 2 if self.active_player_id == 1 else 1
            await self._refresh_unit_lists()

    def _has_combat_units(self, bot: CaptureTheFlagBot) -> bool:
        """
        Robustly checks if a bot has any combat-capable mobile units left.
        Uses all_units to avoid filtered lists and excludes structures and non-combat units.
        """
        if not hasattr(bot, "all_units") or not bot.all_units:
            return False
            
        ignored_types = {
            UnitTypeId.LARVA, 
            UnitTypeId.EGG, 
            UnitTypeId.OVERLORD, 
            UnitTypeId.PYLON,
            UnitTypeId.PROBE,
            UnitTypeId.SCV,
            UnitTypeId.DRONE
        }
        
        # Filter all_units: must not be a structure and must not be in ignored types
        # Also ensure unit is still alive (health > 0) and owned by the bot
        # and NOT a 'memory' unit (artifact from previous frames)
        combat_units = [
            u for u in bot.all_units
            if u.owner_id == bot.player_id
            and not u.is_structure 
            and u.type_id not in ignored_types
            and u.health > 0
            and not getattr(u, "is_memory", False)
        ]
        
        return len(combat_units) > 0

    def _check_termination(self) -> Tuple[bool, bool, float]:
        """Checks if the game should end and returns (done, truncated, reward_delta)."""
        p1_pylons = self.bot1.structures(UnitTypeId.PYLON)
        p2_pylons = self.bot2.structures(UnitTypeId.PYLON)
        
        p1_pylon_dead = p1_pylons.amount == 0
        p2_pylon_dead = p2_pylons.amount == 0
        
        # Identify trainee and opponent bots
        trainee_bot = self.bot1 if self.bot1.is_trainee else self.bot2
        opponent_bot = self.bot2 if self.bot1.is_trainee else self.bot1
        
        trainee_pylon_dead = p1_pylon_dead if self.bot1.is_trainee else p2_pylon_dead
        opponent_pylon_dead = p2_pylon_dead if self.bot1.is_trainee else p1_pylon_dead
        
        # ASYMMETRIC LOSS: If trainee has no combat units left, terminate immediately
        trainee_army_lost = not self._has_combat_units(trainee_bot)
        
        win_reward = self.training_conf.get("rewards", {}).get("win_reward", 100.0)
        loss_penalty = self.training_conf.get("rewards", {}).get("loss_penalty", -win_reward)
        
        reward_delta = 0.0
        done = False
        
        if trainee_pylon_dead:
            done = True
            reward_delta -= win_reward
        elif trainee_army_lost:
            # Trainee lost all units -> Early termination (asymmetric)
            done = True
            reward_delta += loss_penalty 
        elif opponent_pylon_dead:
            done = True
            reward_delta += win_reward
            
        truncated = self.steps_count >= self.max_steps
        return done, truncated, reward_delta


    def action_masks(self):
        """Returns valid actions for the active unit."""
        active_bot = self.bot1 if self.active_player_id == 1 else self.bot2
        if active_bot:
            return active_bot.action_masks()
        
        # Fallback
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

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # --- PROPERTY SANITY CHECK ---
        # Verify that bot.units is not shadowed by a static list
        for bot_idx, bot in enumerate([self.bot1, self.bot2]):
            if bot and "units" in bot.__dict__:
                self._log(f"CRITICAL WARNING: Bot {bot_idx+1} units property is SHADOWED by an instance attribute.")
            elif bot:
                self._log(f"Bot {bot_idx+1} units property is verified dynamic.")
        # -----------------------------
        
        return obs, info

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