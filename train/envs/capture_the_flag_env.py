import gymnasium as gym
import asyncio
import re
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import random
from gymnasium import spaces
from loguru import logger
import json
import os

from train.envs.multi_client_env import MultiClientEnv
from src.utils.layer_constants import LayerConstants
from src.utils.config_manager import ConfigManager
from src.agent.ctf_bot import CaptureTheFlagBot
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM
from sc2.ids.unit_typeid import UnitTypeId
from src.utils.spawning_utils import SpawningUtils
from train.envs.ctf.region_loader import RegionLoader

_PAIR_RE = re.compile(r"^(\d+)_(a|d)$")

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
                 level_name: str = "capture_the_flag_0",
                 **kwargs):
        super().__init__(map_name=map_name, level_name=level_name, training_conf=training_conf, **kwargs)
        self.training_conf = training_conf or {}
        
        # Robustly load parameters from training_conf (handling flattened vs nested)
        env_conf = self.training_conf.get("env", {})

        # Structured per-group roster config (new path).  Falls back to legacy flat params.
        self.unit_pools_config: List[Dict] = (
            self.training_conf.get("unit_pools")
            or env_conf.get("unit_pools")
            or []
        )

        # Legacy flat params — kept for backward compat when unit_pools_config is absent.
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
        
        # Use the config directory from training_conf (universal pipeline passes "config_universal").
        # MultiClientEnv defaults to "config" so we override here.
        config_dir = self.training_conf.get("config_dir", "config_universal")
        self.config_manager = ConfigManager(config_dir)

        self.agent_architecture = self.config_manager.get_agent_architecture()
        self.training_conf.setdefault("agent_architecture", self.agent_architecture)

        # Dynamic Observation Layers from Curriculum
        self.active_layers = self.config_manager.get_cumulative_active_layers(self.level_name)
        self.training_conf["active_layers"] = self.active_layers

        # Dynamic Action Heads from Curriculum
        self.action_heads = self.config_manager.get_cumulative_action_heads(self.level_name)
        if not self.action_heads:
            self.action_heads = env_conf.get("action_heads", ["move", "attack"])
        self.categorical_heads = self.config_manager.get_cumulative_categorical_heads(self.level_name)
        self.training_conf["action_heads"] = self.action_heads
        self.training_conf["categorical_heads"] = self.categorical_heads

        num_layers = len(self.active_layers) if self.active_layers else LayerConstants.get_num_layers()
        obs_dict = {
            "map": spaces.Box(low=0.0, high=1.0, shape=(num_layers, 128, 128), dtype=np.float32),
            "command": spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32),
        }

        if self.agent_architecture == "universal":
            self.active_tactical_layers = self.config_manager.get_cumulative_active_tactical_layers(self.level_name)
            if not self.active_tactical_layers:
                self.active_tactical_layers = self.active_layers
            self.training_conf["active_tactical_layers"] = self.active_tactical_layers
            self.training_conf["fixed_universal_command_dim"] = FIXED_UNIVERSAL_COMMAND_DIM
            n_tactical = len(self.active_tactical_layers)
            obs_dict["tactical"] = spaces.Box(low=0.0, high=1.0, shape=(n_tactical, 32, 32), dtype=np.float32)
            self.observation_space = spaces.Dict(obs_dict)
            self.action_space = spaces.MultiDiscrete([FIXED_UNIVERSAL_COMMAND_DIM, 32 * 32])
        else:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(num_layers, 128, 128), dtype=np.float32)
            spatial_size = len(self.action_heads) * self.grid_size * self.grid_size
            categorical_size = len(self.categorical_heads)
            self.action_space = spaces.Discrete(spatial_size + categorical_size)
        
        self.bot1: Optional[CaptureTheFlagBot] = None
        self.bot2: Optional[CaptureTheFlagBot] = None
        
        self.active_player_id = 1
        self.current_unit_index = 0
        self.player_units = {1: [], 2: []}

        # Arena label → defender (_d) center; used for spawn-time unit_target_map.
        self.label_to_defender_pos: Dict[str, Any] = {}
        
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

    # ------------------------------------------------------------------
    # Region helpers
    # ------------------------------------------------------------------

    def _load_region_pairs(self, map_path: str) -> Dict[int, Dict[str, Any]]:
        """
        Parse map regions and return only COMPLETE pairs following the {n}_{a|d}
        naming scheme (both an '_a' and a '_d' side must be present).

        Pairs where only one side exists (e.g. a defender-only marker) are
        silently skipped — they cannot be used for spawning.

        Returns:
            {pair_index: {"a": RegionDef, "d": RegionDef}}
        """
        regions = RegionLoader().load(map_path)
        raw: Dict[int, Dict[str, Any]] = {}
        for name, region_def in regions.items():
            m = _PAIR_RE.match(name)
            if m:
                idx = int(m.group(1))
                raw.setdefault(idx, {})[m.group(2)] = region_def

        complete = {idx: sides for idx, sides in raw.items() if set(sides) == {"a", "d"}}
        incomplete = [idx for idx in raw if idx not in complete]
        if incomplete:
            logger.warning(
                f"Map '{map_path}': skipping incomplete region pairs {sorted(incomplete)} "
                "(each usable pair must have both '_a' and '_d' regions)."
            )
        return complete

    def _assign_region_pairs(
        self,
        available_pairs: Dict[int, Dict[str, Any]],
        unit_pools_config: List[Dict],
    ) -> Dict[str, int]:
        """
        Randomly assign a distinct region pair index to each unique region label
        found in *unit_pools_config*.

        Labels are processed in sorted order so the mapping is deterministic for
        a given random seed.  Raises RuntimeError if there are fewer available
        pairs than distinct labels.

        Returns:
            {"A": 3, "B": 7, ...}  — label → pair index
        """
        labels = sorted({pool["region"] for pool in unit_pools_config})
        pair_indices = list(available_pairs.keys())
        if len(pair_indices) < len(labels):
            raise RuntimeError(
                f"Need {len(labels)} region pairs for labels {labels} but map "
                f"only has {len(pair_indices)}: {sorted(pair_indices)}"
            )
        chosen = random.sample(pair_indices, len(labels))
        return dict(zip(labels, chosen))

    async def _refresh_bot_states(self):
        """Fetches latest observation and updates bot internal state."""
        from src.utils.sc2_observation_compat import game_state_from_response_observation

        async def refresh(bot, client):
            obs = await client.observation()
            state = game_state_from_response_observation(obs.observation)
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

        # 1. Clear Map
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
            if is_pathable(base_pos):
                return base_pos
            for d in range(1, int(max_dist)):
                for angle in range(0, 360, 45):
                    rad = math.radians(angle)
                    p = Point2((base_pos.x + d * math.cos(rad), base_pos.y + d * math.sin(rad)))
                    if is_pathable(p):
                        return p
            return base_pos

        def to_id(name):
            return getattr(UnitTypeId, name)

        def get_offset_pos(base_pos, index, total):
            angle = (2 * math.pi * index) / max(total, 1)
            dist = 3.0
            p = Point2((base_pos.x + dist * math.cos(angle), base_pos.y + dist * math.sin(angle)))
            return get_valid_pos(p)

        pool_spawn_centers: List[Tuple[int, str, Any, str]] = []
        label_to_defender_pos: Dict[str, Any] = {}
        legacy_targets = None
        if self.unit_pools_config:
            pool_spawn_centers, label_to_defender_pos = await self._spawn_from_unit_pools(
                get_valid_pos, get_offset_pos, to_id
            )
        else:
            legacy_targets = await self._spawn_legacy(get_valid_pos, get_offset_pos, to_id)

        # Advance physics to materialise all units
        for _ in range(3):
            await self._synchronized_step()
            await self._refresh_bot_states()

        # Assign unit_command_map / unit_target_map on each bot after materialisation.
        if pool_spawn_centers:
            self.label_to_defender_pos = label_to_defender_pos
            self._assign_unit_commands(pool_spawn_centers, label_to_defender_pos)
        elif legacy_targets:
            self._assign_legacy_unit_targets(*legacy_targets)

        # Set reduced pylon HP (10 HP + 10 Shield) — applies to all pylons
        p1_pylons = self.bot1.structures(UnitTypeId.PYLON)
        p2_pylons = self.bot2.structures(UnitTypeId.PYLON)
        if p1_pylons.exists and p2_pylons.exists:
            tags = [p1_pylons.first.tag, p2_pylons.first.tag]
            await self.bot1.client.debug_set_unit_value(tags, 2, 10.0)  # Health
            await self.bot1.client.debug_set_unit_value(tags, 3, 10.0)  # Shield

        # Final syncs
        await self._synchronized_step()
        await self._refresh_bot_states()
        await self._synchronized_step()
        await self._refresh_bot_states()

    async def _spawn_from_unit_pools(
        self, get_valid_pos, get_offset_pos, to_id
    ) -> Tuple[List[Tuple[int, str, Any, str]], Dict[str, Any]]:
        """
        Region-based spawn path driven by ``unit_pools_config``.

        Region label → SC2 player mapping: sorted unique labels → player 1, 2, …
        For each pool:
          - high_level_command "attack" → spawn at {pair_index}_a
          - high_level_command "defend" → spawn at {pair_index}_d + place pylon

        Returns:
            pool_spawn_centers: (player_id, high_level_command, spawn_center, region_label)
            label_to_defender_pos: arena label → Point2 of the _d center (flag objective)
        """
        from sc2.ids.unit_typeid import UnitTypeId
        from sc2.position import Point2

        # Resolve the current map file path from the sc2 Map object
        game_map = self._resolve_map(self.map_name)
        map_path = str(game_map.path)

        pairs = self._load_region_pairs(map_path)
        label_to_pair = self._assign_region_pairs(pairs, self.unit_pools_config)

        label_to_defender_pos: Dict[str, Any] = {}
        for pool in self.unit_pools_config:
            if pool.get("high_level_command", "attack") == "defend":
                label = pool["region"]
                if label not in label_to_defender_pos:
                    pair_idx = label_to_pair[label]
                    d_center = pairs[pair_idx]["d"].center
                    label_to_defender_pos[label] = Point2((d_center.x, d_center.y))

        # Player assignment: agent:True → player 1 (trainee), absent/False → player 2.
        # Region label identifies the battle arena — pools from different teams can
        # share a label so their units spawn close to each other and fight.
        client_map = {1: self._manager.clients[0], 2: self._manager.clients[1]}
        cmds_by_player: Dict[int, List] = {1: [], 2: []}
        pool_spawn_centers: List[Tuple[int, str, Any, str]] = []

        for pool in self.unit_pools_config:
            label = pool["region"]
            player_id = 1 if pool.get("agent", False) else 2
            pair_idx = label_to_pair[label]
            command = pool.get("high_level_command", "attack")
            side_char = "a" if command == "attack" else "d"
            region_def = pairs[pair_idx][side_char]

            center = region_def.center
            spawn_center = Point2((center.x, center.y))
            pool_spawn_centers.append((player_id, command, spawn_center, label))

            unit_names = self._calculate_spawns_for_pool(pool)
            for i, name in enumerate(unit_names):
                pos = get_offset_pos(spawn_center, i, len(unit_names))
                cmds_by_player[player_id].append([to_id(name), 1, pos, player_id])

            # Defender pools also own the flag (pylon) at the region centre
            if side_char == "d":
                pylon_pos = get_valid_pos(spawn_center)
                cmds_by_player[player_id].append(
                    [UnitTypeId.PYLON, 1, pylon_pos, player_id]
                )

        # Issue spawn commands
        for player_id, cmds in cmds_by_player.items():
            if cmds:
                await client_map[player_id].debug_create_unit(cmds)

        return pool_spawn_centers, label_to_defender_pos

    def _assign_unit_commands(
        self,
        pool_spawn_centers: List[Tuple[int, str, Any, str]],
        label_to_defender_pos: Dict[str, Any],
    ) -> None:
        """
        Populate ``bot.unit_command_map`` and ``bot.unit_target_map`` by assigning
        each spawned unit to the pool whose spawn center is nearest.

        ``unit_target_map[tag]`` is the _d (defender pylon) center for that unit's
        arena — the navigation objective for attackers and defenders alike.

        Args:
            pool_spawn_centers: (player_id, high_level_command, Point2 center, label)
            label_to_defender_pos: arena label → defender center Point2
        """
        import math
        from sc2.position import Point2

        bot_map = {1: self.bot1, 2: self.bot2}

        for player_id, bot in bot_map.items():
            bot.unit_command_map = {}
            bot.unit_target_map = {}
            player_centers = [
                (cmd, center, label)
                for pid, cmd, center, label in pool_spawn_centers
                if pid == player_id
            ]
            if not player_centers:
                continue

            for unit in bot.units:
                if unit.is_structure:
                    continue
                best_cmd = player_centers[0][0]
                best_label = player_centers[0][2]
                best_dist = math.inf
                for cmd, center, label in player_centers:
                    d = unit.distance_to(Point2((center.x, center.y)))
                    if d < best_dist:
                        best_dist = d
                        best_cmd = cmd
                        best_label = label
                bot.unit_command_map[unit.tag] = best_cmd
                if best_label in label_to_defender_pos:
                    bot.unit_target_map[unit.tag] = label_to_defender_pos[best_label]

    def _assign_legacy_unit_targets(
        self, p1_pylon_pos: Any, p2_pylon_pos: Any
    ) -> None:
        """Legacy spawn: each player's units target the opponent's pylon."""
        for bot, enemy_pylon_pos in (
            (self.bot1, p2_pylon_pos),
            (self.bot2, p1_pylon_pos),
        ):
            bot.unit_command_map = {}
            bot.unit_target_map = {}
            for unit in bot.units:
                if unit.is_structure:
                    continue
                bot.unit_command_map[unit.tag] = "attack"
                bot.unit_target_map[unit.tag] = enemy_pylon_pos

    async def _spawn_legacy(self, get_valid_pos, get_offset_pos, to_id):
        """Original random-position pylon + budget spawn path (backward compat)."""
        from sc2.ids.unit_typeid import UnitTypeId
        from sc2.position import Point2

        w = self.bot1.game_info.playable_area.width
        h = self.bot1.game_info.playable_area.height
        offset_x = self.bot1.game_info.playable_area.x
        offset_y = self.bot1.game_info.playable_area.y

        requested_dist = self.training_conf.get("env", {}).get(
            "flag_distance_min", self.flag_distance_min
        )
        safe_dist = SpawningUtils.calculate_safe_distance(w, h, requested_dist)
        rel_pos1, rel_pos2 = SpawningUtils.get_random_pylon_positions(w, h, safe_dist, margin=1.0)

        pos1 = get_valid_pos(Point2((rel_pos1.x + offset_x, rel_pos1.y + offset_y)))
        pos2 = get_valid_pos(Point2((rel_pos2.x + offset_x, rel_pos2.y + offset_y)))

        await asyncio.gather(
            self.bot1.client.debug_create_unit([[UnitTypeId.PYLON, 1, pos1, 1]]),
            self.bot2.client.debug_create_unit([[UnitTypeId.PYLON, 1, pos2, 2]]),
        )

        spawns1 = self._calculate_spawns(self.mineral_budget, self.vespene_budget)
        spawns2 = self._calculate_spawns(self.mineral_budget, self.vespene_budget)

        cmds1 = [[to_id(u), 1, get_offset_pos(pos1, i, len(spawns1)), 1] for i, u in enumerate(spawns1)]
        cmds2 = [[to_id(u), 1, get_offset_pos(pos2, i, len(spawns2)), 2] for i, u in enumerate(spawns2)]

        await asyncio.gather(
            self.bot1.client.debug_create_unit(cmds1),
            self.bot2.client.debug_create_unit(cmds2),
        )
        return pos1, pos2

    def _calculate_spawns(self, m, v):
        """Legacy budget-greedy spawn for the flat unit_pool path."""
        from train.envs.ctf.budgeted_spawns import calculate_budgeted_spawns

        return calculate_budgeted_spawns(self.unit_pool, self.unit_data, m, v)

    def _calculate_spawns_for_pool(self, pool: Dict) -> List[str]:
        """Budget-greedy spawn list derived from a single unit_pools entry."""
        from train.envs.ctf.budgeted_spawns import calculate_budgeted_spawns

        return calculate_budgeted_spawns(
            pool.get("unit_pool", []),
            self.unit_data,
            pool.get("mineral_budget", 0),
            pool.get("vespene_budget", 0),
        )

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
        Uses library Units filtering for robustness.
        """
        if not hasattr(bot, "units") or not bot.units:
            return False
            
        ignored_types = {
            UnitTypeId.LARVA, 
            UnitTypeId.EGG, 
            UnitTypeId.OVERLORD, 
            UnitTypeId.PROBE, 
            UnitTypeId.SCV, 
            UnitTypeId.DRONE
        }
        
        # bot.units already filters for owned units and excludes structures
        # We further exclude non-combat units and ensure they are alive
        combat_units = bot.units.filter(lambda u: not u.is_structure and u.type_id not in ignored_types and u.health > 0 and not getattr(u, "is_memory", False))
        
        return combat_units.amount > 0

    def _trainee_has_mixed_roles(self) -> bool:
        """
        Returns True when the trainee player has BOTH an "attack" and a "defend"
        pool in unit_pools_config (i.e. a mixed-role team).

        In that case a timeout (draw) is not a pure failure — defenders may have
        kept the pylon alive intentionally — so draw_penalty should be suppressed.
        """
        if not self.unit_pools_config:
            return False

        sorted_labels = sorted({p["region"] for p in self.unit_pools_config})
        label_to_player = {label: i + 1 for i, label in enumerate(sorted_labels)}
        trainee_player_id = 1 if self.bot1.is_trainee else 2

        trainee_commands = {
            pool["high_level_command"]
            for pool in self.unit_pools_config
            if label_to_player.get(pool["region"]) == trainee_player_id
        }
        return "attack" in trainee_commands and "defend" in trainee_commands

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
        
        # ARMY LOSS: Check if both armies are gone (Mutual Destruction -> Draw)
        trainee_army_lost = not self._has_combat_units(trainee_bot)
        opponent_army_lost = not self._has_combat_units(opponent_bot)
        
        win_reward = self.training_conf.get("rewards", {}).get("win_reward", 100.0)
        loss_penalty = self.training_conf.get("rewards", {}).get("loss_penalty", -100.0)
        draw_penalty = self.training_conf.get("rewards", {}).get("draw_penalty", -10.0)
        
        reward_delta = 0.0
        done = False
        
        if trainee_pylon_dead:
            done = True
            reward_delta += loss_penalty
        elif opponent_pylon_dead:
            done = True
            reward_delta += win_reward
        elif trainee_army_lost:
            # Episodes terminate immediately if the trainee loses all units
            done = True
            reward_delta += loss_penalty
        elif opponent_army_lost:
            # Persistence: If only opponent loses army, game continues to allow 
            # trainee to learn secondary objectives (pylon destruction).
            pass
            
        truncated = self.steps_count >= self.max_steps
        if truncated and not done:
            # Only apply draw_penalty when the trainee is a pure attacker.
            # If the team has mixed roles (both "attack" and "defend" pools),
            # a timeout is not a failure — the defenders may have done their job
            # by keeping the pylon alive — so the penalty is suppressed.
            if not self._trainee_has_mixed_roles():
                reward_delta += draw_penalty

        return done, truncated, reward_delta


    def action_masks(self):
        """Returns valid actions for the active unit."""
        active_bot = self.bot1 if self.active_player_id == 1 else self.bot2
        if active_bot:
            return active_bot.action_masks()

        # Fallback before bots are ready
        if isinstance(self.action_space, spaces.MultiDiscrete):
            return np.ones(int(np.sum(self.action_space.nvec)), dtype=bool)
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
        return super().reset(seed=seed, options=options)

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