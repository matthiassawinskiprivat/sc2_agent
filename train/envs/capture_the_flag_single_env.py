
import gymnasium as gym
import asyncio
import numpy as np
import re
from typing import Tuple, List, Dict, Any, Optional, NamedTuple
import random
from gymnasium import spaces
from loguru import logger
import logging

from train.envs.single_client_env import SingleClientEnv
from train.envs.sc2_session_config import every_k_episodes
from src.agent.ctf_bot import CaptureTheFlagBot
from src.utils.config_manager import ConfigManager
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM, universal_fallback_action
from src.utils.layer_constants import LayerConstants
from sc2.ids.unit_typeid import UnitTypeId
from src.utils.spawning_utils import SpawningUtils
from train.envs.ctf.region_loader import RegionLoader
from train.envs.ctf.pool_assignment import (
    alternate_trainee_race_enabled,
    is_pool_trainee,
    pool_player_id,
)
import json
import os

_PAIR_RE = re.compile(r"^(\d+)_(a|d)$")

# Squad micro: one policy unit per physics step; no cross-squad chunk batching.
DEFAULT_STEP_MUL_PER_UNIT = 1
DEFAULT_UNITS_PER_PHYSICS_STEP = 0


class _ChunkEntry(NamedTuple):
    pool_idx: int
    player_id: int
    tag: int
    is_agent: bool


class CaptureTheFlagSingleEnv(SingleClientEnv):
    def __init__(self, 
                 mineral_budget: int = 500, 
                 vespene_budget: int = 0, 
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
        
        # Structured per-group roster config (new path). Falls back to legacy flat params.
        self.unit_pools_config: List[Dict] = (
            self.training_conf.get("unit_pools")
            or env_conf.get("unit_pools")
            or []
        )

        # 1. Minerals
        self.mineral_budget = self.training_conf.get("mineral_budget", 
                                 env_conf.get("mineral_budget", mineral_budget))
        
        # 2. Vespene
        self.vespene_budget = self.training_conf.get("vespene_budget", 
                                 env_conf.get("vespene_budget", vespene_budget))
        
        # 3. Unit Pool
        self.unit_pool = self.training_conf.get("unit_pool", 
                           env_conf.get("unit_pool", unit_pool or ["ZERGLING", "ROACH", "MARINE"]))
        
        # Load unit data for costs
        self.unit_data = self._load_unit_data()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.steps_count = 0
        self.client_count = 1 # God-Mode uses only 1 connection
        
        # Use the config directory from training_conf (universal pipeline passes "config_universal").
        # SingleClientEnv defaults to "config" so we override here.
        config_dir = self.training_conf.get("config_dir", "config_universal")
        self.config_manager = ConfigManager(config_dir)

        self.agent_architecture = self.config_manager.get_agent_architecture()
        self.training_conf.setdefault("agent_architecture", self.agent_architecture)
        if self.agent_architecture == "universal":
            self.grid_size = 32
        self.training_conf["grid_size"] = self.grid_size

        # Prefer cumulative lists pre-computed in the parent process (async/sync pipeline).
        # Subproc workers re-walking every pipeline YAML can race on Windows file reads.
        self.active_layers = self._resolve_cumulative_list(
            "active_layers", self.config_manager.get_cumulative_active_layers
        ) or env_conf.get("active_layers", [])
        self.training_conf["active_layers"] = self.active_layers

        self.action_heads = self._resolve_cumulative_list(
            "action_heads", self.config_manager.get_cumulative_action_heads
        ) or env_conf.get("action_heads", ["move", "attack"])
        self.categorical_heads = self._resolve_cumulative_list(
            "categorical_heads", self.config_manager.get_cumulative_categorical_heads
        )
        self.training_conf["action_heads"] = self.action_heads
        self.training_conf["categorical_heads"] = self.categorical_heads

        num_layers = len(self.active_layers) if self.active_layers else LayerConstants.get_num_layers()
        obs_dict = {
            "map": spaces.Box(low=0.0, high=1.0, shape=(num_layers, 128, 128), dtype=np.float32),
            "command": spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32),
        }

        if self.agent_architecture == "universal":
            self.active_tactical_layers = self._resolve_cumulative_list(
                "active_tactical_layers",
                self.config_manager.get_cumulative_active_tactical_layers,
            ) or self.active_layers
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
        
        self.bot: Optional[CaptureTheFlagBot] = None
        self.active_player_id = 1
        self.current_unit_index = 0
        self.player_units: Dict[int, List[int]] = {1: [], 2: []}
        self.client_count = 1
        self.episode_counter = 0

        # Squad-ordered cycling: each squad = one unit_pool entry.
        # Cycling order follows unit_pools_config order, interleaving P1/P2 squads
        # so that opposing squads (same arena) act consecutively.
        # squad_unit_tags[i] = (player_id, [unit_tags_in_pool_i])
        self.squad_unit_tags: List[Tuple[int, List[int]]] = []
        self.current_squad_index: int = 0
        # Maps unit tag → index in unit_pools_config (set at spawn, survives unit deaths)
        self.unit_pool_index: Dict[int, int] = {}

        # Spawn-time arena metadata (used for endgame defender→attack promotion)
        self.label_to_defender_pos: Dict[str, Any] = {}
        self._promoted_pool_index: Optional[int] = None

        # Deferred terminal step — when the game ends during an opponent's physics step
        # (requires_policy_action=False), the terminal reward from P1's perspective is
        # buffered here and returned on the NEXT phantom agent step so the training buffer
        # always captures the terminal reward.
        self._deferred_done: bool = False
        self._deferred_truncated: bool = False
        self._deferred_p1_terminal_reward: float = 0.0
        self._deferred_terminal_obs: Optional[Any] = None

        # Self-play opponent model — frozen snapshot, synced on hard reset (every_k_episodes).
        # None until the first hard reset; also None when using symmetric self-play.
        self.model: Optional[Any] = None           # latest model ref, set by training loop
        self.opponent_model: Optional[Any] = None  # frozen snapshot used for player 2
        self._opponent_lstm_state: Optional[Any] = None  # stateful LSTM between steps
        self._opponent_last_focus_tag: Optional[int] = None
        self._prepared_chunk: Optional[List[_ChunkEntry]] = None
        self._alternate_trainee_race = alternate_trainee_race_enabled(self.training_conf)

    def _pool_player_id(self, pool: Dict) -> int:
        return pool_player_id(
            pool,
            self.unit_pools_config,
            getattr(self, "episode_counter", 1),
            alternate=getattr(self, "_alternate_trainee_race", False),
        )

    def _is_pool_trainee(self, pool: Dict) -> bool:
        return is_pool_trainee(
            pool,
            self.unit_pools_config,
            getattr(self, "episode_counter", 1),
            alternate=getattr(self, "_alternate_trainee_race", False),
        )

    def _resolve_cumulative_list(self, key: str, getter):
        """Use flattened training_conf list when the pipeline already computed it."""
        precomputed = self.training_conf.get(key)
        if precomputed:
            return list(precomputed)
        return getter(self.level_name)

    def _load_unit_data(self):
        path = os.path.join(os.getcwd(), "data", "unit_data.json")
        if not os.path.exists(path):
            logger.warning(f"unit_data.json not found at {path}. Using empty dict.")
            return {}
        with open(path, "r") as f:
            return json.load(f)

    def set_model(self, model):
        """Receive the latest trained model from the training loop."""
        self.model = model

    def _resolve_opponent_model(self) -> Tuple[Optional[Any], Optional[str]]:
        """
        Model used for opponent inference.

        Prefer the worker-local ``opponent_model`` loaded from disk. ``self.model`` is
        only available in-process (not in SubprocVecEnv workers).
        """
        if self.opponent_model is not None:
            return self.opponent_model, "frozen"
        if self.model is not None:
            return self.model, "live"
        return None, None

    def _preview_computer_opponent(self) -> bool:
        """Debug UI preview: SC2 Computer controls P2 instead of a policy snapshot."""
        return bool(self.training_conf.get("preview_computer_opponent"))

    def _ensure_opponent_model_loaded(self) -> bool:
        """
        Load the frozen opponent snapshot when missing.

        The collector pushes ``initial_model.zip`` at startup, but workers can miss
        that window or fail graft on the first try. Without a retry here, player 2
        falls back to a safe center-cell move until the next hard reset.
        """
        if self._preview_computer_opponent() or self.opponent_model is not None:
            return self.opponent_model is not None
        return self._sync_opponent_model()

    def _opponent_graft_grid_size(self) -> int:
        """32×32 universal spatial head — not env.grid_size (often 128 for legacy masks)."""
        if isinstance(self.action_space, spaces.MultiDiscrete) and len(self.action_space.nvec) >= 2:
            spatial_cells = int(self.action_space.nvec[1])
            side = int(round(spatial_cells ** 0.5))
            if side * side == spatial_cells:
                return side
        return 32

    async def _setup_game_session(self):
        from sc2.player import Bot, Computer
        from sc2.data import Race, Difficulty
        from s2clientprotocol import sc2api_pb2 as sc_pb
        from train.envs.sc2_session_config import session_realtime

        self.bot = CaptureTheFlagBot(player_id=1, training_conf=self.training_conf)
        sim_realtime = session_realtime(self.training_conf)
        env_conf = self.training_conf.get("env", {})
        difficulty_name = self.training_conf.get("difficulty") or env_conf.get("difficulty", "Easy")
        try:
            difficulty = getattr(Difficulty, difficulty_name)
        except AttributeError:
            difficulty = Difficulty.Easy
        players = [Bot(Race.Zerg, self.bot), Computer(Race.Zerg, difficulty)]
        resolved_map = self._resolve_map(self.map_name)
        
        self._log("Starting single-process God-Mode game session...")
        await self._manager.create_join_game(resolved_map, players, realtime=sim_realtime)

        # SC2 5.0.16 requires polling until the game transitions from init_game → in_game
        # before sending any further requests. Sending RequestGameInfo while SC2 is still
        # in init_game causes it to close the WebSocket (WSMsgType.CLOSE / message 257).
        client = self._manager.clients[0]
        self._log("Waiting for SC2 to transition from init_game → in_game...")
        _in_game_wait_secs = 30.0
        _poll_interval = 0.3
        _waited = 0.0
        while _waited < _in_game_wait_secs:
            try:
                ping_resp = await asyncio.wait_for(
                    client._execute(ping=sc_pb.RequestPing()), timeout=5.0
                )
                sc2_status = ping_resp.status  # 3 = in_game in SC2 API proto
                if sc2_status == 3:  # Status.in_game
                    self._log(f"SC2 status = in_game after {_waited:.1f}s.")
                    break
                self._log(f"SC2 status = {sc2_status} (init_game), waiting...")
            except Exception as _ping_exc:
                self._log(f"Ping failed while waiting for in_game: {_ping_exc}")
                raise
            await asyncio.sleep(_poll_interval)
            _waited += _poll_interval
        else:
            raise RuntimeError(
                f"SC2 did not reach in_game status within {_in_game_wait_secs}s after join_game."
            )

        # Prepare bot
        raw_game_info = await client._execute(game_info=sc_pb.RequestGameInfo())
        from sc2.game_info import GameInfo
        game_info = GameInfo(raw_game_info.game_info)
        game_data = await client.get_game_data()
        
        self.bot._prepare_start(client, 1, game_info, game_data, sim_realtime)
        self.bot._proto_game_info = raw_game_info
        await self.bot.on_start()
        
        await client.debug_show_map()
        if not self._preview_computer_opponent():
            await client.debug_control_enemy()
        
        # Initial step and spawn
        await client.step()
        await self._refresh_bot_state()
        await self._spawn_initial_units()
        await self._refresh_unit_lists()

    async def _refresh_bot_state(self):
        from src.utils.sc2_observation_compat import game_state_from_response_observation
        obs = await self.bot.client.observation()
        state = game_state_from_response_observation(obs.observation)
        self.bot._prepare_step(state, self.bot._proto_game_info)

    # ------------------------------------------------------------------
    # Region helpers (mirrors CaptureTheFlagEnv)
    # ------------------------------------------------------------------

    def _load_region_pairs(self, map_path: str) -> Dict[int, Dict[str, Any]]:
        """Return complete {n: {"a": RegionDef, "d": RegionDef}} pairs from the map."""
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
                f"Map '{map_path}': skipping incomplete region pairs {sorted(incomplete)}"
            )
        return complete

    def _assign_region_pairs(
        self,
        available_pairs: Dict[int, Dict[str, Any]],
        unit_pools_config: List[Dict],
    ) -> Dict[str, int]:
        """Randomly assign a distinct pair index to each unique region label."""
        labels = sorted({pool["region"] for pool in unit_pools_config})
        pair_indices = list(available_pairs.keys())
        if len(pair_indices) < len(labels):
            raise RuntimeError(
                f"Need {len(labels)} region pairs for labels {labels} but map "
                f"only has {len(pair_indices)}: {sorted(pair_indices)}"
            )
        chosen = random.sample(pair_indices, len(labels))
        return dict(zip(labels, chosen))

    def _calculate_spawns_for_pool(self, pool: Dict) -> List[str]:
        """Budget-greedy spawn list from a single unit_pools entry."""
        m = pool.get("mineral_budget", 0)
        v = pool.get("vespene_budget", 0)
        unit_pool = pool.get("unit_pool", [])

        def can_afford(u):
            if u not in self.unit_data:
                logger.error(f"Unit {u} not found in unit_data.json. Skipping.")
                return False
            cost = self.unit_data[u]["cost"]
            return cost["minerals"] <= m and cost["vespene"] <= v

        spawns = []
        affordable = [u for u in unit_pool if can_afford(u)]
        while affordable:
            u = random.choice(affordable)
            spawns.append(u)
            cost = self.unit_data[u]["cost"]
            m -= cost["minerals"]
            v -= cost["vespene"]
            affordable = [u for u in unit_pool if can_afford(u)]
        return spawns

    def _assign_unit_commands(
        self,
        pool_spawn_centers: List[Tuple],
        label_to_defender_pos: Dict,
    ) -> None:
        """
        Populate bot.unit_command_map and bot.unit_target_map by proximity to
        spawn centers.

        unit_command_map[tag] → "attack" | "defend"
        unit_target_map[tag]  → Point2 of the _d (defender pylon) center in the
                                 unit's arena — the objective for both attackers
                                 (destroy it) and defenders (guard it).
        """
        import math
        from sc2.position import Point2

        self.bot.unit_command_map = {}
        self.bot.unit_target_map = {}
        self.unit_pool_index = {}
        if not pool_spawn_centers:
            return

        for unit in self.bot.all_units.filter(lambda u: not u.is_structure):
            best_pool_idx = 0
            best_cmd = pool_spawn_centers[0][1]
            best_label = pool_spawn_centers[0][3]
            best_dist = math.inf
            for pool_idx, (player_id, cmd, center, label) in enumerate(pool_spawn_centers):
                if unit.owner_id != player_id:
                    continue
                d = unit.distance_to(Point2((center.x, center.y)))
                if d < best_dist:
                    best_dist = d
                    best_pool_idx = pool_idx
                    best_cmd = cmd
                    best_label = label
            self.bot.unit_command_map[unit.tag] = best_cmd
            if best_label in label_to_defender_pos:
                self.bot.unit_target_map[unit.tag] = label_to_defender_pos[best_label]
            self.unit_pool_index[unit.tag] = best_pool_idx

    async def _spawn_initial_units(self):
        from sc2.ids.unit_typeid import UnitTypeId
        from sc2.position import Point2
        import math
        
        # 1. Clear Map of all existing units and structures
        if self.bot.all_units.exists:
            await self.bot.client.debug_kill_unit(self.bot.all_units.tags)
            await self.bot.client.step()
            await self._refresh_bot_state()

        grid = self.bot.game_info.pathing_grid
        
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

        pool_spawn_centers: List[Tuple] = []
        label_to_defender_pos: Dict = {}

        if self.unit_pools_config:
            pool_spawn_centers, label_to_defender_pos = await self._spawn_from_unit_pools(
                get_valid_pos, get_offset_pos, to_id
            )
        else:
            await self._spawn_legacy(get_valid_pos, get_offset_pos, to_id)

        # Advance physics to materialise all units
        for _ in range(3):
            await self.bot.client.step()
            await self._refresh_bot_state()

        # Assign unit_command_map and unit_target_map by proximity to spawn centers
        if pool_spawn_centers:
            self.label_to_defender_pos = label_to_defender_pos
            self._promoted_pool_index = None
            self._assign_unit_commands(pool_spawn_centers, label_to_defender_pos)

        # Set reduced HP (10 HP + 10 Shield) on all pylons
        all_pylons = self.bot.all_units(UnitTypeId.PYLON)
        if all_pylons.exists:
            tags = [u.tag for u in all_pylons]
            await self.bot.client.debug_set_unit_value(tags, 2, 10.0)  # HP
            await self.bot.client.debug_set_unit_value(tags, 3, 10.0)  # Shield

        await self.bot.client.step()
        await self._refresh_bot_state()

    async def _spawn_from_unit_pools(
        self, get_valid_pos, get_offset_pos, to_id
    ) -> List[Tuple]:
        """
        Region-based spawn driven by unit_pools_config.

        Region label identifies the battle arena, NOT the team.  Pools from
        different teams sharing the same label are placed in the same arena so
        attackers and defenders face each other.

        Player assignment:
          - pool with  agent: True  →  player 1 (trainee)
          - pool without agent flag →  player 2 (opponent)

        For each pool:
          - high_level_command "attack" → spawn at {pair_index}_a center
          - high_level_command "defend" → spawn at {pair_index}_d center + place pylon there

        Returns list of (player_id, high_level_command, spawn_center) tuples for
        unit_command_map assignment after materialisation.
        """
        from sc2.ids.unit_typeid import UnitTypeId
        from sc2.position import Point2

        map_path = str(self._resolve_map(self.map_name).path)
        pairs = self._load_region_pairs(map_path)
        label_to_pair = self._assign_region_pairs(pairs, self.unit_pools_config)

        # Pre-compute the defender (_d) center for each arena label.
        # This is the objective position for every unit in that arena:
        # attackers go there to destroy the pylon; defenders stay there to guard it.
        label_to_defender_pos: Dict[str, Any] = {}
        for pool in self.unit_pools_config:
            if pool.get("high_level_command", "attack") == "defend":
                label = pool["region"]
                if label not in label_to_defender_pos:
                    pair_idx = label_to_pair[label]
                    d_center = pairs[pair_idx]["d"].center
                    label_to_defender_pos[label] = Point2((d_center.x, d_center.y))

        all_cmds: List = []
        # (player_id, command, spawn_center, arena_label)
        pool_spawn_centers: List[Tuple] = []

        for pool in self.unit_pools_config:
            label = pool["region"]
            player_id = self._pool_player_id(pool)
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
                all_cmds.append([to_id(name), 1, pos, player_id])

            # Defender pools own the flag (pylon) at the region centre
            if side_char == "d":
                pylon_pos = get_valid_pos(spawn_center)
                all_cmds.append([UnitTypeId.PYLON, 1, pylon_pos, player_id])

        if all_cmds:
            await self.bot.client.debug_create_unit(all_cmds)

        return pool_spawn_centers, label_to_defender_pos

    async def _spawn_legacy(self, get_valid_pos, get_offset_pos, to_id):
        """Fallback: random pylon positions when unit_pools_config is absent."""
        from sc2.ids.unit_typeid import UnitTypeId
        from sc2.position import Point2

        w = self.bot.game_info.playable_area.width
        h = self.bot.game_info.playable_area.height
        offset_x = self.bot.game_info.playable_area.x
        offset_y = self.bot.game_info.playable_area.y

        requested_dist = self.training_conf.get("env", {}).get("flag_distance_min", 20.0)
        safe_dist = SpawningUtils.calculate_safe_distance(w, h, requested_dist)
        rel_pos1, rel_pos2 = SpawningUtils.get_random_pylon_positions(w, h, safe_dist, margin=1.0)

        pos1 = get_valid_pos(Point2((rel_pos1.x + offset_x, rel_pos1.y + offset_y)))
        pos2 = get_valid_pos(Point2((rel_pos2.x + offset_x, rel_pos2.y + offset_y)))

        spawns1 = self._calculate_spawns(self.mineral_budget, self.vespene_budget)
        spawns2 = self._calculate_spawns(self.mineral_budget, self.vespene_budget)

        cmds1 = [[to_id(u), 1, get_offset_pos(pos1, i, len(spawns1)), 1] for i, u in enumerate(spawns1)]
        cmds2 = [[to_id(u), 1, get_offset_pos(pos2, i, len(spawns2)), 2] for i, u in enumerate(spawns2)]

        await self.bot.client.debug_create_unit(
            [[UnitTypeId.PYLON, 1, pos1, 1], [UnitTypeId.PYLON, 1, pos2, 2]]
            + cmds1 + cmds2
        )

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
        # Refresh all units from the bot's God-Mode perspective
        all_units = self.bot.all_units
        # Filter for alive and not structure
        self.player_units[1] = [u.tag for u in all_units if u.owner_id == 1 and not u.is_structure]
        self.player_units[2] = [u.tag for u in all_units if u.owner_id == 2 and not u.is_structure]
        if self.unit_pool_index and self.unit_pools_config:
            self._rebuild_squad_unit_tags()

    def _pool_alive_count(self, pool_idx: int) -> int:
        alive = set(self.player_units[1] + self.player_units[2])
        return sum(
            1 for tag, pi in self.unit_pool_index.items()
            if pi == pool_idx and tag in alive
        )

    def _agent_attack_units_alive(self) -> int:
        """Count alive agent (P1) units currently in the attack role."""
        alive_p1 = set(self.player_units.get(1, []))
        if not alive_p1:
            return 0

        if self.bot is not None and self.bot.unit_command_map:
            return sum(
                1 for tag in alive_p1
                if self.bot.unit_command_map.get(tag, "attack") == "attack"
            )

        if self.unit_pools_config and self.unit_pool_index:
            return sum(
                self._pool_alive_count(i)
                for i, pool in enumerate(self.unit_pools_config)
                if self._is_pool_trainee(pool)
                and pool.get("high_level_command", "attack") == "attack"
            )

        return len(alive_p1)

    def _enemy_pylon_target_for_pool(self, promoted_pool_idx: int):
        """Return the enemy defender pylon position for a promoted attack squad."""
        from sc2.position import Point2

        if not self.unit_pools_config or not self.label_to_defender_pos:
            return None

        promoted = self.unit_pools_config[promoted_pool_idx]
        promoted_player = self._pool_player_id(promoted)

        for pool in self.unit_pools_config:
            if pool.get("high_level_command", "attack") != "defend":
                continue
            pool_player = self._pool_player_id(pool)
            if pool_player == promoted_player:
                continue
            label = pool.get("region")
            if label in self.label_to_defender_pos:
                return self.label_to_defender_pos[label]
        return None

    def _apply_defender_promotion(self, pool_idx: int) -> None:
        """Switch one defender pool to attack role and retarget the enemy pylon."""
        if self.bot is None or self._promoted_pool_index is not None:
            return

        target = self._enemy_pylon_target_for_pool(pool_idx)
        if target is None:
            return

        alive = set(self.player_units[1] + self.player_units[2])
        promoted_tags = [
            tag for tag, pi in self.unit_pool_index.items()
            if pi == pool_idx and tag in alive
        ]
        if not promoted_tags:
            return

        for tag in promoted_tags:
            self.bot.unit_command_map[tag] = "attack"
            self.bot.unit_target_map[tag] = target

        pool = self.unit_pools_config[pool_idx]
        player_id = self._pool_player_id(pool)
        self.bot.prev_dist_per_squad.pop((player_id, "attack"), None)
        self.bot.prev_dist_per_squad.pop((player_id, "defend"), None)

        self._promoted_pool_index = pool_idx
        self._log(
            f"Promoted defend pool {pool_idx} ({len(promoted_tags)} units, P{player_id}) "
            f"to attack — objective → ({target.x:.1f}, {target.y:.1f})"
        )

    def _maybe_promote_defender_to_attack(self) -> None:
        """
        When every attack squad is eliminated but defender squads remain, promote
        the defender pool with the most surviving units to attack so the episode
        can resolve instead of stalling.
        """
        if (
            self._promoted_pool_index is not None
            or not self.unit_pools_config
            or not self.unit_pool_index
        ):
            return

        attack_indices = [
            i for i, pool in enumerate(self.unit_pools_config)
            if pool.get("high_level_command", "attack") == "attack"
        ]
        defend_indices = [
            i for i, pool in enumerate(self.unit_pools_config)
            if pool.get("high_level_command", "attack") == "defend"
        ]

        if any(self._pool_alive_count(i) > 0 for i in attack_indices):
            return

        defend_alive = [
            (i, self._pool_alive_count(i))
            for i in defend_indices
            if self._pool_alive_count(i) > 0
        ]
        if len(defend_alive) < 2:
            return

        promote_idx = max(defend_alive, key=lambda item: (item[1], -item[0]))[0]
        self._apply_defender_promotion(promote_idx)

    def _rebuild_squad_unit_tags(self) -> None:
        """
        Rebuild squad_unit_tags from unit_pool_index, keeping only alive units.

        squad_unit_tags follows unit_pools_config order so that squads from the
        same arena alternate: (P1 pool-0, P2 pool-1, P1 pool-2, P2 pool-3, ...).
        This is called every time _refresh_unit_lists runs so dead units are dropped.
        """
        alive = set(self.player_units[1] + self.player_units[2])
        self.squad_unit_tags = [
            (
                self._pool_player_id(pool),
                [t for t, pi in self.unit_pool_index.items() if pi == pool_idx and t in alive],
            )
            for pool_idx, pool in enumerate(self.unit_pools_config)
        ]

    def _current_squad(self) -> Tuple[int, List[int]]:
        """Return (player_id, unit_tag_list) for the currently acting squad."""
        if self.squad_unit_tags:
            return self.squad_unit_tags[self.current_squad_index]
        return self.active_player_id, self.player_units.get(self.active_player_id, [])

    def _current_pool_config(self) -> dict:
        if self.unit_pools_config and 0 <= self.current_squad_index < len(self.unit_pools_config):
            return self.unit_pools_config[self.current_squad_index]
        return {}

    def _is_agent_squad(self) -> bool:
        if self.unit_pools_config and self.squad_unit_tags:
            return self._is_pool_trainee(self._current_pool_config())
        return self.active_player_id == 1

    def _squad_tags(self) -> List[int]:
        _, tags = self._current_squad()
        return tags

    def _units_per_physics_step(self) -> int:
        env_conf = self.training_conf.get("env", {})
        n = int(
            self.training_conf.get("units_per_physics_step")
            or env_conf.get("units_per_physics_step", DEFAULT_UNITS_PER_PHYSICS_STEP)
        )
        return max(0, n)

    def _uses_chunk_physics(self) -> bool:
        return self._units_per_physics_step() > 0

    def _next_squad_cursor(self, sq_idx: int, unit_idx: int) -> Tuple[int, int]:
        """Return (squad_index, unit_index) for the next non-empty squad."""
        if not self.squad_unit_tags:
            return sq_idx, unit_idx
        n = len(self.squad_unit_tags)
        for _ in range(n):
            sq_idx = (sq_idx + 1) % n
            _, tags = self.squad_unit_tags[sq_idx]
            if tags:
                return sq_idx, 0
        return sq_idx, unit_idx

    def _peek_acting_chunk(self) -> List[_ChunkEntry]:
        """Collect up to ``units_per_physics_step`` units from the global actor queue."""
        max_n = self._units_per_physics_step()
        if max_n <= 0 or not self.squad_unit_tags:
            return []

        entries: List[_ChunkEntry] = []
        sq_idx = self.current_squad_index
        unit_idx = self.current_unit_index
        start_sq = sq_idx
        start_unit = unit_idx

        while len(entries) < max_n:
            player_id, tags = self.squad_unit_tags[sq_idx]
            if not tags:
                sq_idx, unit_idx = self._next_squad_cursor(sq_idx, unit_idx)
                if sq_idx == start_sq and unit_idx == start_unit:
                    break
                continue
            if unit_idx >= len(tags):
                unit_idx = 0
                sq_idx, unit_idx = self._next_squad_cursor(sq_idx, unit_idx)
                if sq_idx == start_sq and unit_idx == start_unit:
                    break
                continue

            pool_conf = (
                self.unit_pools_config[sq_idx]
                if sq_idx < len(self.unit_pools_config)
                else {}
            )
            is_agent = (
                self._is_pool_trainee(pool_conf)
                if pool_conf
                else player_id == 1
            )
            entries.append(
                _ChunkEntry(sq_idx, player_id, int(tags[unit_idx]), is_agent)
            )
            unit_idx += 1
            if unit_idx >= len(tags):
                unit_idx = 0
                sq_idx, unit_idx = self._next_squad_cursor(sq_idx, unit_idx)
            if sq_idx == start_sq and unit_idx == start_unit:
                break

        return entries

    def _advance_cursor_by(self, n: int) -> None:
        """Advance the global actor cursor by ``n`` units."""
        if n <= 0:
            return
        consumed = 0
        guard = 0
        max_guard = n * max(1, len(self.squad_unit_tags)) + len(self.squad_unit_tags) + 1

        while consumed < n and guard < max_guard:
            guard += 1
            if not self.squad_unit_tags:
                break
            _, tags = self.squad_unit_tags[self.current_squad_index]
            if not tags:
                self.current_squad_index, self.current_unit_index = self._next_squad_cursor(
                    self.current_squad_index, self.current_unit_index
                )
                continue
            if self.current_unit_index >= len(tags):
                self.current_unit_index = 0
                self.current_squad_index, self.current_unit_index = self._next_squad_cursor(
                    self.current_squad_index, self.current_unit_index
                )
                continue

            self.current_unit_index += 1
            consumed += 1
            if self.current_unit_index >= len(tags):
                self.current_unit_index = 0
                self.current_squad_index, self.current_unit_index = self._next_squad_cursor(
                    self.current_squad_index, self.current_unit_index
                )

        if self.squad_unit_tags:
            self.active_player_id = self.squad_unit_tags[self.current_squad_index][0]

    def requires_policy_action(self) -> bool:
        """True when the collector must run a policy forward for this env step.

        Also returns True for the phantom terminal step that flushes a deferred
        terminal reward (from an episode that ended during an opponent's physics
        step) into the training buffer.
        """
        if getattr(self, "_deferred_done", False) or getattr(self, "_deferred_truncated", False):
            return True
        if self._uses_chunk_physics():
            chunk = self._prepared_chunk or self._peek_acting_chunk()
            return any(entry.is_agent for entry in chunk)
        if not self._is_agent_squad():
            return False
        return len(self._squad_tags()) > 0

    async def _get_joint_observation(self):
        player_id, tags = self._current_squad()
        if tags and self.current_unit_index < len(tags):
            self.bot.active_unit_tag = tags[self.current_unit_index]
        return self.bot.get_observation(perspective_player_id=player_id)

    def preview_tick(self):
        """Advance SC2 one frame without policy actions (debug UI training-level preview)."""
        return self._loop.run_until_complete(self._async_preview_tick())

    def _ensure_preview_focal_unit(self) -> None:
        """Pick a focal unit so the tactical fovea is populated in debug preview."""
        if self.bot is None:
            return
        if self.bot.active_unit_tag and self.bot.all_units.find_by_tag(self.bot.active_unit_tag):
            return
        _, tags = self._current_squad()
        if tags:
            self.bot.active_unit_tag = tags[0]
            return
        for tag in self.player_units.get(1, []):
            pool_idx = self.unit_pool_index.get(tag)
            if (
                pool_idx is not None
                and self.unit_pools_config
                and self._is_pool_trainee(self.unit_pools_config[pool_idx])
            ):
                self.bot.active_unit_tag = tag
                return
        if self.player_units.get(1):
            self.bot.active_unit_tag = self.player_units[1][0]

    async def _async_preview_tick(self):
        await self.bot.client.step()
        await self._refresh_bot_state()
        await self._refresh_unit_lists()
        self._ensure_preview_focal_unit()
        return await self._get_joint_observation()

    def _opponent_checkpoint_dir(self) -> str:
        from src.utils.async_buffer_paths import resolve_async_buffer_dir

        return resolve_async_buffer_dir()

    def _opponent_checkpoint_candidates(
        self, explicit_path: Optional[str] = None
    ) -> List[str]:
        if explicit_path:
            return [explicit_path]
        temp_dir = self._opponent_checkpoint_dir()
        # Prefer the learner's published checkpoint (refreshed on hard reset via
        # every_k_episodes). ``initial_model`` is only a bootstrap before the first
        # learner publish. Staging zips are fallbacks when the primary zip is locked.
        return [
            os.path.join(temp_dir, "latest_learner_model.zip"),
            os.path.join(temp_dir, "latest_learner_model_collector_staging.zip"),
            os.path.join(temp_dir, "initial_model.zip"),
            os.path.join(temp_dir, "collector_weights_staging.zip"),
        ]

    def _resolve_opponent_model_conf(self) -> Dict[str, Any]:
        """Model section for opponent load — workers may only have training+env flattened."""
        model_conf = self.training_conf.get("model")
        if model_conf:
            return model_conf
        level_name = self.training_conf.get("level_name", getattr(self, "level_name", None))
        if level_name and hasattr(self, "config_manager"):
            return self.config_manager.get_level_config(level_name).get("model", {})
        return {}

    def _resolve_opponent_algo_bundle(self) -> Tuple[Any, Any, Dict[str, Any], str]:
        """Return (AlgoClass, PolicyClass, policy_kwargs, rnn_type) for opponent load."""
        from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO
        from src.agent.maskable_ppo import MaskablePPO
        from src.agent.models.unit_control_network import UnitControlNetwork
        from src.agent.models.universal_recurrent_policy import UniversalRecurrentPolicy
        from src.agent.models.universal_policy import UniversalActorCriticPolicy

        model_conf = self._resolve_opponent_model_conf()
        algo_name = model_conf.get("algorithm", "MaskablePPO")
        rnn_type = model_conf.get("rnn_type", "gru")
        is_recurrent_algo = algo_name == "MaskableRecurrentPPO"

        policy_kwargs = {**model_conf.get("policy_kwargs", {}), "normalize_images": False}
        policy_kwargs["spatial_resolution"] = 32
        fe_class = policy_kwargs.get("features_extractor_class")
        if fe_class in (
            "src.agent.models.unit_control_network.UnitControlNetwork",
            UnitControlNetwork,
        ):
            policy_kwargs["features_extractor_class"] = UnitControlNetwork
        if is_recurrent_algo:
            policy_kwargs["rnn_type"] = rnn_type

        if is_recurrent_algo:
            return MaskableRecurrentPPO, UniversalRecurrentPolicy, policy_kwargs, rnn_type
        return MaskablePPO, UniversalActorCriticPolicy, policy_kwargs, rnn_type

    def _build_opponent_algo_kwargs(self) -> Dict[str, Any]:
        """Fresh-model kwargs for curriculum graft when direct SB3 load fails."""
        from src.utils.training_device import resolve_opponent_device

        AlgoClass, PolicyClass, policy_kwargs, rnn_type = self._resolve_opponent_algo_bundle()
        training_params = self.training_conf.get("training", self.training_conf)
        model_conf = self._resolve_opponent_model_conf()
        algo_kwargs: Dict[str, Any] = {
            "policy": PolicyClass,
            "env": self,
            "policy_kwargs": policy_kwargs,
            "n_steps": int(training_params.get("n_steps", 2048)),
            "batch_size": int(training_params.get("batch_size", 512)),
            "gamma": float(model_conf.get("gamma", 0.99)),
            "device": resolve_opponent_device(training_params),
        }
        if AlgoClass.__name__ == "MaskableRecurrentPPO":
            algo_kwargs["rnn_type"] = rnn_type
        return algo_kwargs

    def _build_opponent_model_load_kwargs(self) -> Dict[str, Any]:
        """Match collector/learner load kwargs so checkpoint shapes align (512-dim LSTM, UCN)."""
        from src.agent.models.universal_recurrent_policy import UniversalRecurrentPolicy
        from src.utils.training_device import resolve_opponent_device

        _algo, _policy, policy_kwargs, rnn_type = self._resolve_opponent_algo_bundle()
        model_conf = self._resolve_opponent_model_conf()
        algo_name = model_conf.get("algorithm", "MaskablePPO")

        custom_objects = {
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.2,
            "policy_class": UniversalRecurrentPolicy,
            "policy_kwargs": policy_kwargs,
            "rnn_type": rnn_type,
            "optimizer": None,
        }
        training_params = self.training_conf.get("training", self.training_conf)
        load_kwargs: Dict[str, Any] = {
            "env": self,
            "device": resolve_opponent_device(training_params),
            "custom_objects": custom_objects,
            "policy_kwargs": policy_kwargs,
        }
        if "Recurrent" in algo_name:
            load_kwargs["rnn_type"] = rnn_type
        return load_kwargs

    def sync_opponent_model(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Load opponent weights from disk inside this env worker.

        Returns True when ``opponent_model`` is ready. Safe to call from SubprocVecEnv
        workers (picklable path string only — do not pass live model objects).
        """
        return self._sync_opponent_model(checkpoint_path=checkpoint_path)

    def _sync_opponent_model(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Load the latest learner checkpoint from disk as the frozen opponent snapshot.
        """
        candidates = self._opponent_checkpoint_candidates(checkpoint_path)
        existing = [p for p in candidates if os.path.exists(p)]
        if not existing:
            self._log(
                "Opponent sync: no checkpoint found yet "
                f"(searched {candidates}) — skipping."
            )
            return False

        from src.utils.model_utils import load_curriculum_checkpoint

        AlgoClass, _, _policy_kwargs, _rnn = self._resolve_opponent_algo_bundle()
        algo_kwargs = self._build_opponent_algo_kwargs()
        load_kwargs = self._build_opponent_model_load_kwargs()
        grid_size = self._opponent_graft_grid_size()

        errors: List[str] = []
        for model_path in existing:
            try:
                new_model = load_curriculum_checkpoint(
                    AlgoClass,
                    model_path,
                    algo_kwargs,
                    load_kwargs,
                    self.observation_space,
                    self.action_space,
                    grid_size=grid_size,
                )
                new_model.policy.set_training_mode(False)
                self.opponent_model = new_model
                self._opponent_lstm_state = None
                self._opponent_last_focus_tag = None
                self._opponent_infer_source = "frozen"
                self._logged_no_opponent_model = False
                self._log(
                    f"Opponent synced from '{os.path.basename(model_path)}' "
                    f"(episode {self.episode_counter})."
                )
                return True
            except Exception as e:
                import traceback
                errors.append(
                    f"{os.path.basename(model_path)}: {e}\n{traceback.format_exc()}"
                )

        self._log(
            "Opponent sync failed for all checkpoint candidates "
            f"(will retry next hard reset):\n" + "\n---\n".join(errors)
        )
        return False

    async def _predict_opponent_squad_batch(self, tags: List[int]) -> List[np.ndarray]:
        """Batched opponent inference for every unit in the acting squad."""
        from train.squad_batch_inference import opponent_squad_batch_predict

        if not tags:
            return []

        player_id = self.active_player_id
        obs_list, mask_list = self._build_squad_obs_and_masks(tags, player_id)
        default = universal_fallback_action()

        if self.opponent_model is None:
            self._ensure_opponent_model_loaded()

        opp_model, source = self._resolve_opponent_model()
        if opp_model is None:
            if not getattr(self, "_logged_no_opponent_model", False):
                self._log(
                    "Opponent predict: no opponent_model or collector model — "
                    "using fallback move_global + center spatial (not map corner)."
                )
                self._logged_no_opponent_model = True
            return [default.copy() for _ in tags]

        if getattr(self, "_opponent_infer_source", None) != source:
            self._opponent_lstm_state = None
            self._opponent_last_focus_tag = None
            self._opponent_infer_source = source

        opp_model.policy.set_training_mode(False)
        actions, new_state, self._opponent_last_focus_tag = opponent_squad_batch_predict(
            opp_model,
            obs_list,
            mask_list,
            lstm_state=self._opponent_lstm_state,
            deterministic=True,
            focus_tags=[int(t) for t in tags],
            last_focus_tag=getattr(self, "_opponent_last_focus_tag", None),
        )
        self._opponent_lstm_state = new_state
        if actions.ndim == 1:
            return [actions]
        return [actions[i] for i in range(len(tags))]

    async def _apply_fresh_opponent_actions(self) -> None:
        """
        Chunk physics: every opponent unit gets a fresh policy action each step.

        Partial chunks only refresh a subset during apply; re-issuing stale move
        orders for the rest made opponents drift to spatial cell 0 (map corner).
        """
        if self._preview_computer_opponent():
            return
        if self.opponent_model is None:
            self._ensure_opponent_model_loaded()
        tags = list(self.player_units.get(2, []))
        if not tags:
            return

        prev_player = self.active_player_id
        self.active_player_id = 2
        try:
            actions = await self._predict_opponent_squad_batch(tags)
            for tag, action in zip(tags, actions):
                self.bot.active_unit_tag = tag
                self.bot.record_pre_action_retreat_state(tag)
                await self.bot.apply_action(tag, action)
        finally:
            self.active_player_id = prev_player

    async def _predict_opponent_chunk_batch(
        self, entries: List[_ChunkEntry]
    ) -> List[np.ndarray]:
        """Batched opponent inference for opponent entries in a mixed chunk."""
        from train.squad_batch_inference import opponent_squad_batch_predict

        if not entries:
            return []

        obs_list, mask_list = [], []
        for entry in entries:
            self.bot.active_unit_tag = entry.tag
            obs_list.append(
                self.bot.get_observation(perspective_player_id=entry.player_id)
            )
            mask_list.append(self.bot.action_masks())

        default = universal_fallback_action()
        if self.opponent_model is None:
            self._ensure_opponent_model_loaded()
        opp_model, source = self._resolve_opponent_model()
        if opp_model is None:
            return [default.copy() for _ in entries]

        if getattr(self, "_opponent_infer_source", None) != source:
            self._opponent_lstm_state = None
            self._opponent_last_focus_tag = None
            self._opponent_infer_source = source

        opp_model.policy.set_training_mode(False)
        actions, new_state, self._opponent_last_focus_tag = opponent_squad_batch_predict(
            opp_model,
            obs_list,
            mask_list,
            lstm_state=self._opponent_lstm_state,
            deterministic=True,
            focus_tags=[entry.tag for entry in entries],
            last_focus_tag=getattr(self, "_opponent_last_focus_tag", None),
        )
        self._opponent_lstm_state = new_state
        if actions.ndim == 1:
            return [actions]
        return [actions[i] for i in range(len(entries))]

    def _build_squad_obs_and_masks(
        self, tags: List[int], player_id: int
    ) -> Tuple[List[dict], List[np.ndarray]]:
        obs_list, mask_list = [], []
        for tag in tags:
            self.bot.active_unit_tag = tag
            obs_list.append(self.bot.get_observation(perspective_player_id=player_id))
            mask_list.append(self.bot.action_masks())
        return obs_list, mask_list

    def set_squad_batch_actions(self, actions: np.ndarray) -> None:
        """Apply agent unit actions on the next ``step()`` (squad or chunk batch)."""
        self._squad_batch_actions = np.asarray(actions, dtype=np.int64)

    def prepare_acting_chunk(self) -> Optional[dict]:
        """
        Build obs/masks for every agent unit in the next physics chunk.
        Stores the full chunk (agent + opponent) in ``_prepared_chunk``.
        """
        if not self._uses_chunk_physics():
            return None

        chunk = self._peek_acting_chunk()
        if not chunk:
            self._prepared_chunk = None
            return None

        agent_entries = [e for e in chunk if e.is_agent]
        if not agent_entries:
            self._prepared_chunk = chunk
            return None

        obs_list, mask_list, focus_tags = [], [], []
        for entry in agent_entries:
            self.bot.active_unit_tag = entry.tag
            obs_list.append(
                self.bot.get_observation(perspective_player_id=entry.player_id)
            )
            mask_list.append(self.bot.action_masks())
            focus_tags.append(entry.tag)

        self._prepared_chunk = chunk
        return {
            "obs": obs_list,
            "masks": mask_list,
            "focus_tags": focus_tags,
            "chunk_size": len(chunk),
        }

    def prepare_policy_batch(self) -> Optional[dict]:
        """Unified batching hook for squad-atomic or chunk-physics modes."""
        if self._uses_chunk_physics():
            return self.prepare_acting_chunk()
        return self.prepare_squad_policy_batch()

    def prepare_squad_policy_batch(self) -> Optional[dict]:
        """
        Build obs/masks for every unit in the active agent squad when at squad start.
        Returns None when batching is not applicable (opponent squad, partial round, size 1).
        """
        if not self._is_agent_squad() or self.current_unit_index != 0:
            return None
        tags = self._squad_tags()
        if len(tags) <= 1:
            return None
        player_id, _ = self._current_squad()
        obs_list, mask_list = self._build_squad_obs_and_masks(tags, player_id)
        return {
            "obs": obs_list,
            "masks": mask_list,
            "focus_tags": [int(t) for t in tags],
        }

    async def _predict_opponent_action(self):
        """Run frozen opponent model for the current active_unit_tag."""
        obs_p2 = await self._get_joint_observation()
        default = universal_fallback_action()
        opp_model, source = self._resolve_opponent_model()
        if opp_model is None:
            return default

        if getattr(self, "_opponent_infer_source", None) != source:
            self._opponent_lstm_state = None
            self._opponent_last_focus_tag = None
            self._opponent_infer_source = source

        predict_kwargs: dict = dict(
            state=self._opponent_lstm_state,
            deterministic=True,
        )
        opp_model.policy.set_training_mode(False)
        try:
            masks_p2 = self.bot.action_masks()
            action, self._opponent_lstm_state = opp_model.predict(
                obs_p2, action_masks=masks_p2, **predict_kwargs
            )
        except TypeError:
            action, self._opponent_lstm_state = opp_model.predict(
                obs_p2, **predict_kwargs
            )
        return action

    async def _apply_action(self, action):
        """Apply a policy action to the current unit in the active agent squad."""
        _, tags = self._current_squad()
        if not tags or self.current_unit_index >= len(tags):
            return
        await self.bot.apply_action(tags[self.current_unit_index], action)

    async def _reissue_non_squad_orders(
        self, acting_tag_set: set, skip_player_ids: Optional[set] = None
    ) -> None:
        """Re-issue last known orders for alive units outside the acting set."""
        skip_player_ids = skip_player_ids or set()
        all_alive = self.player_units[1] + self.player_units[2]
        for tag in all_alive:
            if tag in acting_tag_set:
                continue
            last_order = self.bot._unit_last_order.get(tag)
            if last_order is None:
                continue
            unit = self.bot.all_units.find_by_tag(tag)
            if unit is None or unit.owner_id in skip_player_ids:
                continue
            head_name, target = last_order
            if "attack" in head_name:
                self.bot.do(unit.attack(target))
            else:
                self.bot.do(unit.move(target))

    async def _flush_bot_actions(self) -> None:
        if hasattr(self.bot, "actions") and self.bot.actions:
            await self.bot.client.actions(self.bot.actions)
            self.bot.actions = []

    async def _advance_squad_index(self) -> None:
        """Move to the next non-empty squad in pool order."""
        self.current_unit_index = 0
        if self.squad_unit_tags:
            await self._refresh_unit_lists()
            n = len(self.squad_unit_tags)
            for _ in range(n):
                self.current_squad_index = (self.current_squad_index + 1) % n
                sq_player, sq_tags = self.squad_unit_tags[self.current_squad_index]
                if sq_tags:
                    self.active_player_id = sq_player
                    break
        else:
            self.active_player_id = 2 if self.active_player_id == 1 else 1
            await self._refresh_unit_lists()

    async def _run_opponent_squad_batch(self):
        """Apply fresh opponent actions for every unit in the squad, then one physics step."""
        acting_player_id = self.active_player_id
        _, tags = self._current_squad()
        if not tags:
            await self._advance_squad_index()
            obs = await self._get_joint_observation()
            return obs, 0.0, False, False, {"squad_physics_step": True}

        if self._preview_computer_opponent():
            return await self._finalize_squad_physics_step(tags, acting_player_id)

        for tag in tags:
            self.bot.active_unit_tag = tag
            self.bot.record_pre_action_retreat_state(tag)

        actions = await self._predict_opponent_squad_batch(tags)
        for tag, action in zip(tags, actions):
            await self.bot.apply_action(tag, action)

        return await self._finalize_squad_physics_step(tags, acting_player_id)

    def _physics_step_size(self, acting_unit_count: int) -> int:
        env_conf = self.training_conf.get("env", {})
        per_unit = int(
            self.training_conf.get("step_mul_per_unit")
            or env_conf.get("step_mul_per_unit", DEFAULT_STEP_MUL_PER_UNIT)
        )
        return max(1, per_unit * max(1, acting_unit_count))

    async def _finalize_chunk_physics_step(
        self, chunk: List[_ChunkEntry], acting_player_id: int
    ):
        """Flush chunk orders, advance SC2 once, compute reward/termination."""
        self.steps_count += 1
        chunk_tag_set = {entry.tag for entry in chunk}
        agent_tags = [entry.tag for entry in chunk if entry.is_agent]

        await self._flush_bot_actions()
        reissue_skip = {2} if self._uses_chunk_physics() else set()
        await self._reissue_non_squad_orders(chunk_tag_set, skip_player_ids=reissue_skip)
        await self._flush_bot_actions()

        await self.bot.client.step(step_size=self._physics_step_size(len(chunk)))
        await self._refresh_bot_state()

        self._maybe_promote_defender_to_attack()

        reward_tags = agent_tags if agent_tags else [entry.tag for entry in chunk]
        reward = self.bot.calculate_squad_step_reward(
            reward_tags, perspective_player_id=acting_player_id
        )

        p1_lost = self.bot.structures(UnitTypeId.PYLON).amount == 0
        p2_lost = self.bot.enemy_structures(UnitTypeId.PYLON).amount == 0
        p1_army_lost = not self._has_combat_units(owner_id=1)
        p2_army_lost = not self._has_combat_units(owner_id=2)

        done = False
        truncated = False
        r_conf = self.training_conf.get("rewards", {})
        win_reward = r_conf.get("win_reward", 100.0)
        loss_penalty = r_conf.get("loss_penalty", -100.0)
        draw_penalty = r_conf.get("draw_penalty", -10.0)

        if p1_lost:
            done = True
            reward += win_reward if acting_player_id == 2 else loss_penalty
        elif p2_lost:
            done = True
            reward += win_reward if acting_player_id == 1 else loss_penalty
        elif p1_army_lost and p2_army_lost:
            done = True
            reward += draw_penalty
        elif p1_army_lost:
            done = True
            reward += loss_penalty if acting_player_id == 1 else win_reward
        elif p2_army_lost:
            done = True
            reward += win_reward if acting_player_id == 1 else loss_penalty

        await self._refresh_unit_lists()
        p1_units_count = len(self.player_units[1])
        p2_units_count = len(self.player_units[2])

        if not done and p1_units_count == 0 and p2_units_count == 0:
            done = True
            reward += draw_penalty

        if not done and self.steps_count >= self.max_steps:
            truncated = True
            if self._agent_attack_units_alive() > 0:
                reward += loss_penalty
            else:
                reward += draw_penalty

        obs = await self._get_joint_observation()
        return obs, reward, done, truncated, {"squad_physics_step": True}

    def _compute_p1_terminal_reward(
        self,
        p1_lost: bool,
        p2_lost: bool,
        p1_army_lost: bool,
        p2_army_lost: bool,
        all_units_gone: bool,
        timeout: bool,
        r_conf: dict,
    ) -> float:
        """P1 (agent) perspective terminal reward for any termination condition."""
        win_reward = r_conf.get("win_reward", 100.0)
        loss_penalty = r_conf.get("loss_penalty", -100.0)
        draw_penalty = r_conf.get("draw_penalty", -10.0)

        if p1_lost:
            return loss_penalty
        if p2_lost:
            return win_reward
        if p1_army_lost and p2_army_lost:
            return draw_penalty
        if all_units_gone:
            return draw_penalty
        if p1_army_lost:
            return loss_penalty
        if p2_army_lost:
            return win_reward
        if timeout:
            if self._agent_attack_units_alive() > 0:
                return loss_penalty
            return draw_penalty
        return 0.0

    async def _finalize_squad_physics_step(
        self, squad_tags: List[int], acting_player_id: int
    ):
        """Flush squad orders, advance SC2 once, compute reward/termination, advance squad.

        Terminal reward correctness guarantee
        ────────────────────────────────────
        When the game ends during the *opponent's* physics step
        (``acting_player_id != 1``) the training buffer has no open slot for
        that step (``requires_policy_action`` returns ``False``).  To ensure the
        terminal reward is captured from P1's perspective we defer it: we store
        the P1-perspective terminal reward and the terminal observation, clear
        the done/truncated flags in the returned tuple, and let
        ``requires_policy_action`` return ``True`` on the next call so the
        collector runs a phantom agent step whose *only* purpose is to flush the
        stored reward into the training buffer.
        """
        self.steps_count += 1
        squad_tag_set = set(squad_tags)

        await self._flush_bot_actions()
        await self._reissue_non_squad_orders(squad_tag_set)
        await self._flush_bot_actions()

        await self.bot.client.step(step_size=self._physics_step_size(len(squad_tags)))
        await self._refresh_bot_state()

        self._maybe_promote_defender_to_attack()

        reward = self.bot.calculate_squad_step_reward(
            squad_tags, perspective_player_id=acting_player_id
        )

        p1_lost = self.bot.structures(UnitTypeId.PYLON).amount == 0
        p2_lost = self.bot.enemy_structures(UnitTypeId.PYLON).amount == 0
        p1_army_lost = not self._has_combat_units(owner_id=1)
        p2_army_lost = not self._has_combat_units(owner_id=2)

        done = False
        truncated = False
        r_conf = self.training_conf.get("rewards", {})
        win_reward = r_conf.get("win_reward", 100.0)
        loss_penalty = r_conf.get("loss_penalty", -100.0)
        draw_penalty = r_conf.get("draw_penalty", -10.0)

        if p1_lost:
            done = True
            reward += win_reward if acting_player_id == 2 else loss_penalty
        elif p2_lost:
            done = True
            reward += win_reward if acting_player_id == 1 else loss_penalty
        elif p1_army_lost and p2_army_lost:
            done = True
            reward += draw_penalty
        elif p1_army_lost:
            done = True
            reward += loss_penalty if acting_player_id == 1 else win_reward
        elif p2_army_lost:
            done = True
            reward += win_reward if acting_player_id == 1 else loss_penalty

        await self._refresh_unit_lists()
        p1_units_count = len(self.player_units[1])
        p2_units_count = len(self.player_units[2])

        all_units_gone = not done and p1_units_count == 0 and p2_units_count == 0
        if all_units_gone:
            done = True
            reward += draw_penalty

        timeout = not done and self.steps_count >= self.max_steps
        if timeout:
            truncated = True
            if self._agent_attack_units_alive() > 0:
                reward += loss_penalty
            else:
                reward += draw_penalty

        # Advance the squad index BEFORE generating the observation.
        # The observation returned from this step is what the collector uses as
        # input to compute the NEXT policy action.  If the next actor is P1
        # squad B but we return P2 squad A's spatially-flipped view, the policy
        # will navigate in completely the wrong direction.  By advancing first,
        # _get_joint_observation always reflects the next acting squad's
        # perspective.
        # Guard: do NOT advance when the episode is ending — terminal obs stays
        # on the current (terminating) squad; reset() will supply a fresh start.
        if not done and not truncated:
            await self._advance_squad_index()

        obs = await self._get_joint_observation()

        # When the episode ends during the opponent's physics step, the
        # training buffer has no slot for this step.  Defer the terminal state
        # so the next phantom agent step carries the P1 reward into the buffer.
        if (done or truncated) and acting_player_id != 1:
            p1_terminal = self._compute_p1_terminal_reward(
                p1_lost, p2_lost, p1_army_lost, p2_army_lost,
                all_units_gone, timeout, r_conf,
            )
            self._deferred_done = done
            self._deferred_truncated = truncated
            self._deferred_p1_terminal_reward = p1_terminal
            self._deferred_terminal_obs = obs
            return obs, 0.0, False, False, {"squad_physics_step": True, "deferred_terminal": True}

        return obs, reward, done, truncated, {"squad_physics_step": True}

    async def _async_step_chunk(self, action_p_active):
        """Apply a fixed-size unit chunk (possibly spanning squads), then one physics step."""
        chunk = self._prepared_chunk or self._peek_acting_chunk()
        if not chunk:
            await self._advance_squad_index()
            obs = await self._get_joint_observation()
            return obs, 0.0, False, False, {"squad_physics_step": False}

        agent_entries = [e for e in chunk if e.is_agent]
        acting_player_id = 1 if agent_entries else chunk[0].player_id

        batch_actions = getattr(self, "_squad_batch_actions", None)
        if batch_actions is not None:
            self._squad_batch_actions = None
            batch_actions = np.asarray(batch_actions, dtype=np.int64)
            if batch_actions.ndim == 1:
                batch_actions = batch_actions.reshape(1, -1)
            if len(batch_actions) != len(agent_entries):
                raise ValueError(
                    f"Chunk batch size mismatch: {len(batch_actions)} actions for "
                    f"{len(agent_entries)} agent units in chunk of {len(chunk)}"
                )

        agent_action_idx = 0
        for entry in chunk:
            if not entry.is_agent:
                continue
            self.bot.active_unit_tag = entry.tag
            self.active_player_id = entry.player_id
            self.bot.record_pre_action_retreat_state(entry.tag)
            if batch_actions is not None:
                action = batch_actions[agent_action_idx]
                agent_action_idx += 1
            else:
                action = action_p_active
            await self.bot.apply_action(entry.tag, action)

        await self._apply_fresh_opponent_actions()

        self._prepared_chunk = None
        self._advance_cursor_by(len(chunk))
        return await self._finalize_chunk_physics_step(chunk, acting_player_id)

    def reset(self, seed=None, options=None):
        self.episode_counter += 1
        self.steps_count = 0
        self.current_unit_index = 0
        self.current_squad_index = 0
        self.active_player_id = 1
        self._promoted_pool_index = None
        self._opponent_lstm_state = None  # always reset LSTM state between episodes
        self._opponent_last_focus_tag = None
        self._squad_batch_actions = None
        self._prepared_chunk = None
        # Clear any deferred terminal state from the previous episode.
        self._deferred_done = False
        self._deferred_truncated = False
        self._deferred_p1_terminal_reward = 0.0
        self._deferred_terminal_obs = None

        every_k = every_k_episodes(self.training_conf)

        session_alive = bool(self._manager.clients)
        is_hard_reset_due = (self.episode_counter % every_k == 0)

        if session_alive and not is_hard_reset_due:
            # --- Soft reset: keep SC2 running, just respawn units ---
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    self._loop.run_until_complete(self._soft_reset_game())
                    obs = self._loop.run_until_complete(self._get_joint_observation())
                    self._last_obs = obs
                    self.game_ended = False
                    self._log(f"Soft reset complete (episode {self.episode_counter}).")
                    if self.opponent_model is None:
                        self._ensure_opponent_model_loaded()
                    return obs, {}
                except Exception as e:
                    import traceback
                    self._log(
                        f"Soft reset failed (attempt {attempt + 1}/{max_retries}): {e}\n"
                        + traceback.format_exc()
                    )
                    if attempt == max_retries - 1:
                        self._log("Soft reset exhausted retries — falling back to hard reset.")
                        break
            # fall through to hard reset
        else:
            # --- Hard reset: full teardown + relaunch (map rotation + opponent sync) ---
            if session_alive:
                resets_done = self.episode_counter // every_k
                map_idx = (self.worker_index + resets_done * self.num_workers) % len(self.map_list)
                self.map_name = self.map_list[map_idx]
                self._log(
                    f"Hard reset at episode {self.episode_counter} "
                    f"(every_k={every_k}), map → {self.map_name}"
                )
            # Sync opponent snapshot from latest learner checkpoint
            self._sync_opponent_model()

        return super().reset(seed=seed, options=options)

    async def _soft_reset_game(self):
        """Reuse the running SC2 session: kill all units, respawn, refresh lists."""
        await self._refresh_bot_state()
        if self.bot is not None:
            self.bot.reset_episode_state()
        await self._spawn_initial_units()
        await self._refresh_unit_lists()

    def _has_combat_units(self, owner_id: int) -> bool:
        """
        Robustly checks if a player has any combat-capable mobile units left.
        """
        if not hasattr(self.bot, "all_units") or not self.bot.all_units:
            return False
            
        ignored_types = {
            UnitTypeId.LARVA, 
            UnitTypeId.EGG, 
            UnitTypeId.OVERLORD, 
            UnitTypeId.PROBE, 
            UnitTypeId.SCV, 
            UnitTypeId.DRONE,
            UnitTypeId.PYLON
        }
        
        # In SingleClientEnv (God Mode), self.bot.all_units contains ALL units on the map
        combat_units = self.bot.all_units.filter(lambda u: u.owner_id == owner_id and not u.is_structure and u.type_id not in ignored_types and u.health > 0)
        
        return combat_units.amount > 0

    async def _async_step(self, action_p_active):
        # Phantom terminal step: flush deferred reward into the training buffer.
        # This fires when the episode ended during the opponent's previous physics
        # step.  No SC2 advance is needed — just return the cached terminal state
        # with the P1-perspective reward so the collector can write it to the
        # rollout buffer with policy_active=True.
        if getattr(self, "_deferred_done", False) or getattr(self, "_deferred_truncated", False):
            obs = self._deferred_terminal_obs
            reward = self._deferred_p1_terminal_reward
            done = self._deferred_done
            truncated = self._deferred_truncated
            self._deferred_done = False
            self._deferred_truncated = False
            self._deferred_p1_terminal_reward = 0.0
            self._deferred_terminal_obs = None
            return obs, reward, done, truncated, {"squad_physics_step": True, "phantom_terminal": True}

        if self._uses_chunk_physics():
            return await self._async_step_chunk(action_p_active)

        if not self._is_agent_squad():
            return await self._run_opponent_squad_batch()

        _, tags = self._current_squad()
        if not tags:
            await self._advance_squad_index()
            obs = await self._get_joint_observation()
            return obs, 0.0, False, False, {"squad_physics_step": False}

        acting_player_id = self.active_player_id
        batch_actions = getattr(self, "_squad_batch_actions", None)
        if batch_actions is not None:
            self._squad_batch_actions = None
            batch_actions = np.asarray(batch_actions, dtype=np.int64)
            if batch_actions.ndim == 1:
                batch_actions = batch_actions.reshape(1, -1)
            if len(batch_actions) != len(tags):
                raise ValueError(
                    f"Squad batch size mismatch: {len(batch_actions)} actions for {len(tags)} units"
                )
            for tag, action in zip(tags, batch_actions):
                self.bot.active_unit_tag = tag
                self.bot.record_pre_action_retreat_state(tag)
                await self.bot.apply_action(tag, action)
            self.current_unit_index = 0
            return await self._finalize_squad_physics_step(list(tags), acting_player_id)

        unit_idx = self.current_unit_index
        tag = tags[unit_idx]

        self.bot.record_pre_action_retreat_state(tag)
        await self.bot.apply_action(tag, action_p_active)

        self.current_unit_index += 1

        if self.current_unit_index < len(tags):
            obs = await self._get_joint_observation()
            return obs, 0.0, False, False, {"squad_physics_step": False}

        completed_tags = tags
        self.current_unit_index = 0
        return await self._finalize_squad_physics_step(completed_tags, acting_player_id)

    def get_obs_focus_unit_tag(self) -> int:
        """
        Tag of the unit whose egocentric observation the policy will receive next.
        Used by the recurrent collector to reset LSTM/GRU hidden state when the
        focal unit changes between steps.  Returns -1 if unknown.
        """
        _, tags = self._current_squad()
        if tags and self.current_unit_index < len(tags):
            return int(tags[self.current_unit_index])
        return -1

    def action_masks(self) -> np.ndarray:
        if self.bot:
            return self.bot.action_masks()
        if isinstance(self.action_space, spaces.MultiDiscrete):
            return np.ones(int(np.sum(self.action_space.nvec)), dtype=bool)
        return np.ones(self.action_space.n, dtype=bool)
