from typing import Dict, Optional, Any, Tuple, List, Iterable
import math
import numpy as np
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.data import Race
from loguru import logger

from src.hierarchical_agent import HierarchicalAgentSystem
from src.utils.perspective_transformer import PerspectiveTransformer
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM

class CaptureTheFlagBot(BotAI):
    """
    Bot for Capture the Flag environment.
    Supports individual unit control and reporting state to the environment.
    """
    # Matches Level7Bot / ObservationTransformer tactical crop (32×32 centered on unit).
    _TACTICAL_FOVEA_MAX_DELTA: float = 17.0
    # Weapon range at or below this is treated as melee (no cooldown-kite shaping).
    _RANGED_WEAPON_RANGE_MIN: float = 1.5
    def __init__(self, player_id: int, training_conf: Optional[Dict] = None):
        super().__init__()
        self.player_id = player_id
        self.training_conf = training_conf or {}
        
        env_conf = self.training_conf.get("env", {})
        self.grid_size = self.training_conf.get("grid_size", env_conf.get("grid_size", 128))
        
        self.agent_system: Optional[HierarchicalAgentSystem] = None
        
        self.active_unit_tag: Optional[int] = None
        self.enemy_pylon_pos: Dict[int, Optional[Point2]] = {1: None, 2: None}
        
        # Asymmetric Self-Play attributes
        self.is_trainee = True # Default to trainee
        self.model = None # Snapshot model for automated inference
        
        # State tracking for rewards
        self.prev_pylon_health = {1: 20.0, 2: 20.0} # Combined HP + Shield
        self.prev_enemy_units_health = {1: 0.0, 2: 0.0}
        self.prev_own_units_health = {1: 0.0, 2: 0.0}
        self.prev_dist_to_pylon = {1: None, 2: None}
        self.prev_dist_per_unit: Dict[int, Optional[float]] = {}
        # Squad CoM distance tracking: (player_id, role) → float distance
        # Updated whenever that role's units are the active squad.
        self.prev_dist_per_squad: Dict[Tuple[int, str], Optional[float]] = {}

        # Maps unit tag → "attack" or "defend" — populated by the env after
        # region-based spawning so calculate_reward can apply role-specific
        # per-step signals.  Falls back to "attack" for unknown tags.
        self.unit_command_map: Dict[int, str] = {}

        # Maps unit tag → explicit objective Point2 — set at spawn time from
        # region pair data.  For attackers: the enemy defender's pylon position
        # in their arena.  For defenders: their own pylon position in their arena.
        # Both map to the _d center of the assigned pair.
        self.unit_target_map: Dict[int, Any] = {}

        # Maps unit tag → (head_name, Point2 target) of the last decoded SC2 order.
        # Re-issued every game step to non-active units so they never sit idle
        # while waiting for their turn in the cycling round-robin.
        self._unit_last_order: Dict[int, Tuple[str, Any]] = {}

        # Cooldown-retreat shaping — populated by record_pre_action_retreat_state()
        # before apply_action so the delta is single-step accurate in squad cycling.
        self._pre_action_weapon_cd_by_unit: Dict[int, float] = {}
        self._pre_action_dist_to_enemy_com: Dict[int, float] = {}
        self._pre_action_dist_to_defend_pylon: Dict[int, float] = {}
        self.last_action_was_move_by_unit: Dict[int, bool] = {}
        
        self.frame_count = 0
        self.cached_observation: Optional[np.ndarray] = None

        # enemy_race must be set to avoid AttributeError in BotAIInternal._prepare_step
        self._enemy_race = Race.Random
        
        # Satisfaction for BotAI initialization
        self._initialize_variables()

    def reset_episode_state(self):
        """Reset all per-episode tracking so reward baselines don't bleed across episodes."""
        self.active_unit_tag = None
        self.enemy_pylon_pos = {1: None, 2: None}
        self.prev_pylon_health = {1: 20.0, 2: 20.0}
        self.prev_enemy_units_health = {1: 0.0, 2: 0.0}
        self.prev_own_units_health = {1: 0.0, 2: 0.0}
        self.prev_dist_to_pylon = {1: None, 2: None}
        self.prev_dist_per_unit = {}
        self.unit_command_map = {}
        self.unit_target_map = {}
        self._unit_last_order = {}
        self.prev_dist_per_squad = {}
        self._pre_action_weapon_cd_by_unit = {}
        self._pre_action_dist_to_enemy_com = {}
        self._pre_action_dist_to_defend_pylon = {}
        self.last_action_was_move_by_unit = {}
        self.cached_observation = None
        self.frame_count = 0

    async def on_start(self):
        env_conf = self.training_conf.get("env", {})
        self.grid_size = self.training_conf.get("grid_size", env_conf.get("grid_size", 128))

        self.agent_system = HierarchicalAgentSystem(
            map_size=(128, 128),
            grid_size=self.grid_size,
            training_conf=self.training_conf
        )
        self.agent_system.create_unit_agent("active_agent")

    def get_observation(self, perspective_player_id: Optional[int] = None) -> np.ndarray:
        """Returns the processed observation from the given player's perspective."""
        # Use provided perspective or default to bot's own player_id
        p_id = perspective_player_id if perspective_player_id is not None else self.player_id
        
        # Defensive check for game info/state
        if not hasattr(self, "game_info") or self.game_info is None:
             logger.warning(f"Bot {self.player_id}: game_info is missing. Returning zero observation.")
             return np.zeros((23, 128, 128), dtype=np.uint8)

        # Flip units based on perspective
        if p_id == self.player_id:
            units = self.units
            enemy_units = self.enemy_units
            structures = self.structures
            enemy_structures = self.enemy_structures
        else:
            # God-Mode: Flip perspectives
            units = self.enemy_units
            enemy_units = self.units
            structures = self.enemy_structures
            enemy_structures = self.structures

        enemy_pylon = enemy_structures(UnitTypeId.PYLON).first if enemy_structures(UnitTypeId.PYLON).exists else None
        
        # Persistent enemy-pylon position tracking (fog-of-war fallback)
        if enemy_pylon:
            self.enemy_pylon_pos[p_id] = enemy_pylon.position

        active_unit = None
        selected_units = []
        if self.active_unit_tag:
            try:
                active_unit = self.all_units.find_by_tag(self.active_unit_tag)
                if active_unit:
                    selected_units = [active_unit]
            except Exception:
                pass

        # command_target: use per-unit spawn-time objective when available.
        # For attackers this is the enemy defender's pylon in their arena;
        # for defenders it is their own pylon in their arena (both = _d center).
        # Fall back to runtime pylon tracking when the map is unknown (e.g. legacy path).
        if self.active_unit_tag and self.active_unit_tag in self.unit_target_map:
            target_pos = self.unit_target_map[self.active_unit_tag]
        else:
            unit_role = self.unit_command_map.get(self.active_unit_tag) if self.active_unit_tag else None
            if unit_role == "defend":
                own_pylon = structures(UnitTypeId.PYLON).first if structures(UnitTypeId.PYLON).exists else None
                target_pos = own_pylon.position if own_pylon else None
            else:
                target_pos = self.enemy_pylon_pos[p_id]

        from src.utils.command_constants import CommandConstants

        raw_hc = self.training_conf.get("high_level_command", 0)
        if self.active_unit_tag and self.active_unit_tag in self.unit_command_map:
            raw_hc = self.unit_command_map[self.active_unit_tag]
        high_cmd = CommandConstants.get_index(raw_hc)

        raw_obs = {
            'units': units,
            'enemy_units': enemy_units,
            'enemy_structures': enemy_structures,
            'structures': structures,
            'effects': getattr(self.state, "effects", []),
            'game_info': self.game_info,
            'game_data': self.game_data,
            'creep': getattr(self.state, "creep", None),
            'command_target': target_pos,
            'selected_units': selected_units,
            'high_level_command': high_cmd,
        }

        # Process with agent system, passing active_unit for the tactical crop
        try:
            processed = self.agent_system.process_observation(
                raw_obs, player_id=p_id, active_unit=active_unit
            )
            obs = processed["active_agent"]

            # Spatial perspective flip for player 2.
            # Universal obs is a dict {map, command, tactical}; original is a plain array.
            if p_id == 2:
                if isinstance(obs, dict):
                    obs = {
                        k: PerspectiveTransformer.flip_observation(v) if v.ndim >= 2 else v
                        for k, v in obs.items()
                    }
                else:
                    obs = PerspectiveTransformer.flip_observation(obs)

            self.cached_observation = obs
        except Exception as e:
            if self.cached_observation is not None:
                logger.warning(f"Bot {self.player_id} (P_ID {p_id}): Observation processing failed ({e}). Returning cached observation.")
                return self.cached_observation
            else:
                # Fallback to zeros
                num_layers = len(self.training_conf.get("active_layers", [])) or 23
                return np.zeros((num_layers, 128, 128), dtype=np.float32)

        return self.cached_observation

    def predict_action(self) -> int:
        """
        Uses the internal snapshot model to predict an action for the current state.
        Only used if self.is_trainee is False.
        """
        if self.is_trainee:
            raise RuntimeError("Trainee bot should not predict actions; it receives them from the external trainer.")
        
        if self.model is None:
            logger.warning(f"Bot {self.player_id}: No model set for automated inference. Returning default action 0.")
            return 0

        obs = self.get_observation()
        masks = self.action_masks()
        
        # SB3 models expect a batch dimension [1, ...]
        obs_batch = np.expand_dims(obs, axis=0)
        
        # Inference (deterministic=True for stable opponent behavior)
        action, _states = self.model.predict(obs_batch, action_masks=np.expand_dims(masks, axis=0), deterministic=True)
        
        # action is typically a numpy array [action_value]
        if isinstance(action, np.ndarray):
            if action.ndim > 0:
                return int(action[0])
            return int(action)
        return int(action)

    def action_masks(self) -> np.ndarray:
        """
        Universal MultiDiscrete masking: flat [command | spatial].

        Mirrors SC2BotUniversal.action_masks — only unconfigured (padded)
        command slots are masked False; all spatial cells are open.
        """
        action_heads = self.training_conf.get("action_heads") or []
        categorical_heads = self.training_conf.get("categorical_heads") or []
        n_configured = min(
            FIXED_UNIVERSAL_COMMAND_DIM,
            len(action_heads) + len(categorical_heads),
        )
        cmd_mask = np.zeros(FIXED_UNIVERSAL_COMMAND_DIM, dtype=bool)
        cmd_mask[:n_configured] = True
        spatial_mask = np.ones(32 * 32, dtype=bool)
        return np.concatenate([cmd_mask, spatial_mask])

    async def apply_action(self, unit_tag: int, action):
        if not self.agent_system or "active_agent" not in self.agent_system.unit_agents:
            return

        unit = self.all_units.find_by_tag(unit_tag)
        if not unit:
            return

        arch = self.training_conf.get("agent_architecture", "original")

        # The observation for the opponent (player 2) is perspective-flipped 180°
        # before being presented to the policy.  The policy therefore generates a
        # spatial action index that is in *flipped* coordinates.  We must un-flip
        # it back to real-world coordinates before issuing the SC2 order.
        is_opponent_unit = unit.owner_id != self.player_id

        if arch == "universal":
            # Universal: action = [head_idx, spatial_idx] on 32×32 grid
            action = np.asarray(action, dtype=np.int64).reshape(-1)
            if action.shape[0] < 2:
                return
            head_idx = int(action[0])
            spatial_idx = int(action[1])

            action_heads = self.training_conf.get("action_heads", [])
            if head_idx >= len(action_heads):
                return  # Padded / unconfigured command slot — no-op

            head_name = action_heads[head_idx]

            if is_opponent_unit:
                # Un-flip: (row, col) → (31-row, 31-col) in a 32×32 grid
                _gs = 32
                _row, _col = divmod(spatial_idx, _gs)
                spatial_idx = ((_gs - 1 - _row) * _gs) + (_gs - 1 - _col)

            world_x, world_y = self.agent_system.unit_agents["active_agent"].convert_spatial_action_to_world(
                head_name, spatial_idx, active_unit=unit
            )
            target = Point2((world_x, world_y))

            if "attack" in head_name:
                self.do(unit.attack(target))
            else:
                self.do(unit.move(target))
            self._unit_last_order[unit_tag] = (head_name, target)
            self.last_action_was_move_by_unit[unit_tag] = "attack" not in head_name
        else:
            # Original: flat integer action on grid_size×grid_size grid
            if is_opponent_unit:
                action = PerspectiveTransformer.flip_action_idx(action, self.grid_size)

            grid_area = self.grid_size * self.grid_size
            is_attack = False
            spatial_idx = action

            if action >= grid_area:
                is_attack = True
                spatial_idx = action - grid_area

            world_x, world_y = self.agent_system.unit_agents["active_agent"].convert_spatial_action_to_world(spatial_idx)
            target = Point2((world_x, world_y))

            if is_attack:
                self.do(unit.attack(target))
            else:
                self.do(unit.move(target))
            head_name_orig = "attack" if is_attack else "move"
            self._unit_last_order[unit_tag] = (head_name_orig, target)
            self.last_action_was_move_by_unit[unit_tag] = not is_attack

    def _enemies_for_unit(self, unit: Unit) -> List[Unit]:
        """Return hostile mobile units for ``unit`` in God-Mode (bot.player_id is always P1)."""
        if unit.owner_id == self.player_id:
            candidates = self.enemy_units
        else:
            candidates = self.units
        return [u for u in candidates if not u.is_structure]

    def _is_enemy_in_tactical_fovea(self, focal: Unit, enemy: Unit) -> bool:
        """True if ``enemy`` lies inside the 32×32 tactical crop centered on ``focal``."""
        if focal is None or enemy is None:
            return False
        h = float(self._TACTICAL_FOVEA_MAX_DELTA)
        dx = abs(float(enemy.position.x) - float(focal.position.x))
        dy = abs(float(enemy.position.y) - float(focal.position.y))
        return dx <= h and dy <= h

    def _enemy_com_in_tactical_fovea(
        self, focal: Unit, enemies: Iterable[Unit]
    ) -> Optional[Point2]:
        """Center of mass of hostile units visible in ``focal``'s tactical fovea."""
        in_fovea = [e for e in enemies if self._is_enemy_in_tactical_fovea(focal, e)]
        if not in_fovea:
            return None
        cx = sum(float(e.position.x) for e in in_fovea) / len(in_fovea)
        cy = sum(float(e.position.y) for e in in_fovea) / len(in_fovea)
        return Point2((cx, cy))

    @staticmethod
    def _dist_to_point(unit: Unit, target: Point2) -> float:
        return math.hypot(
            float(unit.position.x) - float(target.x),
            float(unit.position.y) - float(target.y),
        )

    @staticmethod
    def _weapon_range_attacker_vs_defender(attacker, defender) -> float:
        """Weapon range from attacker to a specific defender (ground vs air-aware)."""
        if attacker is None or defender is None:
            return 0.0
        if bool(getattr(defender, "is_flying", False)):
            air_r = getattr(attacker, "air_range", None)
            if air_r is not None and float(air_r) > 0.0:
                return float(air_r)
        gr = getattr(attacker, "ground_range", None)
        if gr is not None:
            return float(gr)
        return 1.0

    def _is_ranged_combat_unit(self, unit: Unit, enemies: Iterable[Unit]) -> bool:
        """Cooldown-retreat shaping applies only to units that outrange melee contact."""
        enemy_list = [e for e in enemies if e is not None]
        if enemy_list:
            max_range = max(
                self._weapon_range_attacker_vs_defender(unit, e) for e in enemy_list
            )
        else:
            gr = float(getattr(unit, "ground_range", 0.0) or 0.0)
            ar = float(getattr(unit, "air_range", 0.0) or 0.0)
            max_range = max(gr, ar)
        return max_range > self._RANGED_WEAPON_RANGE_MIN

    def _defend_pylon_pos_for_unit(
        self, unit: Unit, fallback_pylon_pos: Optional[Point2] = None
    ) -> Optional[Point2]:
        """Own pylon anchor for a defender unit (spawn-time target preferred)."""
        if unit.tag in self.unit_target_map:
            return self.unit_target_map[unit.tag]
        if fallback_pylon_pos is not None:
            return fallback_pylon_pos
        if unit.owner_id == self.player_id:
            structs = self.structures
        else:
            structs = self.enemy_structures
        pylon = structs(UnitTypeId.PYLON).first if structs(UnitTypeId.PYLON).exists else None
        return pylon.position if pylon is not None else None

    def record_pre_action_retreat_state(self, unit_tag: int) -> None:
        """
        Snapshot pre-action state for per-step distance shaping.

        Captures weapon cooldown / enemy-CoM distance (cooldown retreat) and
        distance to the defender pylon (defender zone penalty).
        """
        unit = self.all_units.find_by_tag(unit_tag)
        if unit is None:
            self._pre_action_weapon_cd_by_unit.pop(unit_tag, None)
            self._pre_action_dist_to_enemy_com.pop(unit_tag, None)
            self._pre_action_dist_to_defend_pylon.pop(unit_tag, None)
            return

        self._pre_action_weapon_cd_by_unit[unit_tag] = float(
            getattr(unit, "weapon_cooldown", 0.0) or 0.0
        )

        enemies = self._enemies_for_unit(unit)
        enemy_com = self._enemy_com_in_tactical_fovea(unit, enemies)
        if enemy_com is None:
            self._pre_action_dist_to_enemy_com.pop(unit_tag, None)
        else:
            self._pre_action_dist_to_enemy_com[unit_tag] = self._dist_to_point(unit, enemy_com)

        if self.unit_command_map.get(unit_tag, "attack") == "defend":
            pylon_pos = self._defend_pylon_pos_for_unit(unit)
            if pylon_pos is not None:
                self._pre_action_dist_to_defend_pylon[unit_tag] = self._dist_to_point(
                    unit, pylon_pos
                )
            else:
                self._pre_action_dist_to_defend_pylon.pop(unit_tag, None)
        else:
            self._pre_action_dist_to_defend_pylon.pop(unit_tag, None)

    def calculate_cooldown_retreat_reward(
        self, unit_tag: int, enemies: Iterable[Unit]
    ) -> float:
        """
        Reward moving away from the enemy cluster CoM while the weapon is on cooldown.

        Gates (all required):
        - ``cooldown_retreat_scale`` > 0 in config
        - Unit has ranged weapon range (> ``_RANGED_WEAPON_RANGE_MIN``)
        - Pre-action weapon cooldown > 0
        - Last issued command was a move (not attack)
        - At least one enemy in the unit's tactical fovea (pre- and post-step)
        - Distance to enemy CoM increased since the pre-action snapshot
        """
        if unit_tag is None:
            return 0.0

        scale = float(
            self.training_conf.get("rewards", {}).get("cooldown_retreat_scale", 0.0)
        )
        if scale == 0.0:
            return 0.0

        unit = self.all_units.find_by_tag(unit_tag)
        if unit is None:
            self._pre_action_dist_to_enemy_com.pop(unit_tag, None)
            self._pre_action_weapon_cd_by_unit.pop(unit_tag, None)
            return 0.0

        if not self._is_ranged_combat_unit(unit, enemies):
            return 0.0

        pre_cd = self._pre_action_weapon_cd_by_unit.get(unit_tag)
        prev_dist = self._pre_action_dist_to_enemy_com.get(unit_tag)
        if pre_cd is None or prev_dist is None:
            return 0.0

        eps = 1e-3
        if float(pre_cd) <= eps:
            return 0.0

        if not self.last_action_was_move_by_unit.get(unit_tag, False):
            return 0.0

        enemy_com = self._enemy_com_in_tactical_fovea(unit, enemies)
        if enemy_com is None:
            return 0.0

        curr_dist = self._dist_to_point(unit, enemy_com)
        if curr_dist <= prev_dist + eps:
            return 0.0

        return scale * (curr_dist - prev_dist)

    def calculate_defender_zone_penalty(
        self, unit_tag: int, own_pylon_pos: Optional[Point2]
    ) -> float:
        """
        Penalize defender units that move farther from their pylon while outside
        the defender zone (distance > ``defender_zone_radius`` map units).

        No positive shaping — defenders are only penalized for stepping away
        beyond the zone until they return inside it.
        """
        if unit_tag is None:
            return 0.0

        if self.unit_command_map.get(unit_tag, "attack") != "defend":
            return 0.0

        r_conf = self.training_conf.get("rewards", {})
        dist_mult = float(r_conf.get("distance_multiplier", 0.1))
        zone_radius = float(r_conf.get("defender_zone_radius", 10.0))
        if dist_mult == 0.0:
            return 0.0

        unit = self.all_units.find_by_tag(unit_tag)
        if unit is None:
            self._pre_action_dist_to_defend_pylon.pop(unit_tag, None)
            return 0.0

        pylon_pos = self._defend_pylon_pos_for_unit(unit, own_pylon_pos)
        if pylon_pos is None:
            return 0.0

        prev_dist = self._pre_action_dist_to_defend_pylon.get(unit_tag)
        if prev_dist is None:
            return 0.0

        curr_dist = self._dist_to_point(unit, pylon_pos)
        eps = 1e-3

        # Inside the defender zone — no penalty from this signal.
        if curr_dist <= zone_radius + eps:
            return 0.0

        # Outside the zone: penalize only when moving farther from the pylon.
        if curr_dist > prev_dist + eps:
            return -dist_mult * (curr_dist - prev_dist)

        return 0.0

    def calculate_squad_step_reward(
        self,
        squad_tags: List[int],
        perspective_player_id: Optional[int] = None,
    ) -> float:
        """
        Reward for one squad physics step.

        Global signals (pylon/army HP, attacker CoM) are applied once.
        Per-unit shaping (cooldown retreat, defender zone, step penalty) is summed
        over ``squad_tags``.
        """
        p_id = perspective_player_id if perspective_player_id is not None else self.player_id

        reward = 0.0
        r_conf = self.training_conf.get("rewards", {})
        
        # Determine Perspective-Correct unit/structure groups
        if p_id == self.player_id:
            own_structs = self.structures
            enemy_structs = self.enemy_structures
            own_units = self.units
            enemy_units = self.enemy_units
        else:
            # God-Mode Perspective Flip
            own_structs = self.enemy_structures
            enemy_structs = self.structures
            own_units = self.enemy_units
            enemy_units = self.units

        pylon_type = UnitTypeId.PYLON
        
        # Resolve Enemy Pylon State
        enemy_pylon = enemy_structs(pylon_type).first if enemy_structs(pylon_type).exists else None
        
        # Update persistent position if currently visible
        if enemy_pylon:
            self.enemy_pylon_pos[p_id] = enemy_pylon.position
            
        target_pos = self.enemy_pylon_pos.get(p_id)

        # 1. Pylon Damage Rewards (Terminal/Heavy signal)
        def get_enemy_hp_safe(pylon_unit, last_hp, target_position):
            if pylon_unit:
                return pylon_unit.health + pylon_unit.shield
            
            # If unit is missing, check visibility of its position
            if target_position and self.is_visible(target_position):
                # We see the position, but no unit -> IT IS DEAD
                return 0.0
            
            # Not visible -> Assume unchanged (Fog of War)
            return last_hp

        def get_own_hp(pylon_unit):
            # We always have info on our own units
            if not pylon_unit: return 0.0
            return pylon_unit.health + pylon_unit.shield

        own_pylon = own_structs(pylon_type).first if own_structs(pylon_type).exists else None
        
        own_pylon_hp = get_own_hp(own_pylon)
        
        own_global_idx = p_id
        enemy_global_idx = 2 if p_id == 1 else 1
        
        enemy_pylon_hp = get_enemy_hp_safe(enemy_pylon, self.prev_pylon_health[enemy_global_idx], target_pos)
        
        # Track global pylon health by persistent player ID (1 or 2)
        p1_pylon_hp = own_pylon_hp if p_id == 1 else enemy_pylon_hp
        p2_pylon_hp = enemy_pylon_hp if p_id == 1 else own_pylon_hp
        
        curr_pylon_hp_map = {1: p1_pylon_hp, 2: p2_pylon_hp}
        
        # Pylon Damage Rewards (Signal for acting player)
        if curr_pylon_hp_map[enemy_global_idx] < self.prev_pylon_health[enemy_global_idx]:
            reward += (self.prev_pylon_health[enemy_global_idx] - curr_pylon_hp_map[enemy_global_idx]) * r_conf.get("pylon_damage_reward", 2.0)
            
        if curr_pylon_hp_map[own_global_idx] < self.prev_pylon_health[own_global_idx]:
            reward -= (self.prev_pylon_health[own_global_idx] - curr_pylon_hp_map[own_global_idx]) * r_conf.get("pylon_damage_penalty", 2.0)

        # Update global health tracking
        self.prev_pylon_health[1] = p1_pylon_hp
        self.prev_pylon_health[2] = p2_pylon_hp
        
        # 2. Combat Rewards (Mobile Units)
        curr_enemy_hp = sum(u.health + u.shield for u in enemy_units if not u.is_structure)
        if curr_enemy_hp < self.prev_enemy_units_health[p_id]:
            reward += (self.prev_enemy_units_health[p_id] - curr_enemy_hp) * r_conf.get("damage_reward", 0.1)
        self.prev_enemy_units_health[p_id] = curr_enemy_hp
        
        curr_own_hp = sum(u.health + u.shield for u in own_units if not u.is_structure)
        if curr_own_hp < self.prev_own_units_health[p_id]:
            reward -= (self.prev_own_units_health[p_id] - curr_own_hp) * r_conf.get("damage_penalty", 0.1)
        self.prev_own_units_health[p_id] = curr_own_hp

        # 3. Progress Rewards — attacker squad Center-of-Mass distance shaping
        #
        # When an attacker unit is active: reward the CoM of ALL attacker units
        # (same player) closing in on the target pylon.
        # Defenders use per-unit zone penalty (section 3c) instead of CoM shaping.
        active_unit_role = self.unit_command_map.get(self.active_unit_tag, "attack") if self.active_unit_tag else "attack"

        if active_unit_role == "attack":
            squad_units = [u for u in own_units if not u.is_structure
                           and self.unit_command_map.get(u.tag, "attack") == "attack"]
            squad_target = next(
                (self.unit_target_map[u.tag] for u in squad_units if u.tag in self.unit_target_map),
                target_pos,
            )
            dist_key = (p_id, active_unit_role)
            dist_mult = r_conf.get("distance_multiplier", 0.1)

            if squad_units and squad_target is not None:
                cx = sum(u.position.x for u in squad_units) / len(squad_units)
                cy = sum(u.position.y for u in squad_units) / len(squad_units)
                curr_com_dist = ((cx - squad_target.x) ** 2 + (cy - squad_target.y) ** 2) ** 0.5
                prev_com_dist = self.prev_dist_per_squad.get(dist_key)
                if prev_com_dist is not None:
                    reward += (prev_com_dist - curr_com_dist) * dist_mult
                self.prev_dist_per_squad[dist_key] = curr_com_dist
            else:
                self.prev_dist_per_squad.pop(dist_key, None)
        else:
            self.prev_dist_per_squad.pop((p_id, "defend"), None)

        # 3b–4. Per-unit shaping summed over the acting squad
        own_pylon_pos = own_pylon.position if own_pylon is not None else None
        mobile_enemies = [u for u in enemy_units if not u.is_structure]
        saved_active = self.active_unit_tag
        for tag in squad_tags:
            self.active_unit_tag = tag
            reward += self.calculate_cooldown_retreat_reward(tag, mobile_enemies)
            reward += self.calculate_defender_zone_penalty(tag, own_pylon_pos)
            unit_role = self.unit_command_map.get(tag, "attack")
            if unit_role == "attack":
                reward += r_conf.get("step_penalty", -0.05)
            else:
                reward += r_conf.get("defender_step_reward", 0.001)
        self.active_unit_tag = saved_active

        return reward

    def calculate_reward(self, perspective_player_id: Optional[int] = None) -> float:
        """Single-unit wrapper around :meth:`calculate_squad_step_reward`."""
        squad_tags = [self.active_unit_tag] if self.active_unit_tag is not None else []
        return self.calculate_squad_step_reward(squad_tags, perspective_player_id)