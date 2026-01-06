from typing import Dict, Optional, Any, Tuple, List
import numpy as np
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.data import Race
from loguru import logger

from src.hierarchical_agent import HierarchicalAgentSystem
from src.utils.perspective_transformer import PerspectiveTransformer

class CaptureTheFlagBot(BotAI):
    """
    Bot for Capture the Flag environment.
    Supports individual unit control and reporting state to the environment.
    """
    def __init__(self, player_id: int, training_conf: Optional[Dict] = None):
        super().__init__()
        self.player_id = player_id
        self.training_conf = training_conf or {}
        self.agent_system: Optional[HierarchicalAgentSystem] = None
        
        self.active_unit_tag: Optional[int] = None
        self.command_targets: Dict[int, Optional[Point2]] = {1: None, 2: None}
        
        # Asymmetric Self-Play attributes
        self.is_trainee = True # Default to trainee
        self.model = None # Snapshot model for automated inference
        
        # Satisfaction for BotAI initialization
        self._initialize_variables()
        from sc2.units import Units
        self.units: Units = Units([], self)
        self.structures: Units = Units([], self)
        self.enemy_units: Units = Units([], self)
        self.enemy_structures: Units = Units([], self)
        self.all_units: Units = Units([], self)
        self.larva: Units = Units([], self)
        self.workers: Units = Units([], self)
        self.townhalls: Units = Units([], self)
        self.gas_buildings: Units = Units([], self)
        self.minerals: int = 0
        self.vespene: int = 0
        self.supply_army: int = 0
        self.supply_workers: int = 0
        self.supply_cap: int = 0
        self.supply_used: int = 0
        self.supply_left: int = 0
        self.idle_worker_count: int = 0
        self.army_count: int = 0
        self.warp_gate_count: int = 0
        
        # State tracking for rewards
        self.prev_pylon_health = {1: 20.0, 2: 20.0} # Combined HP + Shield
        self.prev_enemy_units_health = {1: 0.0, 2: 0.0}
        self.prev_own_units_health = {1: 0.0, 2: 0.0}
        self.prev_dist_to_pylon = {1: None, 2: None}
        
        self.frame_count = 0
        self.cached_observation: Optional[np.ndarray] = None

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
        target_pos = enemy_pylon.position if enemy_pylon else self.command_targets[p_id]
        
        selected_units = []
        if self.active_unit_tag:
            try:
                # We search in ALL units for the active unit tag
                active_unit = self.all_units.find_by_tag(self.active_unit_tag)
                if active_unit:
                    selected_units = [active_unit]
            except Exception:
                pass

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
            'selected_units': selected_units
        }
        
        # Process with agent system, passing p_id for perspective-aware transformation
        try:
            processed = self.agent_system.process_observation(raw_obs, player_id=p_id)
            obs = processed["active_agent"]
            
            # Spatial Perspective Flip for Player 2
            if p_id == 2:
                obs = PerspectiveTransformer.flip_observation(obs)
                
            self.cached_observation = obs
        except Exception as e:
            if self.cached_observation is not None:
                logger.warning(f"Bot {self.player_id} (P_ID {p_id}): Observation processing failed ({e}). Returning cached observation.")
                return self.cached_observation
            else:
                # Fallback to zeros if nothing cached
                num_layers = self.training_conf.get("active_layers_count", 23)
                return np.zeros((num_layers, 128, 128), dtype=np.uint8)

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
        Returns a concatenated mask for Move (first half) and Attack (second half).
        """
        if not self.agent_system or "active_agent" not in self.agent_system.unit_agents:
            # Fallback if system not ready - return all True
            env_conf = self.training_conf.get("env", {})
            grid_size = self.training_conf.get("grid_size", env_conf.get("grid_size", 128))
            grid_area = grid_size * grid_size
            return np.ones(2 * grid_area, dtype=bool)

        # Get pathing mask for movement
        move_mask = self.agent_system.unit_agents["active_agent"].get_action_mask()
        
        # AIR UNIT BYPASS: If unit is flying, it can move anywhere (ignore pathing)
        if self.active_unit_tag is not None:
            active_unit = self.all_units.find_by_tag(self.active_unit_tag)
            if active_unit and active_unit.is_flying:
                move_mask = np.ones_like(move_mask, dtype=bool)

        # Attack mask is fully open (Attack-Move to any location allowed)
        attack_mask = np.ones_like(move_mask, dtype=bool)
        
        mask = np.concatenate([move_mask, attack_mask])

        # Spatial Perspective Flip for Player 2
        if self.player_id == 2:
            mask = PerspectiveTransformer.flip_mask(mask, self.grid_size)
            
        return mask

    async def apply_action(self, unit_tag: int, action: int):
        unit = self.all_units.find_by_tag(unit_tag)
        if not unit: return

        # Spatial Perspective Flip for Player 2
        # If we flipped the observation, we must flip the action back
        if self.player_id == 2:
            action = PerspectiveTransformer.flip_action_idx(action, self.grid_size)

        # Restore dual-head logic parity with Level 7
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
            
        self.command_targets[unit.owner_id] = target

    def calculate_reward(self, perspective_player_id: Optional[int] = None) -> float:
        """
        Calculates CTF rewards from the given player's perspective.
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
        
        # 1. Pylon Damage Rewards (Terminal/Heavy signal)
        def get_hp(pylon):
            if not pylon: return 0.0
            return pylon.health + pylon.shield

        own_pylon = own_structs(pylon_type).first if own_structs(pylon_type).exists else None
        enemy_pylon = enemy_structs(pylon_type).first if enemy_structs(pylon_type).exists else None
        
        own_pylon_hp = get_hp(own_pylon)
        enemy_pylon_hp = get_hp(enemy_pylon)
        
        # Track global pylon health by persistent player ID (1 or 2)
        p1_pylon_hp = own_pylon_hp if p_id == 1 else enemy_pylon_hp
        p2_pylon_hp = enemy_pylon_hp if p_id == 1 else own_pylon_hp
        
        curr_pylon_hp_map = {1: p1_pylon_hp, 2: p2_pylon_hp}
        
        own_global_idx = p_id
        enemy_global_idx = 2 if p_id == 1 else 1
        
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

        # 3. Progress Rewards (Distance to enemy pylon)
        # We reward the ACTIVE unit for getting closer to its perspective-correct target
        if self.active_unit_tag:
            active_unit = self.all_units.find_by_tag(self.active_unit_tag)
            
            if active_unit and enemy_pylon:
                curr_dist = active_unit.distance_to(enemy_pylon)
                if self.prev_dist_to_pylon[p_id] is not None:
                    # Reward for progress
                    dist_change = self.prev_dist_to_pylon[p_id] - curr_dist
                    reward += dist_change * r_conf.get("distance_multiplier", 0.1)
                self.prev_dist_to_pylon[p_id] = curr_dist
            else:
                self.prev_dist_to_pylon[p_id] = None

        return reward