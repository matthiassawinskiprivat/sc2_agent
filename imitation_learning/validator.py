import os
import numpy as np
import asyncio
from typing import Dict, Any, List, Optional
from sc2.bot_ai import BotAI
from sc2.main import run_game
from sc2 import maps
from sc2.player import Bot, Computer
from sc2.data import Race, Difficulty
from sc2.position import Point2
from loguru import logger

from src.hierarchical_agent import HierarchicalAgentSystem
from src.ui.debug_ui import DebugUI
from src.ui.ui_manager import UIManager

class MockAgent:
    def __init__(self):
        self.transformed_observation = None
        self.last_action = None
    def get_action_mask(self): return np.ones(128 * 128, dtype=bool)
    def reset(self): pass
    def get_observation_statistics(self): return {'processing_time_ms': 1.0, 'frame_count': 0}

from PyQt5.QtCore import QObject, pyqtSignal

class ValidatorSignals(QObject):
    frame_updated = pyqtSignal(np.ndarray, int)

class ValidatorBot(BotAI):
    def __init__(self, dataset_path: str, config: Dict[str, Any], mock_agent: MockAgent):
        super().__init__()
        self.signals = ValidatorSignals()
        data = np.load(dataset_path, allow_pickle=True)
        self.dataset_observations = data["observations"]
        self.dataset_actions = data["actions"]
        self.dataset_game_loops = data["game_loops"]
        self.dataset_acting_data = data["acting_data"]
        self.dataset_target_positions = data["target_positions"]
        self.dataset_initial_units = dict(data["initial_units"][0])
        
        self.current_step = 0
        self.total_steps = len(self.dataset_actions)
        self.config = config
        self.mock_agent = mock_agent
        self.action_heads = config.get("env", {}).get("action_heads", ["move"])
        self.grid_size = config.get("env", {}).get("grid_size", 128)
        
        self.tag_map = {}
        self.mapped_initially = False
        self.end_time = -1
        self.unit_locks = {} 

    async def on_step(self, iteration: int):
        if self.total_steps == 0: await self.client.leave(); return

        if not self.mapped_initially and iteration > 0:
            available_units = {u.tag: u for u in self.units}
            for r_tag, unit_data in self.dataset_initial_units.items():
                r_pos = Point2((unit_data[0], unit_data[1]))
                r_type = unit_data[2]
                
                # Filter candidates that are of correct type AND are in available_units
                candidates = [u for u in available_units.values() if u.type_id.value == r_type]
                
                if candidates:
                    # Manual closest distance check since we are using a list
                    closest_unit = min(candidates, key=lambda u: u.distance_to(r_pos))
                    self.tag_map[r_tag] = closest_unit.tag
                    del available_units[closest_unit.tag] # Remove from pool
            self.mapped_initially = True

        if self.current_step < self.total_steps:
            self.mock_agent.transformed_observation = self.dataset_observations[self.current_step]
            self.signals.frame_updated.emit(self.dataset_observations[self.current_step], self.dataset_actions[self.current_step])

        while self.current_step < self.total_steps:
            target_loop = self.dataset_game_loops[self.current_step]
            if self.state.game_loop >= target_loop:
                action_idx = self.dataset_actions[self.current_step]
                acting_data = self.dataset_acting_data[self.current_step]
                
                if action_idx != -1:
                    live_tags = [self.tag_map[u["tag"]] for u in acting_data if u["tag"] in self.tag_map]
                    success = await self._apply_action(action_idx, live_tags)
                    if not success: break 
                
                self.current_step += 1
                if self.current_step == self.total_steps: self.end_time = self.state.game_loop + 112
            else: break
        
        if self.current_step >= self.total_steps and self.state.game_loop >= self.end_time:
            await self.client.leave()

    async def _apply_action(self, action: int, live_tags: List[int]) -> bool:
        from sc2.ids.unit_typeid import UnitTypeId
        from sc2.ids.ability_id import AbilityId
        
        grid_area = self.grid_size * self.grid_size
        total_spatial_indices = len(self.action_heads) * grid_area
        head_name, target_pos_live = None, None
        
        # PROJECTION SETTINGS
        area = getattr(self.game_info, 'playable_area', None)
        crop_x, crop_y = (area.x, area.y) if area else (0, 0)
        mw, mh = (area.width, area.height) if area else (self.game_info.map_size.width, self.game_info.map_size.height)
        scale = min(self.grid_size / max(mw, 1), self.grid_size / max(mh, 1))
        ox, oy = (self.grid_size - int(mw * scale)) // 2, (self.grid_size - int(mh * scale)) // 2

        if action < total_spatial_indices:
            head_idx = action // grid_area
            spatial_idx = action % grid_area
            head_name = self.action_heads[head_idx]
            gy, gx = divmod(spatial_idx, self.grid_size)
            
            # INVERSE PROJECTION
            world_x = ((gx + 0.5 - ox) / scale) + crop_x
            world_y = (((self.grid_size - 1 - gy + 0.5) - oy) / scale) + crop_y
            target_pos_live = Point2((world_x, world_y))
        else:
            cat_idx = action - total_spatial_indices
            cat_heads = self.config.get("env", {}).get("categorical_heads", [])
            if cat_idx < len(cat_heads): head_name = cat_heads[cat_idx]
        
        if not head_name: return True
        acting_units = self.units.filter(lambda u: u.tag in live_tags)
        if not acting_units: return True

        TRANSPORT_TYPES = {UnitTypeId.MEDIVAC, UnitTypeId.OVERLORD, UnitTypeId.WARPPRISM, UnitTypeId.OVERLORDTRANSPORT}
        all_transports = self.units.filter(lambda u: u.type_id in TRANSPORT_TYPES)

        if head_name == "load" and target_pos_live:
            target_transport = all_transports.closest_to(target_pos_live) if all_transports else None
            if target_transport:
                # REFINED LOAD: Distinguish if human selected infantry or medivac
                if acting_units[0].cargo_max == 0:
                    # Marines boarding
                    for u in acting_units: self.do(u.smart(target_transport))
                    # LOCK Medivac until all boarding Marines are inside
                    # Fix: Accumulate locks if multiple groups are sent to same transport
                    current_lock = self.unit_locks.get(target_transport.tag, {"cargo": target_transport.cargo_used, "timeout": 0})
                    new_cargo_req = current_lock["cargo"] + acting_units.amount
                    self.unit_locks[target_transport.tag] = {"cargo": new_cargo_req, "timeout": self.state.game_loop + 224}
                else:
                    # Medivac picking up
                    infantry = self.units.filter(lambda u: u.cargo_max == 0)
                    target_unit = infantry.closest_to(target_pos_live) if infantry else None
                    if target_unit:
                        abil = AbilityId.LOAD_MEDIVAC
                        if target_transport.type_id in {UnitTypeId.OVERLORD, UnitTypeId.OVERLORDTRANSPORT}: abil = AbilityId.LOAD_OVERLORD
                        elif target_transport.type_id == UnitTypeId.WARPPRISM: abil = AbilityId.LOAD_WARPPRISM
                        self.do(target_transport(abil, target_unit))
                        current_lock = self.unit_locks.get(target_transport.tag, {"cargo": target_transport.cargo_used, "timeout": 0})
                        self.unit_locks[target_transport.tag] = {"cargo": current_lock["cargo"] + 1, "timeout": self.state.game_loop + 224}
            return True
        
        elif head_name == "unload":
            # REFINED UNLOAD: Find the specific transport that was commanded
            valid = acting_units.filter(lambda u: u.type_id in TRANSPORT_TYPES and u.cargo_used > 0)
            if not valid: return True 
            
            transport = valid[0]
            
            # FALLBACK: If unload is categorical, recover the spatial target from the dataset
            # This allows verification to work even if the training config is limited.
            if target_pos_live is None:
                orig_target = self.dataset_target_positions[self.current_step]
                if orig_target[0] != 0 or orig_target[1] != 0:
                    target_pos_live = Point2((orig_target[0], orig_target[1]))

            # PROXIMITY GATE: Wait until we are near the target island before unloading
            # This prevents premature unloading if the human clicked far away
            if target_pos_live and transport.distance_to(target_pos_live) > 5.0:
                # If we are not there yet, issue a move/smart command to get there if not already moving
                # Use .ability.id to compare with AbilityId enum, preventing redundant commands
                current_ability = transport.orders[0].ability
                current_id = current_ability.id if hasattr(current_ability, "id") else current_ability
                if not transport.orders or current_id not in {AbilityId.MOVE, AbilityId.SMART}:
                    self.do(transport.move(target_pos_live))
                return False # BLOCK dataset until arrived
            
            abil = AbilityId.UNLOADALLAT_MEDIVAC
            if transport.type_id in {UnitTypeId.OVERLORD, UnitTypeId.OVERLORDTRANSPORT}: abil = AbilityId.UNLOADALLAT_OVERLORD
            elif transport.type_id == UnitTypeId.WARPPRISM: abil = AbilityId.UNLOADALLAT_WARPPRISM
            self.do(transport(abil, target_pos_live if target_pos_live else transport.position))
            self.unit_locks[transport.tag] = {"cargo": 0, "timeout": self.state.game_loop + 44}
            return True
        
        elif head_name == "move":
            for unit in acting_units:
                lock = self.unit_locks.get(unit.tag)
                if lock:
                    # Strictly wait for full cargo or timeout
                    if self.state.game_loop > lock["timeout"] or (lock["cargo"] >= 0 and unit.cargo_used >= lock["cargo"]):
                        del self.unit_locks[unit.tag]
                    else: 
                        return False # BLOCK the whole action if any unit is still busy loading
                self.do(unit.move(target_pos_live))
            return True
        
        elif head_name == "attack":
            for unit in acting_units: self.do(unit.attack(target_pos_live))
            return True
        return True

class Validator:
    def __init__(self, project_root: str = "."):
        self.project_root = os.path.abspath(project_root)
    def start_validation(self, dataset_path: str, level_name: str, map_name: str, config: Dict[str, Any]):
        mock_agent = MockAgent()
        ui_manager = UIManager(mock_agent)
        from imitation_learning.recorder import Recorder
        rec = Recorder(self.project_root)
        map_path = rec.get_map_path(level_name, map_name)
        agent_race, enemy_race = rec._parse_races_from_map_name(map_name)
        bot = ValidatorBot(dataset_path, config, mock_agent)
        return bot, map_path, agent_race, enemy_race, ui_manager
