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
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
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
        self.dataset_target_unit_tags = data.get("target_unit_tags", np.zeros(len(self.dataset_actions), dtype=np.uint64))
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
        self.last_unit_action = {}
        self.unit_locks = {} # tag -> lock_data
        self.boarding_units = {} # boarder_tag -> transport_tag

    async def on_step(self, iteration: int):
        if self.total_steps == 0: await self.client.leave(); return

        if not self.mapped_initially and iteration > 0:
            available_units = {u.tag: u for u in self.all_units}
            for r_tag, unit_data in self.dataset_initial_units.items():
                r_pos = Point2((unit_data[0], unit_data[1]))
                r_type = unit_data[2]
                candidates = [u for u in available_units.values() if u.type_id.value == r_type]
                if candidates:
                    closest_unit = min(candidates, key=lambda u: u.distance_to(r_pos))
                    self.tag_map[int(r_tag)] = closest_unit.tag
                    del available_units[closest_unit.tag]
            self.mapped_initially = True

        if self.current_step < self.total_steps:
            self.mock_agent.transformed_observation = self.dataset_observations[self.current_step]
            self.signals.frame_updated.emit(self.dataset_observations[self.current_step], self.dataset_actions[self.current_step])

        # Boarding Success Detection
        for boarder_tag in list(self.boarding_units.keys()):
            if self.all_units.find_by_tag(boarder_tag) is None:
                logger.info(f"Loop {self.state.game_loop}: Boarding confirmed for {boarder_tag}")
                del self.boarding_units[boarder_tag]

        # SEQUENTIAL SYNC
        current_live_loop = self.state.game_loop
        actions_issued_this_step = 0
        
        while self.current_step < self.total_steps:
            target_loop = self.dataset_game_loops[self.current_step]
            if current_live_loop >= target_loop:
                action_idx = int(self.dataset_actions[self.current_step])
                acting_data = self.dataset_acting_data[self.current_step]
                replay_target_tag = int(self.dataset_target_unit_tags[self.current_step])
                
                if action_idx != -1:
                    live_acting_tags = [self.tag_map[int(u["tag"])] for u in acting_data if int(u["tag"]) in self.tag_map]
                    target_unit_tag_live = self.tag_map.get(replay_target_tag)
                    target_pos_replay = Point2(self.dataset_target_positions[self.current_step])
                    
                    # 1. Lock Check (Physical Gating)
                    for tag in live_acting_tags:
                        if tag in self.boarding_units or tag in self.boarding_units.values():
                            return 
                        
                        lock = self.unit_locks.get(tag)
                        if lock:
                            unit = self.all_units.find_by_tag(tag)
                            if unit is None: # Success: disappeared
                                del self.unit_locks[tag]
                                continue
                            cur_cargo = unit.cargo_used if hasattr(unit, "cargo_used") else 0
                            still_locked = (self.state.game_loop <= lock["timeout"]) and (lock["cargo"] < 0 or cur_cargo < lock["cargo"])
                            if still_locked: return # WAIT
                            else: del self.unit_locks[tag]

                    needs_apply = any(action_idx != self.last_unit_action.get(tag) for tag in live_acting_tags)
                    if needs_apply:
                        # Throttle: Only issue one major action per unit per step to allow physics to update
                        if actions_issued_this_step > 0:
                            return # Wait for next frame to process more commands
                            
                        success = await self._apply_action(action_idx, live_acting_tags, target_unit_tag_live, target_pos_replay)
                        if not success: return # WAIT
                        
                        for tag in live_acting_tags: 
                            self.last_unit_action[tag] = action_idx
                        actions_issued_this_step += 1
                
                self.current_step += 1
                if self.current_step == self.total_steps: self.end_time = self.state.game_loop + 112
            else: break

        if self.current_step >= self.total_steps and self.state.game_loop >= self.end_time:
            await self.client.leave()

    async def _apply_action(self, action: int, live_tags: List[int], target_unit_tag_live: Optional[int] = None, target_pos_replay: Optional[Point2] = None) -> bool:
        grid_area = self.grid_size * self.grid_size
        total_spatial_indices = len(self.action_heads) * grid_area
        head_name, target_pos_live = None, None
        area = getattr(self.game_info, 'playable_area', None)
        crop_x, crop_y = (area.x, area.y) if area else (0, 0)
        mw, mh = (area.width, area.height) if area else (self.game_info.map_size.width, self.game_info.map_size.height)
        scale = min(self.grid_size / max(mw, 1), self.grid_size / max(mh, 1))
        ox, oy = (self.grid_size - int(mw * scale)) // 2, (self.grid_size - int(mh * scale)) // 2

        if action < total_spatial_indices:
            head_idx = action // grid_area
            head_name = self.action_heads[head_idx]
            gy, gx = divmod(action % grid_area, self.grid_size)
            target_pos_live = Point2((((gx + 0.5 - ox) / scale) + crop_x, (((self.grid_size - 1 - gy + 0.5) - oy) / scale) + crop_y))
        else:
            cat_idx = action - total_spatial_indices
            cat_heads = self.config.get("env", {}).get("categorical_heads", [])
            if cat_idx < len(cat_heads): head_name = cat_heads[cat_idx]
        
        if not head_name: return True
        acting_units = self.all_units.filter(lambda u: u.tag in live_tags)
        if not acting_units: return True

        if self.state.game_loop % 8 == 0:
            u_pos = acting_units[0].position if acting_units else "N/A"
            t_pos = target_pos_live or target_pos_replay
            dist = acting_units[0].distance_to(t_pos) if acting_units and t_pos else "N/A"
            logger.debug(f"Loop {self.state.game_loop}: Applying {head_name} for {[u.type_id.name for u in acting_units]}. Pos: {u_pos}, Target: {t_pos}, Dist: {dist}")

        TRANSPORT_TYPES = {UnitTypeId.MEDIVAC, UnitTypeId.OVERLORD, UnitTypeId.WARPPRISM, UnitTypeId.WARPPRISMPHASING, UnitTypeId.OVERLORDTRANSPORT}
        LOAD_ABILITIES = {AbilityId.LOAD_MEDIVAC, AbilityId.LOAD_OVERLORD, AbilityId.LOAD_WARPPRISM, AbilityId.SMART, AbilityId.LOAD, AbilityId.LOADALL}
        
        def get_abil(name):
            try: return getattr(AbilityId, name)
            except AttributeError: return None

        UNLOAD_NAMES = [
            "UNLOADALLAT_MEDIVAC", "UNLOADALLAT_OVERLORD", "UNLOADALLAT_WARPPRISM",
            "UNLOADALL_MEDIVAC", "UNLOADALL_OVERLORD", "UNLOADALL_WARPPRISM",
            "UNLOADALL", "UNLOADALLAT", "UNLOADUNIT"
        ]
        UNLOAD_ABILITIES = {get_abil(n) for n in UNLOAD_NAMES if get_abil(n) is not None}

        def already_has_order(unit, ability_id, target=None):
            if not unit.orders: return False
            order = unit.orders[0]
            current_id = order.ability.id if hasattr(order.ability, "id") else order.ability
            is_match = (current_id == ability_id) or \
                       (ability_id == AbilityId.SMART and current_id in LOAD_ABILITIES) or \
                       (ability_id in LOAD_ABILITIES and current_id in LOAD_ABILITIES) or \
                       (ability_id in UNLOAD_ABILITIES and current_id in UNLOAD_ABILITIES) or \
                       (ability_id == AbilityId.MOVE and current_id in {AbilityId.MOVE, AbilityId.SMART}) or \
                       (ability_id == AbilityId.STOP and current_id in {AbilityId.STOP, AbilityId.STOP_STOP})
            if not is_match: return False
            
            if target is not None:
                ot = order.target
                # Target is a unit tag (int)
                if isinstance(target, (int, np.integer, np.uint64, float)):
                    actual_ot_tag = ot.tag if hasattr(ot, "tag") else ot
                    try: return int(actual_ot_tag) == int(target)
                    except: return False
                # Target is a world position (Point2)
                elif isinstance(target, Point2):
                    if isinstance(ot, (int, np.integer, np.uint64, float)):
                        # Current order targets a unit tag, but we expect a Point2.
                        # This is not a match for our specific target check.
                        return False
                    try: return unit.distance_to(ot) < 1.0
                    except: return False
            return True

        if head_name == "load":
            target_unit = None
            if target_unit_tag_live: target_unit = self.all_units.find_by_tag(target_unit_tag_live)
            if not target_unit and target_pos_live:
                cands = self.all_units.filter(lambda u: u.distance_to(target_pos_live) < 5.0)
                if cands: target_unit = cands.closest_to(target_pos_live)
            if target_unit:
                if acting_units[0].cargo_max == 0:
                    for u in acting_units: 
                        if not already_has_order(u, AbilityId.SMART, target_unit.tag): self.do(u.smart(target_unit))
                        self.boarding_units[u.tag] = target_unit.tag
                    if not already_has_order(target_unit, AbilityId.STOP): self.do(target_unit.stop())
                else:
                    abil = AbilityId.LOAD_MEDIVAC
                    if acting_units[0].type_id in {UnitTypeId.OVERLORD, UnitTypeId.OVERLORDTRANSPORT}: abil = AbilityId.LOAD_OVERLORD
                    elif acting_units[0].type_id in {UnitTypeId.WARPPRISM, UnitTypeId.WARPPRISMPHASING}: abil = AbilityId.LOAD_WARPPRISM
                    for u in acting_units:
                        if target_unit.cargo_max == 0:
                            if not already_has_order(u, abil, target_unit.tag): self.do(u(abil, target_unit))
                            self.boarding_units[target_unit.tag] = u.tag
            return True
        elif head_name == "unload":
            valid = acting_units.filter(lambda u: u.type_id in TRANSPORT_TYPES and u.cargo_used > 0)
            if not valid: return True 
            transport = valid[0]
            
            # Use replay target pos if available for categorical unload
            effective_target_pos = target_pos_live
            if not effective_target_pos and target_pos_replay and (target_pos_replay.x != 0 or target_pos_replay.y != 0):
                effective_target_pos = target_pos_replay
            
            if effective_target_pos and transport.distance_to(effective_target_pos) > 5.0:
                if not already_has_order(transport, AbilityId.MOVE, effective_target_pos): self.do(transport.move(effective_target_pos))
                return False 
            
            abil = AbilityId.UNLOADALLAT_MEDIVAC
            if transport.type_id in {UnitTypeId.OVERLORD, UnitTypeId.OVERLORDTRANSPORT}: abil = AbilityId.UNLOADALLAT_OVERLORD
            elif transport.type_id in {UnitTypeId.WARPPRISM, UnitTypeId.WARPPRISMPHASING}: abil = AbilityId.UNLOADALLAT_WARPPRISM
            
            if not already_has_order(transport, abil): 
                # Use target pos if we have one, otherwise unload at current position
                target = effective_target_pos if effective_target_pos else transport.position
                self.do(transport(abil, target))
            return True
        elif head_name == "move":
            for u in acting_units:
                if not already_has_order(u, AbilityId.MOVE, target_pos_live): self.do(u.move(target_pos_live))
            return True
        elif head_name == "attack":
            t = None
            if target_unit_tag_live: t = self.all_units.find_by_tag(target_unit_tag_live)
            for u in acting_units: 
                if t:
                    if not already_has_order(u, AbilityId.ATTACK, t.tag): self.do(u.attack(t))
                elif not already_has_order(u, AbilityId.ATTACK, target_pos_live): self.do(u.attack(target_pos_live))
            return True
        return True

class Validator:
    def __init__(self, project_root: str = "."):
        self.project_root = os.path.abspath(project_root)
    def start_validation(self, dataset_path: str, level_name: str, map_name: str, config: Dict[str, Any]):
        mock_agent = MockAgent()
        ui_manager = UIManager(mock_agent)
        active_layers = config.get("env", {}).get("active_layers", [])
        if active_layers: ui_manager.active_layers = active_layers
        from imitation_learning.recorder import Recorder
        rec = Recorder(self.project_root)
        map_path = rec.get_map_path(level_name, map_name)
        agent_race, enemy_race = rec._parse_races_from_map_name(map_name)
        bot = ValidatorBot(dataset_path, config, mock_agent)
        return bot, map_path, agent_race, enemy_race, ui_manager
