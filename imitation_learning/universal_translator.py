import os
import numpy as np
import asyncio
from typing import List, Dict, Any, Optional, Union
from sc2.bot_ai import BotAI
from sc2.main import get_replay_version, _setup_replay
from sc2.data import Result, Alliance
from sc2.portconfig import Portconfig
from sc2.sc2process import SC2Process
from sc2.game_state import GameState
from sc2.position import Point2
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from loguru import logger
from src.hierarchical_agent import HierarchicalAgentSystem

class HybridReplayTranslatorBot(BotAI):
    """
    Specialized Translator Bot for Hybrid Architecture.
    Always returns unmasked spatial actions for training speed parity.
    """
    def __init__(self, training_conf: Dict[str, Any], observed_id: int = 1):
        super().__init__()
        self.training_conf = training_conf
        self.observed_id = observed_id
        self.detected_observed_id = None
        self.recorded_data = []
        self.persistent_unit_registry = {}

    async def on_start(self):
        logger.info(f"HybridReplayTranslatorBot started. Default Observed ID: {self.observed_id}")
        grid_size = self.training_conf.get("env", {}).get("grid_size", 128)
        self.agent_system = HierarchicalAgentSystem(
            map_size=(128, 128),
            grid_size=grid_size,
            training_conf=self.training_conf
        )
        self.agent_system.create_unit_agent("agent_0")

    def action_masks(self, unit) -> np.ndarray:
        """
        Hybrid simplified masking: returns all-True flat mask of correct size.
        """
        env_conf = self.training_conf.get("env", {})
        action_heads = env_conf.get("action_heads", ["move_global"])
        categorical_heads = env_conf.get("categorical_heads", [])
        
        spatial_sizes = []
        for h in action_heads:
            if ":" in h: spatial_sizes.append(int(h.split(":")[1])**2)
            else: spatial_sizes.append(1024) # Global locked to 32x32
        
        max_spatial = max(spatial_sizes) if spatial_sizes else 1024
        total_mask_size = len(action_heads) + max_spatial + len(categorical_heads)
        return np.ones(total_mask_size, dtype=bool)

    async def on_step(self, iteration: int):
        if self.detected_observed_id is None and self.state.actions_unit_commands:
            for action in self.state.actions_unit_commands:
                if action.unit_tags:
                    sample_tag = list(action.unit_tags)[0]
                    u = self.all_units.find_by_tag(sample_tag)
                    if u and u.owner_id != 0:
                        self.detected_observed_id = u.owner_id
                        break

        target_id = self.detected_observed_id or self.observed_id
        for u in self.all_units:
            self.persistent_unit_registry[u.tag] = (u.position.x, u.position.y, u.type_id.value, u.owner_id)
            
        for transport in self.units.filter(lambda u: u.cargo_max > 0):
            if hasattr(transport, "passengers"):
                for p in transport.passengers:
                    if p.tag not in self.persistent_unit_registry:
                        self.persistent_unit_registry[p.tag] = (transport.position.x, transport.position.y, p.type_id.value, transport.owner_id)
        
        allied_units, allied_structures = [], []
        enemy_units, enemy_structures = [], []
        processed_tags = set()
        unit_status = {}
        
        beacons = []
        for u in self.all_units:
            if u.tag in processed_tags: continue
            processed_tags.add(u.tag)
            u_name = getattr(u.type_id, "name", "UNKNOWN")
            if "BEACON" in u_name.upper():
                beacons.append(u)
                continue
            if u.owner_id == target_id:
                if u.is_structure: allied_structures.append(u)
                else: 
                    allied_units.append(u)
                    unit_status[u.tag] = {"is_idle": u.is_idle, "has_orders": len(u.orders) > 0}
            elif 0 < u.owner_id < 16:
                if u.is_structure: enemy_structures.append(u)
                else: enemy_units.append(u)

        for b in beacons: enemy_units.append(b)

        # Passengers
        for transport in [u for u in allied_units if u.cargo_max > 0]:
            if hasattr(transport, "passengers"):
                for p in transport.passengers:
                    if p.tag in processed_tags: continue
                    processed_tags.add(p.tag)
                    class PassengerMock:
                        def __init__(self, p, pos, tid):
                            self.tag=p.tag; self.type_id=p.type_id; self.position=pos; self.owner_id=tid
                            self.is_structure=False; self.is_flying=False; self.radius=0.5; self.cargo_max=0; self.cargo_used=0
                            self.orders=[]; self.is_idle=True
                        def distance_to(self, t):
                            tpos = t.position if hasattr(t, "position") else t
                            return ((self.position.x - tpos.x)**2 + (self.position.y - tpos.y)**2)**0.5
                    allied_units.append(PassengerMock(p, transport.position, target_id))
                    unit_status[p.tag] = {"is_idle": True, "has_orders": False}

        unit_masks = {}
        for u in allied_units: unit_masks[u.tag] = self.action_masks(u)

        found_objective = next((u for u in (enemy_structures + [u for u in enemy_units if u.is_structure]) if u.type_id in {UnitTypeId.COMMANDCENTER, UnitTypeId.HATCHERY, UnitTypeId.NEXUS, UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS, UnitTypeId.LAIR, UnitTypeId.HIVE}), None)
        if not found_objective:
            beacons_u = self.all_units.filter(lambda u: "BEACON" in u.type_id.name.upper())
            if beacons_u: found_objective = beacons_u.first

        objective_pos = (found_objective.position.x, found_objective.position.y) if found_objective else None
        actions = self._extract_human_actions()
        raw_obs_data = {
            'units': allied_units, 'structures': allied_structures,
            'enemy_units': enemy_units, 'enemy_structures': enemy_structures,
            'effects': [e for e in self.state.effects], 'creep': self.state.creep,
            'game_loop': self.state.game_loop, 'unit_status': unit_status,
            'unit_masks': unit_masks, 'objective_pos': objective_pos
        }
        unit_snapshot = {tag: data[:3] for tag, data in self.persistent_unit_registry.items()}
        self.recorded_data.append({"raw_obs": raw_obs_data, "actions": actions, "unit_snapshot": unit_snapshot, "game_loop": self.state.game_loop})

    def _extract_human_actions(self) -> List[Dict]:
        target_id = self.detected_observed_id or self.observed_id
        human_actions = []
        for action in self.state.actions_unit_commands:
            if not action.unit_tags: continue
            sample_tag = list(action.unit_tags)[0]
            u_data = self.persistent_unit_registry.get(sample_tag)
            owner = u_data[3] if u_data else None
            if owner is not None and owner != target_id: continue
            target_pos, intent = None, None
            if action.target_unit_tag:
                target_unit = self.all_units.find_by_tag(action.target_unit_tag)
                if target_unit:
                    target_pos = (target_unit.position.x, target_unit.position.y)
                    abil = action.ability_id.value if hasattr(action.ability_id, "value") else action.ability_id
                    if abil == AbilityId.SMART.value:
                        if target_unit.owner_id != target_id and target_unit.owner_id != 0: intent = "attack"
                        elif target_unit.cargo_max > 0: intent = "load"
            elif action.target_world_space_pos:
                target_pos = (action.target_world_space_pos.x, action.target_world_space_pos.y)
            if target_pos or action.ability_id:
                human_actions.append({"ability_id": action.ability_id, "intent_override": intent, "unit_tags": list(action.unit_tags), "target_world_pos": target_pos, "target_unit_tag": action.target_unit_tag or 0})
        return human_actions

class HybridReplayTranslator:
    def __init__(self, project_root: str = "."):
        self.project_root = os.path.abspath(project_root)
        self.translated_dir = os.path.join(self.project_root, "imitation_learning", "replays_translated")

    def translate(self, replay_path: str, level_name: str, config: Dict[str, Any]):
        bot = HybridReplayTranslatorBot(config, observed_id=config.get("env", {}).get("observed_id", 1))
        from .translator import run_replay_fixed # Reuse the runner
        asyncio.get_event_loop().run_until_complete(run_replay_fixed(bot, replay_path=replay_path, realtime=False, observed_id=bot.observed_id))
        initial_units = bot.recorded_data[0].get("unit_snapshot", {}) if bot.recorded_data else {}
        translated_pairs = self.process_raw_data(bot.recorded_data, config, bot.game_info, bot.agent_system)
        self.save_translated_data(translated_pairs, level_name, os.path.basename(replay_path), initial_units)
        return len(translated_pairs)

    def process_raw_data(self, recorded_data: List[Dict], config: Dict[str, Any], game_info: Any, agent_system: HierarchicalAgentSystem) -> List[Dict]:
        from src.transformers.observation_transformer import ObservationTransformer
        final_pairs, env_conf = [], config.get("env", {})
        action_heads = env_conf.get("action_heads", ["move_global"])
        categorical_heads = env_conf.get("categorical_heads", [])
        
        transformer = ObservationTransformer(map_size=(128, 128), training_conf=env_conf)
        
        ABILITY_TO_HEAD = {}
        def add_abil(n, h): 
            if hasattr(AbilityId, n): ABILITY_TO_HEAD[getattr(AbilityId, n).value] = h
        add_abil("SMART","move"); add_abil("MOVE","move"); add_abil("ATTACK","attack")
        add_abil("LOAD","load"); add_abil("UNLOADALLAT","unload")
        
        last_unit_actions = {}
        unit_agent = agent_system.unit_agents["agent_0"]
        unit_agent.last_game_info = game_info

        for entry in recorded_data:
            raw_obs_data, new_human_actions, loop = entry["raw_obs"], entry["actions"], entry["game_loop"]
            unit_status = raw_obs_data.get("unit_status", {})
            just_commanded = set()
            for ha in new_human_actions:
                abil = ha["ability_id"].value if hasattr(ha["ability_id"], "value") else ha["ability_id"]
                base_head = ha.get("intent_override") or ABILITY_TO_HEAD.get(abil, "move")
                
                matched_head = None
                for h in action_heads:
                    if base_head in h: matched_head = h; break
                if not matched_head and base_head in categorical_heads: matched_head = base_head

                if matched_head:
                    action_data = {"target_pos": ha["target_world_pos"], "target_unit_tag": ha.get("target_unit_tag") or 0, "head_name": matched_head, "unit_tags": ha["unit_tags"]}
                    for tag in ha["unit_tags"]: last_unit_actions[tag] = action_data; just_commanded.add(tag)

            for unit in raw_obs_data['units']:
                tag, status = unit.tag, unit_status.get(unit.tag, {})
                if tag in last_unit_actions and (tag in just_commanded or status.get("has_orders", False)):
                    ad = last_unit_actions[tag]; tpos = ad["target_pos"] or (0,0); head_name = ad["head_name"]
                    if head_name in action_heads:
                        h_idx = action_heads.index(head_name)
                        if ":" in head_name:
                            size = int(head_name.split(":")[1])
                            dy = int(unit.position.y - tpos[1] + (size // 2))
                            dx = int(tpos[0] - unit.position.x + (size // 2))
                            if 0 <= dx < size and 0 <= dy < size: spatial_idx = dy * size + dx
                            else: continue 
                        else:
                            old_gs = unit_agent.grid_size; unit_agent.grid_size = 32
                            spatial_idx = unit_agent.get_grid_index_at_world_pos(Point2(tpos)); unit_agent.grid_size = old_gs
                        agent_action = [h_idx, spatial_idx]
                    else:
                        agent_action = [len(action_heads) + categorical_heads.index(head_name), 0]
                    
                    final_command_target = raw_obs_data.get('objective_pos') or tpos
                    frame_raw_obs = {'units': raw_obs_data['units'], 'structures': raw_obs_data['structures'], 'enemy_units': raw_obs_data['enemy_units'], 'enemy_structures': raw_obs_data['enemy_structures'], 'effects': raw_obs_data['effects'], 'creep': raw_obs_data['creep'], 'game_info': game_info, 'game_data': None, 'selected_units': [unit], 'command_target': Point2(final_command_target)}
                    unit_obs = transformer.transform(frame_raw_obs)
                    acting_data = [{"tag": unit.tag, "type": unit.type_id.value, "pos": (unit.position.x, unit.position.y)}]
                    final_pairs.append({"observation": unit_obs, "action": agent_action, "acting_data": acting_data, "target_pos": tpos, "target_unit_tag": ad["target_unit_tag"], "game_loop": loop, "action_mask": raw_obs_data.get("unit_masks", {}).get(tag)})
        return final_pairs

    def save_translated_data(self, pairs: List[Dict], level_name: str, original_filename: str, initial_units: Dict):
        if not pairs: return
        level_dir = os.path.join(self.translated_dir, level_name); os.makedirs(level_dir, exist_ok=True)
        output_path = os.path.join(level_dir, f"{os.path.splitext(original_filename)[0]}.npz")
        np.savez_compressed(output_path, 
                            observations=np.array([p["observation"] for p in pairs], dtype=np.float32), 
                            actions=np.array([p["action"] for p in pairs], dtype=np.int32),
                            game_loops=np.array([p["game_loop"] for p in pairs], dtype=np.int32),
                            acting_data=np.array([p["acting_data"] for p in pairs], dtype=object),
                            target_positions=np.array([p["target_pos"] for p in pairs], dtype=np.float32),
                            target_unit_tags=np.array([(p.get("target_unit_tag") or 0) for p in pairs], dtype=np.uint64),
                            initial_units=np.array([list(initial_units.items())], dtype=object),
                            action_masks=np.array([p["action_mask"] for p in pairs], dtype=bool))
        logger.info(f"Saved {len(pairs)} hybrid examples to {output_path}")
