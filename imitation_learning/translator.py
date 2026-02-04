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
import s2clientprotocol.sc2api_pb2 as sc_pb
from loguru import logger
from src.hierarchical_agent import HierarchicalAgentSystem

class ReplayTranslatorBot(BotAI):
    def __init__(self, training_conf: Dict[str, Any], observed_id: int = 1):
        super().__init__()
        self.training_conf = training_conf
        self.observed_id = observed_id
        self.detected_observed_id = None
        self.recorded_data = []
        self.persistent_unit_registry = {}

    async def on_start(self):
        logger.info(f"ReplayTranslatorBot started. Default Observed ID: {self.observed_id}")
        # Initialize Agent System for masking
        grid_size = self.training_conf.get("env", {}).get("grid_size", 128)
        self.agent_system = HierarchicalAgentSystem(
            map_size=(128, 128),
            grid_size=grid_size,
            training_conf=self.training_conf
        )
        self.agent_system.create_unit_agent("agent_0")

    def action_masks(self, unit) -> np.ndarray:
        """
        Calculates action mask for a specific unit using the same logic as SC2Bot.
        """
        # Ensure agent system has the latest game info
        self.agent_system.unit_agents["agent_0"].last_game_info = self.game_info
        
        move_mask = self.agent_system.unit_agents["agent_0"].get_action_mask()
        
        is_transport_actor = unit.cargo_max > 0
        is_flying = unit.is_flying
        cargo_used = getattr(unit, "cargo_used", 0)

        if is_flying:
            move_mask = np.ones_like(move_mask, dtype=bool)

        unrestricted_mask = np.ones_like(move_mask, dtype=bool)
        zero_mask = np.zeros_like(move_mask, dtype=bool)
        
        # Retrieve Dynamic Ranges
        load_range = 4.0
        heal_range = 4.0
        # In translator, we might not have game_data.abilities easily accessible 
        # like in live bot, but we can try.
        if hasattr(self, "game_data") and self.game_data and hasattr(self.game_data, "abilities"):
            try:
                l_rng = self.game_data.abilities[AbilityId.LOAD_MEDIVAC.value]._proto.cast_range
                if l_rng > 0: load_range = l_rng
            except: pass

        action_heads = self.training_conf.get("env", {}).get("action_heads", ["move"])
        final_masks = []
        for head in action_heads:
            if head == "move":
                final_masks.append(move_mask)
            elif head == "attack":
                # Medivac heal masking logic (simplified for translator if needed)
                if unit.type_id == UnitTypeId.MEDIVAC:
                    damaged_allies = self.units.filter(lambda u: u.tag != unit.tag and u.is_biological and u.health < u.health_max)
                    if not damaged_allies: final_masks.append(zero_mask)
                    else: final_masks.append(move_mask)
                else:
                    final_masks.append(move_mask)
            elif head == "load":
                if is_flying and not is_transport_actor:
                    final_masks.append(zero_mask)
                else:
                    load_mask = self.agent_system.unit_agents["agent_0"].get_transport_mask(
                        self.units, self.game_info, is_transport=is_transport_actor, active_unit=unit, cast_range=load_range
                    )
                    final_masks.append(load_mask)
            else:
                final_masks.append(unrestricted_mask)
        
        # --- CATEGORICAL HEADS MASKING ---
        categorical_masks = []
        categorical_heads = self.training_conf.get("env", {}).get("categorical_heads", [])
        for head in categorical_heads:
            if head == "unload":
                if is_transport_actor and cargo_used > 0:
                    try:
                        unit_agent = self.agent_system.unit_agents["agent_0"]
                        # Advanced mask: Terrain erosion + Footprints
                        blockers = self.structures + self.enemy_structures + self.mineral_field + self.vespene_geyser
                        
                        full_unload_mask = unit_agent.get_unload_mask(blockers, self.game_info)
                        grid_idx = unit_agent.get_grid_index_at_world_pos(unit.position)
                        
                        can_unload = bool(full_unload_mask[grid_idx])
                        categorical_masks.append(can_unload)
                        
                        if self.state.game_loop % 500 == 0:
                            logger.debug(f"Advanced Translator UNLOAD mask for {unit.type_id} at index {grid_idx}: {can_unload}")
                    except Exception as e:
                        logger.error(f"Error in translator advanced mask: {e}")
                        categorical_masks.append(True)
                else:
                    categorical_masks.append(False)
            else:
                categorical_masks.append(True)

        return np.concatenate(final_masks + [np.array(categorical_masks)])

    async def on_step(self, iteration: int):
        if self.detected_observed_id is None and self.state.actions_unit_commands:
            for action in self.state.actions_unit_commands:
                if action.unit_tags:
                    sample_tag = list(action.unit_tags)[0]
                    u = self.all_units.find_by_tag(sample_tag)
                    if u and u.owner_id != 0:
                        self.detected_observed_id = u.owner_id
                        logger.info(f"DYNAMIC PERSPECTIVE: Detected Human Player ID as {self.detected_observed_id}")
                        break

        target_id = self.detected_observed_id or self.observed_id
        for u in self.all_units:
            self.persistent_unit_registry[u.tag] = (u.position.x, u.position.y, u.type_id.value, u.owner_id)
            
        # Also register passengers in registry so they can be tracked even if they start loaded
        for transport in self.units.filter(lambda u: u.cargo_max > 0):
            if hasattr(transport, "passengers"):
                for p in transport.passengers:
                    if p.tag not in self.persistent_unit_registry:
                        self.persistent_unit_registry[p.tag] = (transport.position.x, transport.position.y, p.type_id.value, transport.owner_id)
        allied_units, allied_structures = [], []
        enemy_units, enemy_structures = [], []
        processed_tags = set()
        unit_status = {}
        
        # 1. Primary Allied/Enemy Units
        for u in self.all_units:
            if u.tag in processed_tags: continue
            processed_tags.add(u.tag)
            
            if u.owner_id == target_id:
                if u.is_structure: allied_structures.append(u)
                else: 
                    allied_units.append(u)
                    unit_status[u.tag] = {"is_idle": u.is_idle, "has_orders": len(u.orders) > 0}
            elif 0 < u.owner_id < 16:
                if u.is_structure: enemy_structures.append(u)
                else: enemy_units.append(u)

        # 2. Include Passengers (They are missing from all_units)
        # We need them to record samples for units that just boarded or are about to unboard.
        for transport in [u for u in allied_units if u.cargo_max > 0]:
            if hasattr(transport, "passengers"):
                for p in transport.passengers:
                    if p.tag in processed_tags: continue
                    processed_tags.add(p.tag)
                    
                    # Create a mock-up unit object that ObservationTransformer can handle
                    class PassengerMock:
                        def __init__(self, p, pos, target_id):
                            self.tag = p.tag
                            self.type_id = p.type_id
                            self.position = pos
                            self.owner_id = target_id
                            
                            def safe_get(name, default):
                                try: return getattr(p, name)
                                except: return default
                                
                            self.is_structure = False
                            self.is_flying = False
                            self.radius = 0.5
                            self.is_biological = safe_get("is_biological", False)
                            self.is_mechanical = safe_get("is_mechanical", False)
                            self.is_armored = safe_get("is_armored", False)
                            self.is_massive = safe_get("is_massive", False)
                            self.is_cloaked = safe_get("is_cloaked", False)
                            self.health = safe_get("health", 100)
                            self.health_max = safe_get("health_max", 100)
                            self.shield = safe_get("shield", 0)
                            self.shield_max = safe_get("shield_max", 0)
                            self.energy = safe_get("energy", 0)
                            self.energy_max = safe_get("energy_max", 0)
                            self.is_ready = True
                            self.is_hallucination = False
                            self.is_mine = True
                            self.detect_range = 0
                            self.cargo_max = 0
                            self.cargo_used = 0
                            self.weapon_cooldown = 0
                            self.orders = []
                            self.is_idle = True

                        def distance_to(self, target):
                            """
                            Mock distance calculation. 
                            In SC2, distance is usually from center to center.
                            """
                            if hasattr(target, "position"):
                                target_pos = target.position
                            else:
                                target_pos = target # Assume it's Point2
                            return ((self.position.x - target_pos.x)**2 + (self.position.y - target_pos.y)**2)**0.5
                    
                    mock_p = PassengerMock(p, transport.position, target_id)
                    allied_units.append(mock_p)
                    unit_status[p.tag] = {"is_idle": True, "has_orders": False}

        # 3. Calculate Action Masks for all allied units
        unit_masks = {}
        for u in allied_units:
            unit_masks[u.tag] = self.action_masks(u)

        # 4. Identify Mission Objective (Enemy Main Structure)
        main_structures = enemy_structures + [u for u in enemy_units if u.is_structure]
        objective_pos = None
        target_types = {
            UnitTypeId.COMMANDCENTER, UnitTypeId.HATCHERY, UnitTypeId.NEXUS,
            UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS, UnitTypeId.LAIR, UnitTypeId.HIVE
        }
        found_objective = next((u for u in main_structures if u.type_id in target_types), None)
        if found_objective:
            objective_pos = (found_objective.position.x, found_objective.position.y)

        actions = self._extract_human_actions()
        raw_obs_data = {
            'units': allied_units, 'structures': allied_structures,
            'enemy_units': enemy_units, 'enemy_structures': enemy_structures,
            'effects': [e for e in self.state.effects], 'creep': self.state.creep,
            'game_loop': self.state.game_loop, 'unit_status': unit_status,
            'unit_masks': unit_masks, 'objective_pos': objective_pos
        }
        unit_snapshot = {tag: data[:3] for tag, data in self.persistent_unit_registry.items()}
        self.recorded_data.append({
            "raw_obs": raw_obs_data, "actions": actions,
            "unit_snapshot": unit_snapshot, "game_loop": self.state.game_loop
        })

    def _extract_human_actions(self) -> List[Dict]:
        target_id = self.detected_observed_id or self.observed_id
        human_actions = []
        TRANSPORT_TYPES = {UnitTypeId.MEDIVAC, UnitTypeId.OVERLORD, UnitTypeId.WARPPRISM, UnitTypeId.OVERLORDTRANSPORT, UnitTypeId.WARPPRISMPHASING}
        TRANSPORT_IDS = {t.value for t in TRANSPORT_TYPES}
        
        for action in self.state.actions_unit_commands:
            if not action.unit_tags: continue
            sample_tag = list(action.unit_tags)[0]
            u_data = self.persistent_unit_registry.get(sample_tag)
            owner = u_data[3] if u_data else None
            u_type = u_data[2] if u_data else None
            if owner is None:
                u = self.all_units.find_by_tag(sample_tag)
                if u: owner, u_type = u.owner_id, u.type_id.value
            if owner is not None and owner != target_id: continue

            target_pos, intent = None, None
            target_unit = None
            if action.target_unit_tag:
                target_unit = self.all_units.find_by_tag(action.target_unit_tag)
                if target_unit:
                    target_pos = (target_unit.position.x, target_unit.position.y)
                    abil = action.ability_id
                    if hasattr(abil, "value"): abil = abil.value
                    if abil == AbilityId.SMART.value:
                        if target_unit.owner_id != target_id and target_unit.owner_id != 0: intent = "attack"
                        elif target_unit.type_id in TRANSPORT_TYPES: intent = "load"
                        elif u_type in TRANSPORT_IDS and target_unit.cargo_max == 0: intent = "load"
            elif action.target_world_space_pos:
                target_pos = (action.target_world_space_pos.x, action.target_world_space_pos.y)
            
            if target_pos or action.ability_id:
                human_actions.append({
                    "ability_id": action.ability_id, "intent_override": intent,
                    "unit_tags": list(action.unit_tags), "target_world_pos": target_pos,
                    "target_unit_tag": action.target_unit_tag or 0
                })
        return human_actions

async def _play_replay_fixed(client, ai, realtime=False, player_id=0):
    from sc2.game_state import GameState
    from train.envs.level_0_env import ProtoWrapper
    ai._initialize_variables()
    game_data = await client.get_game_data()
    game_info = await client.get_game_info()
    ping = await client.ping()
    ai._prepare_start(client, player_id, game_info, game_data, realtime=realtime, base_build=ping.ping.base_build)
    state = await client.observation()
    gs = GameState(state.observation)
    proto_game_info = await client._execute(game_info=sc_pb.RequestGameInfo())
    ai._prepare_step(gs, ProtoWrapper(proto_game_info.game_info))
    try: ai._prepare_first_step()
    except Exception: pass
    await ai.on_start()
    iteration = 0
    while True:
        if iteration != 0:
            state = await client.observation() if not realtime else await client.observation(gs.game_loop + client.game_step)
            if client._game_result: return client._game_result[player_id]
            gs = GameState(state.observation)
            proto_game_info = await client._execute(game_info=sc_pb.RequestGameInfo())
            ai._prepare_step(gs, ProtoWrapper(proto_game_info.game_info))
        await ai.issue_events()
        await ai.on_step(iteration)
        await ai._after_step()
        if not realtime and not client.in_game: return Result.Victory
        await client.step()
        iteration += 1

async def run_replay_fixed(ai, replay_path, realtime=False, observed_id=1):
    base_build, data_version = get_replay_version(replay_path)
    async with SC2Process(fullscreen=False, base_build=base_build, data_hash=data_version) as server:
        client = await _setup_replay(server, replay_path, realtime, observed_id)
        return await _play_replay_fixed(client, ai, realtime, player_id=observed_id)

class ReplayTranslator:
    def __init__(self, project_root: str = "."):
        self.project_root = os.path.abspath(project_root)
        self.translated_dir = os.path.join(self.project_root, "imitation_learning", "replays_translated")

    def translate(self, replay_path: str, level_name: str, config: Dict[str, Any]):
        observed_id = config.get("env", {}).get("observed_id", config.get("observed_id", 1))
        bot = ReplayTranslatorBot(config, observed_id=observed_id)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_replay_fixed(bot, replay_path=replay_path, realtime=False, observed_id=observed_id))
        initial_units = bot.recorded_data[0].get("unit_snapshot", {}) if bot.recorded_data else {}
        translated_pairs = self.process_raw_data(bot.recorded_data, config, bot.game_info)
        self.save_translated_data(translated_pairs, level_name, os.path.basename(replay_path), initial_units)
        return len(translated_pairs)

    def process_raw_data(self, recorded_data: List[Dict], config: Dict[str, Any], game_info: Any) -> List[Dict]:
        from src.transformers.observation_transformer import ObservationTransformer
        final_pairs, action_heads = [], config.get("env", {}).get("action_heads", ["move"])
        categorical_heads, grid_size = config.get("env", {}).get("categorical_heads", []), config.get("env", {}).get("grid_size", 128)
        transformer = ObservationTransformer(map_size=(grid_size, grid_size), training_conf=config.get("env", {}))
        area = getattr(game_info, 'playable_area', None)
        crop_x, crop_y = (area.x, area.y) if area else (0, 0)
        mw, mh = (area.width, area.height) if area else (game_info.map_size.width, game_info.map_size.height)
        scale = min(grid_size / max(mw, 1), grid_size / max(mh, 1))
        ox, oy = (grid_size - int(mw * scale)) // 2, (grid_size - int(mh * scale)) // 2
        grid_area = grid_size * grid_size
        ABILITY_TO_HEAD = {}
        def add_abil(n, h): 
            if hasattr(AbilityId, n): ABILITY_TO_HEAD[getattr(AbilityId, n).value] = h
        add_abil("SMART","move"); add_abil("MOVE","move"); add_abil("ATTACK","attack"); add_abil("ATTACK_ATTACK","attack")
        add_abil("LOAD_MEDIVAC","load"); add_abil("LOAD_OVERLORD","load"); add_abil("LOAD_WARPPRISM","load")
        add_abil("LOAD","load"); add_abil("LOADALL","load")
        add_abil("UNLOADALLAT_MEDIVAC","unload"); add_abil("UNLOADALLAT_OVERLORD","unload"); add_abil("UNLOADALLAT_WARPPRISM","unload")
        add_abil("UNLOADALL_MEDIVAC","unload"); add_abil("UNLOADALL_OVERLORD","unload"); add_abil("UNLOADALL_WARPPRISM","unload")
        add_abil("UNLOADALL","unload"); add_abil("UNLOADALLAT","unload")
        last_unit_actions = {}
        for entry in recorded_data:
            raw_obs_data, new_human_actions, loop = entry["raw_obs"], entry["actions"], entry["game_loop"]
            unit_status = raw_obs_data.get("unit_status", {})
            
            # Keep track of units that received a command IN THIS STEP
            just_commanded = set()
            
            for ha in new_human_actions:
                abil = ha["ability_id"].value if hasattr(ha["ability_id"], "value") else ha["ability_id"]
                head_name = ha.get("intent_override") or ABILITY_TO_HEAD.get(abil, "move")
                if head_name in action_heads or head_name in categorical_heads:
                    action_data = {"target_pos": ha["target_world_pos"], "target_unit_tag": ha.get("target_unit_tag") or 0, "head_name": head_name, "unit_tags": ha["unit_tags"]}
                    for tag in ha["unit_tags"]:
                        last_unit_actions[tag] = action_data
                        just_commanded.add(tag)
                    if head_name == "load" and action_data["target_unit_tag"]:
                        target_tag = action_data["target_unit_tag"]
                        if target_tag in last_unit_actions: last_unit_actions.pop(target_tag)
            active_samples = [] 
            for unit in raw_obs_data['units']:
                tag, status = unit.tag, unit_status.get(unit.tag, {})
                if tag in last_unit_actions:
                    # Robust condition: Record if unit is busy OR if it just received a command
                    if tag in just_commanded or not status.get("is_idle", True) or status.get("has_orders", False):
                        ad = last_unit_actions[tag]
                        target_pos = ad["target_pos"]
                        if ad["target_unit_tag"]:
                            all_current = raw_obs_data['units'] + raw_obs_data['structures'] + \
                                          raw_obs_data['enemy_units'] + raw_obs_data['enemy_structures']
                            found = next((u for u in all_current if u.tag == ad["target_unit_tag"])
, None)
                            if found: target_pos = (found.position.x, found.position.y)
                        if not target_pos: target_pos = (0,0)
                        if ad["head_name"] in action_heads:
                            head_idx = action_heads.index(ad["head_name"])
                            px = int((target_pos[0] - crop_x) * scale + ox)
                            py = int(grid_size - 1 - ((target_pos[1] - crop_y) * scale + oy))
                            agent_action = head_idx * (grid_area) + (np.clip(py, 0, grid_size-1) * grid_size + np.clip(px, 0, grid_size-1))
                        else: agent_action = len(action_heads) * (grid_area) + categorical_heads.index(ad["head_name"])
                        
                        # Get pre-calculated action mask
                        action_mask = raw_obs_data.get("unit_masks", {}).get(tag)
                        if action_mask is None:
                            # Fallback to all-True if missing for some reason
                            action_mask = np.ones(grid_area * len(action_heads) + len(categorical_heads), dtype=bool)
                            
                        active_samples.append((unit, agent_action, target_pos, ad["head_name"], ad["target_unit_tag"], action_mask))
                    else: last_unit_actions.pop(tag)
            for unit, agent_action, target_pos, head_name, replay_target_tag, action_mask in active_samples:
                # Use mission objective as command_target if available, otherwise fallback to action target
                final_command_target = raw_obs_data.get('objective_pos') or target_pos
                frame_raw_obs = {'units': raw_obs_data['units'], 'structures': raw_obs_data['structures'], 'enemy_units': raw_obs_data['enemy_units'], 'enemy_structures': raw_obs_data['enemy_structures'], 'effects': raw_obs_data['effects'], 'creep': raw_obs_data['creep'], 'game_info': game_info, 'game_data': None, 'selected_units': [unit], 'command_target': Point2(final_command_target)}
                unit_obs = transformer.transform(frame_raw_obs)
                acting_data = [{"tag": unit.tag, "type": unit.type_id.value, "pos": (unit.position.x, unit.position.y)}]
                final_pairs.append({"observation": unit_obs, "action": agent_action, "acting_data": acting_data, "target_pos": target_pos, "target_unit_tag": replay_target_tag, "game_loop": loop, "action_mask": action_mask})
        return final_pairs

    def save_translated_data(self, pairs: List[Dict], level_name: str, original_filename: str, initial_units: Dict):
        if not pairs: return
        level_dir = os.path.join(self.translated_dir, level_name)
        os.makedirs(level_dir, exist_ok=True)
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
        logger.info(f"Saved {len(pairs)} examples to {output_path}")
