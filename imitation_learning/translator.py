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
    def __init__(self, training_conf: Dict[str, Any]):
        super().__init__()
        self.training_conf = training_conf
        self.agent_system: Optional[HierarchicalAgentSystem] = None
        self.recorded_data = []
        self.persistent_unit_registry = {}

    async def on_start(self):
        self.agent_system = HierarchicalAgentSystem(
            map_size=(128, 128),
            grid_size=128,
            training_conf=self.training_conf
        )
        self.agent_system.create_unit_agent("agent_0")
        logger.info("ReplayTranslatorBot started.")

    async def on_step(self, iteration: int):
        # Update persistent registry with currently visible units
        for u in self.all_units:
            self.persistent_unit_registry[u.tag] = (u.position.x, u.position.y, u.type_id.value)
            
        obs = self._get_agent_observation()
        actions = self._extract_human_actions()
        if actions or (self.state.game_loop % 10 == 0):
            # Use persistent registry for the snapshot to ensure we have data for units that might have just entered a transport
            unit_snapshot = self.persistent_unit_registry.copy()
            self.recorded_data.append({
                "observation": obs,
                "actions": actions,
                "unit_snapshot": unit_snapshot,
                "game_loop": self.state.game_loop
            })

    def _get_agent_observation(self) -> np.ndarray:
        raw_obs = {
            'units': self.units,
            'enemy_units': self.enemy_units,
            'effects': self.state.effects,
            'game_info': self.game_info,
            'game_data': self.game_data,
            'creep': self.state.creep,
            'command_target': None
        }
        return self.agent_system.process_observation(observation=raw_obs)["agent_0"]

    def _extract_human_actions(self) -> List[Dict]:
        human_actions = []
        TRANSPORT_TYPES = {UnitTypeId.MEDIVAC, UnitTypeId.OVERLORD, UnitTypeId.WARPPRISM, UnitTypeId.OVERLORDTRANSPORT}
        for action in self.state.actions_unit_commands:
            target_pos, intent = None, None
            if action.target_world_space_pos:
                target_pos = (action.target_world_space_pos.x, action.target_world_space_pos.y)
            elif action.target_unit_tag:
                target_unit = self.all_units.find_by_tag(action.target_unit_tag)
                if target_unit:
                    target_pos = (target_unit.position.x, target_unit.position.y)
                    abil = action.ability_id
                    if hasattr(abil, "value"): abil = abil.value
                    if abil == AbilityId.SMART.value:
                        if target_unit.alliance == Alliance.Enemy: intent = "attack"
                        elif target_unit.type_id in TRANSPORT_TYPES: intent = "load"
            
            if target_pos or action.ability_id:
                human_actions.append({
                    "ability_id": action.ability_id,
                    "intent_override": intent,
                    "unit_tags": list(action.unit_tags),
                    "target_world_pos": target_pos,
                    "target_unit_tag": action.target_unit_tag
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
    ai._prepare_first_step()
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
        bot = ReplayTranslatorBot(config)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_replay_fixed(bot, replay_path=replay_path, realtime=False, observed_id=1))
        initial_units = bot.recorded_data[0].get("unit_snapshot", {}) if bot.recorded_data else {}
        translated_pairs = self.process_raw_data(bot.recorded_data, config, bot.game_info)
        self.save_translated_data(translated_pairs, level_name, os.path.basename(replay_path), initial_units)
        return len(translated_pairs)

    def process_raw_data(self, recorded_data: List[Dict], config: Dict[str, Any], game_info: Any) -> List[Dict]:
        final_pairs, action_heads = [], config.get("env", {}).get("action_heads", ["move"])
        categorical_heads, grid_size = config.get("env", {}).get("categorical_heads", []), config.get("env", {}).get("grid_size", 128)
        
        # EXACT PROJECTION (Matches ObservationTransformer)
        area = getattr(game_info, 'playable_area', None)
        crop_x, crop_y = (area.x, area.y) if area else (0, 0)
        mw, mh = (area.width, area.height) if area else (game_info.map_size.width, game_info.map_size.height)
        scale = min(grid_size / max(mw, 1), grid_size / max(mh, 1))
        ox, oy = (grid_size - int(mw * scale)) // 2, (grid_size - int(mh * scale)) // 2

        ABILITY_TO_HEAD = {}
        def add_abil(n, h): 
            if hasattr(AbilityId, n): ABILITY_TO_HEAD[getattr(AbilityId, n).value] = h
        add_abil("SMART","move"); add_abil("MOVE","move"); add_abil("ATTACK","attack"); add_abil("ATTACK_ATTACK","attack")
        add_abil("LOAD_MEDIVAC","load"); add_abil("LOAD_OVERLORD","load"); add_abil("LOAD_WARPPRISM","load")
        add_abil("UNLOADALLAT_MEDIVAC","unload"); add_abil("UNLOADALLAT_OVERLORD","unload"); add_abil("UNLOADALLAT_WARPPRISM","unload")

        for entry in recorded_data:
            obs, human_actions, loop = entry["observation"], entry["actions"], entry["game_loop"]
            unit_snapshot = entry["unit_snapshot"]
            if not human_actions:
                final_pairs.append({"observation": obs, "action": -1, "acting_data": [], "target_pos": (0,0), "game_loop": loop})
                continue
            for ha in human_actions:
                abil = ha["ability_id"]
                if hasattr(abil, "value"): abil = abil.value
                head_name = ha.get("intent_override") or ABILITY_TO_HEAD.get(abil, "move")
                if head_name not in action_heads and head_name not in categorical_heads: continue
                
                target_pos = ha["target_world_pos"] or (0,0)
                if head_name in action_heads:
                    head_idx = action_heads.index(head_name)
                    # PROJECT USING CROP
                    px = int((target_pos[0] - crop_x) * scale + ox)
                    py = int(grid_size - 1 - ((target_pos[1] - crop_y) * scale + oy))
                    agent_action = head_idx * (grid_size * grid_size) + (np.clip(py, 0, grid_size-1) * grid_size + np.clip(px, 0, grid_size-1))
                else:
                    agent_action = len(action_heads) * (grid_size * grid_size) + categorical_heads.index(head_name)

                # Track acting units and their types
                acting_data = []
                for tag in ha["unit_tags"]:
                    if tag in unit_snapshot:
                        acting_data.append({"tag": tag, "type": unit_snapshot[tag][2], "pos": (unit_snapshot[tag][0], unit_snapshot[tag][1])})

                final_pairs.append({
                    "observation": obs, 
                    "action": agent_action, 
                    "acting_data": acting_data, 
                    "target_pos": target_pos,
                    "game_loop": loop
                })
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
                            initial_units=np.array([list(initial_units.items())], dtype=object))
        logger.info(f"Saved {len(pairs)} examples to {output_path}")
