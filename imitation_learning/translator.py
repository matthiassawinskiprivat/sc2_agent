import os
import numpy as np
from typing import List, Dict, Any, Optional
from sc2.bot_ai import BotAI
from sc2.main import run_replay
from sc2.data import Result
from loguru import logger
from src.hierarchical_agent import HierarchicalAgentSystem

class ReplayTranslatorBot(BotAI):
    """
    A special bot that runs during replay playback to extract observations and actions.
    """
    def __init__(self, training_conf: Dict[str, Any]):
        super().__init__()
        self.training_conf = training_conf
        self.agent_system: Optional[HierarchicalAgentSystem] = None
        
        # Data collection
        self.recorded_data = []

    async def on_start(self):
        self.agent_system = HierarchicalAgentSystem(
            map_size=(128, 128),
            grid_size=128,
            training_conf=self.training_conf
        )
        self.agent_system.create_unit_agent("agent_0")
        logger.info("ReplayTranslatorBot started.")

    async def on_step(self, iteration: int):
        # 1. Get current observation (Agent's view)
        obs = self._get_agent_observation()
        
        # 2. Extract Human Actions from this frame
        actions = self._extract_human_actions()
        
        if actions:
            self.recorded_data.append({
                "observation": obs,
                "actions": actions
            })

    def _get_agent_observation(self) -> np.ndarray:
        """Constructs the feature layer stack for the current frame."""
        # Use existing logic from SC2Bot but adapted for replay state
        # (Need to mock/find beacon if it's missing in replay)
        raw_obs = {
            'units': self.units,
            'enemy_units': self.enemy_units, # In replay, we see everything or based on player vision
            'effects': self.state.effects,
            'game_info': self.game_info,
            'game_data': self.game_data,
            'creep': self.state.creep,
            'command_target': None # Will be handled by translation logic
        }
        processed = self.agent_system.process_observation(observation=raw_obs)["agent_0"]
        return processed

    def _extract_human_actions(self) -> List[Dict]:
        """Parses the raw actions from the replay state."""
        human_actions = []
        # self.state.observation.actions contains the actions issued in this frame
        raw_actions = self.state.observation.actions
        
        for action in raw_actions:
            # We are interested in ActionRaw (unit commands)
            if action.HasField("action_raw"):
                raw = action.action_raw
                if raw.HasField("unit_command"):
                    cmd = raw.unit_command
                    human_actions.append({
                        "ability_id": cmd.ability_id,
                        "unit_tags": list(cmd.unit_tags),
                        "target_world_pos": (cmd.target_world_space_pos.x, cmd.target_world_space_pos.y) if cmd.HasField("target_world_space_pos") else None,
                        "target_unit_tag": cmd.target_unit_tag if cmd.HasField("target_unit_tag") else None
                    })
        return human_actions

class ReplayTranslator:
    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.translated_dir = os.path.join(project_root, "imitation_learning", "replays_translated")

    def translate(self, replay_path: str, level_name: str, config: Dict[str, Any]):
        bot = ReplayTranslatorBot(config)
        
        logger.info(f"Running replay translation for {replay_path}")
        run_replay(bot, replay_path=replay_path, realtime=False)
        
        # Now translate the collected raw data into (Obs, Action) pairs
        translated_pairs = self.process_raw_data(bot.recorded_data, config)
        
        self.save_translated_data(translated_pairs, level_name, os.path.basename(replay_path))
        return len(translated_pairs)

    def process_raw_data(self, recorded_data: List[Dict], config: Dict[str, Any]) -> List[Dict]:
        """
        Translates raw (Observation, HumanActions) into many (Observation, AgentAction) pairs.
        Handles multi-unit decomposition.
        """
        final_pairs = []
        action_heads = config.get("env", {}).get("action_heads", ["move"])
        grid_size = config.get("env", {}).get("grid_size", 128)
        
        # Map Ability IDs to our Action Heads
        # This is a simplified mapping for now
        # 1 = SMART (Right click), 16 = MOVE, 23 = ATTACK
        ABILITY_TO_HEAD = {
            1: "move", # Smart defaults to move unless we detect an enemy target later
            16: "move",
            23: "attack",
            # Add more as needed (Load/Unload)
        }

        for entry in recorded_data:
            obs = entry["observation"] # This is the full feature stack [C, 128, 128]
            human_actions = entry["actions"]
            
            for ha in human_actions:
                head_name = ABILITY_TO_HEAD.get(ha["ability_id"], "move")
                if head_name not in action_heads:
                    continue
                
                head_idx = action_heads.index(head_name)
                
                # Determine Spatial Index
                if ha["target_world_pos"]:
                    wx, wy = ha["target_world_pos"]
                    # Simplified conversion (assuming map_size 128x128 for now)
                    # Real implementation should use agent's conversion logic
                    gx = int(np.clip(wx, 0, grid_size - 1))
                    gy = int(np.clip(wy, 0, grid_size - 1))
                    spatial_idx = gy * grid_size + gx
                    
                    agent_action = head_idx * (grid_size * grid_size) + spatial_idx
                    
                    # DECOMPOSITION: One example per unit in selection
                    for unit_tag in ha["unit_tags"]:
                        # In a real implementation, the observation might need to be 
                        # slightly modified to highlight the "Active Unit"
                        final_pairs.append({
                            "observation": obs,
                            "action": agent_action
                        })
                        
        return final_pairs

    def save_translated_data(self, pairs: List[Dict], level_name: str, original_filename: str):
        if not pairs:
            logger.warning("No data to save.")
            return

        level_dir = os.path.join(self.translated_dir, level_name)
        os.makedirs(level_dir, exist_ok=True)
        
        base_name = os.path.splitext(original_filename)[0]
        output_path = os.path.join(level_dir, f"{base_name}.npz")
        
        obs_array = np.array([p["observation"] for p in pairs], dtype=np.uint8)
        act_array = np.array([p["action"] for p in pairs], dtype=np.int32)
        
        np.savez_compressed(output_path, observations=obs_array, actions=act_array)
        logger.info(f"Saved {len(pairs)} examples to {output_path}")

