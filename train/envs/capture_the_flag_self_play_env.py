import gymnasium as gym
import asyncio
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import random
from gymnasium import spaces
from loguru import logger
import os
import json

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from sc2.ids.unit_typeid import UnitTypeId

class CaptureTheFlagSelfPlayEnv(CaptureTheFlagSingleEnv):
    """
    Self-Play environment using a single SC2 process and God-Mode (debug_control_enemy).
    This avoids the instability of multi-process handshakes.
    """
    def __init__(self, 
                 opponent_model = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.opponent_model = opponent_model
        
        # Override observation space to ensure it's correct for the perspective flipping
        # (Though SingleEnv already does this)
        num_layers = len(self.active_layers)
        self.observation_space = spaces.Box(low=0, high=255, shape=(num_layers, 128, 128), dtype=np.uint8)

    def set_opponent_model(self, model):
        self.opponent_model = model

    def step(self, action: Any):
        """
        Modified step to automatically handle Player 2's turns using the opponent model.
        """
        # 1. Execute Trainee (P1) Turn
        # If it's currently P2's turn for some reason, we should probably handle it,
        # but normally we start at P1.
        
        obs, reward, terminated, truncated, info = super().step(action)
        
        if terminated or truncated:
            return obs, reward, terminated, truncated, info
            
        # 2. Auto-Play Opponent (P2) until it is P1's turn again or game ends
        # In CaptureTheFlagSingleEnv, the active_player_id cycles automatically.
        while self.active_player_id == 2 and not terminated and not truncated:
            # Get P2 Observation
            # Note: _get_joint_observation in CaptureTheFlagSingleEnv already handles 
            # perspective based on self.active_player_id
            obs_p2 = self._loop.run_until_complete(self._get_joint_observation())
            
            # Predict P2 Action
            if self.opponent_model:
                action_p2, _ = self.opponent_model.predict(obs_p2, deterministic=True)
            else:
                action_p2 = self.action_space.sample()
                
            # Execute P2 Action
            obs_next, r_p2, terminated, truncated, info_p2 = super().step(action_p2)
            
            # Note: The reward returned by super().step() is for the ACTING player.
            # If P2 won, r_p2 is positive. 
            # We don't necessarily return this to the Trainee's learn() call here,
            # but the Trainee needs to know if they LOST.
            # CaptureTheFlagSingleEnv reward logic:
            # if p1_lost: reward += loss_penalty if acting_id == 1 else win_reward
            # Wait, if acting_id == 2 and p1_lost, P2 gets win_reward. 
            # We want P1 to get loss_penalty!
            
            # Let's verify the reward calculation for P1
            if terminated or truncated:
                # We need to calculate the final reward for P1 (the Trainee)
                p1_reward = self._calculate_final_p1_reward(terminated, truncated)
                return obs_next, p1_reward, terminated, truncated, info_p2
            
        # Back to P1 turn
        return obs, reward, terminated, truncated, info

    def _calculate_final_p1_reward(self, terminated, truncated):
        """Helper to get P1's perspective on the game outcome."""
        r_conf = self.training_conf.get("rewards", {})
        win_reward = r_conf.get("win_reward", 100.0)
        loss_penalty = r_conf.get("loss_penalty", -100.0)
        draw_penalty = r_conf.get("draw_penalty", -10.0)
        
        p1_lost = self.bot.structures(UnitTypeId.PYLON).amount == 0
        p2_lost = self.bot.enemy_structures(UnitTypeId.PYLON).amount == 0
        p1_army_lost = not self._has_combat_units(1)
        p2_army_lost = not self._has_combat_units(2)
        
        if p1_lost: return loss_penalty
        if p2_lost: return win_reward
        if p1_army_lost and p2_army_lost: return draw_penalty
        if p1_army_lost: return loss_penalty
        if p2_army_lost: return win_reward
        if truncated: return draw_penalty
        
        return 0.0