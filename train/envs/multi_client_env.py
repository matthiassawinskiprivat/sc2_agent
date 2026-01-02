import gymnasium as gym
import asyncio
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import random
import time

from src.utils.multi_client_manager import MultiClientSC2Manager
from sc2.player import Bot, Computer, AbstractPlayer
from sc2.data import Race, Difficulty
from sc2.maps import Map

class MultiClientEnv(gym.Env):
    """
    Base environment for multi-client SC2 games.
    Manages two SC2 processes and synchronizes their steps.
    """
    def __init__(self, map_name: str = "Simple64", visualize: bool = False, realtime: bool = False):
        self.map_name = map_name
        self.visualize = visualize
        self.realtime = realtime
        
        self._manager = MultiClientSC2Manager()
        self._loop = asyncio.new_event_loop()
        
        # Placeholder spaces - should be defined by subclasses
        self.observation_space = None
        self.action_space = None
        
        self.players: List[AbstractPlayer] = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Ensure clean state
        if self._manager.clients:
             self.close()
             
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._launch_sc2_processes())
            self._loop.run_until_complete(self._setup_game_session())
            
            # Initial observation
            obs = self._loop.run_until_complete(self._get_joint_observation())
            return obs, {}
        except Exception as e:
            self.close()
            raise e

    def step(self, actions: List[Any]):
        """
        Executes a synchronized step for both players.
        actions: [action_p1, action_p2]
        """
        try:
            results = self._loop.run_until_complete(self._async_step(actions))
            return results
        except Exception as e:
            self.close()
            raise e

    async def _launch_sc2_processes(self):
        await self._manager.launch_processes(count=2, fullscreen=self.visualize)

    async def _setup_game_session(self):
        # Must be implemented by subclass to define players/bots
        pass

    async def _async_step(self, actions: List[Any]):
        # 1. Distribute actions
        if len(actions) != 2:
            raise ValueError("Expected 2 actions for 2 players")
            
        await self._apply_actions(actions)
        
        # 2. Step physics
        await asyncio.gather(
            self._manager.host_client.step(),
            self._manager.join_client.step()
        )
        
        # 3. Get results
        return await self._get_step_results()

    async def _apply_actions(self, actions):
        # Override in subclass
        pass

    async def _get_joint_observation(self):
        # Override in subclass
        return [None, None]

    async def _get_step_results(self):
        # Override in subclass
        obs = await self._get_joint_observation()
        return obs, [0, 0], [False, False], [False, False], {}

    def close(self):
        if self._loop and not self._loop.is_closed():
            self._loop.run_until_complete(self._manager.close())
            self._loop.close()
