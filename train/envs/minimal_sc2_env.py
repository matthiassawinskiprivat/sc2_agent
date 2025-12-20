import gymnasium as gym
import numpy as np
import asyncio
from sc2.sc2process import SC2Process
from sc2.client import Client
from sc2.protocol import Protocol
from sc2.data import Race, Difficulty
from sc2.player import Bot, Computer
from sc2.maps import Map
from pathlib import Path
import os
import random
import time

class MinimalSC2Env(gym.Env):
    """
    A minimal SC2 environment that correctly handles the SC2Process lifecycle.
    Used to verify multiprocessing stability with actual SC2 instances.
    """
    def __init__(self, visualize=False, map_name="00_32x32_No_Middle"):
        super().__init__()
        self.visualize = visualize
        self.map_name = map_name
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 64, 64), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        
        self.sc2_process = None
        self.client = None
        self.loop = None

    def reset(self, seed=None, options=None):
        worker_id = os.getpid()
        print(f"[{worker_id}] Resetting environment...")
        if self.sc2_process:
            self.close()
            
        # Randomized sleep to avoid simultaneous SC2 launches
        time.sleep(random.random() * 5)
        
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._launch_sc2())
        except Exception as e:
            print(f"[{worker_id}] Error launching SC2: {e}")
            self.close()
            raise
            
        print(f"[{worker_id}] Reset complete.")
        return self.observation_space.sample(), {}

    async def _launch_sc2(self):
        # Based on dump_sc2process.py, we should use the context manager 
        # to get a Controller which has create_game.
        self.sc2_process = SC2Process(fullscreen=self.visualize)
        self.controller = await self.sc2_process.__aenter__()
        
        self.client = Client(self.sc2_process._ws)
        
        # Minimal game creation
        current_dir = Path(__file__).resolve().parent.parent
        map_path = current_dir / "maps" / "level_0" / f"{self.map_name}.SC2Map"
        
        # controller.create_game is the correct method
        await self.controller.create_game(
            Map(map_path),
            [Bot(Race.Zerg, None), Computer(Race.Zerg, Difficulty.Easy)],
            realtime=self.visualize,
        )
        await self.client.join_game(race=Race.Zerg)

    def step(self, action):
        # Minimal step: just advance the game
        if self.client:
            self.loop.run_until_complete(self.client.step())
        return self.observation_space.sample(), 0.0, False, False, {}

    def close(self):
        if self.sc2_process:
            # We should use __aexit__ to clean up properly
            if self.loop and self.loop.is_running():
                try:
                    self.loop.run_until_complete(self.sc2_process.__aexit__())
                except:
                    self.sc2_process._clean()
            else:
                self.sc2_process._clean()
            self.sc2_process = None
        self.client = None
        if self.loop:
            self.loop.close()
            self.loop = None

if __name__ == "__main__":
    # Test it
    env = MinimalSC2Env()
    try:
        print("Resetting...")
        env.reset()
        print("Stepping...")
        env.step(0)
        print("Closing...")
    finally:
        env.close()
    print("Done.")