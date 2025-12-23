import gymnasium as gym
import numpy as np
import os
import time
import random
import psutil
import asyncio
import aiohttp
from pathlib import Path
from gymnasium import spaces

from sc2 import maps
from sc2.client import Client
from sc2.data import Race, Difficulty
from sc2.main import run_game
from sc2.maps import Map
from sc2.player import Bot, Computer
from sc2.protocol import ConnectionAlreadyClosed, ProtocolError, Protocol
from sc2.sc2process import SC2Process

from src.sc2_bot import SC2Bot
from src.hierarchical_agent import HierarchicalAgentSystem

HARD_RESET_COUNT = 10

class ProtoWrapper:
    def __init__(self, info):
        self.game_info = info

class Level0Env(gym.Env):
    def __init__(self, visualize=False, grid_size: int = 16, worker_index: int = 0, num_workers: int = 1,
                 level_name: str = "level_0"):
        super(Level0Env, self).__init__()
        self.visualize = visualize
        self.grid_size = grid_size
        self.worker_index = worker_index
        self.num_workers = num_workers
        self.level_name = level_name
        self.episode_counter = 0

        self.map_list = self._discover_maps()
        if not self.map_list:
            raise FileNotFoundError(f"No maps found in train/maps/{self.level_name}/")
        self.map_name = self.map_list[self.worker_index % len(self.map_list)]

        self.worker_id = f"{random.randint(1000, 9999)}"
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/worker_{self.worker_id}.log"
        self._log(f"Initialized Worker {self.worker_id} (Index: {self.worker_index}) with map {self.map_name}")

        self.observation_space = spaces.Box(low=0, high=255, shape=(42, 128, 128), dtype=np.uint8)
        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)

        self._sc2_process = None
        self._client = None
        self._controller = None
        self._bot = None
        self._loop = None

        self.steps = 0
        self.max_steps = 300

    def _discover_maps(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        maps_dir = Path(current_dir).parent / "maps" / self.level_name
        if not maps_dir.exists():
            return []
        map_files = [f.stem for f in maps_dir.glob("*.SC2Map")]
        map_files.sort()
        return map_files

    def _log(self, message):
        # Print to stdout too so it gets captured by redirection
        print(f"[Worker {self.worker_id}] {message}")
        try:
            with open(self.log_file, "a") as f:
                timestamp = time.strftime("%H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
        except:
            pass

    def reset(self, seed=None, options=None):
        self._log(f"reset() called (Episode: {self.episode_counter})")
        self.steps = 0
        self.episode_counter += 1
        
        # --- Hard Reset / Map Rotation Logic ---
        if self.episode_counter > 1 and (self.episode_counter + self.worker_index) % HARD_RESET_COUNT == 0:
            self._log(f"Episode {self.episode_counter}: Triggering hard reset.")
            self.close()
            
            resets_done = (self.episode_counter + self.worker_index) // HARD_RESET_COUNT
            map_idx = (self.worker_index + resets_done * self.num_workers) % len(self.map_list)
            self.map_name = self.map_list[map_idx]
            self._log(f"New map: {self.map_name}")
        
        # Robust setup loop
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self._sc2_process is None:
                    self._log(f"Attempting to launch SC2 process (Attempt {attempt+1})...")
                    self._loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._loop)
                    # Launch process with 60s timeout
                    self._loop.run_until_complete(asyncio.wait_for(self._launch_sc2_process(), timeout=60.0))
                    # Setup session with 60s timeout
                    self._loop.run_until_complete(asyncio.wait_for(self._setup_game_session(), timeout=60.0))
                else:
                    self._log("Resetting bot state (with timeout)...")
                    self._loop.run_until_complete(asyncio.wait_for(self._reset_bot_state(), timeout=30.0))
                
                # Success
                break
            except Exception as e:
                import traceback
                self._log(f"Error during environment reset: {e}")
                self.close() # Ensure full cleanup
                if attempt == max_retries - 1:
                    self._log(f"Max retries reached. Raising error.\n{traceback.format_exc()}")
                    raise e
                time.sleep(1.0)

        obs, _ = self._bot._get_observation()
        self._log("reset() complete.")
        return obs, {}

    async def _reset_bot_state(self):
        self._log("Resetting bot state...")
        await self._client.step()
        observation = await self._client.observation()
        from sc2.game_state import GameState
        state = GameState(observation.observation)
        self._bot._prepare_step(state, ProtoWrapper(self._bot.game_info._proto))
        await self._bot.reset_game_state()
        await self._client.step()
        observation = await self._client.observation()
        state = GameState(observation.observation)
        self._bot._prepare_step(state, ProtoWrapper(self._bot.game_info._proto))
    
    async def _launch_sc2_process(self):
        await asyncio.sleep(random.random() * 5)
        self._log("Launching SC2 process...")
        self._sc2_process = SC2Process(fullscreen=self.visualize)
        self._controller = await self._sc2_process.__aenter__()
        self._client = Client(self._sc2_process._ws)
        # Give the client a moment to stabilize
        await asyncio.sleep(0.5)

    async def _setup_game_session(self):
        # Verify connection
        try:
             await self._client.ping()
        except Exception as e:
             self._log(f"Initial ping failed: {e}")
             raise e

        game_map = self._resolve_map(self.map_name)
        if self._bot is None:
            self._log("Creating SC2Bot...")
            class DummyConn:
                def send(self, data): pass
                def recv(self): return "continue", None
            self._bot = SC2Bot(conn=DummyConn())
            self._bot.distance_calculation_method = 2
            self._bot.grid_size = self.grid_size

        self._log("Creating game...")
        await self._controller.create_game(
            game_map,
            [Bot(Race.Zerg, self._bot), Computer(Race.Zerg, Difficulty.Easy)],
            realtime=self.visualize,
        )
        await self._client.join_game(race=Race.Zerg)
        self._bot._generated_frame = -1
        game_info = await self._client.get_game_info()
        game_data = await self._client.get_game_data()

        self._log("Preparing bot start...")
        try:
            from sc2.unit import UnitGameData
            UnitGameData._game_data = game_data
            UnitGameData._bot_object = self._bot
        except ImportError:
            pass
        
        self._bot._prepare_start(client=self._client, player_id=1, game_info=game_info, game_data=game_data)
        initial_observation = await self._client.observation()
        from sc2.game_state import GameState
        state = GameState(initial_observation.observation)
        self._bot._prepare_step(state, ProtoWrapper(game_info._proto))
        try:
            await self._bot.on_step(0)
        except Exception as e:
            self._log(f"Error in on_step(0): {e}")
            raise

    def step(self, action):
        self.steps += 1
        try:
            obs, reward, done, truncated, info = self._loop.run_until_complete(self._async_step(action))
        except (ProtocolError, ConnectionAlreadyClosed) as e:
            self._log(f"Connection to game lost: {e}")
            obs, reward, done, truncated, info = np.zeros(self.observation_space.shape, dtype=np.uint8), 0, True, True, {}
        if self.steps >= self.max_steps:
            truncated = True
            done = True
            reward = -10
        return obs, reward, done, truncated, info

    async def _async_step(self, action):
        world_x, world_y = self._bot.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world(action)
        from sc2.position import Point2
        self._bot.last_target_pos = Point2((world_x, world_y))
        if self._bot.zergling:
            cmd = self._bot.zergling.move(self._bot.last_target_pos)
            if cmd and not isinstance(cmd, bool):
                await self._bot.do(cmd)
        await self._client.step()
        response = await self._client.observation()
        from sc2.game_state import GameState
        state = GameState(response.observation)
        self._bot._prepare_step(state, ProtoWrapper(self._bot.game_info._proto))
        return self._bot._get_step_results()

    def close(self):
        self._log("close() called.")
        if self._sc2_process:
            if self._loop and not self._loop.is_closed():
                try:
                    self._log("Attempting to close SC2 connection (with timeout)...")
                    if hasattr(self._sc2_process, "_close_connection"):
                         try:
                             self._loop.run_until_complete(asyncio.wait_for(self._sc2_process._close_connection(), timeout=5.0))
                         except asyncio.TimeoutError:
                             self._log("Timeout while closing SC2 connection.")
                    
                    self._log("Attempting to kill SC2 process (with timeout)...")
                    try:
                        self._loop.run_until_complete(asyncio.wait_for(self._sc2_process.__aexit__(None, None, None), timeout=5.0))
                    except asyncio.TimeoutError:
                        self._log("Timeout while killing SC2 process.")
                        self._sc2_process._clean()
                except Exception as e:
                    self._log(f"Error during close: {e}")
                    self._sc2_process._clean()
            else:
                self._log("Cleaning up SC2 process (loop closed/missing)...")
                self._sc2_process._clean()
            self._sc2_process = None
        
        self._client = None
        self._controller = None
        self._bot = None
        
        if self._loop:
            if not self._loop.is_closed():
                try:
                    self._log("Closing event loop...")
                    self._loop.close()
                except Exception as e:
                    self._log(f"Error closing loop: {e}")
            self._loop = None
        self._log("close() complete.")

    def _resolve_map(self, map_name):
        map_path = Path(__file__).resolve().parent.parent / "maps" / self.level_name / f"{map_name}.SC2Map"
        if not map_path.exists():
            raise FileNotFoundError(f"Map not found: {map_path}")
        return Map(map_path)

    def action_masks(self):
        mask = None
        if self._bot and hasattr(self._bot, "action_masks"):
            mask = self._bot.action_masks()
        else:
            mask = np.ones(self.action_space.n, dtype=bool)
        
        # Validation logging and fallback
        mask_sum = np.sum(mask)
        if mask_sum == 0:
            self._log(f"CRITICAL: All actions masked! (size {mask.size}). Falling back to all-True.")
            mask = np.ones(mask.size, dtype=bool)
        
        return mask

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_sc2_process'] = None
        state['_client'] = None
        state['_controller'] = None
        state['_bot'] = None
        state['_loop'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)