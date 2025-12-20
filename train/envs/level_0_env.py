import gymnasium as gym
import numpy as np
import os
import time
import random
import asyncio
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

class Level0Env(gym.Env):
    def __init__(self, visualize=False, grid_size: int = 16, worker_index: int = 0, num_workers: int = 1,
                 level_name: str = "level_0"):
        super(Level0Env, self).__init__()
        self.visualize = visualize
        self.grid_size = grid_size
        self.worker_index = worker_index
        self.num_workers = num_workers
        self.level_name = level_name
        self.map_rotation_index = 0

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

        self.sc2_process = None
        self.client = None
        self.protocol = None
        self.bot = None
        self.loop = None

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
        try:
            with open(self.log_file, "a") as f:
                timestamp = time.strftime("%H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
        except:
            pass

    def reset(self, seed=None, options=None):
        self._log("Resetting environment...")
        if self.sc2_process:
            self.close()

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._launch_and_setup_game())
        except Exception as e:
            self._log(f"Error during game launch: {e}")
            self.close()
            raise

        obs, _ = self.bot._get_observation()
        self._log("Environment reset complete.")
        return obs, {}

    async def _launch_and_setup_game(self):
        self._log("Launching SC2 process...")
        self.sc2_process = SC2Process(fullscreen=self.visualize)
        self.controller = await self.sc2_process.__aenter__()
        self.client = Client(self.sc2_process._ws)
        
        game_map = self._resolve_map(self.map_name)

        self._log("Creating game...")
        await self.controller.create_game(
            game_map,
            [Bot(Race.Zerg, None), Computer(Race.Zerg, Difficulty.Easy)],
            realtime=self.visualize,
        )
        
        self._log("Joining game...")
        await self.client.join_game(race=Race.Zerg)

        self._log("Creating SC2Bot...")
        class DummyConn:
            def send(self, data): pass
            def recv(self): return "continue", None
        self.bot = SC2Bot(conn=DummyConn())
        self.bot.distance_calculation_method = 2
        self.bot._generated_frame = -1
        
        self._log("Getting game info and data...")
        # Client.get_game_info() returns a GameInfo object.
        # However, _prepare_step expects 'proto_game_info' which has .game_info.start_raw.pathing_grid
        # In burnysc2, GameInfo is a wrapper.
        # Let's check how GameInfo is constructed or if we have the raw proto.
        
        # We can re-request the raw info via the protocol to get the protobuf directly if needed.
        # Or checking GameInfo implementation.
        # Based on the error: AttributeError: game_info
        # The object passed as proto_game_info does NOT have a .game_info attribute.
        # We passed game_info._proto.
        
        # Let's inspect what game_info._proto actually is.
        # But to fix it quickly: The logic in bot_ai.py is:
        # proto_game_info.game_info.start_raw.pathing_grid
        
        # If we look at sc2/client.py:
        # result = await self._execute(game_info=sc_pb.RequestGameInfo())
        # return GameInfo(result.game_info)
        
        # So client.get_game_info() returns GameInfo(result.game_info).
        # result.game_info IS the ResponseGameInfo protobuf.
        # But bot_ai.py expects something that HAS .game_info.
        
        # It seems bot_ai.py expects the 'result' from the _execute call, NOT just the game_info part.
        # OR it expects a structure that mimics it.
        
        # Let's try to pass a dummy object that has the structure it wants, 
        # using the game_info we already have.
        
        game_info = await self.client.get_game_info()
        game_data = await self.client.get_game_data()

        self._log("Preparing bot start...")
        self.bot._prepare_start(client=self.client, player_id=1, game_info=game_info, game_data=game_data)
        
        self._log("Getting initial observation...")
        initial_observation = await self.client.observation()
        
        # We need to create a GameState from the observation
        from sc2.game_state import GameState
        state = GameState(initial_observation)
        
        self._log("Preparing bot step...")
        
        # Hack/Fix: construct a dummy object to satisfy bot_ai.py:789
        class ProtoWrapper:
            def __init__(self, info):
                self.game_info = info
        
        self.bot._prepare_step(state, ProtoWrapper(game_info._proto))

        self._log("Running bot on_step(0)...")
        # We need to monkey-patch or avoid on_step if it uses self.conn
        # Actually, let's see what happens.
        try:
            await self.bot.on_step(0)
        except Exception as e:
            self._log(f"Error in on_step(0): {e}")
            raise

    def step(self, action):
        self.steps += 1

        try:
            obs, reward, done, truncated, info = self.loop.run_until_complete(self._async_step(action))
        except (ProtocolError, ConnectionAlreadyClosed) as e:
            self._log(f"Connection to game lost: {e}")
            obs, reward, done, truncated, info = np.zeros(self.observation_space.shape, dtype=np.uint8), 0, True, True, {}
        
        if self.steps >= self.max_steps:
            truncated = True
            done = True
            reward = -10

        return obs, reward, done, truncated, info

    async def _async_step(self, action):
        world_x, world_y = self.bot.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world(action)
        self.bot.last_target_pos = (world_x, world_y)

        if self.bot.zergling:
            cmd = self.bot.zergling.move(self.bot.last_target_pos)
            if cmd and not isinstance(cmd, bool):
                await self.bot.do(cmd)

        await self.client.step()

        observation = await self.client.observation()
        self.bot.units = []
        self.bot.structures = []
        self.bot.enemy_units = []
        self.bot.enemy_structures = []
        self.bot.all_units = []
        self.bot._prepare_step(observation, self.bot.game_info._proto)
        
        return self.bot._get_step_results()

    def close(self):
        self._log("Closing environment...")
        if self.sc2_process:
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
        self._log("Environment closed.")

    def _resolve_map(self, map_name):
        map_path = Path(__file__).resolve().parent.parent / "maps" / self.level_name / f"{map_name}.SC2Map"
        if not map_path.exists():
            raise FileNotFoundError(f"Map not found: {map_path}")
        return Map(map_path)

    def action_masks(self):
        return np.ones(self.action_space.n, dtype=bool)
