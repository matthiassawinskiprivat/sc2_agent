
import gymnasium as gym
import asyncio
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import random
import time
import os
from pathlib import Path

from src.utils.single_client_manager import SingleClientSC2Manager
from sc2.player import Bot, Computer, AbstractPlayer
from sc2.data import Race, Difficulty
from sc2.maps import Map

class SingleClientEnv(gym.Env):
    """
    Base environment for single-client SC2 games.
    Manages one SC2 process.
    """
    def __init__(self, map_name: str = "Simple64", visualize: bool = False, realtime: bool = False,
                 level_name: str = "capture_the_flag", sc2_process_cls=None, client_cls=None,
                 training_conf: Dict = None, worker_index: int = 0, num_workers: int = 1, 
                 fullscreen: bool = False, **kwargs):
        self.map_name = map_name
        self.visualize = visualize
        self.fullscreen = fullscreen
        self.realtime = realtime
        self.level_name = level_name
        self.training_conf = training_conf or {}
        self.worker_index = worker_index
        self.num_workers = num_workers
        
        # Logging setup
        self.worker_id = f"{random.randint(1000, 9999)}"
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/worker_{self.worker_id}.log"
        self._log(f"Initialized SingleClient Worker {self.worker_id} (Index: {self.worker_index}) with map {self.map_name}")

        # Dependency Injection for testing
        from sc2.sc2process import SC2Process
        from sc2.client import Client
        self._sc2_process_cls = sc2_process_cls if sc2_process_cls else SC2Process
        self._client_cls = client_cls if client_cls else Client

        self._manager = SingleClientSC2Manager(
            sc2_process_cls=self._sc2_process_cls, 
            client_cls=self._client_cls,
            worker_index=self.worker_index
        )
        self._loop = asyncio.new_event_loop()

        # Map discovery
        self.map_list = self._discover_maps()
        if self.map_list:
            self.map_name = self.map_list[self.worker_index % len(self.map_list)]
        else:
            self.map_list = [self.map_name]
            
        self.observation_space = None
        self.action_space = None
        
        # State tracking
        self._last_obs: Optional[np.ndarray] = None
        self.game_ended: bool = False

    def _discover_maps(self):
        from pathlib import Path
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        maps_dir = Path(current_dir).parent / "maps" / self.level_name
        if not maps_dir.exists():
            return []
        map_files = [f.stem for f in maps_dir.glob("*.SC2Map")]
        map_files.sort()
        return map_files

    def _resolve_map(self, map_name):
        from pathlib import Path
        from sc2.maps import Map
        import sc2
        try:
            return sc2.maps.get(map_name)
        except KeyError:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            map_path = Path(current_dir).parent / "maps" / self.level_name / f"{map_name}.SC2Map"
            if not map_path.exists():
                raise FileNotFoundError(f"Map not found: {map_path}")
            return Map(map_path)

    def _log(self, message):
        print(f"[Worker {self.worker_id}] {message}")
        try:
            with open(self.log_file, "a") as f:
                import time
                timestamp = time.strftime("%H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
        except:
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._log(f"reset() called (Level: {self.level_name})")
        
        if self._manager.clients:
             self._log("Cleaning up previous session...")
             self.close()
             
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            client_count = getattr(self, "client_count", 1)
            self._log(f"Launching SC2 process with {client_count} clients...")
            self._loop.run_until_complete(self._manager.launch_process(
                client_count=client_count, 
                visualize=self.visualize, 
                fullscreen=self.fullscreen
            ))
            self._log("Setting up game session...")
            self._loop.run_until_complete(self._setup_game_session())
            
            self._log("Getting initial observation...")
            obs = self._loop.run_until_complete(self._get_joint_observation())
            self._last_obs = obs
            self.game_ended = False
            return obs, {}
        except Exception as e:
            self._log(f"CRITICAL: Error during reset: {e}")
            import traceback
            self._log(traceback.format_exc())
            self.close()
            raise e

    def step(self, action: Any):
        from sc2.protocol import ProtocolError
        
        if self.game_ended:
             return self._last_obs, 0.0, True, False, {"game_ended": True, "already_ended": True}

        info = {}
        try:
            results = self._loop.run_until_complete(self._async_step(action))
            obs, reward, terminated, truncated, info = results
            self._last_obs = obs
            if terminated or truncated:
                self.game_ended = True
            return results
        except ProtocolError as e:
            if "Game has already ended" in str(e):
                self._log("ProtocolError: Game ended during step. Transitioning to terminal state.")
                self.game_ended = True
                reward, terminal_info = self._get_terminal_info(str(e))
                info.update(terminal_info)
                info["game_ended"] = True
                return self._last_obs, reward, True, False, info
            raise e
        except Exception as e:
            self._log(f"CRITICAL: Error during step: {e}")
            self.close()
            raise e

    async def _setup_game_session(self):
        pass

    async def _async_step(self, action: Any):
        await self._apply_action(action)
        await self._manager.client.step()
        return await self._get_step_results()

    async def _apply_action(self, action):
        pass

    async def _get_joint_observation(self):
        return None

    async def _get_step_results(self):
        obs = await self._get_joint_observation()
        return obs, 0.0, False, False, {}

    def _get_terminal_info(self, error_msg: str) -> Tuple[float, Dict]:
        return 0.0, {"error": error_msg}

    def close(self):
        if self._loop and not self._loop.is_closed():
            self._loop.run_until_complete(self._manager.close())
            self._loop.close()
