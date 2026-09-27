
import gymnasium as gym
import asyncio
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import random
import time
import os
from pathlib import Path

from src.utils.single_client_manager import SingleClientSC2Manager
from src.utils.config_manager import ConfigManager
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
        
        # Load configuration manager (config_dir comes from the training pipeline;
        # pipeline.yaml ``env`` holds shared SC2 startup timing defaults).
        config_dir = (
            self.training_conf.get("config_dir")
            or (self.training_conf.get("env") or {}).get("config_dir")
            or "config"
        )
        self.config_manager = ConfigManager(config_dir)
        
        # Logging setup
        self.worker_id = f"{random.randint(1000, 9999)}"
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/worker_{self.worker_id}.log"
        self._log(f"Initialized SingleClient Worker {self.worker_id} (Index: {self.worker_index}) with map {self.map_name}")
        
        # Base layers from Level 0 (for logging context)
        base_layers = self.config_manager.get_cumulative_active_layers("level_0")
        cumulative_layers = self.config_manager.get_cumulative_active_layers(self.level_name)
        self._log(f"Observation layers: ({len(base_layers)}) -> ({len(cumulative_layers)}) via ConfigManager")

        # Dependency Injection for testing
        from sc2.sc2process import SC2Process
        from sc2.client import Client
        self._sc2_process_cls = sc2_process_cls if sc2_process_cls else SC2Process
        self._client_cls = client_cls if client_cls else Client

        from src.utils.single_client_manager import (
            DEFAULT_CLOSE_SETTLE_SEC,
            DEFAULT_POST_CREATE_DELAY_SEC,
            DEFAULT_PROCESS_TREE_WAIT_SEC,
        )

        stagger = self._conf_float("sc2_startup_stagger_sec", None)
        self._manager = SingleClientSC2Manager(
            sc2_process_cls=self._sc2_process_cls,
            client_cls=self._client_cls,
            worker_index=self.worker_index,
            num_workers=self.num_workers,
            startup_stagger_sec=stagger,
            post_create_delay_sec=self._conf_float(
                "sc2_post_create_delay_sec", DEFAULT_POST_CREATE_DELAY_SEC
            ),
            close_settle_sec=self._conf_float(
                "sc2_close_settle_sec", DEFAULT_CLOSE_SETTLE_SEC
            ),
            process_tree_wait_sec=self._conf_float(
                "sc2_process_tree_wait_sec", DEFAULT_PROCESS_TREE_WAIT_SEC
            ),
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

    def _conf_float(self, key: str, default: Optional[float]) -> Optional[float]:
        """Read a float from training_conf, then pipeline.yaml ``env``, then default.

        Shared SC2 launch timing lives in ``config_universal/pipeline.yaml`` under
        ``env:`` so every level inherits the same hard-reset / startup knobs.
        Per-level YAML may still override by setting the same key.
        """
        env_conf = self.training_conf.get("env", {})
        if not isinstance(env_conf, dict):
            env_conf = {}
        if key in self.training_conf:
            raw = self.training_conf[key]
        elif key in env_conf:
            raw = env_conf[key]
        else:
            pipe_env = {}
            cm = getattr(self, "config_manager", None)
            if cm is not None:
                pipe = getattr(cm, "pipeline_config", None) or {}
                pipe_env = pipe.get("env", {}) if isinstance(pipe, dict) else {}
                if not isinstance(pipe_env, dict):
                    pipe_env = {}
            if key in pipe_env:
                raw = pipe_env[key]
            else:
                return default
        if raw is None:
            return default
        return float(raw)

    def _discover_maps(self):
        from pathlib import Path
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_maps_dir = (
            self.training_conf.get("maps_dir")
            or self.training_conf.get("env", {}).get("maps_dir")
        )
        if config_maps_dir:
            maps_dir = Path(config_maps_dir)
            if not maps_dir.is_absolute():
                maps_dir = Path(current_dir).parent.parent / config_maps_dir
        else:
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

        # Priority 1: maps_dir from training_conf (CTF and other non-standard levels).
        # Check both the flattened top level and the nested env dict since
        # SingleClientEnv-based envs don't flatten the config the way Level0Env does.
        config_maps_dir = (
            self.training_conf.get("maps_dir")
            or self.training_conf.get("env", {}).get("maps_dir")
        )
        if config_maps_dir:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            maps_dir = Path(config_maps_dir)
            if not maps_dir.is_absolute():
                maps_dir = Path(current_dir).parent.parent / config_maps_dir
            map_path = maps_dir / f"{map_name}.SC2Map"
            if not map_path.exists():
                raise FileNotFoundError(f"Map not found: {map_path}")
            return Map(map_path)

        # Priority 2: SC2 library map registry
        try:
            return sc2.maps.get(map_name)
        except (KeyError, FileNotFoundError, OSError):
            pass

        # Priority 3: project maps directory
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

    def _startup_max_retries(self) -> int:
        env_conf = self.training_conf.get("env", {})
        return int(
            self.training_conf.get("startup_max_retries")
            or env_conf.get("startup_max_retries", 3)
        )

    def _startup_stagger_sec(self) -> float:
        return self._manager._startup_stagger_sec()

    def _startup_timeout_sec(self) -> float:
        env_conf = self.training_conf.get("env", {})
        if not isinstance(env_conf, dict):
            env_conf = {}
        raw_base = (
            self.training_conf.get("startup_timeout_sec")
            if self.training_conf.get("startup_timeout_sec") is not None
            else env_conf.get("startup_timeout_sec")
        )
        if raw_base is None:
            raw_base = self._conf_float("startup_timeout_sec", 90.0)
        base = float(raw_base if raw_base is not None else 90.0)
        # launch_process sleeps worker_index × stagger before SC2 starts.
        # join_game also stretches under parallel DirectX load, and CTF then
        # polls up to sc2_in_game_wait_sec before spawn/obs. Those must sit
        # inside this wait_for or later workers raise TimeoutError mid-join.
        in_game = float(self._conf_float("sc2_in_game_wait_sec", 15.0) or 0.0)
        join_slack = float(self.worker_index) * 8.0
        return base + self.worker_index * self._startup_stagger_sec() + in_game + join_slack

    async def _startup_session(self):
        """Launch SC2, join the game, and fetch the first observation."""
        client_count = getattr(self, "client_count", 1)
        self._log(f"Launching SC2 process with {client_count} client(s)...")
        await self._manager.launch_process(
            client_count=client_count,
            visualize=self.visualize,
            fullscreen=self.fullscreen,
        )
        self._log("Setting up game session...")
        await self._setup_game_session()
        self._log("Getting initial observation...")
        return await self._get_joint_observation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._log(f"reset() called (Level: {self.level_name})")

        if self._manager.clients:
            self._log("Cleaning up previous session...")
            self.close()

        max_retries = self._startup_max_retries()
        timeout_sec = self._startup_timeout_sec()
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                self._log(
                    f"Starting SC2 session (attempt {attempt + 1}/{max_retries}, "
                    f"worker {self.worker_index}, timeout={timeout_sec:.0f}s)..."
                )
                obs = self._loop.run_until_complete(
                    asyncio.wait_for(self._startup_session(), timeout=timeout_sec)
                )
                self._last_obs = obs
                self.game_ended = False
                self._log("reset() complete.")
                return obs, {}
            except Exception as e:
                import traceback
                last_error = e
                self._log(
                    f"SC2 startup failed (attempt {attempt + 1}/{max_retries}, "
                    f"worker {self.worker_index}): {e}"
                )
                self._log(traceback.format_exc())
                self.close()
                if attempt < max_retries - 1:
                    # Desync retries across parallel workers; short backoff still
                    # reduces GPU/port races without multi-second dead air.
                    base = self._conf_float("sc2_retry_backoff_sec", 2.0) or 2.0
                    time.sleep(base + self.worker_index * 1.0)

        assert last_error is not None
        self._log(
            f"Max startup retries ({max_retries}) exhausted for level "
            f"'{self.level_name}' (worker {self.worker_index})."
        )
        raise last_error

    @staticmethod
    def _is_sc2_connection_error(exc: BaseException) -> bool:
        """True when SC2's WebSocket closed unexpectedly mid-game (not a Python bug)."""
        type_name = type(exc).__name__
        module = type(exc).__module__ or ""
        # aiohttp: ClientConnectionResetError, ClientConnectionError, ServerDisconnectedError
        if "aiohttp" in module or "ClientConnection" in type_name or "ServerDisconnected" in type_name:
            return True
        # Websockets / asyncio transport errors
        msg = str(exc).lower()
        if "closing transport" in msg or "connection reset" in msg or "connection closed" in msg:
            return True
        # Raw OS-level connection drops
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, EOFError)):
            return True
        return False

    def step(self, action: Any):
        from sc2.protocol import ProtocolError, ConnectionAlreadyClosed
        
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
        except ConnectionAlreadyClosed as e:
            self._log(
                f"SC2 connection closed during step ({e}). "
                "Marking episode as done — SC2 will relaunch on next reset()."
            )
            self.close()
            self.game_ended = True
            return self._last_obs, 0.0, True, False, {"sc2_connection_lost": True}
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
            if self._is_sc2_connection_error(e):
                # SC2 process exited or dropped its WebSocket mid-game.
                # Treat as a terminal episode so the env worker survives and
                # the next reset() relaunches SC2 instead of crashing the collector.
                self._log(
                    f"SC2 connection lost during step ({type(e).__name__}: {e}). "
                    "Marking episode as done — SC2 will relaunch on next reset()."
                )
                self.close()
                self.game_ended = True
                return self._last_obs, 0.0, True, False, {"sc2_connection_lost": True}
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
        try:
            if self._loop and not self._loop.is_closed():
                self._loop.run_until_complete(self._manager.close())
                self._loop.close()
        except Exception as e:
            self._log(f"Error during close: {e}")
