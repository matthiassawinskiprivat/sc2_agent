import gymnasium as gym
import numpy as np
import os
import time
import random
import psutil
import asyncio
import aiohttp
from typing import Dict
from pathlib import Path
from gymnasium import spaces
from unittest.mock import MagicMock

from sc2 import maps
from sc2.client import Client
from sc2.data import Race, Difficulty
from sc2.main import run_game
from sc2.maps import Map
from sc2.player import Bot, Computer
from sc2.protocol import ConnectionAlreadyClosed, ProtocolError, Protocol
from sc2.sc2process import SC2Process

from src.sc2_bot import SC2Bot, MapCleanupFailedError
from src.level_0_bot import Level0Bot
from src.hierarchical_agent import HierarchicalAgentSystem
from src.utils.config_manager import ConfigManager
from src.utils.layer_constants import LayerConstants
from src.utils.proto_utils import ProtoWrapper
from train.envs.sc2_session_config import use_computer_player
from src.agent.universal_constants import (
    FIXED_UNIVERSAL_COMMAND_DIM,
    build_action_head_manifest,
)

HARD_RESET_COUNT = 100

class Level0Env(gym.Env):
    def __init__(self, visualize=False, grid_size: int = 16, worker_index: int = 0, num_workers: int = 1,
                 level_name: str = "level_0", sc2_process_cls=None, client_cls=None, training_conf: Dict = None,
                 max_steps: int = 300, initial_steps: int = 0):
        super(Level0Env, self).__init__()
        self.visualize = visualize
        self.grid_size = grid_size
        self.worker_index = worker_index
        self.num_workers = num_workers
        self.level_name = level_name
        self.episode_counter = 0
        self.training_conf = training_conf or {}
        
        # Load Additive Configuration
        try:
            config_dir = self.training_conf.get("config_dir", "config")
            self.config_manager = ConfigManager(config_dir)
            
            # CUMULATIVE CONFIGS
            self.active_layers = self.config_manager.get_cumulative_active_layers(self.level_name)
            self.active_tactical_layers = self.config_manager.get_cumulative_active_tactical_layers(self.level_name)
            self.action_heads = self.config_manager.get_cumulative_action_heads(self.level_name)
            self.categorical_heads = self.config_manager.get_cumulative_categorical_heads(self.level_name)
            self.agent_architecture = self.config_manager.get_agent_architecture()

            print(f"DEBUG [{self.level_name}]: action_heads={self.action_heads}, categorical_heads={self.categorical_heads}, arch={self.agent_architecture}")

            # SENSIBLE DEFAULTS for Universal Architecture
            if self.agent_architecture == "universal":
                if not self.active_layers:
                    self.active_layers = ["allied_unit_density", "enemy_unit_density", "terrain_pathable", "command_objective", "command_assignment"]
                if not self.active_tactical_layers:
                    self.active_tactical_layers = self.active_layers
                if not self.action_heads:
                    self.action_heads = ["move_global", "move_local"]
            
            # Load level-specific config for rewards and other params
            level_conf = self.config_manager.get_level_config(self.level_name)
            
            # FLAT CONFIG: Move training/env flags to root for ObservationTransformer and bot
            for section in ["training", "env"]:
                if section in level_conf:
                    for k, v in level_conf[section].items():
                        self.training_conf[k] = v

            # Ensure training_conf has the cumulative lists so the Bot can use them
            self.training_conf['active_layers'] = self.active_layers
            self.training_conf['active_tactical_layers'] = self.active_tactical_layers
            self.training_conf['action_heads'] = self.action_heads
            self.training_conf['categorical_heads'] = self.categorical_heads
            self.training_conf['agent_architecture'] = self.agent_architecture

            if 'env' in level_conf and 'rewards' in level_conf['env']:
                self.training_conf['rewards'] = level_conf['env']['rewards']
            
            # Imitation Learning Configuration
            training_section = level_conf.get('training', {})
            self.imitation_data_percentage = training_section.get('imitation_data_percentage', 0)
            self.imitation_data_max_steps = training_section.get('imitation_data_max_steps', max_steps)
            
            # Inject into training_conf for bot/agent visibility
            self.training_conf['imitation_data_percentage'] = self.imitation_data_percentage
            self.imitation_data_max_steps = self.imitation_data_max_steps

            if self.agent_architecture == "universal":
                manifest = build_action_head_manifest(self.training_conf, self.action_heads)
                self.training_conf["fixed_universal_command_dim"] = FIXED_UNIVERSAL_COMMAND_DIM
                self.training_conf["action_head_manifest"] = manifest

        except Exception as e:
            print(f"Warning: Failed to load config for {self.level_name}: {e}")
            self.active_layers = []
            self.action_heads = ["move"] 
            self.categorical_heads = []
            self.agent_architecture = "original"

        # Dependency Injection for testing
        self._sc2_process_cls = sc2_process_cls if sc2_process_cls else SC2Process
        
        # Branch Bot class by architecture
        if self.agent_architecture == "universal":
            from src.sc2_bot_universal import SC2BotUniversal
            self._bot_cls = SC2BotUniversal
        else:
            self._bot_cls = SC2Bot
            
        self._client_cls = client_cls if client_cls else Client

        self.map_list = self._discover_maps()
        if not self.map_list:
            raise FileNotFoundError(f"No maps found in train/maps/{self.level_name}/")
        self.map_name = self.map_list[self.worker_index % len(self.map_list)]

        self.worker_id = f"{random.randint(1000, 9999)}"
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/worker_{self.worker_id}.log"
        self._log(f"Initialized Worker {self.worker_id} (Index: {self.worker_index}) with map {self.map_name}")

        num_channels = len(self.active_layers) if self.active_layers else LayerConstants.get_num_layers()
        
        # New Dict Observation Space
        obs_dict = {
            "map": spaces.Box(low=0.0, high=1.0, shape=(num_channels, 128, 128), dtype=np.float32),
            "command": spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        }
        
        if self.agent_architecture == "universal":
            self.active_tactical_layers = self.config_manager.get_cumulative_active_tactical_layers(self.level_name)
            if not self.active_tactical_layers:
                self.active_tactical_layers = self.active_layers
            self.training_conf['active_tactical_layers'] = self.active_tactical_layers
            n_tactical = len(self.active_tactical_layers) if self.active_tactical_layers else num_channels
            obs_dict["tactical"] = spaces.Box(low=0.0, high=1.0, shape=(n_tactical, 32, 32), dtype=np.float32)

        self.observation_space = spaces.Dict(obs_dict)
        
        if self.agent_architecture == "universal":
            # Universal: fixed command axis; unused slots masked via action_head_manifest.
            spatial_res = 32
            self.action_space = spaces.MultiDiscrete(
                [FIXED_UNIVERSAL_COMMAND_DIM, spatial_res * spatial_res]
            )
        else:
            # ORIGINAL FLAT DISCRETE SPACE
            spatial_size = len(self.action_heads) * self.grid_size * self.grid_size
            categorical_size = len(self.categorical_heads)
            self.action_space = spaces.Discrete(spatial_size + categorical_size)

        self._sc2_process = None
        self._client = None
        self._controller = None
        self._bot = None
        self._loop = None

        self.steps = 0
        self.max_steps = max_steps
        # Initialize total steps from the provided initial_steps (adjusted for per-worker count)
        self.total_steps_in_level = initial_steps // max(1, self.num_workers)
        self.training_steps_limit = self.training_conf.get("training_steps", 100_000)

    def _discover_maps(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Priority: 1. training_conf['maps_dir'], 2. standard level_name path
        config_maps_dir = self.training_conf.get("maps_dir")
        if config_maps_dir:
            # If absolute or relative to project root
            maps_dir = Path(config_maps_dir)
            if not maps_dir.is_absolute():
                # Assume relative to project root (which is parent of train/)
                maps_dir = Path(current_dir).parent.parent / config_maps_dir
        else:
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
            except MapCleanupFailedError as e:
                # Expected recovery path: teardown SC2 and relaunch on next loop iteration.
                self._log(
                    f"Map cleanup incomplete ({e}); forcing SC2 hard reset "
                    f"(attempt {attempt + 1}/{max_retries}, worker {self.worker_index})."
                )
                self.close()
                if attempt == max_retries - 1:
                    self._log(f"Max retries reached for level '{self.level_name}'. Raising error to main process.")
                    raise e
                time.sleep(2.0)
            except Exception as e:
                import traceback
                self._log(f"CRITICAL: Error during environment reset for level '{self.level_name}' (Worker {self.worker_index}): {e}")
                self._log(f"Traceback:\n{traceback.format_exc()}")
                self.close() # Ensure full cleanup
                if attempt == max_retries - 1:
                    self._log(f"Max retries reached for level '{self.level_name}'. Raising error to main process.")
                    raise e
                time.sleep(2.0) # Longer sleep between retries to allow system cleanup

        obs, _ = self._bot._get_observation()
        self._log("reset() complete.")
        return obs, {}

    async def _reset_bot_state(self):
        self._log("Resetting bot state...")
        await self._client.step()
        observation = await self._client.observation()

        from src.utils.sc2_observation_compat import game_state_from_response_observation
        state = game_state_from_response_observation(observation.observation)

        self._bot._prepare_step(state, ProtoWrapper(self._bot.game_info._proto))
        await self._bot.reset_game_state()
        await self._client.step()
        observation = await self._client.observation()

        state = game_state_from_response_observation(observation.observation)
            
        self._bot._prepare_step(state, ProtoWrapper(self._bot.game_info._proto))
    
    async def _launch_sc2_process(self):
        max_launch_attempts = 3
        for attempt in range(max_launch_attempts):
            await asyncio.sleep(random.random() * 5 + attempt * 2)
            self._log(f"Launching SC2 process (Attempt {attempt+1}/{max_launch_attempts})...")
            
            self._sc2_process = self._sc2_process_cls(fullscreen=self.visualize)
            try:
                self._controller = await self._sc2_process.__aenter__()
                
                # Verify port was assigned correctly
                # SC2Process uses _port internally in burnysc2
                current_port = getattr(self._sc2_process, "_port", None)
                if current_port is None:
                    self._log(f"WARNING: SC2Process launched with port=None on attempt {attempt+1}")
                    await self._sc2_process.__aexit__(None, None, None)
                    if attempt == max_launch_attempts - 1:
                        raise RuntimeError("Failed to obtain a valid port for SC2Process after multiple attempts.")
                    continue

                # Use _ws if available (real SC2Process), fallback to ws (for some mocks)
                ws = getattr(self._sc2_process, "_ws", getattr(self._sc2_process, "ws", None))
                if ws is None:
                    self._log(f"WARNING: WebSocket is None on attempt {attempt+1}")
                    await self._sc2_process.__aexit__(None, None, None)
                    continue

                self._client = self._client_cls(ws)
                # Give the client a moment to stabilize
                await asyncio.sleep(0.5)
                # Success
                return
            except Exception as e:
                self._log(f"Error during SC2 launch attempt {attempt+1}: {e}")
                try:
                    await self._sc2_process.__aexit__(None, None, None)
                except:
                    pass
                if attempt == max_launch_attempts - 1:
                    raise e

    async def _setup_game_session(self):
        # Verify connection
        try:
             await self._client.ping()
        except Exception as e:
             self._log(f"Initial ping failed: {e}")
             raise e

        game_map = self._resolve_map(self.map_name)
        if self._bot is None:
            self._log(f"Creating {self._bot_cls.__name__}...")
            class DummyConn:
                def send(self, data): pass
                def recv(self): return "continue", None
            self._bot = self._bot_cls(conn=DummyConn(), training_conf=self.training_conf)
            self._bot.distance_calculation_method = 2
            self._bot.grid_size = self.grid_size

        self._log("Creating game...")
        use_computer = use_computer_player(self.level_name, self.training_conf)
        
        players = [Bot(Race.Zerg, self._bot)]
        if use_computer:
            difficulty_name = self.training_conf.get("env", {}).get("difficulty", "Easy")
            try:
                difficulty = getattr(Difficulty, difficulty_name)
            except AttributeError:
                difficulty = Difficulty.Easy
            players.append(Computer(Race.Zerg, difficulty))
            self._log(f"Starting game with Computer ({difficulty_name})")
        else:
            self._log("Starting game WITHOUT Computer (Movement Level)")

        await self._controller.create_game(
            game_map,
            players,
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
            self._bot.client = self._client
            self._bot._game_info = game_info
            self._bot._game_data = game_data
        except ImportError:
            pass
        
        self._bot._prepare_start(client=self._client, player_id=1, game_info=game_info, game_data=game_data)
        initial_observation = await self._client.observation()

        from src.utils.sc2_observation_compat import game_state_from_response_observation
        state = game_state_from_response_observation(initial_observation.observation)
            
        self._bot._prepare_step(state, ProtoWrapper(game_info._proto))
        try:
            await self._bot.on_step(0)
        except Exception as e:
            self._log(f"Error in on_step(0): {e}")
            raise

    def step(self, action):
        """
        GYMNASIUM INTERFACE: Action Index -> Next Observation
        This is called by model.learn() or stable_baselines3.
        It bridges the synchronous Gym API with the asynchronous SC2 simulation.
        """
        self.steps += 1
        self.total_steps_in_level += 1
        try:
            # 1. Run the async logic in our dedicated event loop
            obs, reward, done, truncated, info = self._loop.run_until_complete(self._async_step(action))
        except (ProtocolError, ConnectionAlreadyClosed) as e:
            self._log(f"Connection to game lost: {e}")
            # Robust dummy observation for Dict spaces
            dummy_obs = {}
            if isinstance(self.observation_space, spaces.Dict):
                for k, v in self.observation_space.spaces.items():
                    dummy_obs[k] = np.zeros(v.shape, dtype=v.dtype)
            else:
                dummy_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            obs, reward, done, truncated, info = dummy_obs, 0, True, True, {}
        
        # 2. Handle Episode Truncation (Max Steps reached without target)
        if self.steps >= self.max_steps and not done:
            truncated = True
            # We don't set done=True or apply a penalty here. 
            # This allows PPO to bootstrap the value from the last state, 
            # effectively telling the agent: "You were on the right track, just ran out of time."
        
        return obs, reward, done, truncated, info

    async def _async_step(self, action):
        """
        IMPLEMENTATION: How one step actually happens in the game.
        """
        # 1. TRANSMIT: Tell the bot which action the model picked
        if self._bot.zergling:
            # Note: This calls SC2Bot.apply_action which we commented earlier
            await self._bot.apply_action(self._bot.zergling.tag, action)
        
        # 2. FLUSH: Send all queued StarCraft commands to the game server
        actions = getattr(self._bot, "_actions", None)
        if actions is None:
            actions = getattr(self._bot, "actions", None)
        
        if actions:
            await self._client.actions(actions[:])
            actions.clear()

        # 3. ADVANCE: Tell the SC2 engine to simulate the next few frames
        await self._client.step()
        
        # 4. OBSERVE: Download the new game state from the engine
        response = await self._client.observation()

        from src.utils.sc2_observation_compat import game_state_from_response_observation
        state = game_state_from_response_observation(response.observation)
            
        # Update the bot's internal tracking with the fresh state
        self._bot._prepare_step(state, ProtoWrapper(self._bot.game_info._proto))
        
        # 5. RESULT: Calculate pixels and reward for the transition
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
        config_maps_dir = self.training_conf.get("maps_dir")
        if config_maps_dir:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            maps_dir = Path(config_maps_dir)
            if not maps_dir.is_absolute():
                maps_dir = Path(current_dir).parent.parent / config_maps_dir
            map_path = maps_dir / f"{map_name}.SC2Map"
        else:
            map_path = Path(__file__).resolve().parent.parent / "maps" / self.level_name / f"{map_name}.SC2Map"
            
        if not map_path.exists():
            raise FileNotFoundError(f"Map not found: {map_path}")
        return Map(map_path)

    def get_obs_focus_unit_tag(self) -> int:
        """
        Unit tag for the observation the policy is about to see (single-unit levels: the zergling).
        Used by async/sync recurrent collectors to reset RNN state when the focal unit changes
        (critical for Level3/Level7 cycling). Return -1 if unknown.
        """
        if self._bot and getattr(self._bot, "zergling", None):
            return int(self._bot.zergling.tag)
        return -1

    def action_masks(self):
        mask = None
        if self._bot and hasattr(self._bot, "action_masks"):
            # Calculate training progress
            is_training = self.training_conf.get("is_training", True)
            if is_training:
                global_steps_estimate = self.total_steps_in_level * self.num_workers
                progress = min(1.0, global_steps_estimate / max(1, self.training_steps_limit))
            else:
                progress = 1.0
                
            mask = self._bot.action_masks(curriculum_progress=progress)
        else:
            # MultiDiscrete or Discrete fallback
            if isinstance(self.action_space, spaces.MultiDiscrete):
                mask = [np.ones(dim, dtype=bool) for dim in self.action_space.nvec]
            else:
                mask = np.ones(self.action_space.n, dtype=bool)
        
        # Validation logging and fallback
        # Handle both list of arrays (MultiDiscrete) and single array (Discrete)
        if isinstance(mask, list):
            mask_sum = sum(np.sum(m) for m in mask)
            total_size = sum(m.size for m in mask)
        else:
            mask_sum = np.sum(mask)
            total_size = mask.size

        if mask_sum == 0:
            self._log(f"CRITICAL: All actions masked! (size {total_size}). Falling back to all-True.")
            if isinstance(mask, list):
                mask = [np.ones(m.size, dtype=bool) for m in mask]
            else:
                mask = np.ones(total_size, dtype=bool)
        
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