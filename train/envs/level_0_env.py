import gymnasium as gym
import numpy as np
import asyncio
import os
import time
import random
import psutil
import sys
from pathlib import Path
from gymnasium import spaces

# SC2 Imports
from sc2.data import Race, Difficulty
from sc2.player import Bot
from sc2.client import Client
from sc2.sc2process import SC2Process
from sc2.portconfig import Portconfig
from sc2.ids.unit_typeid import UnitTypeId
from sc2 import maps
from sc2.maps import Map
from sc2.units import Units
from sc2.unit import Unit
from sc2.position import Point2
from sc2.protocol import ProtocolError, ConnectionAlreadyClosed

# Project Imports
from src.hierarchical_agent import HierarchicalAgentSystem

HARD_RESET_COUNT = 10

class MockState:
    def __init__(self, game_loop):
        self.game_loop = game_loop
        self.effects = []
        self.creep = None


class Level0Env(gym.Env):
    def __init__(self, visualize=False, grid_size: int = 16, worker_index: int = 0, num_workers: int = 1):
        super(Level0Env, self).__init__()
        self.visualize = visualize
        self.grid_size = grid_size
        self.worker_index = worker_index
        self.num_workers = num_workers
        self.episode_counter = 0

        # --- Map Selection Logic ---
        self.map_list = self._discover_maps()
        if not self.map_list:
            raise FileNotFoundError("No maps found in train/maps/level_0/. Please add maps to this directory.")
        self.map_name = self.map_list[self.worker_index % len(self.map_list)]
        # --------------------------

        self.worker_id = f"{random.randint(1000, 9999)}"
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/worker_{self.worker_id}.log"
        self._log(f"Initialized Worker {self.worker_id} (Index: {self.worker_index}) with map {self.map_name}")

        self.observation_space = spaces.Box(low=0, high=255, shape=(42, 128, 128), dtype=np.uint8)
        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)

        self.system = HierarchicalAgentSystem(
            map_size=(128, 128),
            grid_size=self.grid_size,
            training_conf={'render_beacon': True, 'larger_units': True}
        )
        self.agent_id = "zergling"
        self.agent = self.system.create_unit_agent(self.agent_id)

        self.unit_command_uses_self_do = False

        self.sc2_process = None
        self.client = None
        self.loop = None

        self.game_data = None
        self.game_info = None
        self.state = None
        self.zergling = None
        self.beacon = None

        # State tracking for Command Layers
        self.last_target_pos = None

        self.steps = 0
        self.max_steps = 300
        self.first_launch = True
        self.valid_spawn_points = []
        self.previous_distance = None

    def _discover_maps(self):
        """Finds and sorts maps in the training directory."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level from 'train/envs' to 'train'
        maps_dir = Path(current_dir).parent / "maps" / "level_0"
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

    def do(self, action, **kwargs):
        return action

    def _distance_squared_unit_to_unit(self, unit1, unit2):
        return ((unit1.position.x - unit2.position.x) ** 2 + (unit1.position.y - unit2.position.y) ** 2)**0.5

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.last_target_pos = None
        self.previous_distance = None
        self.episode_counter += 1

        # --- Hard Reset Logic ---
        # Hard reset every 50 episodes (and on the very first run)
        if self.episode_counter > 1 and self.episode_counter % HARD_RESET_COUNT == 0:
            self._log(f"Episode {self.episode_counter}: Performing hard reset and changing map.")
            self.close() # Ensure client and process are killed

            # Calculate next map
            round_num = (self.episode_counter // 50)
            map_offset = round_num * self.num_workers
            map_idx = (map_offset + self.worker_index) % len(self.map_list)
            self.map_name = self.map_list[map_idx]
            self._log(f"New map: {self.map_name}")
        # --------------------------

        if self.client:
            try:
                self.loop.run_until_complete(self._soft_reset_logic())
                obs = self._get_observation()

                if self.zergling and self.beacon:
                    self.previous_distance = self.zergling.distance_to(self.beacon)

                return obs, {}
            except Exception as e:
                self._log(f"Soft Reset Failed: {e}. Forcing hard reset.")
                self.close()

        # This part will execute on the first run and after a hard reset
        attempts = 0
        while attempts < 5:
            try:
                self._launch_client()
                self.valid_spawn_points = [] # Reset spawn points as it's a new map
                if not self.valid_spawn_points: self._calculate_spawn_points()
                self.loop.run_until_complete(self._soft_reset_logic())

                if self.first_launch:
                    time.sleep(random.uniform(1, 5))
                    self.first_launch = False

                obs = self._get_observation()
                if self.zergling and self.beacon:
                    self.previous_distance = self.zergling.distance_to(self.beacon)
                return obs, {}
            except Exception as e:
                self._log(f"Launch Error: {e}")
                self.close()
                attempts += 1
                time.sleep(5 + attempts)

        raise RuntimeError(f"Worker {self.worker_id} failed to launch SC2 after hard reset.")

    async def _soft_reset_logic(self):
        for i in range(20):
            try:
                response = await self.client.observation()
                if not response: continue
                raw_units = response.observation.observation.raw_data.units
                tags = [u.tag for u in raw_units]
                if not tags: break
                await self.client.debug_kill_unit(tags)
                await self.client.step()
            except (ProtocolError, ConnectionAlreadyClosed) as e:
                self._log(f"Connection lost during cleanup: {e}")
                raise ConnectionAlreadyClosed("Lost connection during cleanup")
            except Exception as e:
                self._log(f"Error during unit cleanup: {e}")
                # Don't re-raise, try to continue
                
        if not self.valid_spawn_points:
             self._calculate_spawn_points()

        if self.valid_spawn_points:
            while True:
                start = random.choice(self.valid_spawn_points)
                end = random.choice(self.valid_spawn_points)
                if ((start.x - end.x)**2 + (start.y - end.y)**2)**0.5 > 8:
                    break
        else:
            # Fallback if pathing grid is empty
            w, h = self.game_info.map_size
            start = Point2((random.uniform(w*0.1, w*0.9), random.uniform(h*0.1, h*0.9)))
            end = Point2((random.uniform(w*0.1, w*0.9), random.uniform(h*0.1, h*0.9)))


        await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 1, start, 1]])
        await self.client.debug_create_unit([[UnitTypeId.BEACON_TERRAN, 1, end, 2]])

        await self.client.step()
        await self.client.step()

    def _calculate_spawn_points(self):
        grid = self.game_info.pathing_grid
        self.valid_spawn_points = []
        for x in range(4, grid.width - 4, 4):
            for y in range(4, grid.height - 4, 4):
                if grid[(x, y)] != 0:
                    self.valid_spawn_points.append(Point2((x, y)))
        self._log(f"Map Loaded. {len(self.valid_spawn_points)} spawn points.")

    def _resolve_map(self, map_name):
        try:
            return maps.get(map_name)
        except KeyError:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            map_path = Path(project_root) / "train" / "maps" / "level_0" / f"{map_name}.SC2Map"
            if not map_path.exists():
                raise FileNotFoundError(f"Map not found: {map_path}")
            return Map(map_path)

    def _launch_client(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        portconfig = Portconfig()
        self.sc2_process = SC2Process(host="127.0.0.1", port=portconfig.shared, fullscreen=self.visualize)

        try:
            self.loop.run_until_complete(self.sc2_process.__aenter__())
        except AttributeError:
            self.loop.run_until_complete(self.sc2_process.launch())

        self.client = Client(self.sc2_process._ws)

        try:
            if hasattr(self.client, "create_game"):
                self.loop.run_until_complete(
                    self.client.create_game(self.map_name, [Bot(Race.Zerg, None)], realtime=False, disable_fog=True)
                )
            else:
                from sc2.controller import Controller
                controller = Controller(self.sc2_process._ws, self.sc2_process)
                game_map = self._resolve_map(self.map_name)
                self.loop.run_until_complete(
                    controller.create_game(game_map, [Bot(Race.Zerg, None)], realtime=False, disable_fog=True)
                )
        except Exception as e:
            raise e

        self.loop.run_until_complete(
            self.client.join_game(self.map_name, Race.Zerg, Difficulty.VeryEasy)
        )

        self.game_data = self.loop.run_until_complete(self.client.get_game_data())
        self.game_info = self.loop.run_until_complete(self.client.get_game_info())
        try:
            self.client.game_step = 4
        except:
            pass
        self.loop.run_until_complete(self.client.step())

    def step(self, action):
        self.steps += 1

        try:
            # 1. Convert Spatial Action to World Coordinates
            world_x, world_y = self.agent.convert_spatial_action_to_world(action)
            self.last_target_pos = Point2((world_x, world_y))

            # 2. Execute Action (Async wrapper)
            async def async_action_step():
                if self.zergling:
                    # Send Move Command
                    cmd = self.zergling.move(self.last_target_pos)
                    # Check if cmd is a valid object (not bool/None)
                    if cmd and not isinstance(cmd, bool):
                        await self.client.actions(cmd)

                # Advance Game Simulation
                try:
                    await self.client.step()
                except ProtocolError as e:
                    if "Game has already ended" in str(e): return "ENDED"
                    raise e

            # Run the async step synchronously
            result = self.loop.run_until_complete(async_action_step())

            # Handle Game End
            if result == "ENDED":
                self._log("Game Ended Unexpectedly")
                self.close()
                return np.zeros((42, 128, 128), dtype=np.uint8), 0, True, False, {}

            # 3. Get Observation (State after move)
            obs = self._get_observation()

            # 4. Calculate Reward
            # This returns the scalar reward and a bool if we won
            reward, reached_target = self._calculate_reward()

            done = reached_target

            # 5. Handle Continuous Task Logic (Respawn)
            if reached_target:
                # Kill old beacon, spawn new one (Async)
                self.loop.run_until_complete(self._soft_reset_logic())

                # CRITICAL: Refresh observation immediately!
                # The agent needs to see the NEW beacon position for the next inference step.
                obs = self._get_observation()

                # CRITICAL: Reset Distance Tracker for the NEW beacon.
                # If we don't do this, the next step will calculate:
                # diff = (Old_Zero_Distance) - (New_Far_Distance) = Huge Negative Value.
                # We reset it so the next step starts fresh.
                if self.zergling and self.beacon:
                    self.previous_distance = self.zergling.distance_to(self.beacon)
                else:
                    self.previous_distance = None

            # 6. Check Episode Termination (Time Limit)
            truncated = False
            if self.steps >= self.max_steps:
                truncated = True
                reward = -10

            return obs, reward, done, truncated, {}

        except Exception as e:
            self._log(f"Step Error: {e}")
            # Return safe fallback observation on crash to prevent training pipeline collapse
            return np.zeros((42, 128, 128), dtype=np.uint8), 0, True, False, {}


    def _get_observation(self):
        response = self.loop.run_until_complete(self.client.observation())

        if hasattr(response, "observation"):
            resp_obs = response.observation
        else:
            resp_obs = response
        if hasattr(resp_obs, "observation"):
            actual_obs = resp_obs.observation
        else:
            actual_obs = resp_obs

        raw_data = actual_obs.raw_data
        game_loop = getattr(actual_obs, "game_loop", 0)
        self.state = MockState(game_loop)

        proto_units = raw_data.units
        sc2_units = [self._safe_create_unit(u) for u in proto_units]
        all_units = Units(sc2_units, self)

        self.zergling = all_units.filter(lambda u: u.alliance == 1 and u.type_id == UnitTypeId.ZERGLING).first
        beacons = all_units.filter(lambda u: "BEACON" in u.name.upper())
        self.beacon = beacons.first if beacons.exists else None

        enemies = []
        if self.beacon: enemies.append(self.beacon)

        raw_obs = {
            'units': all_units.filter(lambda u: u.alliance == 1),
            'enemy_units': enemies,
            'effects': self.state.effects,
            'game_info': self.game_info,
            'game_data': self.game_data,
            'creep': self.state.creep,
            'command_target': self.last_target_pos  # FIX: Pass command target
        }
        processed = self.system.process_observation(
            observation=raw_obs
        )

        return processed[self.agent_id]

    def _calculate_reward(self):
        if not self.zergling or not self.beacon:
            return 0, False # No reward if zergling or beacon are missing

        current_dist = self._distance_squared_unit_to_unit(self.zergling, self.beacon)
        reward = 0.0
        reached_target = False

        # Win Condition: Agent reached the beacon
        if current_dist < 2.0:  # Slightly larger radius to ensure hit
            self._log(f"[Worker {self.worker_id}] SUCCESS! Reached Beacon.")
            reward = 100.0
            reached_target = True
        elif self.previous_distance is not None:
            # Reward for getting closer (positive) or penalty for moving away (negative)
            reward = (self.previous_distance - current_dist) * 1.0 # Scale the reward
        
        # Update previous distance for the next step
        self.previous_distance = current_dist

        return reward, reached_target

    def _safe_create_unit(self, proto_unit):
        try:
            return Unit(proto_unit, self)
        except TypeError:
            u = Unit(proto_unit)
            u._bot_object = self
            return u

    def action_masks(self):
        if self.agent.last_game_info is None and self.game_info:
            self.agent.last_game_info = self.game_info
        return self.agent.get_action_mask()

    def close(self):
        self._log("Close method called.")
        if self.client:
            try:
                self._log("Leaving game.")
                self.loop.run_until_complete(self.client.leave())
            except (asyncio.CancelledError, ConnectionResetError, ProtocolError) as e:
                self._log(f"Error leaving game: {e}")
            except Exception as e:
                self._log(f"An unexpected error occurred while leaving game: {e}")
            self.client = None

        if self.sc2_process:
            try:
                self._log("Closing SC2 process.")
                self.loop.run_until_complete(self.sc2_process.__aexit__(None, None, None))
            except (asyncio.CancelledError, ConnectionResetError) as e:
                self._log(f"Error closing SC2 process: {e}")
            except Exception as e:
                self._log(f"An unexpected error occurred while closing SC2 process: {e}")
            self.sc2_process = None

        if self.loop and self.loop.is_running():
            try:
                self._log("Closing event loop.")
                self.loop.close()
            except Exception as e:
                self._log(f"An unexpected error occurred while closing the event loop: {e}")
        self.loop = None
        self._log("Close method finished.")