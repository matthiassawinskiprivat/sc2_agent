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


class MockState:
    def __init__(self, game_loop):
        self.game_loop = game_loop
        self.effects = []
        self.creep = None


class Level0Env(gym.Env):
    def __init__(self, visualize=False):
        super(Level0Env, self).__init__()
        self.visualize = visualize
        self.map_name = "Middle_Only"

        self.worker_id = f"{random.randint(1000, 9999)}"
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/worker_{self.worker_id}.log"
        self._log(f"Initialized Worker {self.worker_id}")

        self.observation_space = spaces.Box(low=0, high=255, shape=(42, 128, 128), dtype=np.uint8)
        self.action_space = spaces.Discrete(128 * 128)

        self.system = HierarchicalAgentSystem(map_size=(128, 128))
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
        self.max_steps = 50000000
        self.first_launch = True
        self.valid_spawn_points = []
        self.previous_distance = None

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
        return (unit1.position.x - unit2.position.x) ** 2 + (unit1.position.y - unit2.position.y) ** 2

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.last_target_pos = None  # Reset target on new episode
        self.previous_distance = None

        if self.client:
            try:
                self.loop.run_until_complete(self._soft_reset_logic())
                obs = self._get_observation()

                if self.zergling and self.beacon:
                    self.previous_distance = self.zergling.distance_to(self.beacon)

                return obs, {}
            except Exception as e:
                self._log(f"Soft Reset Failed: {e}")
                self.close()

        attempts = 0
        while attempts < 5:
            try:
                self._launch_client()
                if not self.valid_spawn_points: self._calculate_spawn_points()
                self.loop.run_until_complete(self._soft_reset_logic())

                if self.first_launch:
                    time.sleep(random.uniform(1, 5))
                    self.first_launch = False

                return self._get_observation(), {}
            except Exception as e:
                self._log(f"Launch Error: {e}")
                self.close()
                attempts += 1
                time.sleep(5 + attempts)

        raise RuntimeError(f"Worker {self.worker_id} failed to launch SC2")

    async def _soft_reset_logic(self):
        for i in range(20):
            try:
                response = await self.client.observation()
                raw_units = response.observation.observation.raw_data.units
                tags = [u.tag for u in raw_units]
                if not tags: break
                await self.client.debug_kill_unit(tags)
                await self.client.step()
            except Exception:
                raise ConnectionAlreadyClosed("Lost connection during cleanup")

        if self.valid_spawn_points:
            while True:
                start = random.choice(self.valid_spawn_points)
                end = random.choice(self.valid_spawn_points)
                if ((start.x - end.x)**2 + (start.y - end.y)**2)**0.5 > 8:
                    break
        else:
            start, end = Point2((10, 10)), Point2((20, 20))

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

            # 5. Handle Continuous Task Logic (Respawn)
            if reached_target:
                # Kill old beacon, spawn new one (Async)
                self.loop.run_until_complete(self._respawn_beacon_async())

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
            done = False
            if self.steps >= self.max_steps:
                done = True

            return obs, reward, done, False, {}

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
        processed = self.system.process_observation(raw_obs)
        return processed[self.agent_id]

    def _calculate_reward(self):
        """
        Calculates reward based on a Potential Field (Distance Difference).

        Returns:
            reward (float): The calculated reward for this step.
            reached_target (bool): True if the beacon was collected.
        """
        # 1. Safety Check: If units are missing, return 0
        if not self.zergling or not self.beacon:
            return 0.0, False

        # 2. Calculate Current Distance
        dist = self.zergling.distance_to(self.beacon)

        # 3. WIN CONDITION (Jackpot)
        # If we are close enough to "touch" it
        if dist < 2.5:
            self._log(f"SUCCESS! Reached Beacon at step {self.steps}")

            # CRITICAL: Reset previous_distance so the next step (after respawn)
            # doesn't calculate a huge penalty for the distance jumping from 0 -> 50.
            self.previous_distance = None

            return 10.0, True

        # 4. INITIALIZATION (First step of episode or after respawn)
        if self.previous_distance is None:
            self.previous_distance = dist
            return 0.0, False

        # 5. PROGRESS REWARD (Potential Field)
        # diff > 0: Moved Closer (Good)
        # diff < 0: Moved Away (Bad)
        diff = self.previous_distance - dist
        self.previous_distance = dist

        # Scaling: Moving 1 game meter gives +0.5 reward.
        # This provides a dense signal to guide the agent.
        progress_reward = diff * 0.5

        # 6. TIME PENALTY (Urgency)
        # A small negative value to force the agent to take the shortest path.
        # Without this, the agent might zig-zag unnecessarily.
        step_penalty = -0.01

        total_reward = progress_reward + step_penalty

        # 7. Stability Clipping
        # Prevents exploding gradients if units glitch or teleport.
        total_reward = np.clip(total_reward, -1.0, 1.0)

        return total_reward, False

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
        if self.client:
            try:
                self.loop.run_until_complete(self.client.leave())
            except:
                pass
            self.client = None
        if self.sc2_process:
            try:
                self.loop.run_until_complete(self.sc2_process.__aexit__(None, None, None))
            except:
                pass
            try:
                pid = self.sc2_process._process.pid
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except:
                        pass
                parent.kill()
            except:
                pass
            self.sc2_process = None
        if self.loop:
            try:
                self.loop.close()
            except:
                pass
            self.loop = None