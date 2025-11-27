import gymnasium as gym
import numpy as np
import asyncio
import os
import time
import traceback
import random
import psutil
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
        self.map_name = "Zerg_Lower_Left"

        self.worker_id = f"{random.randint(1000, 9999)}"
        print(f"[Worker {self.worker_id}] Initialized")

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

        self.steps = 0
        self.max_steps = 300  # Give it enough time to cross the map
        self.first_launch = True
        self.valid_spawn_points = []

    def do(self, action, **kwargs):
        return action

    def _distance_squared_unit_to_unit(self, unit1, unit2):
        return (unit1.position.x - unit2.position.x) ** 2 + (unit1.position.y - unit2.position.y) ** 2

    def _safe_create_unit(self, proto_unit):
        try:
            return Unit(proto_unit, self)
        except TypeError:
            u = Unit(proto_unit)
            u._bot_object = self
            return u

    def reset(self, seed=None, options=None):
        self.steps = 0

        if self.client:
            try:
                self.loop.run_until_complete(self._soft_reset_logic())
                return self._get_observation(), {}
            except Exception as e:
                print(f"[Worker {self.worker_id}] Soft Reset Failed: {e}")
                self.close()

        attempts = 0
        while attempts < 3:
            try:
                self._launch_client()
                if not self.valid_spawn_points: self._calculate_spawn_points()
                self.loop.run_until_complete(self._soft_reset_logic())

                if self.first_launch:
                    time.sleep(random.uniform(1, 5))
                    self.first_launch = False

                return self._get_observation(), {}
            except Exception as e:
                print(f"[Worker {self.worker_id}] Launch Error: {e}")
                self.close()
                attempts += 1
                time.sleep(5 + attempts)

        raise RuntimeError(f"Worker {self.worker_id} failed to launch SC2")

    async def _soft_reset_logic(self):
        # 1. AGGRESSIVE CLEANUP
        # Loop until 0 units remain.
        for i in range(20):
            try:
                # Get raw observation directly to see EVERYTHING
                response = await self.client.observation()
                raw_units = response.observation.observation.raw_data.units

                # Kill EVERYTHING on the map.
                # This includes Zergling, Beacons, and any leftover trash.
                tags = [u.tag for u in raw_units]

                if not tags:
                    break  # Map is empty!

                await self.client.debug_kill_unit(tags)
                await self.client.step()
            except Exception:
                raise ConnectionAlreadyClosed("Lost connection during cleanup")

        # 2. Spawn New
        if self.valid_spawn_points:
            start = random.choice(self.valid_spawn_points)
            end = random.choice(self.valid_spawn_points)
        else:
            start, end = Point2((10, 10)), Point2((20, 20))

        # Use Player 2 for Beacon so it registers as 'Enemy'
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
        print(f"[Worker {self.worker_id}] Map Loaded. {len(self.valid_spawn_points)} spawn points.")

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
        game_map = self._resolve_map(self.map_name)

        # FIX: disable_fog=True is CRITICAL for cleanup to work
        try:
            if hasattr(self.client, "create_game"):
                self.loop.run_until_complete(
                    self.client.create_game(self.map_name, [Bot(Race.Zerg, None)], realtime=False, disable_fog=True)
                )
            else:
                from sc2.controller import Controller
                controller = Controller(self.sc2_process._ws, self.sc2_process)
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
            world_x, world_y = self.agent.convert_spatial_action_to_world(action)

            async def async_step():
                if self.zergling:
                    cmd = self.zergling.move(Point2((world_x, world_y)))
                    await self.client.actions(cmd)

                try:
                    await self.client.step()
                except ProtocolError as e:
                    if "Game has already ended" in str(e): return "ENDED"
                    raise e

            result = self.loop.run_until_complete(async_step())
            if result == "ENDED":
                self.close()
                return np.zeros((42, 128, 128), dtype=np.uint8), 0, True, False, {}

            obs = self._get_observation()
            reward, done, info = self._calculate_reward()

            if self.steps >= self.max_steps:
                done = True
                # Penalty for timeout
                reward -= 10

            return obs, reward, done, False, info

        except Exception:
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
        if hasattr(actual_obs, "raw_data"):
            raw_data = actual_obs.raw_data
        else:
            raise AttributeError("Could not find raw_data")

        game_loop = getattr(actual_obs, "game_loop", 0)
        self.state = MockState(game_loop)
        proto_units = raw_data.units
        sc2_units = [self._safe_create_unit(u) for u in proto_units]
        all_units = Units(sc2_units, self)

        self.zergling = all_units.filter(lambda u: u.alliance == 1 and u.type_id == UnitTypeId.ZERGLING).first

        # Filter for Enemy (2) or Neutral (3) Beacons
        # Beacons can sometimes spawn as Neutral (3) even if requested as 2, depending on map settings.
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
            'creep': self.state.creep
        }
        processed = self.system.process_observation(raw_obs)
        return processed[self.agent_id]

    def _calculate_reward(self):
        if not self.zergling or not self.beacon:
            return 0, False, {"is_success": False}

        dist = self.zergling.distance_to(self.beacon)

        # Win Condition
        if dist < 2.0:  # Slightly larger radius to ensure hit
            print(f"[Worker {self.worker_id}] SUCCESS! Reached Beacon.")
            return 100, True, {"is_success": True}

        # Shaping: -0.01 (far) to +0.99 (on top)
        # This forces movement
        return (1.0 - (dist / 50.0)), False, {"is_success": False}

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