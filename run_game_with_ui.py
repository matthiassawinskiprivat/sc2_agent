import asyncio
import signal
import sys
import subprocess
import time
import os
import aiohttp
import inspect
from pathlib import Path
from typing import Optional

# --- 1. Force Windows Selector Policy for Qt Compatibility ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- Qt Imports ---
from PyQt5.QtWidgets import QApplication
from qasync import QEventLoop

# --- SC2 Imports ---
import sc2
import sc2.unit
from sc2 import maps
from sc2.player import Bot, Computer
from sc2.data import Race, Difficulty
from sc2.client import Client
from sc2.controller import Controller
from sc2.protocol import ProtocolError
from sc2.game_info import GameInfo
from sc2.game_state import GameState
from sc2.units import Units
from sc2.unit import Unit

# --- Protocol Imports ---
try:
    from s2clientprotocol import sc2api_pb2 as sc_pb
except ImportError:
    print("Error: s2clientprotocol not found. Please install it via 'pip install s2clientprotocol'")
    sys.exit(1)

# --- Project Imports ---
from src.agent.hierarchical_agent import HierarchicalAgentSystem
from src.sc2_bot import UnitControlBot
from src.ui.debug_ui import DebugUI
from src.ui.ui_manager import UIManager


def find_sc2_installation():
    """Finds the SC2 installation directory and executable."""
    possible_paths = []
    if os.environ.get("SC2PATH"):
        possible_paths.append(Path(os.environ.get("SC2PATH")))

    possible_paths.extend([
        Path("C:/Program Files (x86)/StarCraft II"),
        Path("C:/Program Files/StarCraft II"),
        Path("D:/StarCraft II"),
        Path("E:/StarCraft II")
    ])

    base_dir = None
    for p in possible_paths:
        if p.exists() and (p / "Versions").exists():
            base_dir = p
            break

    if not base_dir:
        raise FileNotFoundError("Could not find StarCraft II installation.")

    versions_dir = base_dir / "Versions"
    versions = []
    for p in versions_dir.iterdir():
        if p.is_dir() and p.name.startswith("Base"):
            try:
                versions.append((int(p.name[4:]), p))
            except ValueError:
                continue

    if not versions:
        raise FileNotFoundError(f"No 'Base' versions found in {versions_dir}")

    latest_version_path = sorted(versions, key=lambda x: x[0], reverse=True)[0][1]

    executable = latest_version_path / "SC2_x64.exe"
    if not executable.exists():
        executable = latest_version_path / "SC2.exe"

    if not executable.exists():
        raise FileNotFoundError(f"SC2 executable not found at {executable}")

    return executable, base_dir


def resolve_map(map_name, sc2_base_dir):
    """Resolves the map object."""
    try:
        m = maps.get(map_name)
        if m: return m
    except:
        pass

    print(f"Map '{map_name}' not found in library. Searching Maps folder...")
    maps_dir = sc2_base_dir / "Maps"

    found_file = None
    for path in maps_dir.rglob(f"{map_name}*.SC2Map"):
        found_file = path
        break

    if found_file:
        print(f"Found map file: {found_file}")

        class CustomMap:
            def __init__(self, path):
                self.path = path
                self.name = path.stem
                with open(path, "rb") as f:
                    self.data = f.read()
                self.relative_path = path.relative_to(maps_dir)

        return CustomMap(found_file)

    print(f"Warning: Map file for '{map_name}' not found. Attempting to use name directly.")
    return map_name


class SC2QtIntegration:
    def __init__(self):
        self.game_task: Optional[asyncio.Task] = None
        self._running = False
        self.sc2_process = None

        # Initialize System and UI
        self.system = HierarchicalAgentSystem(map_size=(128, 128))
        self.agent_id = "main_agent"
        self.main_agent = self.system.create_unit_agent(self.agent_id)
        self.bot = UnitControlBot(self.system, agent_id=self.agent_id)

        self.ui_manager = UIManager(self.main_agent)
        self.debug_ui = DebugUI(self.main_agent, self.ui_manager)
        self.debug_ui.show()

    def launch_sc2_manual(self, executable, port=9168):
        """Launch SC2 with correct CWD."""
        install_root = executable.parent.parent.parent

        if "x64" in executable.name:
            cwd = install_root / "Support64"
        else:
            cwd = install_root / "Support"

        if not cwd.exists():
            cwd = install_root

        print(f"Launching SC2 from CWD: {cwd}")

        args = [
            str(executable),
            "-listen", "127.0.0.1",
            "-port", str(port),
            "-displayMode", "0",
            "-windowwidth", "1024",
            "-windowheight", "768",
            "-windowx", "50",
            "-windowy", "50",
        ]

        self.sc2_process = subprocess.Popen(args, cwd=cwd)
        return f"ws://127.0.0.1:{port}/sc2api"

    async def run_sc2_game(self):
        print("SC2 coroutine starting...")
        self._running = True

        # Config
        target_map_name = "TorchesAIE_v4"
        bot_race = Race.Protoss
        opponent_race = Race.Zerg
        difficulty = Difficulty.Easy
        port = 9168

        try:
            executable, base_dir = find_sc2_installation()
            ws_url = self.launch_sc2_manual(executable, port)
            map_obj = resolve_map(target_map_name, base_dir)

            print("Waiting for SC2 to start (approx 5s)...")
            for i in range(60):
                if self.sc2_process.poll() is not None:
                    raise RuntimeError("SC2 process exited immediately.")
                await asyncio.sleep(1)
                if i > 4:
                    break

            print(f"Connecting to {ws_url}...")
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                connected = False

                for i in range(30):
                    try:
                        async with session.ws_connect(ws_url) as ws:
                            print("Connected to SC2 API!")

                            # 1. Create Game
                            controller = Controller(ws, self.sc2_process)
                            if hasattr(controller, "__aenter__"):
                                await controller.__aenter__()

                            try:
                                players = [
                                    Bot(bot_race, self.bot),
                                    Computer(opponent_race, difficulty)
                                ]
                                print(f"Creating game on {target_map_name}...")
                                await controller.create_game(map_obj, players, realtime=True)
                            except ProtocolError as e:
                                if "Already in the process" in str(e):
                                    print("Game creation skipped (already in progress).")
                                else:
                                    raise e
                            except Exception as e:
                                raise e

                            # 2. Join Game
                            print("Switching to Client for Join...")
                            client = Client(ws)
                            if hasattr(client, "__aenter__"):
                                await client.__aenter__()

                            print("Joining game...")
                            await client.join_game(
                                race=bot_race,
                                observed_player_id=1,
                                portconfig=None,
                                rgb_render_config=None
                            )

                            # 3. MANUAL BOT INITIALIZATION
                            print("Initializing Bot...")
                            proto_game_info = await client._execute(game_info=sc_pb.RequestGameInfo())
                            game_info_wrapper = GameInfo(proto_game_info.game_info)
                            game_data = await client.get_game_data()

                            # --- FIX 1: Inject GameData into SC2 Classes ---
                            if hasattr(sc2.unit, "UnitGameData"):
                                sc2.unit.UnitGameData._game_data = game_data

                            for base in Unit.__mro__:
                                if base.__module__.startswith("sc2."):
                                    try:
                                        setattr(base, "_game_data", game_data)
                                    except:
                                        pass
                            Unit._game_data = game_data

                            # --- FIX 2: Calculate Missing ID Sets ---
                            if not hasattr(game_data, "mineral_ids"):
                                print("Computing mineral_ids manually...")
                                game_data.mineral_ids = set()
                                game_data.vespene_ids = set()
                                for unit_id, unit_type in game_data.units.items():
                                    try:
                                        name = getattr(unit_type, "name", "").upper()
                                        if "MINERALFIELD" in name:
                                            game_data.mineral_ids.add(unit_id)
                                        if "VESPENEGEYSER" in name:
                                            game_data.vespene_ids.add(unit_id)
                                    except Exception:
                                        continue
                                if not game_data.vespene_ids:
                                    game_data.vespene_ids = {342, 344, 882, 608, 2270}

                                    # Inject Dependencies
                            self.bot._client = client
                            self.bot.client = client
                            self.bot._game_info = game_info_wrapper
                            self.bot._game_data = game_data
                            self.bot.player_id = 1
                            self.bot.race = bot_race
                            self.bot._race = bot_race
                            self.bot._previous_upgrades = set()

                            # Initialize Containers
                            for attr in ['units', 'all_units', 'workers', 'townhalls', 'structures',
                                         'minerals', 'vespene', 'resources', 'destructables',
                                         'enemy_units', 'enemy_structures']:
                                setattr(self.bot, attr, Units([]))

                            if inspect.iscoroutinefunction(self.bot.on_start):
                                await self.bot.on_start()
                            else:
                                self.bot.on_start()

                            # Reveal Map
                            print("Revealing map for debugging...")
                            await client.debug_show_map()

                            print("Game started! Running manual loop...")
                            connected = True
                            iteration = 0

                            # --- GAME LOOP ---
                            while self._running:
                                try:
                                    # A. Get Raw Response
                                    response = await client.observation()
                                    if hasattr(response, "observation"):
                                        response_observation = response.observation
                                    else:
                                        response_observation = response

                                    if response_observation.player_result:
                                        print("Game Finished.")
                                        break

                                    # C. Convert to GameState
                                    try:
                                        state = GameState(response_observation)
                                    except TypeError:
                                        state = GameState(response_observation, game_data)

                                    # D. MANUAL UNIT PARSING
                                    raw_units = response_observation.observation.raw_data.units
                                    all_wrapped_units = []
                                    counts = {1: 0, 2: 0, 3: 0, 4: 0}

                                    for u in raw_units:
                                        if u.alliance in counts: counts[u.alliance] += 1
                                        try:
                                            new_unit = Unit(u)
                                        except TypeError:
                                            new_unit = Unit(u, game_data)

                                        if not hasattr(new_unit, "_game_data") or new_unit._game_data is None:
                                            new_unit._game_data = game_data
                                        if not hasattr(new_unit, "_bot") or new_unit._bot is None:
                                            new_unit._bot = self.bot

                                        all_wrapped_units.append(new_unit)

                                    if iteration % 100 == 0:
                                        print(f"Frame {iteration} Raw Counts: Self={counts[1]}, Enemy={counts[4]}")

                                    # E. Populate State Objects
                                    state.own_units = Units([u for u in all_wrapped_units if u.alliance == 1])
                                    state.enemy_units = Units([u for u in all_wrapped_units if u.alliance == 4])

                                    state.mineral_field = Units(
                                        [u for u in all_wrapped_units if u.type_id in game_data.mineral_ids])
                                    state.vespene_geyser = Units(
                                        [u for u in all_wrapped_units if u.type_id in game_data.vespene_ids])
                                    state.destructables = Units(
                                        [u for u in all_wrapped_units if u.alliance == 3 and u.radius > 1])

                                    # F. Prepare Bot Step
                                    if hasattr(self.bot, "_prepare_step"):
                                        if inspect.iscoroutinefunction(self.bot._prepare_step):
                                            await self.bot._prepare_step(state, proto_game_info)
                                        else:
                                            self.bot._prepare_step(state, proto_game_info)

                                    # --- CRITICAL FIX: FORCE INJECT DATA ---
                                    # _prepare_step might have reset the state or cached properties to empty lists
                                    # We force our correctly parsed units back into the bot's state
                                    self.bot.state.own_units = state.own_units
                                    self.bot.state.enemy_units = state.enemy_units
                                    # Also update legacy private attributes if they exist
                                    if hasattr(self.bot, '_units'): self.bot._units = state.own_units
                                    if hasattr(self.bot, '_enemy_units'): self.bot._enemy_units = state.enemy_units
                                    # ----------------------------------------

                                    # G. Run Bot Logic
                                    if inspect.iscoroutinefunction(self.bot.on_step):
                                        await self.bot.on_step(iteration)
                                    else:
                                        self.bot.on_step(iteration)

                                    # H. Issue Events
                                    if hasattr(self.bot, "issue_events"):
                                        if inspect.iscoroutinefunction(self.bot.issue_events):
                                            await self.bot.issue_events()
                                        else:
                                            self.bot.issue_events()

                                    iteration += 1
                                    await asyncio.sleep(0.01)

                                except Exception as e:
                                    print(f"Game Loop Error: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    break

                            print("Game loop ended.")
                            break

                    except aiohttp.ClientConnectorError:
                        await asyncio.sleep(1)
                    except Exception as e:
                        print(f"Connection/Game error: {e}")
                        import traceback
                        traceback.print_exc()
                        await asyncio.sleep(2)

                if not connected:
                    print("Failed to connect/start game after multiple attempts.")

        except asyncio.CancelledError:
            print("SC2 coroutine received cancel.")
        except Exception as e:
            print(f"Error in SC2 execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("SC2 coroutine exiting.")
            self._running = False
            self.debug_ui.close()
            if self.sc2_process:
                print("Terminating SC2 process...")
                try:
                    self.sc2_process.terminate()
                except:
                    pass

    async def start(self):
        loop = asyncio.get_event_loop()
        if self.game_task is None or self.game_task.done():
            self.game_task = loop.create_task(self.run_sc2_game())
        return self.game_task

    async def shutdown(self):
        self._running = False
        if self.game_task:
            self.game_task.cancel()
            try:
                await self.game_task
            except asyncio.CancelledError:
                pass


def main():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    integration = SC2QtIntegration()

    try:
        loop.add_signal_handler(signal.SIGINT, lambda: asyncio.ensure_future(integration.shutdown()))
    except (NotImplementedError, ValueError):
        pass

    app.aboutToQuit.connect(lambda: asyncio.ensure_future(integration.shutdown()))

    with loop:
        loop.create_task(integration.start())
        loop.run_forever()


if __name__ == "__main__":
    main()