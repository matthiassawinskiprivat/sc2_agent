import sys
import os
import time
import traceback
import signal
from pathlib import Path

# --- Qt Imports ---
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread

# --- SC2 Imports ---
import sc2
from sc2 import maps
from sc2.maps import Map  # <--- CRITICAL IMPORT
from sc2.player import Bot, Computer
from sc2.data import Race, Difficulty
from sc2.main import run_game

# --- Project Imports ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.hierarchical_agent import HierarchicalAgentSystem
from src.sc2_bot import UnitControlBot
from src.ui.debug_ui import DebugUI
from src.ui.ui_manager import UIManager

# --- Configuration ---
# Change this to "Zerg_Lower_Left" to test your training map
MAP_NAME = "Middle_Only"
BOT_RACE = Race.Terran
OPPONENT_RACE = Race.Zerg
DIFFICULTY = Difficulty.Easy


class GameThread(QThread):
    """
    Runs the StarCraft II game loop in a background thread.
    """

    def __init__(self, bot_instance, map_name):
        super().__init__()
        self.bot = bot_instance
        self.map_name = map_name

    def run(self):
        print(f"[GameThread] Starting SC2 on map: {self.map_name}")

        # --- 1. Signal Hack ---
        original_signal = signal.signal

        def no_op_signal(sig, handler):
            pass

        signal.signal = no_op_signal

        try:
            # --- 2. Resolve Map Object (FIXED) ---
            game_map = None

            # A. Try Standard Library
            try:
                game_map = maps.get(self.map_name)
            except KeyError:
                pass

            # B. Try Local File Search (if library lookup failed)
            if game_map is None:
                print(f"[GameThread] Map '{self.map_name}' not in library. Searching local files...")

                base_dir = Path(os.path.dirname(os.path.abspath(__file__)))

                # Check probable locations
                candidates = [
                    base_dir / "train" / "maps" / "level_0" / f"{self.map_name}.SC2Map",
                    base_dir / "Maps" / f"{self.map_name}.SC2Map",
                    base_dir / f"{self.map_name}.SC2Map"
                ]

                for path in candidates:
                    if path.exists():
                        print(f"[GameThread] Found map at: {path}")
                        game_map = Map(path)  # Create proper Map object
                        break

                if game_map is None:
                    raise FileNotFoundError(f"Could not find map file '{self.map_name}.SC2Map' in project folders.")

            # --- 3. Run Game ---
            run_game(game_map, [
                Bot(BOT_RACE, self.bot),
                Computer(OPPONENT_RACE, DIFFICULTY)
            ], realtime=True, disable_fog=True)

        except Exception as e:
            print(f"[GameThread] SC2 Crashed: {e}")
            traceback.print_exc()
        finally:
            signal.signal = original_signal
            print("[GameThread] Game Ended.")


class SC2Visualizer:
    def __init__(self):
        print("[Main] Initializing Agent System...")

        # 1. Init Agent System
        self.system = HierarchicalAgentSystem(map_size=(128, 128))
        self.agent_id = "main_agent"
        self.main_agent = self.system.create_unit_agent(self.agent_id)

        # 2. Init Bot Logic
        self.bot = UnitControlBot(self.system, agent_id=self.agent_id)

        # 3. Init UI Manager
        self.ui_manager = UIManager(self.main_agent)

        # 4. Show Window
        self.debug_ui = DebugUI(self.main_agent, self.ui_manager)
        self.debug_ui.show()

        # 5. Start Game in Background
        self.game_thread = GameThread(self.bot, MAP_NAME)
        self.game_thread.start()


def main():
    app = QApplication(sys.argv)
    visualizer = SC2Visualizer()
    app.aboutToQuit.connect(lambda: visualizer.game_thread.terminate())
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()