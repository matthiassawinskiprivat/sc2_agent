import sys

import os

import time

import traceback

import signal
from contextlib import contextmanager

from pathlib import Path



# --- Qt Imports ---

from PyQt5.QtWidgets import QApplication

from PyQt5.QtCore import QThread, QTimer



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

from src.ui.level_preview import (

    load_pipeline_levels,

    build_preview_training_conf,

    discover_level_maps,

    create_preview_env,

    push_obs_dict_to_agent,

    reconfigure_hierarchical_system,

    sync_preview_debug_click_overlay,

)

from src.utils.layer_constants import LayerConstants

from src.utils.sc2_window_focus import try_focus_starcraft_window





def _layer_names_in_map_order(layer_dict: dict) -> list:

    """Stable channel order for observation tensors (sorted by layer index)."""

    return [name for name, _ in sorted(layer_dict.items(), key=lambda kv: kv[1])]



# --- Configuration ---

LADDER_MAPS_DIR = "train/maps/ladder"

DEFAULT_MAP_NAME = "PersephoneAIE_v4"

DEFAULT_ALLIED_RACE = Race.Terran

DEFAULT_ENEMY_RACE = Race.Zerg

DIFFICULTY = Difficulty.Easy


@contextmanager
def _disable_signal_handlers():
    """python-sc2 registers SIGINT when SC2 starts; that only works on the main thread."""
    original_signal = signal.signal

    def no_op_signal(sig, handler):
        pass

    signal.signal = no_op_signal
    try:
        yield
    finally:
        signal.signal = original_signal



def _resolve_map(map_name: str) -> Map:

    """Resolve a map from project files first; fall back to the SC2 library registry."""

    project_root = Path(__file__).resolve().parent

    candidates = [

        project_root / LADDER_MAPS_DIR / f"{map_name}.SC2Map",

        project_root / "train" / "maps" / f"{map_name}.SC2Map",

        project_root / "Maps" / f"{map_name}.SC2Map",

        project_root / f"{map_name}.SC2Map",

    ]



    for path in candidates:

        if path.exists():

            print(f"[GameThread] Found map at: {path}")

            return Map(path)



    try:

        return maps.get(map_name)

    except (KeyError, FileNotFoundError, OSError):

        pass



    raise FileNotFoundError(

        f"Could not find map file '{map_name}.SC2Map' in project folders "

        f"(checked ladder and other local paths)."

    )





class GameThread(QThread):

    """Runs free-play StarCraft II via python-sc2 ``run_game``."""



    def __init__(self, bot_instance, map_name, allied_race: Race, enemy_race: Race):

        super().__init__()

        self.bot = bot_instance

        self.map_name = map_name

        self.allied_race = allied_race

        self.enemy_race = enemy_race



    def run(self):

        print(f"[GameThread] Starting SC2 on map: {self.map_name}")



        try:
            with _disable_signal_handlers():
                game_map = _resolve_map(self.map_name)
                run_game(game_map, [
                    Bot(self.allied_race, self.bot),
                    Computer(self.enemy_race, DIFFICULTY)
                ], realtime=True, disable_fog=True)
        except Exception as e:
            print(f"[GameThread] SC2 Crashed: {e}")
            traceback.print_exc()
        finally:
            print("[GameThread] Game Ended.")





class TrainingLevelGameThread(QThread):

    """Runs a real training env (same setup as curriculum) for observation preview."""



    def __init__(self, visualizer, level_name: str, map_name: str):

        super().__init__()

        self.visualizer = visualizer

        self.level_name = level_name

        self.map_name = map_name

        self._stop = False
        self.env = None



    def request_stop(self):

        self._stop = True



    def run(self):

        env = None

        print(f"[TrainingLevel] Starting level={self.level_name} map={self.map_name}")

        try:
            with _disable_signal_handlers():
                training_conf = reconfigure_hierarchical_system(
                    self.visualizer.system,
                    self.level_name,
                )
                env = create_preview_env(self.level_name, self.map_name, training_conf)
                self.env = env
                obs, _ = env.reset()
                push_obs_dict_to_agent(
                    self.visualizer.main_agent,
                    obs,
                    training_conf,
                )
                sync_preview_debug_click_overlay(env, self.visualizer.main_agent)

                while not self._stop:
                    try:
                        obs = env.preview_tick()
                        push_obs_dict_to_agent(
                            self.visualizer.main_agent,
                            obs,
                            training_conf,
                        )
                        sync_preview_debug_click_overlay(env, self.visualizer.main_agent)
                    except Exception as tick_err:
                        print(f"[TrainingLevel] preview_tick failed: {tick_err}")
                        traceback.print_exc()
                        break
                    time.sleep(1.0 / 22.4)
        except Exception as e:
            print(f"[TrainingLevel] Failed: {e}")
            traceback.print_exc()
        finally:
            self.env = None
            if env is not None:
                try:
                    with _disable_signal_handlers():
                        env.close()
                except Exception as close_err:
                    print(f"[TrainingLevel] close error: {close_err}")
            print("[TrainingLevel] Session ended.")





class SC2Visualizer:

    def __init__(self):

        print("[Main] Initializing Agent System...")



        training_conf = {

            'agent_architecture': 'universal',

            'render_beacon': True,

            'larger_units': False,

            'action_heads': ['move_global', 'move_local'],

            'active_layers': _layer_names_in_map_order(LayerConstants.LAYERS),

            'active_tactical_layers': _layer_names_in_map_order(LayerConstants.TACTICAL_LAYERS),

        }

        self.system = HierarchicalAgentSystem(map_size=(128, 128), training_conf=training_conf)

        self.agent_id = "main_agent"

        self.main_agent = self.system.create_unit_agent(self.agent_id, dilation_size=0)



        self.bot = UnitControlBot(self.system, agent_id=self.agent_id)



        self.ui_manager = UIManager(self.main_agent)



        self.game_thread = None

        self.pipeline_levels = load_pipeline_levels()

        self.debug_ui = DebugUI(

            self.main_agent,

            self.ui_manager,

            maps_dir=LADDER_MAPS_DIR,

            on_start_game=self.start_free_play_game,

            on_start_training_level=self.start_training_level_game,

            pipeline_levels=self.pipeline_levels,

            on_training_level_changed=self._maps_for_level,

            on_manual_spatial_move=self.request_manual_spatial_move,

            default_allied_race=DEFAULT_ALLIED_RACE.name,

            default_enemy_race=DEFAULT_ENEMY_RACE.name,

            default_map_name=DEFAULT_MAP_NAME,

        )

        self.debug_ui.show()



    def _maps_for_level(self, level_name: str):

        training_conf = build_preview_training_conf(level_name)

        return discover_level_maps(level_name, training_conf)



    def _schedule_sc2_window_focus(self):

        """SC2 spawns behind the Qt window; bring it forward for audio + visibility."""

        for delay_ms in (1500, 4000, 8000):

            QTimer.singleShot(delay_ms, try_focus_starcraft_window)



    def _clear_terrain_cache(self):

        self.system.shared_transformer.clear_cache()

        tr = self.system.shared_transformer.transformer

        tr._terrain_caches.clear()

        tr._last_map_name = None



    def start_free_play_game(self, map_name: str, allied_race: Race, enemy_race: Race):

        if self.game_thread is not None and self.game_thread.isRunning():

            return



        self._clear_terrain_cache()

        self.debug_ui.set_game_controls_enabled(False)

        self.game_thread = GameThread(self.bot, map_name, allied_race, enemy_race)

        self.game_thread.finished.connect(self._on_game_finished)

        self.game_thread.start()

        self._schedule_sc2_window_focus()



    def start_training_level_game(self, level_name: str, map_name: str):

        if self.game_thread is not None and self.game_thread.isRunning():

            return



        self._clear_terrain_cache()

        self.debug_ui.set_game_controls_enabled(False)

        self.game_thread = TrainingLevelGameThread(self, level_name, map_name)

        self.game_thread.finished.connect(self._on_game_finished)

        self.game_thread.start()

        self._schedule_sc2_window_focus()



    def _on_game_finished(self):

        self.debug_ui.set_game_controls_enabled(True)

        self.game_thread = None



    def request_manual_spatial_move(self, command_name: str, gx: int, gy: int) -> None:

        """Issue a policy-equivalent spatial move in the active SC2 session."""

        thread = self.game_thread

        if thread is None or not thread.isRunning():

            return

        if isinstance(thread, TrainingLevelGameThread):

            env = getattr(thread, "env", None)

            if env is not None and hasattr(env, "queue_preview_spatial_move"):

                env.queue_preview_spatial_move(command_name, gx, gy)

            return

        bot = getattr(thread, "bot", None)

        if bot is not None and hasattr(bot, "queue_manual_spatial_move"):

            bot.queue_manual_spatial_move(command_name, gx, gy)





def main():

    app = QApplication(sys.argv)

    visualizer = SC2Visualizer()



    def terminate_game():

        if visualizer.game_thread is not None and visualizer.game_thread.isRunning():

            if hasattr(visualizer.game_thread, "request_stop"):

                visualizer.game_thread.request_stop()

            visualizer.game_thread.terminate()



    app.aboutToQuit.connect(terminate_game)

    sys.exit(app.exec_())





if __name__ == "__main__":

    main()


