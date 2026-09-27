import multiprocessing
import multiprocessing.connection
import traceback
from loguru import logger
from typing import Dict
from pathlib import Path

from sc2 import maps
from sc2.player import Bot, Computer
from sc2.data import Race, Difficulty
from sc2.main import run_game
from sc2.maps import Map
from src.sc2_bot import SC2Bot
from src.utils.sc2_cleanup import kill_switch_sc2


class GameProcess(multiprocessing.Process):
    def __init__(self, game_config: Dict, conn: multiprocessing.connection.Connection):
        super().__init__()
        self.conn = conn
        self.game_config = game_config

    def _resolve_map(self, map_name):
        logger.info(f"GameProcess ({self.pid}) _resolve_map called for {map_name}")
        try:
            resolved_map = maps.get(map_name)
            logger.info(f"GameProcess ({self.pid}) Map '{map_name}' resolved via maps.get().")
            return resolved_map
        except KeyError:
            project_root = Path(__file__).resolve().parent.parent
            logger.info(f"GameProcess ({self.pid}) project_root: {project_root}")
            
            level_name = self.game_config.get("level_name", "level_0")
            map_name_without_ext = map_name.replace(".SC2Map", "")
            
            map_path = project_root / "train" / "maps" / level_name / f"{map_name_without_ext}.SC2Map"
            logger.info(f"GameProcess ({self.pid}) Constructed map_path: {map_path}")
            if not map_path.exists():
                logger.error(f"GameProcess ({self.pid}) Map not found: {map_path}")
                raise FileNotFoundError(f"Map not found: {map_path}")
            logger.info(f"GameProcess ({self.pid}) Map '{map_name}' resolved via direct path.")
            return Map(map_path)

    def run(self):
        # Configure logging for the child process
        import os
        import sys
        logger.remove()
        log_level = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
        logger.add(sys.stderr, level=log_level, colorize=False)
        from train.training_process_logging import configure_process_alert_logging

        configure_process_alert_logging()

        logger.info(f"GameProcess ({self.pid}) started. Args: {self.game_config}")
        
        try:
            logger.info(f"GameProcess ({self.pid}) Initializing SC2Bot...")
            bot = SC2Bot(self.conn)
            
            logger.info(f"GameProcess ({self.pid}) Resolving map...")
            game_map = self._resolve_map(self.game_config["map_name"])
            
            logger.info(f"GameProcess ({self.pid}) Calling run_game with map={game_map.path}, realtime={self.game_config['realtime']}...")
            run_game(
                game_map,
                [
                    Bot(Race.Zerg, bot),
                    Computer(Race.Zerg, Difficulty.Easy),
                ],
                realtime=self.game_config["realtime"],
            )
            logger.info(f"GameProcess ({self.pid}) run_game returned.")
        except Exception as e:
            logger.error(f"GameProcess ({self.pid}) encountered an error: {e}")
            logger.error(traceback.format_exc())
            self.conn.send(e)
        finally:
            logger.info(f"GameProcess ({self.pid}) finished. Closing connection and cleaning up SC2.")
            self.conn.close()
            kill_switch_sc2()
