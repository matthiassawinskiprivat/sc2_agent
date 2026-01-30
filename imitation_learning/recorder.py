import os
import datetime
from pathlib import Path
from sc2 import maps
from sc2.player import Human, Computer
from sc2.data import Race, Difficulty
from sc2.main import run_game
from loguru import logger

class Recorder:
    """
    Handles launching SC2 games for human recording of training data.
    """
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.replays_raw_root = self.project_root / "imitation_learning" / "replays_raw"
        self.maps_root = self.project_root / "train" / "maps"

    def _parse_races_from_map_name(self, map_name: str) -> tuple[Race, Race]:
        """
        Parses agent and enemy races from map name (e.g., ...Terran_vs_Zerg).
        Returns (agent_race, enemy_race).
        """
        agent_race = Race.Terran
        enemy_race = Race.Terran

        if "_vs_" in map_name:
            try:
                parts = map_name.split("_vs_")
                agent_part = parts[0].split("_")[-1]
                enemy_part = parts[1].split("_")[0]

                mapping = {
                    "Terran": Race.Terran,
                    "Zerg": Race.Zerg,
                    "Protoss": Race.Protoss
                }

                agent_race = mapping.get(agent_part, Race.Terran)
                enemy_race = mapping.get(enemy_part, Race.Terran)
            except Exception as e:
                logger.error(f"Error parsing races from map name {map_name}: {e}")

        return agent_race, enemy_race

    def get_map_path(self, level_name: str, map_name: str) -> Path:
        """Resolves the absolute path to a map file."""
        return self.maps_root / level_name / f"{map_name}.SC2Map"

    def start_recording(self, level_name: str, map_name: str):
        """
        Launches a game in human mode and saves the replay.
        """
        map_path = self.get_map_path(level_name, map_name)
        if not map_path.exists():
            raise FileNotFoundError(f"Map not found: {map_path}")

        agent_race, enemy_race = self._parse_races_from_map_name(map_name)
        
        # Prepare replay destination
        level_replay_dir = self.replays_raw_root / level_name
        level_replay_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        replay_filename = f"{timestamp}_{map_name}.SC2Replay"
        replay_path = level_replay_dir / replay_filename

        logger.info(f"Starting recording on {map_name} as {agent_race.name}")
        
        try:
            run_game(
                maps.Map(map_path),
                [
                    Human(agent_race),
                    Computer(enemy_race, Difficulty.Easy)
                ],
                realtime=True,
                save_replay_as=str(replay_path)
            )
            logger.info(f"Recording finished. Replay saved to {replay_path}")
            return str(replay_path)
        except Exception as e:
            logger.error(f"Failed to run game for recording: {e}")
            raise e
