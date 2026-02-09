import os
import datetime
import random
import yaml
from pathlib import Path
from sc2 import maps
from sc2.player import Human, Computer, Bot
from sc2.data import Race, Difficulty
from sc2.main import run_game
from sc2.bot_ai import BotAI
from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId
from loguru import logger
from src.level_0_bot import Level0Bot
from src.utils.proto_utils import ProtoWrapper

class ReplayObserver(BotAI):
    """
    An interactive bot that does nothing, allowing a human to take control.
    Supports level-specific setup (e.g. random unit placement) and chat commands.
    """
    def __init__(self, level_name: str, config: dict, disable_fog: bool = False):
        super().__init__()
        self.level_name = level_name
        self.config = config
        self.disable_fog = disable_fog
        self.zergling = None
        self.beacon = None
        self.previous_distance = None

    async def on_start(self):
        if self.disable_fog:
            logger.info("Disabling Fog of War for recording...")
            await self.client.debug_show_map()

        await self._setup_level_scenario()

    async def _setup_level_scenario(self):
        """
        Clears the map and sets up the level scenario.
        The objective beacon is spawned for Player 2 (Computer) to maintain the game session 
        without needing extra dummy units, ensuring only 1 beacon is visible.
        """
        env_conf = self.config.get("env", {})
        # Defaults to Level0Env if not specified, matching train_pipeline.py logic
        env_class = env_conf.get("env_class", "Level0Env")
        
        if env_class == "Level0Env":
             logger.info(f"Preparing scenario for {self.level_name} (Env: {env_class})...")
             
             # Reset internal tracking
             self.zergling = None
             self.beacon = None
             self.previous_distance = None

             # 1. Robust Cleanup Loop
             await self._refresh_state()
             for cleanup_attempt in range(3):
                 if not self.all_units:
                     break
                 logger.info(f"Cleanup attempt {cleanup_attempt+1} (Units remaining: {self.all_units.amount})")
                 await self.client.debug_kill_unit(self.all_units.tags)
                 await self.client.step()
                 await self._refresh_state()

             # 2. Determine positions (Parity with Level0Bot logic)
             area = self.game_info.playable_area
             margin = 5.0
             grid = self.game_info.pathing_grid
             valid_points = []
             # Sample points to avoid huge list
             for x in range(int(area.x + margin), int(area.x + area.width - margin), 2):
                 for y in range(int(area.y + margin), int(area.y + area.height - margin), 2):
                     if grid[(x, y)] != 0:
                         valid_points.append(Point2((x, y)))

             if len(valid_points) < 2:
                 p1 = Point2((area.x + area.width/2, area.y + area.height/2))
                 p2 = Point2((area.x + area.width/2 + 5, area.y + area.height/2))
             else:
                 p1, p2 = random.sample(valid_points, 2)
                 attempts = 10
                 while p1.distance_to(p2) < 10.0 and attempts > 0:
                     p1, p2 = random.sample(valid_points, 2)
                     attempts -= 1

             # 3. Spawn Units
             # Player 1 (Human) gets the Zergling
             await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 1, p1, self.player_id]])
             # Player 2 (Computer) gets the Beacon to maintain the game session and serve as objective
             await self.client.debug_create_unit([[UnitTypeId.BEACON_TERRANSMALL, 1, p2, 2]])
             
             await self.client.step()
             await self._refresh_state()
             
             # Identify units for internal tracking (Level0Bot logic expects these)
             self.zergling = self.units(UnitTypeId.ZERGLING).first if self.units(UnitTypeId.ZERGLING).exists else None
             self.beacon = self.all_units(UnitTypeId.BEACON_TERRANSMALL).first if self.all_units(UnitTypeId.BEACON_TERRANSMALL).exists else None
             
             if self.zergling and self.beacon:
                 logger.info(f"Scenario ready. Zergling (P1) at {p1}, Beacon (P2 Objective) at {p2}")
             
             await self.chat_send(f"Recording started. Reach the beacon and type 'q' to finish.")

    async def _refresh_state(self):
        """
        Refreshes the bot's internal state from a new observation.
        Required by Level0Bot.setup_scenario.
        """
        obs = await self.client.observation()
        from sc2.game_state import GameState
        
        # Ensure we have proto info for prepare_step
        if not hasattr(self.game_info, "_proto") or not self.game_info._proto:
             game_info = await self.client.get_game_info()
             proto = game_info._proto
        else:
             proto = self.game_info._proto
        
        # Wrap the proto to match the library's expected structure
        self._prepare_step(GameState(obs.observation), ProtoWrapper(proto))

    async def on_step(self, iteration: int):
        # Listen for chat commands from the human player
        if self.state.chat:
            for msg in self.state.chat:
                command = msg.message.lower().strip()
                if command in ["q", "quit", "exit", "done", "end"]:
                    logger.info(f"Graceful quit requested via chat command: '{command}'")
                    await self.client.leave()
                elif command in ["r", "reset"]:
                    logger.info("Manual level reset requested via chat command.")
                    await self._setup_level_scenario()

    async def on_end(self, game_result):
        logger.info(f"Game ended with result: {game_result}")

class Recorder:
    """
    Handles launching SC2 games for human recording of training data.
    """
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
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

    def start_recording(self, level_name: str, map_name: str, disable_fog: bool = False):
        """
        Launches a game in human mode and saves the replay.
        """
        map_path = self.get_map_path(level_name, map_name)
        if not map_path.exists():
            raise FileNotFoundError(f"Map not found: {map_path}")

        agent_race, enemy_race = self._parse_races_from_map_name(map_name)
        
        # Load level configuration
        config_path = self.project_root / "config" / f"{level_name}.yaml"
        config = {}
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed to load config for {level_name}: {e}")
        
        # Prepare replay destination
        level_replay_dir = self.replays_raw_root / level_name
        level_replay_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        replay_filename = f"{timestamp}_{map_name}.SC2Replay"
        replay_path = level_replay_dir / replay_filename

        logger.info(f"Starting recording on {map_name} as {agent_race.name} (Fog of War: {not disable_fog})")
        logger.info("NOTE: You are playing as a 'Bot' proxy. You can still use the mouse/keyboard to move units.")
        
        try:
            run_game(
                maps.Map(map_path),
                [
                    Bot(agent_race, ReplayObserver(level_name=level_name, config=config, disable_fog=disable_fog)),
                    Computer(enemy_race, Difficulty.Easy)
                ],
                realtime=True,
                save_replay_as=str(replay_path)
            )
            
            if replay_path.exists():
                logger.info(f"Recording finished. Replay saved to {replay_path}")
            else:
                logger.warning(f"Game ended but replay file was not found at {replay_path}.")
                
            return str(replay_path)
        except Exception as e:
            logger.error(f"Failed to run game for recording: {e}")
            raise e