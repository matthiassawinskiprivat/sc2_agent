import os
import datetime
import random
from pathlib import Path
from sc2 import maps
from sc2.player import Human, Computer, Bot
from sc2.data import Race, Difficulty
from sc2.main import run_game
from sc2.bot_ai import BotAI
from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId
from loguru import logger

class ReplayObserver(BotAI):
    """
    An interactive bot that does nothing, allowing a human to take control.
    Supports level-specific setup (e.g. random unit placement) and chat commands.
    """
    def __init__(self, level_name: str, disable_fog: bool = False):
        super().__init__()
        self.level_name = level_name
        self.disable_fog = disable_fog

    async def on_start(self):
        if self.disable_fog:
            logger.info("Disabling Fog of War for recording...")
            await self.client.debug_show_map()
            
        if self.level_name == "level_0":
            await self._setup_level_0()

    async def _setup_level_0(self):
        """Randomizes Zergling and Beacon positions, favoring edges for the beacon."""
        logger.info("Setting up Level 0: Randomizing Zergling and Beacon (Edge Bias enabled)...")
        
        # 1. Clear Map
        if self.all_units:
            await self.client.debug_kill_unit([u.tag for u in self.all_units])
            # Materialize death
            await self.client.step()
        
        # 2. Determine spawn area
        area = self.game_info.playable_area
        margin = 2.0 # Use smaller margin to allow getting closer to actual edges
        
        # Find valid pathable points
        grid = self.game_info.pathing_grid
        valid_points = []
        
        # Sample points with higher density (step 1)
        for x in range(int(area.x + margin), int(area.x + area.width - margin)):
            for y in range(int(area.y + margin), int(area.y + area.height - margin)):
                if grid[(x, y)] != 0:
                    valid_points.append(Point2((x, y)))

        if len(valid_points) < 2:
            p1 = Point2((area.x + area.width/2, area.y + area.height/2))
            p2 = Point2((area.x + area.width/2 + 5, area.y + area.height/2))
        else:
            # Favor edges for the Beacon (p2)
            min_x = min(p.x for p in valid_points)
            max_x = max(p.x for p in valid_points)
            min_y = min(p.y for p in valid_points)
            max_y = max(p.y for p in valid_points)
            
            # Define 'edge points' as those within 4 units of any boundary
            edge_threshold = 4.0
            edge_points = [
                p for p in valid_points
                if p.x < min_x + edge_threshold or p.x > max_x - edge_threshold or
                   p.y < min_y + edge_threshold or p.y > max_y - edge_threshold
            ]
            
            # 80% chance to pick an edge point for the beacon to focus on difficult cases
            if edge_points and random.random() < 0.8:
                p2 = random.choice(edge_points)
            else:
                p2 = random.choice(valid_points)
            
            # Zergling (p1) can be anywhere
            p1 = random.choice(valid_points)
        
        # Ensure minimum distance between them
        attempts = 0
        while p1.distance_to(p2) < 10.0 and attempts < 30:
            p1 = random.choice(valid_points)
            attempts += 1

        # 3. Spawn units
        await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 1, p1, self.player_id]])
        await self.client.debug_create_unit([[UnitTypeId.BEACON_TERRANSMALL, 1, p2, self.player_id]])
        
        # Materialize units
        await self.client.step()
        logger.info(f"Units ready. Zergling at {p1}, Beacon (Edge-Favored) at {p2}")

    async def on_step(self, iteration: int):
        # Listen for chat commands from the human player
        if self.state.chat:
            for msg in self.state.chat:
                command = msg.message.lower().strip()
                if command in ["quit", "exit", "done", "end"]:
                    logger.info(f"Graceful quit requested via chat command: '{command}'")
                    await self.client.leave()
                elif command == "reset":
                    logger.info("Manual level reset requested via chat command.")
                    if self.level_name == "level_0":
                        await self._setup_level_0()

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
                    Bot(agent_race, ReplayObserver(level_name=level_name, disable_fog=disable_fog)),
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