
import asyncio
import logging
from typing import List, Optional, Type
from sc2.sc2process import SC2Process
from sc2.client import Client
from sc2.player import Bot, Computer, AbstractPlayer
from sc2.data import Race, Difficulty
from sc2.maps import Map

logger = logging.getLogger(__name__)

class SingleClientSC2Manager:
    """
    Manages a single SC2 process and client.
    """
    def __init__(self, sc2_process_cls: Type[SC2Process] = SC2Process, client_cls: Type[Client] = Client, worker_index: int = 0):
        self.sc2_process_cls = sc2_process_cls
        self.client_cls = client_cls
        self.worker_index = worker_index
        self.process: Optional[SC2Process] = None
        self.client: Optional[Client] = None
        self.controller = None

    async def launch_process(self, visualize: bool = False, fullscreen: bool = False):
        """Launches a single SC2 process."""
        self.process = self.sc2_process_cls(
            fullscreen=fullscreen,
            render=visualize
        )
        self.controller = await self.process.__aenter__()
        
        # Extract websocket connection
        ws = getattr(self.process, "_ws", getattr(self.process, "ws", None))
        self.client = self.client_cls(ws)
        
        # Give it a moment to stabilize
        await asyncio.sleep(0.2)

    async def create_join_game(self, game_map: Map, players: List[AbstractPlayer], realtime: bool = True):
        """
        Creates and joins a game with the single client.
        In "God-Mode", we expect one Bot and possibly one Computer or just one Bot.
        """
        if not self.client:
            raise RuntimeError("Process not launched.")

        logger.info(f"Creating game on map: {game_map.name}...")
        try:
            result = await self.controller.create_game(
                game_map,
                players,
                realtime=realtime
            )
            if result.create_game.HasField("error"):
                logger.error(f"Engine Error: {result.create_game.error}")
            
            # Map loading wait - optimized to 10s based on previous track findings
            await asyncio.sleep(10)
        except Exception as e:
            logger.error(f"Exception during create_game: {e}")
            raise e

        logger.info("Joining game...")
        try:
            # Join as the first player (Host)
            # In God-Mode, we can control all units regardless of which player we joined as,
            # as long as we use debug commands.
            await self.client.join_game(
                name="GodBot",
                race=players[0].race,
                observed_player_id=1
            )
            logger.info("Single-client session established.")
        except Exception as e:
            logger.error(f"Error during join_game: {e}")
            await self.close()
            raise e

    async def close(self):
        """Closes connection and kills SC2 process."""
        if self.process:
             try:
                 await self.process.__aexit__(None, None, None)
             except:
                 pass
        
        # Aggressive cleanup for Windows
        try:
            from src.utils.sc2_cleanup import kill_switch_sc2
            kill_switch_sc2()
        except:
            pass
        
        await asyncio.sleep(0.5)
        
        self.process = None
        self.client = None
        self.controller = None
