
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
    Manages a single SC2 process and one or more clients.
    """
    def __init__(self, sc2_process_cls: Type[SC2Process] = SC2Process, client_cls: Type[Client] = Client, worker_index: int = 0):
        self.sc2_process_cls = sc2_process_cls
        self.client_cls = client_cls
        self.worker_index = worker_index
        self.process: Optional[SC2Process] = None
        self.clients: List[Client] = []
        self.controller = None

    @property
    def client(self) -> Optional[Client]:
        return self.clients[0] if self.clients else None

    async def launch_process(self, client_count: int = 1, visualize: bool = False, fullscreen: bool = False):
        """Launches a single SC2 process and connects clients."""
        self.process = self.sc2_process_cls(
            fullscreen=fullscreen,
            render=visualize
        )
        self.controller = await self.process.__aenter__()
        
        # Primary client
        ws = getattr(self.process, "_ws", getattr(self.process, "ws", None))
        self.clients.append(self.client_cls(ws))
        
        # Secondary clients sharing the same process
        for i in range(1, client_count):
            ws_extra = await self.process._session.ws_connect(self.process.ws_url)
            self.clients.append(self.client_cls(ws_extra))
        
        # Give it a moment to stabilize
        await asyncio.sleep(0.2)

    async def create_join_game(self, game_map: Map, players: List[AbstractPlayer], realtime: bool = True):
        """
        Creates a game and joins with all connected clients.
        """
        if not self.clients:
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
            
            await asyncio.sleep(10)
        except Exception as e:
            logger.error(f"Exception during create_game: {e}")
            raise e

        logger.info(f"Joining game with {len(self.clients)} clients...")
        try:
            join_tasks = []
            for i, client in enumerate(self.clients):
                join_tasks.append(client.join_game(
                    name=f"Bot{i+1}",
                    race=players[i].race if i < len(players) else Race.Zerg,
                    observed_player_id=i+1
                ))
            await asyncio.gather(*join_tasks)
            logger.info("Single-process multi-client session established.")
        except Exception as e:
            logger.error(f"Error during join_game: {e}")
            await self.close()
            raise e

    async def close(self):
        """Closes connections and kills SC2 process."""
        # Close additional clients first
        for client in self.clients[1:]:
            try:
                await client._ws.close()
            except:
                pass

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
        self.clients.clear()
        self.controller = None
