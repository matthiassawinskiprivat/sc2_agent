import asyncio
import logging
from typing import List, Optional, Type
from sc2.sc2process import SC2Process
from sc2.client import Client
from sc2.player import Bot, Computer, AbstractPlayer
from sc2.data import Race, Difficulty
from sc2.maps import Map

logger = logging.getLogger(__name__)

class MultiClientSC2Manager:
    """
    Manages multiple SC2 processes and clients for self-play or multi-agent scenarios.
    """
    def __init__(self, sc2_process_cls: Type[SC2Process] = SC2Process, client_cls: Type[Client] = Client):
        self.sc2_process_cls = sc2_process_cls
        self.client_cls = client_cls
        self.processes: List[SC2Process] = []
        self.clients: List[Client] = []
        self.controllers = []
        
        self.host_client: Optional[Client] = None
        self.join_client: Optional[Client] = None

    async def launch_processes(self, count: int = 2, visualize: bool = False, fullscreen: bool = False):
        """Launches the specified number of SC2 processes."""
        for i in range(count):
            # When visualizing, we prefer windowed mode with a specific resolution for multiple instances
            resolution = (800, 600) if visualize else None
            # Place windows side by side if visualizing
            placement = (i * 810, 50) if visualize else None
            
            # Note: Do NOT use 'render=True' on Windows as it tries to load 'libEGL.so' (Linux only)
            # Just setting fullscreen=False is enough to show the game window.
            process = self.sc2_process_cls(
                fullscreen=fullscreen, 
                resolution=resolution,
                placement=placement
            )
            controller = await process.__aenter__()
            
            # Extract websocket connection
            ws = getattr(process, "_ws", getattr(process, "ws", None))
            client = self.client_cls(ws)
            
            self.processes.append(process)
            self.controllers.append(controller)
            self.clients.append(client)
            
            # Give it a moment to stabilize
            await asyncio.sleep(0.5)

        if len(self.clients) >= 1:
            self.host_client = self.clients[0]
        if len(self.clients) >= 2:
            self.join_client = self.clients[1]

    async def create_join_game(self, game_map: Map, players: List[AbstractPlayer], realtime: bool = False):
        """
        Creates a game on the host and joins with all clients.
        Assumes 2 players for now.
        """
        if len(self.clients) < 2:
            raise RuntimeError("Need at least 2 clients launched.")

        # Port configuration for multi-client communication
        from sc2.portconfig import Portconfig
        # contiguous_ports is often more stable for local multi-client sessions
        try:
            portconfig = Portconfig.contiguous_ports()
        except Exception:
            portconfig = Portconfig()
            
        logger.info(f"Using Portconfig: {portconfig.as_json}")

        # Host creates the game
        logger.info("Host is creating the game...")
        try:
            # We use the controller of the first process to create the game
            await self.controllers[0].create_game(
                game_map,
                players,
                realtime=realtime
            )
            logger.info("Game created successfully. Waiting for map to load...")
            # Increased delay to ensure SC2 is ready to accept join requests
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error during create_game: {e}")
            raise e

        # Both join in parallel - this is crucial for Portconfig handshake
        logger.info("Both players are joining the game...")
        try:
            join_tasks = [
                self.clients[0].join_game(
                    name="Player1",
                    race=Race.Zerg, 
                    portconfig=portconfig
                ),
                self.clients[1].join_game(
                    name="Player2",
                    race=Race.Zerg, 
                    portconfig=portconfig
                )
            ]
            await asyncio.gather(*join_tasks)
            logger.info("Multi-client session established (both players joined).")
        except Exception as e:
            logger.error(f"Error during parallel join_game: {e}")
            raise e

    async def close(self):
        """Closes all connections and processes."""
        # Close clients first
        # We don't explicit call client.leave() usually if we are just killing the process,
        # but closing connection is good practice.
        
        # Kill processes
        # Using aexit on the process controller/manager
        for process in self.processes:
             try:
                 await process.__aexit__(None, None, None)
             except Exception as e:
                 logger.error(f"Error closing process: {e}")
        
        self.processes.clear()
        self.clients.clear()
        self.controllers.clear()
        self.host_client = None
        self.join_client = None
