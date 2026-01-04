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
    def __init__(self, sc2_process_cls: Type[SC2Process] = SC2Process, client_cls: Type[Client] = Client, worker_index: int = 0):
        self.sc2_process_cls = sc2_process_cls
        self.client_cls = client_cls
        self.worker_index = worker_index
        self.processes: List[SC2Process] = []
        self.clients: List[Client] = []
        self.controllers = []
        
        self.host_client: Optional[Client] = None
        self.join_client: Optional[Client] = None

    async def launch_processes(self, count: int = 2, visualize: bool = False, fullscreen: bool = False):
        """Launches the specified number of SC2 processes."""
        for i in range(count):
            # Let burnysc2 handle host/port selection for maximum stability
            process = self.sc2_process_cls(
                fullscreen=fullscreen,
                render=visualize
            )
            controller = await process.__aenter__()
            
            # Extract websocket connection
            ws = getattr(process, "_ws", getattr(process, "ws", None))
            client = self.client_cls(ws)
            
            self.processes.append(process)
            self.controllers.append(controller)
            self.clients.append(client)
            
            # Give it a moment to stabilize
            await asyncio.sleep(0.2)

        if len(self.clients) >= 1:
            self.host_client = self.clients[0]
        if len(self.clients) >= 2:
            self.join_client = self.clients[1]

    async def create_join_game(self, game_map: Map, players: List[AbstractPlayer], realtime: bool = True):
        """
        Creates a game on the host and joins with all clients.
        Assumes 2 players for now.
        """
        if len(self.clients) < 2:
            raise RuntimeError("Need at least 2 clients launched.")

        # Port configuration for multi-client communication
        from sc2.portconfig import Portconfig
        # Use unique high ports based on worker_index to avoid conflicts
        # Each worker needs 4 ports (2 server, 2 player)
        base_port = 20000 + (self.worker_index * 10)
        portconfig = Portconfig(
            server_ports=[base_port, base_port + 1], 
            player_ports=[[base_port + 2, base_port + 3]]
        )
        logger.info(f"Worker {self.worker_index} using Portconfig: {portconfig.as_json}")

        # Host creates the game
        logger.info(f"Host is creating the game on map: {game_map.name}...")
        try:
            # We use realtime=True for better engine-level loading sync during debug
            result = await self.controllers[0].create_game(
                game_map,
                players,
                realtime=realtime
            )
            if result.create_game.HasField("error"):
                logger.error(f"Engine Error: {result.create_game.error}")
                
            logger.info("Game created. Waiting 10s for map loading...")
            await asyncio.sleep(10)
        except Exception as e:
            logger.error(f"Exception during create_game: {e}")
            raise e

        # Both join in parallel - MANDATORY for Portconfig handshake stability
        logger.info("Both players are joining the game (Parallel Handshake)...")
        try:
            # We use name and race explicitly to ensure the engine assigns slots correctly
            join_tasks = [
                self.clients[0].join_game(
                    name="HostPlayer",
                    race=players[0].race, 
                    observed_player_id=1,
                    portconfig=portconfig
                ),
                self.clients[1].join_game(
                    name="JoinerPlayer",
                    race=players[1].race, 
                    observed_player_id=2,
                    portconfig=portconfig
                )
            ]
            results = await asyncio.gather(*join_tasks)
            logger.info(f"Join results received: {results}")
            logger.info("Multi-client session established.")
        except Exception as e:
            logger.error(f"Error during parallel join_game: {e}")
            await self.close()
            raise e

    async def close(self):
        """Closes all connections and aggressively kills SC2 processes."""
        for process in self.processes:
             try:
                 await process.__aexit__(None, None, None)
             except:
                 pass
        
        # Aggressive cleanup for Windows to prevent zombie processes
        try:
            from src.utils.sc2_cleanup import kill_switch_sc2
            kill_switch_sc2()
        except:
            pass
        
        # Wait for OS to reclaim ports and resources
        await asyncio.sleep(0.5)
        
        self.processes.clear()
        self.clients.clear()
        self.controllers.clear()
        self.host_client = None
        self.join_client = None

