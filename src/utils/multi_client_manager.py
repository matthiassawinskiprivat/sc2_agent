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
            logger.info(f"Launching SC2 Process {i+1}/{count}...")
            process = self.sc2_process_cls(
                fullscreen=fullscreen,
                render=visualize
            )
            controller = await process.__aenter__()
            
            # Verify controller is responsive
            ping = await controller.ping()
            logger.info(f"Process {i+1} responsive. Ping: {ping}")
            
            # Robustly extract websocket connection
            ws = getattr(process, "_ws", getattr(process, "ws", None))
            if ws is None:
                ws = getattr(controller, "_ws", getattr(controller, "ws", None))
            
            client = self.client_cls(ws)
            
            self.processes.append(process)
            self.controllers.append(controller)
            self.clients.append(client)
            
            # Give each process significant time to open its listening port
            await asyncio.sleep(5.0)

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
        # Each worker needs:
        # 2 Server Ports (one for each SC2 process to listen for setup)
        # 2 Ports per Player for game traffic (GamePort, BasePort) * 2 Players = 4
        # Total 6 ports.
        base_port = 30000 + (self.worker_index * 100) # Higher base port
        
        # Server Ports: [P1_Server, P2_Server]
        server_ports = [base_port + 0, base_port + 1]
        
        # Player Ports: [[P1_Game, P1_Base], [P2_Game, P2_Base]]
        player_ports = [
            [base_port + 2, base_port + 3], # Player 1
            [base_port + 4, base_port + 5]  # Player 2
        ]
        
        portconfig = Portconfig(
            server_ports=server_ports, 
            player_ports=player_ports
        )
        logger.info(f"Worker {self.worker_index} using Portconfig: {portconfig.as_json}")

        # Host creates the game
        logger.info(f"Host is creating the game on map: {game_map.name}...")
        try:
            # We use realtime=False for maximum compatibility during handshake
            # Add a timeout to prevent infinite hang
            create_task = self.controllers[0].create_game(
                game_map,
                players,
                realtime=realtime
            )
            result = await asyncio.wait_for(create_task, timeout=60.0)
            
            logger.info(f"create_game Request sent successfully. Result: {result.create_game.HasField('error')}")
            if result.create_game.HasField("error"):
                logger.error(f"Engine Error: {result.create_game.error}")
                
            logger.info("Waiting 15s for map loading and engine stability...")
            await asyncio.sleep(15)
        except asyncio.TimeoutError:
            logger.error("Timeout during create_game!")
            raise
        except Exception as e:
            logger.error(f"Exception during create_game: {e}")
            raise e

        # Both join in parallel - MANDATORY for Portconfig handshake stability
        logger.info("Both players are joining the game (Parallel Handshake)...")
        try:
            # We use name and race explicitly to ensure the engine assigns slots correctly
            async def join_with_log(client, name, player_id, p_config):
                logger.info(f"Client {player_id} ({name}) initiating join_game...")
                res = await client.join_game(
                    name=name,
                    race=players[player_id-1].race, 
                    observed_player_id=player_id,
                    portconfig=p_config
                )
                logger.info(f"Client {player_id} ({name}) join_game SUCCESS.")
                return res

            join_tasks = [
                join_with_log(self.clients[0], "HostPlayer", 1, portconfig),
                join_with_log(self.clients[1], "JoinerPlayer", 2, portconfig)
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

