
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
    def __init__(self, sc2_process_cls: Type[SC2Process] = SC2Process, client_cls: Type[Client] = Client, worker_index: int = 0, num_workers: int = 1):
        self.sc2_process_cls = sc2_process_cls
        self.client_cls = client_cls
        self.worker_index = worker_index
        self.num_workers = max(1, int(num_workers))
        self.process: Optional[SC2Process] = None
        self.clients: List[Client] = []
        self.controller = None

    def _startup_stagger_sec(self) -> float:
        """Delay between parallel worker SC2 launches (matches single_client_env timeout budget)."""
        return 8.0 if self.num_workers >= 5 else 6.0

    @property
    def client(self) -> Optional[Client]:
        return self.clients[0] if self.clients else None

    async def launch_process(self, client_count: int = 1, visualize: bool = False, fullscreen: bool = False):
        """Launches a single SC2 process and connects clients."""
        
        # Deterministic port allocation to prevent race conditions in parallel environments
        # Base port 12000 leaves plenty of room above standard 5000+ ports
        port = 12000 + (self.worker_index * 1000)

        # Stagger SC2 startup across parallel workers to avoid DirectX/GPU init race.
        # Without this, 5 instances launching simultaneously share GPU init paths and
        # one or more crash with 0xC0000005 (access violation).
        stagger_sec = self._startup_stagger_sec()
        startup_delay = self.worker_index * stagger_sec
        if startup_delay > 0:
            logger.info(
                f"Worker {self.worker_index} waiting {startup_delay:.0f}s before SC2 launch "
                f"(staggered startup, {stagger_sec:.0f}s × index)..."
            )
            await asyncio.sleep(startup_delay)

        logger.info(f"Worker {self.worker_index} launching SC2 on port {port}")
        
        self.process = self.sc2_process_cls(
            fullscreen=fullscreen,
            render=visualize,
            port=port,
            host="127.0.0.1"
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
            
            # SC2 needs a moment to finish lobby setup before join_game; too short a pause
            # yields WSMsgType.CLOSE (257) on join when multiple instances start in parallel.
            await asyncio.sleep(3.0)
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
             # Capture PID before exiting for targeted cleanup
             # SC2Process typically has a ._process which is a subprocess.Popen
             pid = getattr(self.process, "_process", None)
             if pid and hasattr(pid, "pid"):
                 pid = pid.pid
             elif hasattr(self.process, "pid"):
                 pid = self.process.pid
             else:
                 pid = None

             try:
                 await self.process.__aexit__(None, None, None)
             except:
                 pass
             
             # Targeted cleanup if PID is known
             if pid:
                 import psutil
                 try:
                     proc = psutil.Process(pid)
                     if proc.is_running():
                         logger.info(f"Targeted kill of SC2 process PID: {pid}")
                         proc.kill()
                 except (psutil.NoSuchProcess, psutil.AccessDenied):
                     pass
        
        await asyncio.sleep(0.5)
        
        self.process = None
        self.clients.clear()
        self.controller = None
