
import asyncio
import os
from pathlib import Path
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.data import Race, Difficulty
from sc2.sc2process import SC2Process
from sc2.client import Client
from sc2.game_state import GameState
from sc2.maps import Map

# Mocking ProtoWrapper from level_0_env.py
class ProtoWrapper:
    def __init__(self, proto):
        self._proto = proto

    def __getattr__(self, name):
        try:
            return getattr(self._proto, name)
        except AttributeError:
            if hasattr(self._proto, "game_info"):
                return getattr(self._proto.game_info, name)
            raise AttributeError(f"'{type(self._proto).__name__}' has no attribute '{name}'")

async def reproduce():
    print("Starting reproduction...")
    process = SC2Process()
    controller = await process.__aenter__()
    ws = getattr(process, "_ws", getattr(process, "ws", None))
    client = Client(ws)

    map_path = Path("train/maps/level_0/00_32x32_No_Middle.SC2Map").resolve()
    game_map = Map(map_path)

    print("Creating game...")
    await controller.create_game(
        game_map,
        [Bot(Race.Zerg, None), Computer(Race.Zerg, Difficulty.Easy)],
        realtime=False,
    )
    
    print("Joining game...")
    await client.join_game(race=Race.Zerg)
    
    print("Getting observation...")
    initial_observation = await client.observation()
    
    print("Attempting to initialize GameState...")
    try:
        state = GameState(initial_observation)
        print("GameState initialized successfully (Unexpected!)")
    except AttributeError as e:
        print(f"Caught expected error: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"Caught unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await process.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(reproduce())
