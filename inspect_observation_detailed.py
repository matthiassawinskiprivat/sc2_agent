
import asyncio
import os
from pathlib import Path
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.data import Race, Difficulty
from sc2.sc2process import SC2Process
from sc2.client import Client
from sc2.maps import Map

async def inspect():
    print("Starting inspection...")
    process = SC2Process()
    controller = await process.__aenter__()
    ws = getattr(process, "_ws", getattr(process, "ws", None))
    client = Client(ws)

    map_path = Path("train/maps/level_0/00_32x32_No_Middle.SC2Map").resolve()
    game_map = Map(map_path)

    await controller.create_game(
        game_map,
        [Bot(Race.Zerg, None), Computer(Race.Zerg, Difficulty.Easy)],
        realtime=False,
    )
    await client.join_game(race=Race.Zerg)
    
    obs = await client.observation()
    print(f"Observation type: {type(obs)}")
    print(f"Observation attributes: {dir(obs)}")
    
    if hasattr(obs, "observation"):
        print(f"obs.observation type: {type(obs.observation)}")
        print(f"obs.observation attributes: {dir(obs.observation)}")
        
        # In some versions of burnysc2, it might be obs.observation.raw_data
        # but here the error says GameState(initial_observation) fails.
        # GameState expects a ResponseObservation (which has 'observation' field)
        # or it expects the 'observation' field itself.
    
    await process.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(inspect())
