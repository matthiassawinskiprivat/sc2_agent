
from sc2.client import Client
from sc2.sc2process import SC2Process
import asyncio

async def inspect_proto():
    async with SC2Process() as controller:
        client = Client(controller._ws)
        game_info = await client.get_game_info()
        print("GameInfo proto fields:")
        print(dir(game_info._proto))
        if hasattr(game_info._proto, "game_info"):
            print("Has game_info field")
        else:
            print("Does NOT have game_info field")

asyncio.run(inspect_proto())
