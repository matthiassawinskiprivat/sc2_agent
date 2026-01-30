import sys
from pathlib import Path
from sc2 import maps
from sc2.player import Bot, Computer
from sc2.data import Race, Difficulty
from sc2.main import run_game
from src.level_drop_bot import LevelDropBot

# Create a dummy bot that just moves somewhere to generate actions
class RandomMoverBot(LevelDropBot):
    async def on_step(self, iteration):
        if iteration == 0:
            await self.client.debug_show_map()
        
        # Just pick a unit and move it randomly to generate "actions" in the replay
        if self.units:
            u = self.units.random
            # Move to a random spot
            u.move(u.position.random_on_distance(5))

def generate_test_replay():
    print("Generating test replay...")
    map_name = "00_64x64_Upper_Right_Terran_vs_Protoss"
    base_dir = Path("D:/sc2_unit_agent")
    map_path = base_dir / "train" / "maps" / "level_drop_2" / f"{map_name}.SC2Map"

    run_game(
        maps.Map(map_path),
        [
            Bot(Race.Terran, RandomMoverBot(conn=None)),
            Computer(Race.Protoss, Difficulty.Easy)
        ],
        realtime=False,
        save_replay_as="replays/test_replay.SC2Replay"
    )
    print("Replay saved to replays/test_replay.SC2Replay")

if __name__ == "__main__":
    generate_test_replay()
