import sys
import os
import argparse
from pathlib import Path
from sc2 import maps
from sc2.player import Human, Computer
from sc2.data import Race, Difficulty
from sc2.main import run_game

def play_level_drop_2(map_name="00_64x64_Upper_Right_Terran_vs_Protoss"):
    """
    Launches a game of Drop Level 2 allowing the user to play as Terran.
    """
    print(f"Starting Drop Level 2 on map: {map_name}")
    print("You are playing as TERRAN (Human).")
    print("Goal: Load units, fly to enemy island, unload, and destroy the enemy.")
    
    # 1. Resolve Map Path
    # The maps are in train/maps/level_drop_2/
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    map_path = base_dir / "train" / "maps" / "level_drop_2" / f"{map_name}.SC2Map"
    
    if not map_path.exists():
        print(f"Error: Map file not found at {map_path}")
        # List available maps
        map_dir = base_dir / "train" / "maps" / "level_drop_2"
        if map_dir.exists():
            print("Available maps:")
            for f in map_dir.glob("*.SC2Map"):
                print(f" - {f.stem}")
        return

    # 2. Run Game
    try:
        run_game(
            maps.Map(map_path),
            [
                Human(Race.Terran),
                Computer(Race.Protoss, Difficulty.Easy) # Enemy race might vary by map name, but Protoss/Zerg/Terran work
            ],
            realtime=True,
            save_replay_as=f"replays/manual_{map_name}.SC2Replay"
        )
        print(f"Game finished. Replay saved to replays/manual_{map_name}.SC2Replay")
    except Exception as e:
        print(f"Error running game: {e}")

if __name__ == "__main__":
    # You can change the map name here if you want to try a different variation
    play_level_drop_2()
