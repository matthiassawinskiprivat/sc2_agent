import os
import time
import numpy as np
from sb3_contrib import MaskablePPO
from train.envs.level_0_env import Level0Env

MODEL_PATH = "models/level_0/final_model"


def watch():
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print(f"Model not found at {MODEL_PATH}")
        return

    # Load Model
    model = MaskablePPO.load(MODEL_PATH)

    # Start Environment (Visual Mode)
    env = Level0Env(visualize=True)

    # Pick a specific map to verify
    env.map_name = "Zerg_Lower_Left"

    obs, _ = env.reset()
    done = False

    print("\n--- STARTING WATCH MODE ---")
    try:
        while not done:
            # 1. Get Mask
            masks = env.action_masks()

            # 2. Predict
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)

            # 3. DEBUG: Print Action
            # Convert the flat index (0-16384) to Matrix coords (0-128)
            y_cnn, x_cnn = divmod(int(action), 128)

            # Convert to World Coords using the agent's internal logic
            wx, wy = env.agent.convert_spatial_action_to_world(int(action))

            print(f"AI Output Index: {action} | CNN Grid: ({x_cnn}, {y_cnn}) | SC2 World: ({wx:.1f}, {wy:.1f})")

            # 4. Step
            obs, reward, done, truncated, info = env.step(action)

            if info.get('is_success'):
                print(">>> SUCCESS! Goal Reached.")


    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    watch()