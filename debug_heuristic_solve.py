import numpy as np
import time
import cv2
import os
from train.envs.level_0_env import Level0Env
from src.utils.layer_constants import LayerConstants


def heuristic_solve():
    print("Initializing Environment...")
    env = Level0Env(visualize=True)
    obs, _ = env.reset()

    # Identify the Beacon Layer (Enemy Unit Density)
    # Check LayerConstants.LAYERS for the index.
    # ENEMY_UNIT_DENSITY is usually index 7.
    BEACON_LAYER_IDX = LayerConstants.LAYERS[LayerConstants.ENEMY_UNIT_DENSITY]
    PLAYER_LAYER_IDX = LayerConstants.LAYERS[LayerConstants.ALLIED_UNIT_DENSITY]

    print(f"Beacon Layer Index: {BEACON_LAYER_IDX}")
    print(f"Player Layer Index: {PLAYER_LAYER_IDX}")

    for step in range(500):
        # 1. Get the Enemy Layer
        beacon_map = obs[BEACON_LAYER_IDX]

        # 2. Find the brightest pixel (The Beacon)
        # argmax gives the flattened index of the max value
        max_val = np.max(beacon_map)

        if max_val == 0:
            print(f"[Step {step}] No Beacon visible on layer {BEACON_LAYER_IDX}!")
            # Save debug image to verify
            debug_layers(obs, step)
            # Random move to try and find it?
            action = env.action_space.sample()
        else:
            # np.argmax returns flat index for the array flattened (Row-Major usually)
            # The environment expects a flat index 0..16383

            # NOTE: numpy arrays are [y, x]. Flattening aligns with (y * w + x).
            # This should match your agent's flat action space logic.
            flat_index = np.argmax(beacon_map)

            # Decode for logging
            y_img, x_img = np.unravel_index(flat_index, beacon_map.shape)

            print(f"[Step {step}] Beacon found at Pixel (x={x_img}, y={y_img}). Sending Action: {flat_index}")
            action = flat_index

        # 3. Step the Environment
        obs, reward, done, truncated, info = env.step(action)

        if reward > 50:
            print(f"!!! SUCCESS !!! Reached Beacon at step {step}")
            break

        if done:
            print("Episode finished.")
            obs, _ = env.reset()

        time.sleep(0.1)  # Slow down to watch

    env.close()


def debug_layers(obs, step):
    """Save layers to disk if beacon is missing"""
    os.makedirs("debug_failure", exist_ok=True)

    # Save Enemy Layer
    beacon_layer = obs[LayerConstants.LAYERS[LayerConstants.ENEMY_UNIT_DENSITY]]
    norm = cv2.normalize(beacon_layer, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(f"debug_failure/step_{step}_enemy.png", norm)

    # Save Allied Layer
    player_layer = obs[LayerConstants.LAYERS[LayerConstants.ALLIED_UNIT_DENSITY]]
    norm_p = cv2.normalize(player_layer, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(f"debug_failure/step_{step}_player.png", norm_p)

    print(f"Saved debug images to debug_failure/")


if __name__ == "__main__":
    heuristic_solve()