import cv2
import numpy as np
import os
from train.envs.level_0_env import Level0Env
from src.utils.layer_constants import LayerConstants


def debug_view():
    print("Initializing Debug Environment...")
    env = Level0Env(visualize=False)  # Headless for snapshot
    obs, _ = env.reset()

    # 1. Get Unit Layer (Layer 0) - Should show Zergling
    # 2. Get Enemy Layer (Layer 7) - Should show Beacon
    # 3. Get Mask - Should be white where walkable

    idx_units = 0  # ALLIED_UNIT_DENSITY
    idx_enemy = 7  # ENEMY_UNIT_DENSITY

    layer_units = obs[idx_units]
    layer_enemy = obs[idx_enemy]

    # Get Mask from Agent logic
    mask_flat = env.agent.get_action_mask()
    mask_2d = mask_flat.reshape((128, 128))

    # --- VISUALIZATION ---
    # Create a combined RGB image
    # Blue = Units (Zergling)
    # Red = Enemy (Beacon)
    # Green = Walkable Terrain (Mask)

    h, w = layer_units.shape
    debug_img = np.zeros((h, w, 3), dtype=np.float32)

    # Normalize Mask to 0.0-0.5 for background visibility
    debug_img[:, :, 1] = mask_2d.astype(np.float32) * 0.3

    # Add Units (Bright Blue)
    debug_img[:, :, 0] += layer_units  # B

    # Add Beacon (Bright Red)
    debug_img[:, :, 2] += layer_enemy  # R

    # Scale up for humans (512x512)
    debug_img = cv2.resize(debug_img, (512, 512), interpolation=cv2.INTER_NEAREST)
    debug_img = np.clip(debug_img * 255, 0, 255).astype(np.uint8)

    # Draw Grid Center for reference
    cv2.line(debug_img, (256, 0), (256, 512), (100, 100, 100), 1)
    cv2.line(debug_img, (0, 256), (512, 256), (100, 100, 100), 1)

    # Save
    os.makedirs("debug_output", exist_ok=True)
    path = "debug_output/alignment_check.png"
    cv2.imwrite(path, debug_img)

    print(f"\n>>> DEBUG IMAGE SAVED TO: {path} <<<")
    print("Open this image. You should see:")
    print("1. A Green Box (The valid map area).")
    print("2. A Blue Dot (Your Zergling).")
    print("3. A Red Dot (The Beacon).")
    print("\nIF THE DOTS ARE OUTSIDE THE GREEN BOX, YOUR COORDINATES ARE BROKEN.")

    env.close()


if __name__ == "__main__":
    debug_view()