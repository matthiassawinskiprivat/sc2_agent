import time
import numpy as np
from train.envs.level_0_env import Level0Env
import traceback


def test_coords():
    print("Initializing Environment...")
    try:
        env = Level0Env(visualize=True)
        env.reset()
    except Exception as e:
        print(f"AN ERROR OCCURRED DURING ENV INITIALIZATION: {e}")
        print(traceback.format_exc())
        raise

    # 1. Define Test Clicks (Image Coordinates)
    # Recall: Image (0,0) is Top-Left. (127,127) is Bottom-Right.
    test_actions = [
        (64, 64, "CENTER"),  # Should move to middle
        (20, 64, "TOP"),  # Should move North
        (100, 64, "BOTTOM"),  # Should move South
        (64, 20, "LEFT"),  # Should move West
        (64, 100, "RIGHT"),  # Should move East
    ]

    # Get access to the internal agent logic for conversion
    agent = env._bot.agent_system.unit_agents["agent_0"]

    for y_img, x_img, label in test_actions:
        print(f"\n--- TESTING CLICK: {label} (Img: {x_img}, {y_img}) ---")

        # Convert manually to verify math
        flat_index = y_img * 128 + x_img

        # Run the actual Env step
        # The Env will use agent.convert_spatial_action_to_world internally
        env.step(flat_index)

        # Wait to see movement
        time.sleep(2)

        # Get Unit Position
        unit = env._bot.zergling
        if unit:
            print(f"Unit Position: ({unit.position.x:.1f}, {unit.position.y:.1f})")
        else:
            print("Unit lost!")

    print("\nTest Complete. Closing...")
    env.close()


if __name__ == "__main__":
    test_coords()