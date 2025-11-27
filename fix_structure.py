import os
import shutil

def touch_init(path):
    init_path = os.path.join(path, "__init__.py")
    if not os.path.exists(init_path):
        print(f"Creating missing package file: {init_path}")
        with open(init_path, "w") as f:
            f.write("")

def check_file(path, description):
    if not os.path.exists(path):
        print(f"MISSING CRITICAL FILE: {path}")
        return False
    return True

# 1. Create __init__.py files to make imports work
dirs_to_fix = [
    "src",
    "src/agent",
    "src/transformers",
    "src/utils",
    "train",
    "train/envs",
    "train/maps",
]

print("--- Fixing Package Structure ---")
for d in dirs_to_fix:
    if os.path.exists(d):
        touch_init(d)
    else:
        print(f"Warning: Directory '{d}' does not exist.")

# 2. Check hierarchical_agent.py
expected_path = os.path.join("src", "hierarchical_agent.py")
wrong_path = os.path.join("src", "agent", "hierarchical_agent.py")

if os.path.exists(wrong_path) and not os.path.exists(expected_path):
    print(f"\nFound hierarchical_agent.py in '{wrong_path}'.")
    print(f"Moving it to '{expected_path}' to match imports...")
    shutil.move(wrong_path, expected_path)
    print("Move successful.")

if check_file(expected_path, "Hierarchical Agent"):
    print("\nSUCCESS: File structure looks correct.")
    print("You can try running the training command again.")
else:
    print("\nFAILURE: src/hierarchical_agent.py is still missing.")
    print("Please check where you saved the 'Hierarchical Agent System' code.")