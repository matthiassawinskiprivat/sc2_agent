import os
import shutil
from pathlib import Path

# Where files should be (from project root)
FILE_LOCATIONS = {
    # Agent files
    "hierarchical_agent.py": "src/agent/",
    "unit_control_agent.py": "src/agent/",
    "unit_data_extractor.py": "src/agent/",

    # Transformer files
    "observation_transformer.py": "src/transformers/",
    "shared_observation_transformer.py": "src/transformers/",
    "ui_transformer.py": "src/transformers/",

    # UI files
    "debug_ui.py": "src/ui/",
    "ui_manager.py": "src/ui/",

    # Utility files
    "layer_constants.py": "src/utils/",

    # Bot files
    "sc2_bot.py": "src/",
}


def organize_files():
    project_root = Path(__file__).parent

    # Find all Python files in project root and subdirs
    python_files = list(project_root.glob("**/*.py"))

    for file_path in python_files:
        file_name = file_path.name

        if file_name in FILE_LOCATIONS:
            target_dir = project_root / FILE_LOCATIONS[file_name]
            target_path = target_dir / file_name

            # Create target directory if needed
            target_dir.mkdir(parents=True, exist_ok=True)

            # Move file if it's not already in the right place
            if file_path != target_path:
                try:
                    shutil.move(str(file_path), str(target_path))
                    print(f"Moved {file_name} → {FILE_LOCATIONS[file_name]}")
                except Exception as e:
                    print(f"Error moving {file_name}: {e}")


if __name__ == "__main__":
    print("Organizing project files...")
    organize_files()
    print("Done!")