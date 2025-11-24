import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
print(f"Project root: {project_root}")
print("=" * 60)

# Expected structure
expected_files = [
    "src/__init__.py",
    "src/agent/__init__.py",
    "src/transformers/__init__.py",
    "src/ui/__init__.py",
    "src/utils/__init__.py",
    "src/agent/hierarchical_agent.py",
    "src/agent/unit_control_agent.py",
    "src/agent/unit_data_extractor.py",
    "src/transformers/observation_transformer.py",
    "src/transformers/shared_observation_transformer.py",
    "src/transformers/ui_transformer.py",
    "src/ui/debug_ui.py",
    "src/ui/ui_manager.py",
    "src/sc2_bot.py",
    "src/utils/layer_constants.py",
    "data/unit_data.json"
]

print("Checking files and directories:\n")

issues = []
for file_path in expected_files:
    full_path = os.path.join(project_root, file_path)
    if os.path.exists(full_path):
        print(f"✓ {file_path}")
    else:
        print(f"✗ MISSING: {file_path}")
        issues.append(file_path)

print("\n" + "=" * 60)
if issues:
    print(f"Found {len(issues)} missing files!")
    print("Please move files to correct locations.")
else:
    print("All files present! Structure is correct.")
print("=" * 60)

# Print actual structure for comparison
print("\nActual directory structure:")
for root, dirs, files in os.walk(project_root):
    level = root.replace(project_root, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")