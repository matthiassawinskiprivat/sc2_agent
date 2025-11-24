"""
Setup script for StarCraft 2 Unit Control Agent
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required.")
        sys.exit(1)
    print(f"Python version: {sys.version}")


def install_dependencies():
    """Install required packages"""
    print("Installing dependencies...")
    
    # Install PyTorch first (if needed)
    try:
        import torch
        print("PyTorch already installed")
    except ImportError:
        print("Installing PyTorch...")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision"], check=True)
    
    # Install other dependencies
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_file):
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], check=True)
    else:
        print("requirements.txt not found, installing basic dependencies...")
        basic_deps = [
            "numpy", "opencv-python", "matplotlib", "PyQt5", 
            "pysc2", "sc2", "s2clientprotocol", "protobuf",
            "PyYAML", "websockets", "aiohttp"
        ]
        subprocess.run([sys.executable, "-m", "pip", "install"] + basic_deps, check=True)
    
    print("Dependencies installed successfully!")


def check_starcraft2_installation():
    """Check if StarCraft 2 is installed"""
    sc2_paths = []
    
    # Windows paths
    if sys.platform == "win32":
        sc2_paths.extend([
            "C:\\Program Files (x86)\\StarCraft II",
            "C:\\Program Files\\StarCraft II",
            os.path.expanduser("~\\Documents\\StarCraft II")
        ])
    
    # macOS paths
    elif sys.platform == "darwin":
        sc2_paths.extend([
            "/Applications/StarCraft II",
            os.path.expanduser("~/Library/Application Support/Blizzard/StarCraft II")
        ])
    
    # Linux paths (using Wine or similar)
    else:
        sc2_paths.extend([
            os.path.expanduser("~/.wine/drive_c/Program Files (x86)/StarCraft II"),
            os.path.expanduser("~/.wine/drive_c/Program Files/StarCraft II")
        ])
    
    for path in sc2_paths:
        if os.path.exists(path):
            print(f"StarCraft 2 found at: {path}")
            return path
    
    print("Warning: StarCraft 2 installation not found automatically.")
    print("Please ensure StarCraft 2 is installed and maps are available.")
    return None


def setup_project_structure():
    """Create necessary directories and __init__.py files for proper package structure"""

    # Project directories
    directories = [
        "data",
        "debug_output",
        "logs",
        "models",
        "replays",
        # Add src subdirectories
        "src",
        "src/agent",
        "src/transformers",
        "src/ui",
        "src/utils"
    ]

    # Create directories
    for directory in directories:
        path = os.path.join(os.path.dirname(__file__), directory)
        os.makedirs(path, exist_ok=True)
        print(f"✓ Directory: {path}")

    # Create __init__.py files for all packages
    init_files = [
        "src/__init__.py",
        "src/agent/__init__.py",
        "src/transformers/__init__.py",
        "src/ui/__init__.py",
        "src/utils/__init__.py"
    ]

    for init_file in init_files:
        full_path = os.path.join(os.path.dirname(__file__), init_file)
        # Only create if it doesn't exist (don't overwrite)
        if not os.path.exists(full_path):
            with open(full_path, 'w') as f:
                f.write('"""Auto-generated package file for SC2 Unit Control Agent"""\n')
            print(f"✓ Created: {init_file}")
        else:
            print(f"✓ Already exists: {init_file}")


def create_default_config(sc2_path: str = None):
    """Create default configuration file"""
    config = {
        "map_size": [200, 200],
        "display_size": [1200, 800],
        "unit_data_file": "data/unit_data.json",
        "fps_limit": 60,
        "enable_ui": True,
        "save_debug_data": False,
        "debug_output_dir": "debug_output",
        "sc2_path": sc2_path,
        "game_version": "latest"
    }
    
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Default configuration created: {config_path}")
    return config_path


def extract_unit_data():
    """Extract StarCraft 2 unit data"""
    print("Extracting unit data...")
    
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from src.sc2api.unit_data_extractor import UnitDataExtractor
        
        extractor = UnitDataExtractor()
        data_file = extractor.save_unit_data("data/unit_data.json")
        print(f"Unit data extracted to: {data_file}")
        
    except Exception as e:
        print(f"Error extracting unit data: {e}")
        print("This is normal if StarCraft 2 is not installed or pysc2 is not available.")
        print("The agent will use default unit data.")


def create_run_scripts():
    """Create convenient run scripts"""
    
    # Windows batch file
    if sys.platform == "win32":
        batch_content = """@echo off
echo Starting StarCraft 2 Unit Control Agent Demo...
python main.py --demo
pause
"""
        with open("run_demo.bat", "w") as f:
            f.write(batch_content)
        print("Created run_demo.bat")
        
        setup_content = """@echo off
echo Setting up StarCraft 2 Unit Control Agent...
python setup.py
pause
"""
        with open("setup.bat", "w") as f:
            f.write(setup_content)
        print("Created setup.bat")
    
    # Unix shell script
    else:
        shell_content = """#!/bin/bash
echo "Starting StarCraft 2 Unit Control Agent Demo..."
python3 main.py --demo
"""
        with open("run_demo.sh", "w") as f:
            f.write(shell_content)
        os.chmod("run_demo.sh", 0o755)
        print("Created run_demo.sh")
        
        setup_shell_content = """#!/bin/bash
echo "Setting up StarCraft 2 Unit Control Agent..."
python3 setup.py
"""
        with open("setup.sh", "w") as f:
            f.write(setup_shell_content)
        os.chmod("setup.sh", 0o755)
        print("Created setup.sh")


def main():
    """Main setup function"""
    print("=" * 60)
    print("StarCraft 2 Unit Control Agent - Setup")
    print("=" * 60)

    # Check Python version
    check_python_version()

    # Setup project structure FIRST (before any imports)
    print("\nSetting up project structure...")
    setup_project_structure()

    # Install dependencies
    try:
        install_dependencies()
    except Exception as e:
        print(f"Error installing dependencies: {e}")

    # Check for StarCraft 2
    sc2_path = check_starcraft2_installation()

    # Create configuration
    create_default_config(sc2_path)

    # Extract unit data
    extract_unit_data()

    # Create run scripts
    create_run_scripts()

    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print("\nYour project structure is ready.")
    print("Run: python run_game_with_ui.py")


if __name__ == "__main__":
    main()