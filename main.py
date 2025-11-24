"""
StarCraft 2 Unit Control Agent - Main Entry Point
Now with real SC2 integration using python-sc2
"""

import sys
import os
import argparse
import json
import time
import threading
import queue
import numpy as np
from typing import Dict, Any

from src.utils.layer_constants import LayerConstants

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.layer_constants import LayerConstants
from src.agent.unit_control_agent import UnitControlAgent
from src.sc2api.unit_data_extractor import UnitDataExtractor
from src.sc2api.sc2_bot import run_sc2_game, UnitControlBot
from src.transformers.observation_transformer import ObservationTransformer
from ui.ui_manager import UIManager
from ui.debug_ui import create_debug_ui


def setup_project():
    """Setup the project by extracting unit data"""
    print("Setting up StarCraft 2 Unit Control Agent project...")

    # Create data directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Extract unit data
    print("Extracting unit data from StarCraft 2...")
    extractor = UnitDataExtractor()
    unit_data_file = extractor.save_unit_data(os.path.join(data_dir, 'unit_data.json'))

    print(f"Unit data saved to: {unit_data_file}")
    print("Project setup complete!")

    return unit_data_file


def create_default_config(unit_data_file: str = None) -> Dict[str, Any]:
    """Create default configuration"""
    return {
        'cnn_input_size': [128, 128],
        'display_size': [1200, 800],
        'unit_data_file': unit_data_file,
        'fps_limit': 60,
        'enable_ui': True,
        'save_debug_data': False,
        'debug_output_dir': './debug_output',
        'sc2_map': 'AcidPlantLE',  # Default map
        'sc2_difficulty': None,    # None for solo, e.g., 'VeryEasy' for AI opponent
        'game_version': 'latest',
        'enabled_layers': LayerConstants.DEFAULT_UNIT_CONTROL_LAYERS
    }


def save_config(config: Dict[str, Any], filename: str = 'config.json'):
    """Save configuration to file"""
    config_path = os.path.join(os.path.dirname(__file__), filename)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")


def load_config(filename: str = 'config.json') -> Dict[str, Any]:
    """Load configuration from file with fallback to defaults"""
    config_path = os.path.join(os.path.dirname(__file__), filename)
    default_config = create_default_config()

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
            # Merge with defaults to ensure all keys exist
            default_config.update(loaded_config)
            print(f"Configuration loaded from: {config_path}")
            return default_config
    else:
        print(f"Config file {config_path} not found. Using default configuration.")
        return default_config


def demo_mode(agent: UnitControlAgent, ui_manager: UIManager, config: Dict[str, Any]):
    """Run in demo mode with simulated data"""
    print("Running in demo mode with simulated data...")

    def create_simulated_observation():
        obs = np.zeros((6, 200, 200), dtype=np.float32)

        # Allied units
        # Marine at (50, 50)
        x, y = 50, 50
        radius = 10
        damage = 6 / 100.0

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    px, py = x + dx, y + dy
                    if 0 <= px < 200 and 0 <= py < 200:
                        obs[0, py, px] += damage

        # Stalker at (70, 60)
        x, y = 70, 60
        radius = 12
        damage = 13 / 100.0

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    px, py = x + dx, y + dy
                    if 0 <= px < 200 and 0 <= py < 200:
                        obs[0, py, px] += damage
                        obs[1, py, px] += damage

        # Enemy units
        # Zergling at (150, 100)
        x, y = 150, 100
        radius = 1
        damage = 5 / 100.0

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    px, py = x + dx, y + dy
                    if 0 <= px < 200 and 0 <= py < 200:
                        obs[2, py, px] += damage

        # Health and energy layers
        obs[4, 50, 50] = 0.8
        obs[4, 70, 60] = 0.9
        obs[4, 150, 100] = 1.0

        # Clip attack values to 1.0
        obs[:4] = np.clip(obs[:4], 0, 1.0)

        return obs

    frame_count = 0

    # Start the UI manager's real-time mode
    if ui_manager:
        ui_manager.start_real_time_mode()

    try:
        while True:
            # Create simulated observation
            simulated_obs = create_simulated_observation()

            # Add some animation
            if frame_count % 30 < 15:
                # Move marine
                simulated_obs[0, 50, 50] = 0
                simulated_obs[4, 50, 50] = 0
                simulated_obs[0, 51, 51] = 6 / 100.0
                simulated_obs[4, 51, 51] = 0.8

            # Process with agent
            agent.process_transformed_observation(simulated_obs)

            # Update UI if enabled
            if ui_manager:
                # Generate frame
                frame = ui_manager.ui_transformer.update_display(
                    simulated_obs,
                    ui_manager.enabled_layers,
                    fps=ui_manager.current_fps if hasattr(ui_manager, 'current_fps') else 30
                )

                # Put frame in queue for UI
                try:
                    ui_manager.update_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        ui_manager.update_queue.get_nowait()  # Remove old
                        ui_manager.update_queue.put_nowait(frame)  # Add new
                    except:
                        pass

            frame_count += 1
            time.sleep(1 / 30)  # 30 FPS

            if frame_count % 100 == 0:
                print(f"Demo frame: {frame_count}")

    except KeyboardInterrupt:
        print("\nDemo mode stopped by user.")


def real_sc2_mode(agent: UnitControlAgent, ui_manager: UIManager, config: Dict[str, Any]):
    """Run with real StarCraft 2 game"""
    print("Launching StarCraft 2...")
    print(f"Map: {config['sc2_map']}")
    print(f"UI Enabled: {config['enable_ui']}")

    # Create observation transformer
    transformer = ObservationTransformer(
        map_size=tuple(config['cnn_input_size']),
        unit_data_file=config['unit_data_file']
    )

    # UI queue for cross-thread communication
    ui_queue = queue.Queue(maxsize=5)

    # Start UI thread
    ui_thread = None
    if config['enable_ui']:
        ui_manager.start_real_time_mode()
        ui_queue = ui_manager.ui_queue

    # Start SC2 game in separate thread (blocking call)
    sc2_thread = threading.Thread(
        target=run_sc2_game,
        args=(agent, transformer),
        kwargs={
            'map_name': config['sc2_map'],
            'difficulty': config['sc2_difficulty'],
            'ui_queue': ui_queue
        }
    )
    sc2_thread.daemon = True
    sc2_thread.start()

    # Start UI if enabled
    if config['enable_ui']:
        try:
            from ui.debug_ui import create_debug_ui
            app, window = create_debug_ui(agent)

            # Override agent's get_latest_ui_frame to use our queue
            def get_frame():
                try:
                    return ui_manager.ui_queue.get_nowait()
                except:
                    return None

            agent.get_latest_ui_frame = get_frame
            print("Debug: About to start UI event loop...")
            sys.exit(app.exec_())
        except ImportError as e:
            print(f"Could not load UI: {e}")
            print("Running in headless mode...")
            sc2_thread.join()
    else:
        print("Running in headless mode. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="StarCraft 2 Unit Control Agent")
    parser.add_argument('--setup', action='store_true', help='Setup project (extract unit data)')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with simulated data')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file path')
    parser.add_argument('--no-ui', action='store_true', help='Disable UI (headless mode)')
    parser.add_argument('--save-debug', action='store_true', help='Enable debug data saving')
    parser.add_argument('--map', type=str, help='SC2 map name (default: AcidPlantLE)')
    parser.add_argument('--difficulty', type=str, help='AI difficulty (e.g., VeryEasy, Easy, Medium)')

    args = parser.parse_args()

    # Setup project if requested
    if args.setup:
        unit_data_file = setup_project()
        config = create_default_config(unit_data_file)
        save_config(config)
        return

    # Load configuration
    config = load_config(args.config)

    # Override with command line args
    if args.no_ui:
        config['enable_ui'] = False
    if args.save_debug:
        config['save_debug_data'] = True
    if args.map:
        config['sc2_map'] = args.map
    if args.difficulty:
        config['sc2_difficulty'] = args.difficulty

    # Create core agent
    agent = UnitControlAgent(
        cnn_input_size=tuple(config['cnn_input_size'])
    )

    # Create UI manager if enabled
    ui_manager = None
    if config['enable_ui']:
        ui_manager = UIManager(agent, display_size=tuple(config['display_size']))

    # Run modes
    if config['enable_ui']:
        # Create UI first
        try:
            from ui.debug_ui import create_debug_ui

            # Create UI (this shows the window)
            app, window = create_debug_ui(agent)

            # Use the UI manager from the window
            ui_manager = window.ui_manager

            # Ensure agent has ui_queue
            if not hasattr(agent, 'ui_queue'):
                agent.ui_queue = ui_manager.update_queue

            # Start demo loop in background thread
            demo_thread = threading.Thread(
                target=demo_mode,
                args=(agent, ui_manager, config)
            )
            demo_thread.daemon = True
            demo_thread.start()

            # Run Qt event loop - this blocks until you close the window
            print("Demo UI started. Close window to exit.")
            sys.exit(app.exec_())

        except ImportError as e:
            print(f"Could not load UI: {e}")
            print("Running in headless demo mode...")
            demo_mode(agent, None, config)
    else:
        # Headless demo (no UI)
        demo_mode(agent, None, config)

    return


if __name__ == "__main__":
    main()