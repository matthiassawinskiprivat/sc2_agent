#!/usr/bin/env python3
"""
StarCraft 2 Unit Control Agent - Example Usage
Demonstrates how to integrate the agent into your SC2 bot
"""

import sys
import os
import numpy as np
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.unit_control_agent import UnitControlAgent
from src.transformers.observation_transformer import ObservationTransformer


class ExampleSC2Bot:
    """Example SC2 bot that uses the unit control agent"""
    
    def __init__(self):
        """Initialize the bot with unit control agent"""
        # Create agent with CNN input size
        self.agent = UnitControlAgent(cnn_input_size=(128, 128))
        self.step_count = 0
        
    def on_game_start(self):
        """Called when the game starts"""
        print("Game started! Unit control agent ready.")
        # print(f"Agent config: {self.agent.get_config()}")
        
        # Show available layers
        from src.utils.layer_constants import LayerConstants
        print(f"Available layers: {len(LayerConstants.LAYERS)} total")
        print(f"Default layers: {LayerConstants.DEFAULT_UNIT_CONTROL_LAYERS}")
    
    def on_game_step(self, observation: Dict[str, Any]):
        """Process each game step"""
        self.step_count += 1
        
        # Process observation with agent
        transformed_obs = self.agent.process_observation(observation, player_id=1)
        
        if transformed_obs is not None:
            # Get observation statistics
            if self.step_count % 100 == 0:  # Log every 100 steps
                stats = self.agent.get_observation_statistics()
                print(f"Step {self.step_count}: FPS={stats.get('fps', 0)}, "
                      f"Shape={stats.get('shape', 'N/A')}")
            
            # Here you would typically:
            # 1. Feed transformed_obs to your RL model
            # 2. Get action predictions
            # 3. Convert actions back to SC2 commands
            # 4. Send commands to the game
            
            # For now, just demonstrate the data format
            if self.step_count == 1:
                self._analyze_first_observation(transformed_obs)
    
    def on_game_end(self):
        """Called when the game ends"""
        print(f"Game ended after {self.step_count} steps")
        
        # Stop real-time mode
        if self.agent.is_running:
            self.agent.stop_real_time_mode()
        
        # Save final debug data
        self.agent.save_debug_data("final_game_state")
    
    def _analyze_first_observation(self, obs: np.ndarray):
        """Analyze the first observation to demonstrate data format"""
        print("\n=== First Observation Analysis ===")
        print(f"Observation shape: {obs.shape}")
        print(f"Observation dtype: {obs.dtype}")
        
        # Get layer names from transformer
        layer_names = self.agent.obs_transformer.get_enabled_layer_names()
        
        for i, name in enumerate(layer_names):
            layer_data = obs[i]
            min_val = np.min(layer_data)
            max_val = np.max(layer_data)
            mean_val = np.mean(layer_data)
            non_zero = np.count_nonzero(layer_data)
            
            print(f"\nLayer {i}: {name}")
            print(f"  Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}")
            print(f"  Non-zero pixels: {non_zero}")
            
            if non_zero > 0:
                # Find positions with highest values
                max_positions = np.where(layer_data == max_val)
                if len(max_positions[0]) > 0:
                    y, x = max_positions[0][0], max_positions[1][0]
                    print(f"  Max value position: ({x}, {y})")
    
    def select_layers_for_training(self, layer_names: List[str]):
        """Select specific layers for training"""
        selected = self.agent.select_layers(layer_names)
        if selected is not None:
            print(f"Selected {len(layer_names)} layers for training: {layer_names}")
            print(f"Selected data shape: {selected.shape}")
        return selected
    
    def demonstrate_layer_selection(self):
        """Demonstrate different layer selections"""
        from src.utils.layer_constants import LayerConstants
        
        print("\n=== Layer Selection Examples ===")
        
        # Select only attack layers
        attack_layers = [
            LayerConstants.ALLIED_GROUND_ATTACK,
            LayerConstants.ALLIED_AIR_ATTACK,
            LayerConstants.ENEMY_GROUND_ATTACK,
            LayerConstants.ENEMY_AIR_ATTACK
        ]
        attack_data = self.select_layers_for_training(attack_layers)
        
        # Select only state layers
        state_layers = [
            LayerConstants.HEALTH_PERCENTAGE,
            LayerConstants.ENERGY_PERCENTAGE
        ]
        state_data = self.select_layers_for_training(state_layers)
        
        # Select custom combination
        custom_layers = [
            LayerConstants.ALLIED_GROUND_ATTACK,
            LayerConstants.ENEMY_GROUND_ATTACK,
            LayerConstants.HEALTH_PERCENTAGE
        ]
        custom_data = self.select_layers_for_training(custom_layers)
        
        return attack_data, state_data, custom_data


def create_mock_game_loop():
    """Create a mock game loop for demonstration"""
    
    def create_mock_observation(step):
        """Create mock observation that changes over time"""
        class MockUnit:
            def __init__(self, unit_type, x, y, owner, health, health_max, shield=0, energy=0):
                self.unit_type = unit_type
                self.x = x
                self.y = y
                self.owner = owner
                self.health = health
                self.health_max = health_max
                self.shield = shield
                self.shield_max = shield
                self.energy = energy
                self.energy_max = energy
        
        # Create units that move and change health over time
        units = []
        
        # Allied marine that moves right
        marine_x = 50 + (step % 100) * 0.5
        marine_health = max(20, 45 - (step // 50))
        units.append(MockUnit(48, marine_x, 50, 1, marine_health, 45))
        
        # Allied stalker that moves down
        stalker_y = 60 + (step % 80) * 0.3
        units.append(MockUnit(74, 70, stalker_y, 1, 80, 80, 50))
        
        # Enemy zergling that moves left
        zergling_x = 150 - (step % 120) * 0.4
        units.append(MockUnit(105, zergling_x, 100, 2, 35, 35))
        
        # Enemy hydralisk with spell caster energy
        energy = min(200, (step % 150) * 2)
        units.append(MockUnit(107, 160, 120, 2, 80, 80, energy=energy))
        
        return {'feature_units': units}
    
    return create_mock_observation


def demonstrate_agent_features():
    """Demonstrate various agent features"""
    
    print("=" * 60)
    print("StarCraft 2 Unit Control Agent - Feature Demonstration")
    print("=" * 60)
    
    # Create bot
    bot = ExampleSC2Bot()
    
    # Simulate game start
    bot.on_game_start()
    
    # Create mock game loop
    game_loop = create_mock_game_loop()
    
    print("\nStarting simulated game loop...")
    print("(Press Ctrl+C to stop)")
    
    try:
        for step in range(100):  # Run for 100 steps
            # Create mock observation
            observation = game_loop(step)
            
            # Process with bot
            bot.on_game_step(observation)
            
            # Demonstrate layer selection
            if step == 25:
                print("\n--- Demonstrating layer selection ---")
                bot.demonstrate_layer_selection()
            
            if step == 50:
                print("\n--- Saving debug data ---")
                bot.agent.save_debug_data(f"step_{step}")
            
            # Small delay to simulate real-time
            # time.sleep(0.05)
            
            # Print progress every 25 steps
            if step % 25 == 0:
                print(f"Progress: {step}/100 steps")
    
    except KeyboardInterrupt:
        print("\n\nGame loop interrupted by user")
    
    except Exception as e:
        print(f"\n\nError in game loop: {e}")
    
    finally:
        # Simulate game end
        bot.on_game_end()
        
        print("\n" + "=" * 60)
        print("Feature demonstration completed!")
        print("=" * 60)
        
        # Show final statistics
        final_stats = bot.agent.get_observation_statistics()
        print(f"Total steps processed: {bot.step_count}")
        print(f"Processing time: {final_stats.get('processing_time_ms', 'N/A')}ms avg")
        print(f"Final observation shape: {final_stats.get('shape', 'N/A')}")


def show_integration_guide():
    """Show how to integrate with actual SC2 bot"""
    
    print("\n" + "=" * 60)
    print("Integration Guide")
    print("=" * 60)
    
    print("""
To integrate this agent with your actual StarCraft 2 bot:

1. Install requirements:
   pip install -r requirements.txt

2. In your SC2 bot code, import the agent:
   
   from src.agent.unit_control_agent import UnitControlAgent
   
   class MySC2Bot(sc2.BotAI):
       def __init__(self):
           self.unit_agent = UnitControlAgent()
       
       async def on_step(self, iteration):
           # Get SC2 observation
           observation = self.get_observation()
           
           # Transform with agent
           cnn_input = self.unit_agent.process_observation(observation)""")