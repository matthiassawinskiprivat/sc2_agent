#!/usr/bin/env python3
"""
StarCraft 2 Unit Control Agent - Hierarchical System Example
Demonstrates shared observation transformation and multiple agents
"""

import sys
import os
import numpy as np
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.hierarchical_agent import HierarchicalAgentSystem
from src.utils.layer_constants import LayerConstants


class ExampleSC2BotWithHierarchicalSystem:
    """Example SC2 bot that uses the hierarchical agent system"""
    
    def __init__(self):
        """Initialize the bot with hierarchical agent system"""
        # Create hierarchical system
        self.system = HierarchicalAgentSystem(map_size=(128, 128))
        
        # Create different types of unit control agents
        self._create_agents()
        
        self.step_count = 0
        
    def _create_agents(self):
        """Create different types of agents for different purposes"""
        
        # Combat-focused agent (attack layers only)
        combat_agent = self.system.create_unit_agent(
            agent_id="combat_agent",
            cnn_input_size=(128, 128),
            layer_indices=[0, 1, 2, 3]  # Attack layers only
        )
        
        # Support agent (health and energy monitoring)
        support_agent = self.system.create_unit_agent(
            agent_id="support_agent", 
            cnn_input_size=(128, 128),
            layer_indices=[4, 5]  # Health and energy only
        )
        
        # Full agent (all layers for comprehensive analysis)
        full_agent = self.system.create_unit_agent(
            agent_id="full_agent",
            cnn_input_size=(128, 128),
            layer_indices=[0, 1, 2, 3, 4, 5]  # All 6 layers
        )
        
        print(f"Created {len(self.system.unit_agents)} agents:")
        for agent_id, config in self.system.agent_configs.items():
            layer_names = [LayerConstants.get_layer_name(idx) for idx in config['layer_indices']]
            print(f"  - {agent_id}: {layer_names}")
    
    def on_game_start(self):
        """Called when the game starts"""
        print("Game started! Hierarchical agent system ready.")
        print(f"System config: {len(self.system.unit_agents)} agents")
        
        # Show available layers
        print(f"Available layers: {len(LayerConstants.LAYERS)} total")
        print(f"Core layers: {LayerConstants.CORE_LAYERS}")
    
    def on_game_step(self, observation: Dict[str, Any]):
        """Process each game step"""
        self.step_count += 1
        
        # Process observation through hierarchical system
        results = self.system.process_observation(observation, player_id=1)
        
        if results:
            # Log every 50 steps
            if self.step_count % 50 == 0:
                print(f"Step {self.step_count}: Processed with {len(results)} agents")
                
                for agent_id, data in results.items():
                    print(f"  - {agent_id}: {data.shape}")
                    
                    # Show some statistics
                    min_val = np.min(data)
                    max_val = np.max(data)
                    mean_val = np.mean(data)
                    print(f"    Min: {min_val:.3f}, Max: {max_val:.3f}, Mean: {mean_val:.3f}")
            
            # Here you would typically:
            # 1. Feed each agent's data to their respective RL models
            # 2. Get action predictions for each agent
            # 3. Coordinate actions between agents
            # 4. Send commands to the game
            
            # For demonstration, just show the different layer selections
            if self.step_count == 1:
                self._demonstrate_layer_selections(results)
    
    def _demonstrate_layer_selections(self, results: Dict[str, np.ndarray]):
        """Demonstrate how different agents see different layers"""
        print("\n=== Agent Layer Selections ===")
        
        for agent_id, data in results.items():
            config = self.system.agent_configs[agent_id]
            layer_indices = config['layer_indices']
            
            print(f"\nAgent '{agent_id}' uses {len(layer_indices)} layers:")
            for i, idx in enumerate(layer_indices):
                layer_name = LayerConstants.get_layer_name(idx)
                print(f"  Layer {i} (index {idx}): {layer_name}")
                
                # Show some statistics for this layer
                if i < data.shape[0]:
                    layer_data = data[i] if len(layer_indices) > 1 else data
                    non_zero = np.count_nonzero(layer_data)
                    print(f"    Non-zero pixels: {non_zero}")
    
    def on_game_end(self):
        """Called when the game ends"""
        print(f"Game ended after {self.step_count} steps")
        
        # Show final system statistics
        stats = self.system.get_system_statistics()
        print("\nFinal System Statistics:")
        print(f"Total transformations: {stats['system_stats']['total_transformations']}")
        print(f"Total agent processes: {stats['system_stats']['total_agent_processes']}")
        print(f"Avg transformation time: {stats['system_stats']['avg_transformation_time']*1000:.2f}ms")
        
        # Save final debug data
        self.system.reset_system()
    
    def get_system(self) -> HierarchicalAgentSystem:
        """Get the hierarchical agent system"""
        return self.system


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


def demonstrate_hierarchical_system():
    """Demonstrate the hierarchical agent system"""
    
    print("=" * 70)
    print("StarCraft 2 Hierarchical Agent System - Feature Demonstration")
    print("=" * 70)
    
    # Create bot
    bot = ExampleSC2BotWithHierarchicalSystem()
    
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
            
            # Small delay to simulate real-time
            time.sleep(0.05)
            
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
        
        print("\n" + "=" * 70)
        print("Hierarchical system demonstration completed!")
        print("=" * 70)


if __name__ == "__main__":
    demonstrate_hierarchical_system()