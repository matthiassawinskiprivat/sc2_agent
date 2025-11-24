# StarCraft 2 Unit Control Agent - Corrected Architecture

## Overview
This document describes the corrected architecture that addresses all the issues raised, implementing a clean separation of concerns with shared observation transformation and multiple agent support.

## Core Design Principles

### 1. **Fixed 6-Layer System**
- Only the 6 layers you specified are implemented
- No extensible layer system (as requested)
- Layer indices are fixed (0-5)

### 2. **Shared Observation Transformation**
- Transform once, share among all agents
- No duplicate transformation work
- Efficient for multiple agents

### 3. **Clean Agent Architecture**
- UnitControlAgent is pure business logic
- No UI responsibilities
- Receives pre-transformed observations

### 4. **Multiple Agent Support**
- Hierarchical system manages multiple agents
- Each agent can select different layer subsets
- Shared transformer for efficiency

## Architecture Components

### 1. SharedObservationTransformer
**Location**: `src/transformers/shared_observation_transformer.py`

```python
class SharedObservationTransformer:
    def transform_observation(self, observation, observation_id=None, player_id=1):
        # Transforms SC2 observation to 6-layer format
        # Caches results to avoid duplicate work
        # Returns [6×H×W] numpy array
```

**Features**:
- Transforms observation once per game step
- Caches results using observation ID
- Always produces 6 layers in fixed order:
  1. Allied Ground Attack (index 0)
  2. Allied Air Attack (index 1)
  3. Enemy Ground Attack (index 2)
  4. Enemy Air Attack (index 3)
  5. Health Percentage (index 4)
  6. Energy Percentage (index 5)

### 2. UnitControlAgent (Core Agent)
**Location**: `src/agent/unit_control_agent.py`

```python
class UnitControlAgent:
    def __init__(self, cnn_input_size=(128, 128)):
        # Pure agent - no transformer, no UI
        
    def process_transformed_observation(self, transformed_observation):
        # Processes pre-transformed observation
        # Returns the data (no transformation)
        
    def select_layers(self, layer_indices):
        # Selects layers by index (0-5)
        # Returns subset of the 6 layers
```

**Key Changes**:
- No longer transforms observations
- Receives pre-transformed data
- Layer selection by indices (0-5)
- No UI dependencies
- Pure business logic only

### 3. HierarchicalAgentSystem
**Location**: `src/agent/hierarchical_agent.py`

```python
class HierarchicalAgentSystem:
    def __init__(self, map_size=(128, 128), unit_data_file=None):
        self.shared_transformer = SharedObservationTransformer(...)
        self.unit_agents = {}  # Multiple agents
        
    def process_observation(self, observation, player_id=1):
        # 1. Transform once with shared transformer
        # 2. Distribute to all unit agents
        # 3. Each agent gets its selected layers
        # 4. Return results from all agents
        
    def create_unit_agent(self, agent_id, cnn_input_size, layer_indices):
        # Creates new agent with specific layer selection
```

**Features**:
- Manages shared transformer
- Creates and manages multiple unit agents
- Handles layer distribution
- Tracks system statistics

### 4. LayerConstants (Fixed)
**Location**: `src/utils/layer_constants.py`

```python
class LayerConstants:
    # Only the 6 layers you specified
    ALLIED_GROUND_ATTACK = "allied_ground_attack"
    ALLIED_AIR_ATTACK = "allied_air_attack"
    ENEMY_GROUND_ATTACK = "enemy_ground_attack"
    ENEMY_AIR_ATTACK = "enemy_air_attack"
    HEALTH_PERCENTAGE = "health_percentage"
    ENERGY_PERCENTAGE = "energy_percentage"
    
    # Fixed mapping
    LAYERS = {name: index for name, index in ...}
    CORE_LAYERS = [all 6 layer names]
```

**Key Changes**:
- Only 6 layers (as specified)
- No extensible system
- Fixed indices (0-5)

## Usage Patterns

### Basic Usage with Single Agent
```python
from src.agent.unit_control_agent import UnitControlAgent
from src.utils.layer_constants import LayerConstants

# Create agent
agent = UnitControlAgent(cnn_input_size=(128, 128))

# In your SC2 bot:
transformed_obs = shared_transformer.transform_observation(observation)
agent_data = agent.process_transformed_observation(transformed_obs)

# Select specific layers
combat_data = agent.select_layers([0, 1, 2, 3])  # Attack layers only
support_data = agent.select_layers([4, 5])       # Health/energy only
```

### Multiple Agents with Hierarchical System
```python
from src.agent.hierarchical_agent import HierarchicalAgentSystem

# Create system
system = HierarchicalAgentSystem(map_size=(128, 128))

# Create different agents
combat_agent = system.create_unit_agent(
    "combat", 
    cnn_input_size=(128, 128),
    layer_indices=[0, 1, 2, 3]  # Attack layers only
)

support_agent = system.create_unit_agent(
    "support",
    cnn_input_size=(128, 128), 
    layer_indices=[4, 5]  # Health/energy only
)

# Process observation once, distribute to all agents
results = system.process_observation(observation, player_id=1)

# results = {
#     "combat": [4 layers of attack data],
#     "support": [2 layers of health/energy data]
# }
```

### In SC2 Bot Integration
```python
class MySC2Bot:
    def __init__(self):
        self.system = HierarchicalAgentSystem(map_size=(128, 128))
        
        # Create agents for different unit types
        self.marine_agent = self.system.create_unit_agent(
            "marine", layer_indices=[0, 2, 4]  # Ground attacks + health
        )
        self.air_agent = self.system.create_unit_agent(
            "air", layer_indices=[1, 3, 4]     # Air attacks + health
        )
    
    async def on_step(self, iteration):
        # Get SC2 observation
        observation = self.get_observation()
        
        # Process with hierarchical system (transforms once)
        agent_results = self.system.process_observation(observation)
        
        # Get data for each agent
        marine_data = agent_results.get("marine")
        air_data = agent_results.get("air")
        
        # Use with RL models
        marine_actions = self.marine_model.predict(marine_data)
        air_actions = self.air_model.predict(air_data)
        
        # Execute actions...
```

## Performance Benefits

### 1. **No Duplicate Transformation**
- Single transformation per game step
- Shared among all agents
- Significant performance improvement with multiple agents

### 2. **Clean Architecture**
- Separation of concerns
- Easy to test components independently
- No UI dependencies in core logic

### 3. **Flexible Agent Design**
- Different agents can focus on different aspects
- Layer selection by indices (simple and fast)
- Easy to add new agent types

### 4. **Efficient Memory Usage**
- Only 6 layers always
- No extensible overhead
- Minimal memory footprint

## Key Corrections Made

### 1. **Fixed Layer System**
- ❌ Removed extensible layer system
- ✅ Only 6 core layers as specified
- ✅ Fixed indices (0-5)

### 2. **Shared Transformation**
- ❌ Removed per-agent transformation
- ✅ Single transformation shared among agents
- ✅ Hierarchical system manages distribution

### 3. **Clean Agent Architecture**
- ❌ Removed UI from UnitControlAgent
- ❌ Removed transformer from UnitControlAgent
- ✅ Pure business logic only
- ✅ Receives pre-transformed data

### 4. **Proper Layer Selection**
- ❌ Removed layer selection by name
- ✅ Layer selection by indices (0-5)
- ✅ Simple array slicing

## Testing

Run the corrected system:
```bash
# Test the hierarchical system
python test_system.py

# Run hierarchical example
python example_hierarchical.py

# Run demo with UI
python main.py --demo
```

## Conclusion

This corrected architecture:
- ✅ Implements exactly the 6 layers you specified
- ✅ Transforms observations once and shares among agents
- ✅ Provides clean separation of concerns
- ✅ Supports multiple agents efficiently
- ✅ Ready for SC2 bot integration
- ✅ Production-ready performance

The system now follows your requirements precisely while maintaining clean, efficient, and extensible architecture for multiple agents.