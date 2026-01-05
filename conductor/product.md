# Initial Concept
The goal is for myself learning more about reinforcement learning algorithms.

# Product Vision
The **StarCraft 2 Unit Control Agent** (expanding into a full-scale AI) is a sophisticated reinforcement learning project designed to master the complexities of real-time strategy through a hierarchical agent architecture. 

It aims to evolve from its current high-fidelity unit micro-management capabilities into a complete competitive bot. This evolution includes a strategic Master Agent for high-level coordination, robust Macro-Management systems for economic growth, and advanced multi-agent tactical execution.

## Core Objectives
*   **Hierarchical Learning:** Implement a multi-layered RL architecture where strategic decisions (Master Agent) delegate tasks to tactical specialists (Unit Control Agents).
*   **Complete Strategic Play:** Expand beyond micro-management to include base building, resource optimization, and tech-tree progression.
*   **Multi-Front Coordination:** Enable the agent to handle large-scale engagements across multiple locations simultaneously.
*   **Algorithmic Mastery:** Serve as a primary environment for implementing, testing, and understanding state-of-the-art RL algorithms (e.g., PPO, DQN) in complex, high-dimensional spaces.

## Target Experience
*   **Educational Depth:** The codebase and architecture are balanced to provide both competitive performance and clear, modular implementations that facilitate learning and experimentation.
*   **State-of-the-Art Micro:** Maintain and enhance the CNN-based observation system that allows for professional-level unit control.
*   **Visual Transparency:** Utilize the real-time debug UI to provide deep insights into the agent's perception and decision-making processes.


## Feature Highlights

- Multi-agent control using hierarchical agent systems.

- Dynamic observation transformation with customizable layer stacks.

- Curriculum learning through progressively complex training levels.

- Competitive self-play in specialized environments like Capture the Flag, utilizing perspective flipping for symmetric agent training.



## Training Levels

The project follows a curriculum starting from basic movement to complex combat:

- **Level 0-2:** Basic movement and single-unit navigation.

- **Level 3-6:** Multi-unit coordination and target collection.

- **Level 7-10:** Advanced combat micro and tactical positioning.

- **Capture the Flag:** Competitive self-play with resource-based army spawning and objective-based goals.

## Core Mechanisms

### Dynamic Layer Stack
The project employs a curriculum-driven observation system where the fidelity of the agent's perception increases over time.

- **Additive Accumulation:** Each training level defines a set of `active_layers` in its configuration. As the agent progresses through the curriculum (Level 0 → Level 10 → Capture the Flag), the `ConfigManager` dynamically accumulates all layers defined in previous levels.
- **Persistence:** This ensures that skills learned in early levels (e.g., following beacons, respecting terrain) remain available as the agent learns more complex behaviors (e.g., combat micro, air unit management).
- **Transparency:** Environments log the layer expansion during initialization (e.g., `Observation layers: (5) -> (23)`), providing clear feedback on the agent's current state space.

