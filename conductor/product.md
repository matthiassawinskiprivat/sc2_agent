# Initial Concept
The goal is for myself learning more about reinforcement learning algorithms.

# Product Vision
The **StarCraft 2 Unit Control Agent** (expanding into a full-scale AI) is a sophisticated reinforcement learning project designed to master the complexities of real-time strategy through a hierarchical agent architecture. 

It aims to evolve from its current high-fidelity unit micro-management capabilities into a complete competitive bot. This evolution includes a strategic Master Agent for high-level coordination, robust Macro-Management systems for economic growth, and advanced multi-agent tactical execution.

(See `conductor/tech-stack.md` for the detailed "Long Term Plan / Final Architecture" specification).

## Core Objectives
*   **Hierarchical Learning:** Implement a multi-layered RL architecture where strategic decisions (Manager Agent) delegate tasks to specialized sub-agents (Build Agent, Unit Control Agent).
*   **Complete Strategic Play:** Expand beyond micro-management to include base building, resource optimization, and tech-tree progression.
*   **Multi-Front Coordination:** Enable the agent to handle large-scale engagements across multiple locations simultaneously.
*   **Algorithmic Mastery:** Serve as a primary environment for implementing, testing, and understanding state-of-the-art RL algorithms (e.g., PPO, DQN, Pointer Networks, Recurrent Reasoning Models) in complex, high-dimensional spaces.

## Target Experience
*   **Educational Depth:** The codebase and architecture are balanced to provide both competitive performance and clear, modular implementations that facilitate learning and experimentation.
*   **State-of-the-Art Micro:** Maintain and enhance the CNN-based observation system that allows for professional-level unit control.
*   **Visual Transparency:** Utilize the real-time debug UI to provide deep insights into the agent's perception and decision-making processes.


## Feature Highlights

- Multi-agent control using hierarchical agent systems.

- Dynamic observation transformation with customizable layer stacks.

- Curriculum learning through progressively complex training levels.

- Competitive self-play in specialized environments like Capture the Flag, utilizing perspective flipping and asymmetric training modes.

## Training Levels
...
- **Capture the Flag:** Competitive asymmetric self-play with resource-based army spawning and objective-based goals.

## Core Mechanisms
...
### Asymmetric Self-Play
To maximize training efficiency and minimize data noise, the project supports asymmetric self-play.

- **Automated Opponent:** During training, one player acts as the trainee while the opponent is controlled by a snapshot of the current model.
- **Trainee-Centric Termination:** Episodes terminate immediately if the trainee loses all units, providing a clear and rapid penalty signal.
- **Persistence:** If the opponent army is defeated, the game continues, allowing the trainee to learn secondary objectives (like destroying enemy structures) without noise from the "dead" opponent.
The project employs a curriculum-driven observation system where the fidelity of the agent's perception increases over time.

- **Additive Accumulation:** Each training level defines a set of `active_layers` and `action_heads` in its configuration. As the agent progresses through the curriculum (Level 0 → Level 10 → Capture the Flag → Level Drop), the `ConfigManager` dynamically accumulates all layers and action capabilities defined in previous levels.
- **Persistence:** This ensures that skills learned in early levels (e.g., following beacons, respecting terrain) remain available as the agent learns more complex behaviors (e.g., combat micro, air unit management).
- **Transparency:** Environments log the layer expansion during initialization (e.g., `Observation layers: (5) -> (23)`), providing clear feedback on the agent's current state space.

### Air Unit Mechanics
Air units (e.g., Medivacs, Void Rays) are handled differently than ground units to respect their ability to cross any terrain.

- **Unrestricted Movement:** Action masks for flying units are fully open (all `True`). This allows the agent to command air units to any grid coordinate, even those marked as unpathable in the `pathing_grid`.
- **Combat Versatility:** Attack masks for air units are also unrestricted, enabling them to engage targets over cliffs or voids.
- **Ground Parity:** Ground units continue to be constrained by the `pathing_grid` to ensure realistic movement behaviors.