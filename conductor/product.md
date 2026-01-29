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

### Optimized Observation Pipeline
To maximize training stability and performance, the project utilizes an optimized observation data flow.

- **Float32 Representation:** Observations are represented as `float32` tensors in the `[0, 1]` range. This eliminates expensive redundant bounds-checking and normalization steps in the Stable Baselines 3 training loop.
- **Bypass Library Preprocessing:** By pre-normalizing data in the `ObservationTransformer` and explicitly disabling internal library normalization (`normalize_images=False`), the system achieves a significant speedup in policy evaluation, preventing watchdog timeouts during large-scale training.

### Air Unit Mechanics

Air units (e.g., Medivacs, Void Rays) are handled differently than ground units to respect their ability to cross any terrain.



- **Unrestricted Movement:** Action masks for flying units are fully open (all `True`). This allows the agent to command air units to any grid coordinate, even those marked as unpathable in the `pathing_grid`.

- **Combat Versatility:** Attack masks for air units are also unrestricted, enabling them to engage targets over cliffs or voids.

- **Ground Parity:** Ground units continue to be constrained by the `pathing_grid` to ensure realistic movement behaviors.



### Safe Unloading Mechanics

To prevent agents from attempting to unload units on invalid terrain or inside buildings, a specialized mask is applied to the `unload` action head.



- **Terrain Erosion:** The mask excludes unpathable terrain with a 2-pixel safety margin, ensuring units are not dropped too close to cliff edges or voids.

- **Exact Footprints:** Buildings, resource patches, and destructible debris are excluded using their exact footprints (calculated via `footprint_radius`), ensuring that units can be dropped adjacent to these objects but not directly on top of them.

### Robust Loading Mechanics
To ensure reliable transport operations, the agent implements a dual-mode loading system that supports all three races.

- **Dual Interaction:** The system automatically distinguishes between a unit entering a transport and a transport picking up a unit.
- **Explicit Ability Usage:** Transports (Medivacs, Overlords, Warp Prisms) utilize explicit `LOAD` AbilityIds rather than generic right-click commands, ensuring robust behavior in dense unit environments.
- **Targeting Precision:** The system uses zero-dilation action masks for entity interaction (e.g., loading), forcing the agent to target the exact unit location and eliminating "ghost target" artifacts that cause command fallbacks.
- **Race-Agnostic:** The implementation handles race-specific transport abilities (e.g., `LOAD_MEDIVAC` vs. `LOAD_OVERLORD`) through a unified action head, facilitating multi-race training.

### Refined Transport Rewards
To prevent excessive unit shuffling and incentivize efficient tactical maneuvers, the transport reward system utilizes a context-aware scaling mechanism.

- **Initial Load Scaling:** Loading a unit into a transport is rewarded once per unit per episode. The reward scales positively with the transport's current cargo occupancy, encouraging the agent to fill transports before departing.

- **Emergency Pickups:** Recurring rewards for loading previously-handled units are granted only if the unit is significantly damaged (below 50% HP), incentivizing rescue operations while penalizing redundant load/unload cycles for healthy units.

- **Efficient Bulk Unloading:** Unloading rewards in the target zone scale based on the transport's fullness at the start of the drop sequence. The first units dropped from a full transport yield the highest rewards, discouraging inefficient single-unit shuttle runs.

# Future Plans

### Hybrid Action Heads (Multi-Race Optimization)
To scale the agent to handle the full complexity of all three StarCraft II races, the action architecture will evolve from a flat joint distribution to a hybrid system that distinguishes between spatial targeting and state management.

- **Spatial Heads:** High-resolution (128x128) grids will be reserved exclusively for actions requiring precise world coordinates (e.g., Move, Attack, Yamato Cannon, Abduct, Force Field, Psionic Storm).
- **Categorical Toggle Head:** A compact categorical head will manage all "instant" or "state-based" abilities that do not require a spatial target.
    - **Terran:** Stimpack, Siege/Unsiege, Cloak, Lift/Land.
    - **Zerg:** Burrow/Unburrow, Morph to Overseer, Generate Creep.
    - **Protoss:** Activate Guardian Shield, Chrono Boost (on self/nearby), Warp Gate Transformation.
- **Parameter Efficiency:** This optimization eliminates the massive parameter waste of 16k-neuron spatial heads for non-spatial toggles, significantly accelerating training convergence and reducing VRAM usage across all race models.
- **Hierarchical Conditioning:** The system will move toward a hierarchical model where the categorical selection can provide context to the spatial heads, allowing the agent to learn "Targeted Abilities" (like Abduct or Snipe) more efficiently by sharing spatial targeting logic.
