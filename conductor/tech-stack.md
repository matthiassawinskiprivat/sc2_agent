# Technology Stack

## Core Language & Runtime
*   **Python 3.10**: The primary development language, chosen for its extensive RL and AI ecosystem.

## Game Integration
*   **burnysc2**: A fast and modern StarCraft 2 Python library used to interface with the game client.

## Reinforcement Learning & Deep Learning
*   **Stable-Baselines3 (SB3)**: Provides reliable implementations of RL algorithms like PPO and DQN.     
*   **Gymnasium**: The standard API for reinforcement learning environments.
*   **PyTorch (v1.13.1)**: The underlying deep learning framework for model training and inference.       
*   **sb3-contrib**: Extra SB3 algorithms and tools.

## Data Processing & Utilities
*   **NumPy (<2.0)**: Essential for high-performance tensor and matrix operations in observation transformation.
*   **OpenCV**: Used for image-like processing of spatial observation layers.
*   **psutil**: Used for robust system process management and StarCraft 2 instance cleanup.
*   **PyYAML**: Configuration management via YAML files.
*   **Loguru**: Modern, structured logging for debugging training and game loops.

## Visualization & Monitoring
*   **PyQt5**: Powers the real-time interactive debug UI.
*   **Matplotlib**: Used for plotting and secondary visualizations.
*   **Tensorboard**: Real-time tracking of training metrics (reward, loss, etc.).

## Quality Assurance
*   **Pytest**: The framework for writing and running unit and integration tests.


## Infrastructure

- **Multi-Client Manager:** Orchestrates multiple SC2 processes for self-play scenarios.

- **Perspective-Aware Transformers:** Ensures symmetric observations and actions for adversarial agents in self-play.

- **Dynamic Configuration Manager:** Orchestrates additive accumulation of observation layers and action heads across the training curriculum.

---

# Long Term Plan / Final Architecture

The project aims to achieve a fully autonomous, competitive StarCraft II agent through a **Hierarchical Multi-Agent Reinforcement Learning (H-MARL)** architecture. The system mimics a military command structure, dividing responsibilities into strategic, logistical, and tactical layers.

## 1. System Components & Flow

The architecture operates in a strict top-down hierarchy.

### A. Manager Agent (The General)
*   **Role:** Strategic decision-making and high-level orchestration.
*   **Responsibilities:**
    *   Economy Management: Decides when to Expand or Build infrastructure.
    *   Military Strategy: Decides when to Attack, Defend, or Scout.
    *   **Unit Drafting:** Explicitly selects (drafts) which combat units are assigned to each command.
*   **Frequency:** Runs on a slower tick rate (e.g., every 8-16 game steps) to simulate "planning time."
*   **Output:** Generates a structured **Command Object** and updates the **Unit Assignment Table**.

### B. Build Agent (The Logistics Officer)
*   **Role:** Execution of economic and production commands.
*   **Trigger:** Activates *only* when the Manager issues a `BUILD`, `TRAIN`, or `EXPAND` command.
*   **Responsibilities:**
    *   **Worker Selection:** Intelligent selection of the optimal worker (SCV) for the job based on distance and current task (e.g., weighing pulling a miner vs. an idle worker).
    *   **Placement Optimization:** Refines coarse location instructions into precise, valid build coordinates, handling complex constraints like walling.

### C. Unit Control Agent (The Field Commander)
*   **Role:** Tactical micro-management and combat execution.
*   **Trigger:** Always active.
*   **Responsibilities:**
    *   **Squad Execution:** Executes commands for dynamically drafted squads. If the Manager creates 2 squads (Alpha and Bravo), the Unit Control Agent runs twice per step (batch processing), once for each squad context.
    *   **Micro:** Stutter-stepping, focus-firing, and terrain usage.

---

## 2. Unit Selection Mechanisms

Both the Manager and Build Agent use a shared **Entity Selection Architecture** (Pointer Network), but with different activation logic.

### A. Build Agent Selection (One-Shot)
*   **Goal:** Select exactly **ONE** worker to build a structure.
*   **Input:** List of Candidate Workers (Features: Position, Load, Idle Status).
*   **Mechanism:**
    1.  Encodes all workers using `EntityEncoder`.
    2.  Compares encoded workers to the `Command Context`.
    3.  Applies **Softmax** across the worker list.
    4.  Selects the worker with the highest probability.

### B. Manager Agent Selection (Multi-Select Drafting)
*   **Goal:** Select **ANY NUMBER** of combat units to join a Squad/Command.
*   **Input:** List of all Combat Units (Features: Type, Health, Position, Current Command).
*   **Mechanism:**
    1.  Encodes all units using `EntityEncoder`.
    2.  Compares encoded units to the `Command Context`.
    3.  Applies **Sigmoid** to each unit independently.
    4.  Any unit with a score `> Threshold` (e.g., 0.5) is "drafted" into the new command. This allows the Manager to split forces (e.g., "Take these 10 healthy Marines to attack, leave the 5 injured ones to defend").

---

## 3. Neural Network Architectures

### A. Manager Agent Network ("The Reasoning Brain")
A recurrent network designed for long-term strategic planning ("System 2" thinking), utilizing a **Scripted Adaptive Depth** mechanism.

*   **Inputs:**
    *   **Global Vector:** `[Batch, 128]` (Resources, Supply, Army Count, Enemy Intel).
    *   **Map Features:** `[Batch, 256]` (CNN encoding of the global minimap).
*   **Core:**
    *   **Reasoning Core:** Uses a specialized GRUCell that loops over the *same* input multiple times to deepen reasoning.
    *   **Adaptive Depth (Scripted):** The number of reasoning loops (thinking steps) is controlled programmatically based on game state heuristics.
        *   **Low Intensity:** 1 Step.
        *   **High Intensity (Combat/Crisis):** 3-5 Steps.
*   **Outputs (Multi-Head):**
    1.  **Command Type Head (Softmax):** Selects the action type (e.g., `ATTACK`, `BUILD`).
    2.  **Parameter Head (Softmax):** Selects the specific unit or building type.
    3.  **Coarse Spatial Head (Spatial Softmax):** Selects an approximate 16x16 grid sector.
    4.  **Unit Selection Head (Sigmoid Pointer):** Selects which units execute this command.

### B. Build Agent Network ("The Pointer Network")
*   **Inputs:** Context Vector (Command), Entity List (Workers), Local Map (Terrain).
*   **Core:**
    *   **Entity Encoder:** Shared MLP that creates an embedding for each worker independently.
    *   **Reasoning Core (Fixed Depth):** A GRU-based reasoning loop fixed at **3 steps**. This allows the agent to reason about spatial constraints (e.g., walling connectivity) before committing to a placement.
*   **Outputs:**
    1.  **Selection Head (Softmax Pointer):** Selects 1 worker.
    2.  **Fine Placement Head (Spatial):** Precise `(dx, dy)` offset.

### C. Unit Control Agent Network ("The Spatial Actor")
The existing spatial agent, enhanced with reasoning capabilities for tactical simulation.

*   **Inputs:**
    *   **Spatial Observation:** `128x128` grid (Units, Health, Terrain).
    *   **Command Overlay:** The Manager's command target is injected as a specific "Objective Channel".
    *   **Context Injection:** The `Command Type` is injected into the fully connected layers.
*   **Core:**
    *   **CNN Encoder:** Processes the spatial observation.
    *   **Reasoning Core (Fixed Depth):** A GRU-based reasoning loop fixed at **3 steps**. This enables "Tactical Prediction" (e.g., simulating a move into tank range) and handling object permanence.
*   **Outputs (Dynamic):**
    *   **Configurable Heads:** The network supports adding/removing heads based on configuration (e.g., `action_heads: ["move", "attack", "load", "unload"]`).
    *   **Move Head:** `128x128` grid.
    *   **Attack Head:** `128x128` grid.
    *   **Load Head:** `128x128` grid (Targeting Transport).
    *   **Unload Head:** `128x128` grid (Targeting Ground).
    *   **Unload Masking:** Implements a safety-first drop policy. It uses `cv2.erode` on the pathing grid for terrain safety and `cv2.circle` with `footprint_radius` for exact entity exclusion.
    *   **Ability Selector:** Softmax over the available heads.

---

## 4. Command Protocol

*   **Command Object:** Contains `type`, `target_pos`, `unit_type`, `priority`.
*   **Execution Loop:**
    1.  Manager generates a Command and selects a list of Unit Tags.
    2.  System updates the global `UnitAssignment` dictionary.
    3.  **For Build Commands:** The single drafted worker is passed to the **Build Agent**.
    4.  **For Combat Commands:** Units are grouped by Command ID. The **Unit Control Agent** runs a batch inference pass for each active group.

---

## 5. Drop Operations Prototype (Immediate Goal)

To validate the "Reasoning" and "Dynamic Heads" architecture, a specific vertical slice will be implemented.

### A. Environment: `Level_Drop_Env`
A specialized gym environment designed to force the agent to learn the `Load` -> `Move` -> `Unload` sequence.
*   **Map Structure:** Two islands separated by `PathingGrid.VOID` (unwalkable).
    *   **Island A (Start):** Contains 4 Marines + 1 Medivac.
    *   **Island B (Goal):** Contains a Target Beacon.
*   **Fog of War:** Enabled (unlike previous basic levels) to test the Reasoning Core's memory.
*   **Reward Function:**
    *   `+0.1` per Marine loaded into the Medivac.
    *   `+0.5` when Medivac crosses the void gap.
    *   `+1.0` per Marine unloaded on Island B.
    *   `+10.0` (Episode Termination) when all Marines are on Island B.

### B. Observation Updates
The `ObservationTransformer` will be updated to expose transport mechanics:
*   **`transport_slots` Layer:** Visualizes transport units (Medivacs/Overlords).
    *   `0.0`: Non-transport.
    *   `0.5`: Transport (Empty).
    *   `1.0`: Transport (Full).
*   **`passenger_density` Layer:** Visualizes the "hidden" density of units inside a transport. If a Medivac has 4 Marines, this layer shows a value of `4.0` at the Medivac's pixel location.

### C. Network Updates
The `UnitControlAgent` will be upgraded to support **Dynamic Action Heads**.
*   **Configuration:** `action_heads: ["move", "attack", "load", "unload"]`.
*   **Weight Surgery:** A utility will be created to load pre-trained weights (from Level 7) into the new model, initializing the `move` and `attack` heads with learned knowledge while initializing `load` and `unload` randomly.