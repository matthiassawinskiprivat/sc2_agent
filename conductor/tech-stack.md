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
*   **PyYAML**: Configuration management via YAML files.
*   **Loguru**: Modern, structured logging for debugging training and game loops.

## Visualization & Monitoring
*   **PyQt5**: Powers the real-time interactive debug UI.
*   **Matplotlib**: Used for plotting and secondary visualizations.
*   **Tensorboard**: Real-time tracking of training metrics (reward, loss, etc.).

## Quality Assurance
*   **Pytest**: The framework for writing and running unit and integration tests.
