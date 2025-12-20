# Product Guidelines

These guidelines define the development philosophy, communication standards, and visual identity for the StarCraft 2 Unit Control Agent project.

## Development Philosophy
*   **Educational Explainer:** The codebase serves as a primary learning resource. Documentation and comments must prioritize explaining *why* specific reinforcement learning concepts (e.g., reward shaping, observation normalization, layer selection) are implemented.
*   **Test-Driven Development (TDD):** Every functional component, especially those involving game logic or RL data processing, must be backed by unit tests to ensure reliability and facilitate experimentation.
*   **Modular & Decoupled Architecture:** Maintain strict separation between the SC2 API interface, the observation transformers, the RL agent logic, and the visualization tools to allow for easy framework swapping and testing.

## Communication & Documentation
*   **Deep Commentary:** Use extensive code comments for complex RL logic. Aim to bridge the gap between theoretical papers and practical implementation.
*   **Architecture Transparency:** Maintain clear documentation of data flow, particularly how raw game states are transformed into fixed-size CNN tensors.

## Visual Identity (Debug UI)
*   **Information-Rich Dashboard:** The UI must provide a high-fidelity representation of the agent's "world view." Use color-coded heatmaps and intuitive overlays to visualize damage zones, unit health, and energy states.
*   **Interactive Clarity:** Design for real-time monitoring, ensuring that layer toggles and performance metrics are accessible and responsive.

## Testing Standards
*   **Logic Verification:** Tests should focus on verifying that observation layers are correctly populated based on known unit positions and stats.
*   **Integration Checks:** Ensure that the agent-to-environment loop remains stable across different SC2 maps and game updates.
