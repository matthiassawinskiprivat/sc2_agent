# Specification: JIT Recurrent Loop Optimization

## Context
Recurrent models (Double GRU) are currently bottlenecked by a Python `for` loop that processes frames step-by-step to handle episode resets. This causes high CPU load and underutilizes the GPU during training.

## Goal
Optimize the recurrent sequence processing logic using `torch.jit.script`. This will compile the step-by-step loop into a high-performance binary that runs entirely on the GPU, bypassing the Python interpreter.

## Requirements
1.  **JIT Scripting**: Implement standalone helper functions for processing sequences with LSTM and GRU that are compatible with `torch.jit.script`.
2.  **Compatibility**: These functions must handle the same logic as the current `_process_sequence`, specifically:
    -   Resetting hidden states when `episode_starts == 1.0`.
    -   Correctly handling the batch/sequence dimensions.
3.  **Integration**: Update `MaskableRecurrentActorCriticPolicy` to use these JIT-optimized functions.
4.  **Performance**: Significantly reduce CPU usage during the `train/` phase and increase Training FPS.

## Technical Details
-   Standalone functions avoid the complexity of JIT-compiling an entire class.
-   Functions will use standard loops (e.g., `range(len)`) instead of specialized utilities like `zip_strict`.
-   Explicit type hints are required for JIT compilation.
