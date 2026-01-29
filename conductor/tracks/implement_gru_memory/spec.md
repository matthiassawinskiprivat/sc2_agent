# Specification: GRU Temporal Memory for MaskableRecurrentPPO

## Context
The user wants to use a GRU for temporal memory instead of an LSTM. 
The agent currently uses a GRU for "Thinking" (inside `UnitControlNetwork`).
The PPO algorithm currently uses an LSTM for "Remembering" (temporal memory).
The goal is to switch the temporal memory to a GRU, resulting in a "Double GRU" architecture.

## Requirements
1.  **Refactor `MaskableRecurrentActorCriticPolicy`**:
    -   Support an `rnn_type` argument ("lstm" or "gru").
    -   If "gru", use `nn.GRU` instead of `nn.LSTM`.
    -   Handle the different hidden state structure (GRU has one state, LSTM has two).
2.  **Refactor `MaskableRecurrentPPO`**:
    -   Adjust hidden state initialization and buffer shapes based on the chosen `rnn_type`.
    -   Ensure `RNNStates` (the NamedTuple) can handle both or provide a compatible abstraction.
3.  **Refactor `MaskableRecurrentRolloutBuffer`**:
    -   Store the correct hidden state shape for GRU.
4.  **Verification**:
    -   Update `tests/test_maskable_recurrent_ppo.py` to test both LSTM and GRU modes.
    -   Ensure masking still works with GRU temporal memory.

## Technical Details
-   `nn.LSTM` hidden state: `(h, c)`
-   `nn.GRU` hidden state: `h`
-   `sb3_contrib` uses a specific `RNNStates` NamedTuple which is hardcoded to `(h, c)` tuples. We may need to pass a zero-tensor as `c` for GRU or implement a custom State wrapper to maintain compatibility with the rest of the PPO logic.
