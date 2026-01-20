# Plan: VRAM Cache Implementation

- [x] **Buffer Enhancement**
    - [x] Update `MaskableRecurrentRolloutBuffer` in `src/agent/maskable_recurrent_ppo.py`.
    - [x] Add `to_device(device)` method to pre-load all arrays into a dictionary of Tensors.
    - [x] Add `clear_vram_cache()` method.
    - [x] Modify `_get_samples` to check for cached tensors before calling `self.to_torch()`.

- [x] **Algorithm Integration**
    - [x] Update `MaskableRecurrentPPO.train()` to call `rollout_buffer.to_device(self.device)` before the epoch loop.
    - [x] Ensure cache is cleared/invalidated during `reset()` or at the start of `collect_rollouts()`.

- [x] **Verification**
    - [x] Run existing tests in `tests/test_maskable_recurrent_ppo.py` to ensure functional correctness.
    - [x] (Manual) Observe CPU/GPU usage shift during training.
