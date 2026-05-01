# AGENTS.md — Technical Reference for AI Assistants

This file contains internal wiring rules, contracts, and conventions for the SC2 Unit Control Agent codebase. It is intended for AI coding assistants (Cursor, Copilot, etc.) working on this project.

For a human-oriented overview, see [README.md](README.md).

**Important:** The `original` architecture (`agent_architecture: "original"`, `config/`, `train/train_pipeline.py`, `SC2Bot`) is **deprecated** and will be removed. All new development targets the **`universal`** architecture (`agent_architecture: "universal"`, `config_universal/`, `train/async_train_universal.py`, `SC2BotUniversal`). When making changes, focus exclusively on the universal path unless explicitly asked to modify legacy code.

---

## Source of truth files

When making changes to observations, actions, or config wiring, always consult these files first:

| Area | File |
|------|------|
| Layer names and indices | `src/utils/layer_constants.py` |
| Cumulative config accumulation | `src/utils/config_manager.py` |
| Observation tensor construction | `src/transformers/observation_transformer.py` |
| Base environment and obs space | `train/envs/level_0_env.py` |
| Universal bot / action decoding | `src/sc2_bot_universal.py` |
| Universal action masking (flat mask, spatial policy behavior, combat command gating) | `src/sc2_bot_universal.py`, `src/level_7_bot.py` (`action_masks`), `src/agent/models/universal_recurrent_policy.py` (`_disable_spatial_mask`) |
| Combat bot actuation (`Level7Bot`, universal branch) | `src/level_7_bot.py` (`apply_action`) — must stay agent-faithful; see **Agent-faithful actuation** |
| Multi-unit beacon assignment (Level 3) | `src/level_3_bot.py`, `train/envs/level_3_env.py` |
| Network architecture | `src/agent/models/unit_control_network.py` |
| Universal curriculum | `config_universal/pipeline.yaml` |

---

## Class hierarchy

### Environments

```
Level0Env (gym.Env)        — base: single unit, single target, obs space, config wiring
  └─ Level3Env             — multi-unit, multi-target (beacon collection)
       └─ Level7Env        — combat (attack actions)
```

### Bots

```
SC2Bot (BotAI)             — original architecture, base game logic
  └─ Level0Bot             — single-unit beacon collection
  └─ Level3Bot             — multi-unit beacon assignment and cycling
       └─ Level7Bot        — combat reward/observation extensions

SC2BotUniversal (BotAI)    — universal architecture, MultiDiscrete action decoding
```

### Key components

```
ObservationTransformer     — raw SC2 state → (C, H, W) tensor stacks
HierarchicalAgentSystem    — shared transformer + per-unit agent management
UnitControlNetwork         — CNN feature extractor (global + tactical streams → fused features)
```

---

## Shared code, inheritance, and refactoring discipline

Training environments and bots **inherit from each other** (see hierarchy above). The same methods are often **reused across levels** and **called from distant code** (vectorized env wrappers, async collectors, universal vs original policy paths, tests). A change that looks correct in one file can **silently break** another level or subsystem.

**Before changing a method on a base or shared class** (`Level0Env`, `Level3Bot`, `SC2Bot`, `reset_game_state`, reward helpers, observation builders, `ConfigManager` getters, rollout collection, etc.):

1. **Find every caller** — use search (e.g. ripgrep) for the method name, class name, and related hooks (`env_method`, `super()` overrides). Do not assume only one level uses it.
2. **Reason about subclasses** — overrides may depend on base behavior; changing the base can bypass or invert intent in `Level3Env` / `Level7Env` / `Level3Bot` / `Level7Bot` without a compile error.
3. **Run tests for all affected areas** — movement, multi-unit, combat, config accumulation, and any recurrent / vec-env tests tied to the change.

**When a fix would force awkward branching or risks breaking unrelated callers, prefer isolation over mutation:**

- **Override in the specific subclass** (`Level3Env` only, `Level7Bot` only) instead of widening `Level0Env` behavior.
- **Add a new helper or small module** with a clear name instead of overloading an existing function with new parameters or hidden global state.
- **Add a dedicated code path** (e.g. env flag, `agent_architecture` branch) rather than changing semantics for all architectures at once.

Default stance: **minimize blast radius**. Shared bases are contracts for the whole curriculum; treat edits there like API changes.

---

## ConfigManager — cumulative config accumulation

**Module:** `src/utils/config_manager.py`

The ConfigManager walks `pipeline.yaml`'s `levels` list in order and accumulates four additive properties. For a target level `L`:

1. Start from the **pipeline root** list (if present).
2. Walk levels in order, loading each `<level>.yaml`'s `env` section.
3. **Append** new items in order, **skip duplicates** (case-insensitive).
4. Stop when the walk reaches `L`.
5. If `L` is not in `pipeline.levels`, load `L.yaml` and merge on top.

| Getter | YAML key (under `env`) |
|--------|----------------------|
| `get_cumulative_active_layers(level)` | `active_layers` |
| `get_cumulative_active_tactical_layers(level)` | `active_tactical_layers` |
| `get_cumulative_action_heads(level)` | `action_heads` |
| `get_cumulative_categorical_heads(level)` | `categorical_heads` |

Lists are **union-in-order**, not per-level replacement. Adding a layer or head in a later level **never removes** earlier ones.

**Consumed by:** `Level0Env.__init__()` — calls all four getters, stores results in `training_conf` so bots and the `ObservationTransformer` see a single flat config dict.

---

## Observation layer ordering — CRITICAL CONTRACT

`active_layers` (global map) and `active_tactical_layers` (tactical map) are **independent** cumulative configs. They may contain different layer sets and different numbers of layers.

| Stream | Config key | Getter | Index mapping used for slicing |
|--------|-----------|--------|-------------------------------|
| Global 128×128 | `active_layers` | `get_cumulative_active_layers` | `LayerConstants.LAYERS` |
| Tactical 32×32 | `active_tactical_layers` | `get_cumulative_active_tactical_layers` | `LayerConstants.TACTICAL_LAYERS` |

### Ordering rules

1. **Config order IS channel order.** The `ObservationTransformer` slices channels in the exact order they appear in the accumulated list. No sorting or reordering is applied. Channel 0 = first entry, channel 1 = second, etc.
2. **Accumulation is append-only.** Each level can add new layer names; they are appended to the end. Duplicates are skipped. Layers are never removed.
3. **Existing channels never shift.** A channel at position N in an earlier level stays at position N in all subsequent levels. The network's learned weights for that channel remain valid across the curriculum.
4. **Levels that define no layers inherit everything.** If a level's YAML omits `active_layers` / `active_tactical_layers` (or the list is empty), it uses the accumulated list from prior levels, unchanged.

### Example pipeline accumulation

```
level_move_single_0:
  active_layers: [allied_density, enemy_density, terrain, objective, assignment]
  active_tactical_layers: [allied_density, enemy_density, objective, assignment, terrain]

level_move_multi_0 through level_move_multi_4:
  (no new layers → inherits same lists)

level_attack_0:
  active_layers: [allied_g_dps_base, enemy_g_dps_base, allied_health, enemy_health]
  active_tactical_layers: [allied_g_dps_base, enemy_g_dps_base, allied_health, enemy_health]

→ At level_attack_0, cumulative active_layers =
  [allied_density, enemy_density, terrain, objective, assignment,
   allied_g_dps_base, enemy_g_dps_base, allied_health, enemy_health]
  (channels 0-4 unchanged, channels 5-8 appended)
```

### When adding new layers

Only list **new** layer names in a level YAML. They get appended after all previously accumulated layers. Never redefine the full list.

If you add a new layer name, you must:
1. Add a string constant in `LayerConstants` (`src/utils/layer_constants.py`).
2. Add it to `LayerConstants.LAYERS` (and `TACTICAL_LAYERS` if applicable) with the next available index.
3. Implement the rendering logic in `ObservationTransformer._process_single_unit` or the relevant method.
4. Add the name to the appropriate level YAML under `active_layers` / `active_tactical_layers`.

---

## Observation construction flow

### Global map (128×128)

```
Bot._get_observation() → raw_obs dict
  → ObservationTransformer.transform(raw_obs)
    → allocates (NUM_LAYERS, 128, 128) buffer
    → renders all units/beacons/terrain into full buffer
    → slices to active_layers channels (config order)
    → returns {"map": (N, 128, 128), "command": (10,)}
```

### Tactical map (32×32, universal only)

```
Bot._get_observation() → raw_obs dict
  → ObservationTransformer.transform_tactical(raw_obs, active_unit)
    → allocates (NUM_TACTICAL_LAYERS, 32, 32) buffer
    → renders units/beacons within local crop around active_unit
    → slices to active_tactical_layers channels (config order)
    → returns (N_tactical, 32, 32)
```

### Key details

- `transform_tactical` temporarily overrides internal state (scale, offsets, crop, layer map) to render at tactical resolution, then restores.
- Both `transform` and `transform_tactical` use their respective config lists for slicing — they are independent.
- The `raw_obs` dict must include `high_level_command` for the command vector to be set correctly. Level-specific bots are responsible for including this field.

---

## UnitControlNetwork — spatial fusion architecture

The network processes three inputs:

- **`map`** (global, 128×128) → Global CNN trunk → AdaptiveAvgPool2d to 16×16
- **`tactical`** (32×32) → Tactical CNN → downsample to 16×16
- **`command`** (10,) → one-hot/embedding

The 16×16 outputs from both CNNs are:
1. **Concatenated** channel-wise → `_spatial_fused` (stashed for the SpatialDecoder)
2. **Global Average Pooled** separately, then concatenated with `command` → fed through Linear + ReasoningCore → `thought` features

The policy head uses `thought` for the command head and `_spatial_fused` + latent for the SpatialDecoder.

---

## Action space

### Universal architecture (`agent_architecture == "universal"`)

- **Space:** `MultiDiscrete([n_heads, spatial_res²])` where `spatial_res = 32`.
- `action[0]` = command/head index, `action[1]` = spatial cell on 32×32 grid.
- If `head_idx >= len(action_heads)`, it indexes into `categorical_heads` (no spatial target).
- `SC2BotUniversal` decodes head index → head name, spatial index → game coordinate.
- `UniversalRecurrentPolicy` disables masking on the spatial slice (shared spatial head assumption).

### Original architecture

- **Space:** `Discrete(len(action_heads) * grid_size² + len(categorical_heads))`.
- Flattened `(head, y, x)` with categoricals appended. Decoded by `SC2Bot`.

### action_heads vs categorical_heads

- **`action_heads`** — spatial heads needing a grid target (e.g. `move`, `attack`, `load`). Cumulative.
- **`categorical_heads`** — discrete toggles with no spatial target. Cumulative.

---

## Universal action masking

**Mask vector shape:** `action_masks()` returns a **single flat** `bool` array of length `FIXED_UNIVERSAL_COMMAND_DIM + 32*32` (64 + 1024), concatenated as `[command_slice | spatial_slice]`. This must match `MultiDiscrete.nvec` (`[64, 1024]`) for stacking in vectorized envs and for `MaskableRecurrentPPO`.

| Piece | Location | Behavior |
|--------|----------|----------|
| Base mask builder | `SC2BotUniversal.action_masks` | **Command slice:** `True` for the first `len(action_heads) + len(categorical_heads)` indices (pipeline-configured heads), `False` for padded `_unused` slots only. **Spatial slice:** all `True` (no pathing or guidance masking in the env mask). |
| Policy forward | `UniversalRecurrentPolicy._disable_spatial_mask` | When masks are passed into the policy, the **spatial tail is forced to all `True`** before sampling. Only **command** logits respect the env mask. **Reason:** one shared `SpatialDecoder` serves every head; per-head spatial masks would contradict that architecture. |
| Combat (Level 7, universal) | `Level7Bot.action_masks` | Same as `SC2BotUniversal.action_masks` (no extra attack / tactical-fovea command masking). |
| `action_heads` on bots | `SC2Bot.__init__` / `SC2BotUniversal.__init__` | **`training_conf['action_heads']` (flattened, cumulative) must win** over nested `env['action_heads']`. Per-level YAML only lists **new** heads to append; nested `env.action_heads` is not the full command list. If the bot used only the nested list, **policy `MultiDiscrete` indices and `apply_action` decoding would disagree** (silent combat bugs). |

**Tests:** `tests/test_sc2bot_action_heads_flattened_priority.py`, `tests/test_level_7_tactical_attack_mask.py`.

---

## Agent-faithful actuation (universal) — no hidden “help”

**Principle:** What the policy outputs for a spatial head (command index + cell → world position) must become the **actual** StarCraft order target. The agent is supposed to **learn** targeting and motion from observations and rewards, not inherit a second, hidden decision layer that overrides or substitutes its chosen point or command semantics.

**Do not** (for `agent_architecture: "universal"`, in `apply_action` / actuation paths):

- Remap `attack` to a different target than the decoded world point (e.g. force `unit.attack(enemy_unit)` when the policy chose a coordinate, pick “nearest enemy to click,” or override with Hungarian / sticky combat assignment).
- Similarly override `move` (or other spatial heads) to a manager-chosen position while still consuming the policy’s action as if it were honored.
- Add new “convenience” actuation that **replaces** the policy’s spatial choice with heuristics (assignment, closest unit, centroid, etc.) **without** the same choice being explicit in the observation and action space.

**Fine (not actuation overrides):**

- **Observations and rewards** may use assignments, distances to preferred objectives, or other structure—as long as that structure is either fixed by an explicit curriculum contract (see Level 3 beacon assignment) or clearly exposed so learning can account for it.
- **Pathability / legality** masking (invalid cells) may be enforced outside the universal env mask if you add an explicit curriculum mechanism; the default universal env mask does not gate spatial cells.
- **Command-head masking** in the default universal path only hides **unconfigured** padded command indices (`_unused`); configured heads from the pipeline are not masked by observation-derived heuristics.

**Regression intent:** Combat and micro levels must not quietly teach a different MDP than the one implied by `action_heads` and the spatial decoder. If you add actuation “help,” you are changing the effective action space without updating the policy or metrics.

---

## Level YAML structure

Each level file (e.g. `level_move_single_0.yaml`) has three top-level sections:

```yaml
training:
  training_steps: 2_000_000
  num_envs: 4
  n_steps: 4096
  batch_size: 1024
  learning_rate: 0.0003
  # ...

env:
  env_class: "Level3Env"       # which Gym env class to instantiate
  targets: 2
  max_steps: 200
  maps_dir: "train/maps/128x128"
  high_level_command: "scout"
  active_layers: [...]         # NEW layers to add (appended to accumulated)
  active_tactical_layers: [...] # NEW tactical layers to add
  action_heads: [...]          # NEW action heads to add
  rewards:
    success: 40.0
    step_penalty: -0.1
    distance_multiplier: 1.0

model:
  algorithm: "MaskableRecurrentPPO"
  policy: "CnnLstmPolicy"
  policy_kwargs:
    features_extractor_class: "src.agent.models.unit_control_network.UnitControlNetwork"
    features_extractor_kwargs:
      features_dim: 512
      thinking_steps: 3
    lstm_hidden_size: 512
```

`Level0Env` flattens `training` and `env` into a single `training_conf` dict consumed by bots and the transformer.

---

## Environment ↔ Bot ↔ Transformer wiring

| Level type | Env class | Bot class | Targets |
|-----------|-----------|-----------|---------|
| Single-unit move | `Level0Env` | `Level0Bot` / `SC2BotUniversal` | `Point2` beacons |
| Multi-unit move | `Level3Env` | `Level3Bot` | `Unit` beacons (player_id=1) |
| Combat | `Level7Env` | `Level7Bot` | `Unit` enemies |

### Level3Env / Level3Bot — unit-to-beacon assignment (CRITICAL CONTRACT)

This behavior is intentional for curriculum stability and credit assignment. **Do not “optimize” it away** by re-running global matching when beacons disappear.

| Rule | Details |
|------|---------|
| **When assignments are created** | Exactly **once per episode**, after spawn, in `Level3Bot.reset_game_state`: `self.current_assignments = self.get_optimal_assignments()`. |
| **Algorithm** | `get_optimal_assignments()` uses `scipy.optimize.linear_sum_assignment` (Hungarian) on unit–beacon distances. With `targets` units and `targets` beacons (verified at reset), the matching is **one-to-one**: each beacon tag appears at most once as a value in `current_assignments`. |
| **Frozen for the episode** | **Never** call `get_optimal_assignments()` mid-episode to **re-match remaining** units and beacons. Survivors **keep** their initial beacon tag until that beacon is collected or the pairing is cleaned up. |
| **When a beacon is collected** | `Level3Env._async_step` removes **only** the collecting unit’s entry from `current_assignments` after kill. Other units’ `unit_tag → beacon_tag` entries are **unchanged**. |
| **Stale pairing (beacon gone, unit still in dict)** | In `Level3Bot._get_observation`, if this unit’s assigned beacon no longer exists in the game, **only** `pop` that unit’s key from `current_assignments` (and reset its distance baseline). **Do not** run Hungarian on all remaining units to assign “leftover” beacons. |
| **Empty `current_assignments`** | Helpers such as `get_assigned_distances`, `check_collections`, and original-arch `action_masks` must **not** lazily refill via `get_optimal_assignments()` when the dict is empty mid-episode. Empty means “no pairing”; refilling would violate the frozen contract. |
| **Surplus units (`num_units > num_beacons`)** | `get_optimal_assignments()` still assigns extra units to **closest** beacon (many-to-one). Normal Level 3 episodes assert equal counts at spawn; do not rely on surplus behavior for standard multi-beacon levels. |

**Why this matters:** Re-running Hungarian after each collection changes every survivor’s designated beacon, observation `command_target`, and distance-shaping baseline. That contradicts the training assumption that each unit’s objective stays the **same beacon for the whole episode** (aside from completion/removal of that pair).

**Regression coverage:** `tests/test_level_3_assignment_logic.py::test_frozen_assignments_no_rehungarian_when_peer_beacon_missing` asserts that a missing peer beacon does not trigger a global re-match and that surviving units keep their original beacon in `command_target`.

### Level3Env — per-step reward attribution (and recurrent training)

| Piece | Behavior |
|-------|----------|
| **Who acts** | `Level3Env._async_step` uses `current_tag = active_unit_tags[current_unit_index]`, applies the action only to that unit, then one `client.step()`. |
| **Distance shaping** | `calculate_distance_reward(current_tag)` updates `previous_distances[current_tag]` only. No other unit’s baseline is touched on that env step. |
| **Collection / success** | `check_collections` buffers `pending_success_rewards[u_tag]` per collecting unit. The env pops **`pending_success_rewards.pop(current_tag, 0.0)`** so the success bonus is attributed on the **step where that unit is the actor** (may be a later step if another unit’s physics completed a pickup first — still the correct unit tag, not the wrong actor). |
| **RNN / GRU reset when cycling units** | One env step = one **different** unit’s egocentric observation. The shared policy must **not** carry LSTM hidden state from unit A into the forward pass for unit B. Envs expose `get_obs_focus_unit_tag()`; async collectors (`train/async_train_universal.py`) and `MaskableRecurrentPPO.collect_rollouts` OR **true episode starts** with **focal-unit change** so `episode_starts` resets the RNN whenever `obs_focus_unit_tag` changes. Level0 single-unit: stable tag → same behavior as before. |

### Important: `command_target` type differences

- **Single-unit levels (Level0Bot):** `command_target` is a `Point2` object. In `ObservationTransformer._draw_command_info`, this triggers the default branch which draws `COMMAND_OBJECTIVE` with radius `r=3`.
- **Multi-unit levels (Level3Bot):** `command_target` is a beacon `Unit` object. This triggers the `is_beacon` branch. The radius for `COMMAND_OBJECTIVE` must be explicitly set to `max(3, int(1.5 * scale))` to match the effective size from single-unit levels.
- **Combat levels (Level7Bot):** `command_target` is a `Point2` (enemy position). Same default branch as single-unit.

If beacon markers appear too small for the CNN to detect (especially after Global Average Pooling), check the radius logic in `_draw_command_info`.

---

## Common pitfalls

1. **Layer ordering:** Never apply `sorted()` to channel indices during slicing. Config list order = channel order. Sorting scrambles the semantic alignment.
2. **Beacon marker size:** Beacon `Unit` objects and `Point2` targets follow different code paths in `_draw_command_info`. Verify both produce appropriately sized markers on the `COMMAND_OBJECTIVE` layer.
3. **Missing `high_level_command`:** Level-specific bots must include `high_level_command` in `raw_obs`. Without it, the command vector defaults to index 0 regardless of config.
4. **`debug_show_map()` is a toggle:** Calling it twice re-enables fog of war. Guard with a flag if used in `reset_game_state`.
5. **Observation space changes:** When a new level adds layers/heads, `Level0Env.reset()` rebuilds the observation space. The model's feature extractor input channels must match.
6. **Level 3 assignment:** Do not add mid-episode `get_optimal_assignments()` calls (e.g. in `_get_observation` when a beacon disappears) to “rebalance” targets. See **Level3Env / Level3Bot — unit-to-beacon assignment (CRITICAL CONTRACT)** above.
7. **Level 3 + recurrent PPO:** If you change rollout collection, keep **LSTM/GRU resets on focal-unit change** in sync with `get_obs_focus_unit_tag()`. Dropping that OR turns multi-unit cycling into cross-contaminated hidden state and breaks learning.
8. **Inherited / shared methods:** Editing base env or bot methods without checking **all callers and subclasses** often breaks other levels. See **Shared code, inheritance, and refactoring discipline** above; when in doubt, **override locally** or **add isolated code** instead of changing shared behavior.
9. **Universal actuation overrides:** Do not reintroduce hidden retargeting of move/attack orders. See **Agent-faithful actuation (universal) — no hidden “help”** above.
10. **Nested `env.action_heads` vs cumulative list:** `Level0Env` writes the **full cumulative** `action_heads` onto `training_conf` at the root. Bots must **not** prefer `training_conf['env']['action_heads']` over that flattened list, or command indices from the policy will not match `apply_action`. See **Universal action masking** table row `action_heads` on bots.

---

## Testing

```bash
pytest
```

Key test files:
- `tests/test_additive_config.py` — ConfigManager accumulation correctness
- `tests/test_level_3_universal_architecture.py` — Level3 obs/action wiring
- `tests/test_level_3_assignment_logic.py` — Hungarian pairing, collection rules, **episode-frozen assignment** (no mid-episode global re-match)
- `tests/test_vec_env_focus.py` — `get_obs_focus_unit_tag` / `vec_env_obs_focus_tags` for recurrent resets when cycling units
- `tests/test_level_7_env.py` — Level7 combat env
- `tests/test_level_7_mask_mismatch.py` — action mask shape validation
- `tests/test_sc2bot_action_heads_flattened_priority.py` — cumulative `action_heads` on `SC2Bot` must beat nested `env` lists
- `tests/test_level_7_tactical_attack_mask.py` — Level 7 universal: attack heads not masked by tactical fovea; spatial tail all-`True`

---

## TDD best practices

**Every code change must include corresponding test updates.** When adding a feature, add tests that cover it. When fixing a bug, add a regression test that reproduces the bug and verifies the fix. When refactoring, ensure existing tests still pass and update any that relied on the old structure. Do not submit changes without tests — if existing tests break due to your change, fix them as part of the same change.

**Always run and validate tests before considering a feature complete.** After implementing a change, execute the relevant tests with `pytest` and confirm they pass. A feature is not done until its tests run green. If tests fail, diagnose and fix before moving on.

### 1. Always mock ConfigManager — never read real YAML in unit tests

Configs change frequently. Tests that open `config/level_2.yaml` or `config_universal/level_move_multi_0.yaml` and assert specific values (reward scalars, step counts, layer counts) **will break** every time the config is tuned.

**Do this:** Patch `ConfigManager` at the import site and return controlled values.

```python
from unittest.mock import patch, MagicMock

def test_env_obs_space():
    mock_cm = MagicMock()
    mock_cm.get_cumulative_active_layers.return_value = ["layer_a", "layer_b", "layer_c"]
    mock_cm.get_cumulative_active_tactical_layers.return_value = ["layer_a", "layer_b"]
    mock_cm.get_cumulative_action_heads.return_value = ["move"]
    mock_cm.get_cumulative_categorical_heads.return_value = []
    mock_cm.get_agent_architecture.return_value = "universal"
    mock_cm.get_level_config.return_value = {"env": {"rewards": {}}}

    with patch("train.envs.level_0_env.ConfigManager", return_value=mock_cm), \
         patch("train.envs.level_0_env.Level0Env._discover_maps", return_value=["map.SC2Map"]):
        env = Level0Env(training_conf={...})
        assert env.observation_space["map"].shape[0] == 3  # matches mock, not YAML
```

**Don't do this:**

```python
def test_config_values():
    cm = ConfigManager("config")
    cfg = cm.get_level_config("level_2")
    assert cfg["training"]["n_steps"] == 2048      # breaks when tuned
    assert cfg["env"]["rewards"]["success"] == 50.0  # breaks when tuned
```

### 2. Use `tmp_path` for ConfigManager accumulation tests

When testing `ConfigManager` itself (accumulation order, dedup, off-pipeline levels), create synthetic YAML files under pytest's `tmp_path`. This isolates the test from any real pipeline changes.

```python
def test_cumulative_layers(tmp_path):
    (tmp_path / "pipeline.yaml").write_text("levels: [lvl0, lvl1]")
    (tmp_path / "lvl0.yaml").write_text("env:\n  active_layers: [A, B]")
    (tmp_path / "lvl1.yaml").write_text("env:\n  active_layers: [C]")

    cm = ConfigManager(str(tmp_path))
    assert cm.get_cumulative_active_layers("lvl1") == ["a", "b", "c"]
```

### 3. Assert structural properties, not magic numbers

Tests should verify that components **agree with each other**, not that they match a hard-coded constant that mirrors the current config.

**Good — assert consistency:**

```python
assert env.observation_space["map"].shape[0] == len(env.active_layers)
assert env.action_space.n == len(env.action_heads) * grid_size ** 2 + len(env.categorical_heads)
assert mask.shape[0] == env.action_space.n
```

**Bad — assert a snapshot of today's config:**

```python
assert env.observation_space.shape[0] == 23      # why 23? breaks when layers change
assert env.action_space.n == 32768               # breaks when grid or heads change
assert np.array_equal(env.action_space.nvec, [2, 1024])  # breaks when heads change
```

If a test needs a specific number, derive it from the mock config that the test itself defines, not from the real pipeline.

### 4. Keep fixtures self-contained

Every test fixture should define its own complete `training_conf` with all values the code under test needs. Don't rely on defaults that come from real YAML files.

```python
@pytest.fixture
def mock_training_conf():
    return {
        "grid_size": 128,
        "max_steps": 100,
        "config_dir": "config_universal",
        "level_name": "test_level",
        "active_layers": ["allied_unit_density", "enemy_unit_density"],
        "active_tactical_layers": ["allied_unit_density", "enemy_unit_density"],
        "action_heads": ["move"],
        "categorical_heads": [],
        "agent_architecture": "universal",
        "high_level_command": 0,
        "rewards": {"success": 10.0, "step_penalty": -0.1},
        "env": {"targets": 1, "maps_dir": "train/maps/128x128"},
    }
```

### 5. Common mock stack for environment tests

Most env tests need the same set of patches. Use this pattern:

```python
with patch("train.envs.level_0_env.ConfigManager", return_value=mock_cm), \
     patch("train.envs.level_0_env.SC2Process"), \
     patch("train.envs.level_0_env.Client"), \
     patch("train.envs.level_0_env.Level0Env._discover_maps", return_value=["test.SC2Map"]):
    env = SomeEnv(training_conf=mock_training_conf)
```

Consider extracting this into a shared fixture if many test files need it.

### 6. Test behavior, not config values

Tests should answer questions like:
- "Does the reward function compute correctly given these inputs?" (mock the inputs)
- "Does the observation space shape match the configured layers?" (mock the layer list)
- "Does the action mask have the right length for this action space?" (mock the heads)

Tests should NOT answer:
- "Does level_2.yaml have gamma=0.995?" (config validation, not behavior)
- "Does the CTF pipeline produce exactly 23 layers?" (snapshot, not invariant)

### 7. Existing tests that need refactoring

These tests currently depend on real config files and are known to be brittle:

| Test file | Issue |
|-----------|-------|
| `test_config_reward.py` | Reads `config/level_2.yaml`, asserts exact reward values and hyperparams |
| `test_level_7_config.py` | Reads `config/level_7.yaml`, asserts exact env_class, unit types |
| `test_capture_the_flag_single_env.py` | Asserts `observation_space.shape[0] == 23` from real pipeline |
| `test_fix_ctf_space_mismatch.py` | Asserts `action_space.n == 32768` from real pipeline |
| `test_multi_client_config_integration.py` | Asserts `config_dir == "config"` |
| `test_action_mask_shape.py` | Constructs env with real `config_universal` pipeline, prints instead of asserts |

When touching these files, refactor them to use mocked configs.
