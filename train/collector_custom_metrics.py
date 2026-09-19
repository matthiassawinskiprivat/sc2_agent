"""Turn env ``info`` scalars into TensorBoard tags for the async collector."""

from collections import deque
from typing import Any, FrozenSet, Mapping, MutableMapping

import numpy as np

CUSTOM_METRIC_WINDOW = 100

# Rare per-step counts: a mean of the last 100 infos is almost always 0
# (one heal in a 200-step episode is ~0.005/step; TensorBoard rounds that to 0).
ROLLOUT_SUM_INFO_KEYS: FrozenSet[str] = frozenset(
    {
        "heal_casts",
        "heal_attempts",
        "move_commands",
        "attack_commands",
        "cloak_commands",
        "uncloak_commands",
        "cooldown_retreat",
    }
)


def record_info_scalars(
    custom_metrics: MutableMapping[str, deque],
    rollout_sums: MutableMapping[str, float],
    info: Mapping[str, Any],
    sum_keys: FrozenSet[str] = ROLLOUT_SUM_INFO_KEYS,
) -> None:
    """Append windowed means, or add rare counters into ``rollout_sums``."""
    for key, val in info.items():
        if not isinstance(val, (int, float, np.number)):
            continue
        if isinstance(val, (bool, np.bool_)):
            continue
        if key in sum_keys:
            rollout_sums[key] = rollout_sums.get(key, 0.0) + float(val)
            continue
        if key not in custom_metrics:
            custom_metrics[key] = deque(maxlen=CUSTOM_METRIC_WINDOW)
        custom_metrics[key].append(float(val))


def publish_custom_metrics(
    signal_meta: MutableMapping[str, Any],
    custom_metrics: Mapping[str, deque],
    rollout_sums: Mapping[str, float] | None = None,
) -> None:
    """Write ``env/<key>`` (or keep an existing ``/`` prefix) onto ``signal_meta``."""
    for key, vals in custom_metrics.items():
        if vals:
            prefix = "" if "/" in key else "env/"
            signal_meta[f"{prefix}{key}"] = float(np.mean(vals))
    if not rollout_sums:
        return
    for key, total in rollout_sums.items():
        prefix = "" if "/" in key else "env/"
        signal_meta[f"{prefix}{key}"] = float(total)
