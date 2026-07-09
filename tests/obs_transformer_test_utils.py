"""Shared helpers for observation-transformer unit tests."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.layer_constants import LayerConstants


def transform_map(transformer, raw_obs: Dict[str, Any]) -> np.ndarray:
    """Run ``transform()`` and return the map tensor (full or pruned)."""
    result = transformer.transform(raw_obs)
    if isinstance(result, dict):
        return result["map"]
    return result


def get_layer(
    map_data: np.ndarray,
    name: str,
    active_layers: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Index a layer by semantic name.

    When ``active_layers`` is provided (or map channels are pruned), resolve the
    channel by config order. Otherwise use the global ``LayerConstants.LAYERS`` index.
    """
    key = name.lower()
    if active_layers is not None:
        indices = [
            i for i, ln in enumerate(active_layers) if ln.lower() == key
        ]
        if not indices:
            raise KeyError(f"Layer {name!r} not in active_layers {active_layers}")
        return map_data[indices[0]]

    num_global = LayerConstants.get_num_layers()
    if map_data.shape[0] < num_global:
        raise ValueError(
            f"Sliced map has {map_data.shape[0]} channels; pass active_layers to get_layer()"
        )
    return map_data[LayerConstants.LAYERS[name]]


def layer_sum(
    map_data: np.ndarray,
    name: str,
    active_layers: Optional[List[str]] = None,
) -> float:
    return float(np.sum(get_layer(map_data, name, active_layers)))


def layer_val(
    map_data: np.ndarray,
    name: str,
    x: int,
    y: int,
    active_layers: Optional[List[str]] = None,
) -> float:
    return float(get_layer(map_data, name, active_layers)[y, x])
