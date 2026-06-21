from __future__ import annotations

import numpy as np
import pandas as pd


def normalise_weights(raw: np.ndarray, max_weights: np.ndarray | None = None) -> np.ndarray:
    """Map raw continuous actions into long-only weights that sum to one.

    Uses a softmax transform followed by iterative clipping when max weights are supplied.
    """
    raw = np.asarray(raw, dtype="float64")
    raw = raw - np.nanmax(raw)
    weights = np.exp(raw)
    weights = weights / weights.sum()
    if max_weights is None:
        return weights
    return apply_max_weight_constraints(weights, np.asarray(max_weights, dtype="float64"))


def apply_max_weight_constraints(weights: np.ndarray, max_weights: np.ndarray) -> np.ndarray:
    """Project long-only weights into simple max-weight constraints."""
    weights = np.clip(np.asarray(weights, dtype="float64"), 0, None)
    max_weights = np.asarray(max_weights, dtype="float64")
    if max_weights.sum() < 1:
        raise ValueError("Sum of max_weights must be at least one.")
    weights = weights / weights.sum()
    for _ in range(20):
        over = weights > max_weights
        if not over.any():
            break
        excess = (weights[over] - max_weights[over]).sum()
        weights[over] = max_weights[over]
        under = ~over
        capacity = max_weights[under] - weights[under]
        if capacity.sum() <= 1e-12:
            break
        weights[under] += excess * capacity / capacity.sum()
    return weights / weights.sum()


def weights_from_config(asset_names: list[str], max_weight_config: dict[str, float] | None = None) -> np.ndarray:
    if not max_weight_config:
        return np.ones(len(asset_names))
    return np.array([max_weight_config.get(asset, 1.0) for asset in asset_names], dtype="float64")


def validate_weights(weights: pd.DataFrame | pd.Series, tolerance: float = 1e-6) -> bool:
    arr = weights.to_frame().T if isinstance(weights, pd.Series) else weights
    non_negative = (arr >= -tolerance).all().all()
    sums_to_one = np.allclose(arr.sum(axis=1).to_numpy(), 1.0, atol=tolerance)
    return bool(non_negative and sums_to_one)
