from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio.constraints import apply_max_weight_constraints


def equal_weight(returns: pd.DataFrame) -> pd.DataFrame:
    n = returns.shape[1]
    weights = np.repeat(1 / n, n)
    return pd.DataFrame(np.tile(weights, (len(returns), 1)), index=returns.index, columns=returns.columns)


def static_reserve_benchmark(returns: pd.DataFrame) -> pd.DataFrame:
    template = {"SHY": 0.40, "IEF": 0.25, "TLT": 0.15, "GLD": 0.10, "FXE": 0.05, "FXY": 0.05}
    raw = np.array([template.get(col, 1 / returns.shape[1]) for col in returns.columns], dtype="float64")
    raw = raw / raw.sum()
    return pd.DataFrame(np.tile(raw, (len(returns), 1)), index=returns.index, columns=returns.columns)


def rolling_minimum_variance(
    returns: pd.DataFrame,
    window: int = 63,
    max_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    weights = []
    for i in range(len(returns)):
        if i < window:
            w = np.repeat(1 / returns.shape[1], returns.shape[1])
        else:
            cov = returns.iloc[i - window : i].cov().to_numpy()
            inv_diag = 1 / np.maximum(np.diag(cov), 1e-8)
            w = inv_diag / inv_diag.sum()
        if max_weights is not None:
            w = apply_max_weight_constraints(w, max_weights)
        weights.append(w)
    return pd.DataFrame(weights, index=returns.index, columns=returns.columns)


def rolling_mean_variance(
    returns: pd.DataFrame,
    window: int = 63,
    risk_aversion: float = 5.0,
    max_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    weights = []
    for i in range(len(returns)):
        if i < window:
            w = np.repeat(1 / returns.shape[1], returns.shape[1])
        else:
            hist = returns.iloc[i - window : i]
            mu = hist.mean().to_numpy()
            vol = np.maximum(hist.std().to_numpy(), 1e-8)
            score = mu / (risk_aversion * vol**2)
            score = np.maximum(score, 0)
            w = score / score.sum() if score.sum() > 0 else np.repeat(1 / returns.shape[1], returns.shape[1])
        if max_weights is not None:
            w = apply_max_weight_constraints(w, max_weights)
        weights.append(w)
    return pd.DataFrame(weights, index=returns.index, columns=returns.columns)
