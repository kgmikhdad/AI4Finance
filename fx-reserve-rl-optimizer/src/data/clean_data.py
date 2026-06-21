from __future__ import annotations

import pandas as pd


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Clean, sort, forward-fill, and align asset prices."""
    cleaned = prices.copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned = cleaned.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.replace([float("inf"), float("-inf")], pd.NA)
    cleaned = cleaned.ffill().dropna(how="any")
    cleaned = cleaned.loc[:, cleaned.nunique(dropna=True) > 1]
    return cleaned


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple percentage returns."""
    return prices.pct_change().dropna(how="any")


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns."""
    import numpy as np

    return np.log(prices / prices.shift(1)).dropna(how="any")
