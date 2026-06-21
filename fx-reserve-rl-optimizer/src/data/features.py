from __future__ import annotations

import pandas as pd


def compute_features(returns: pd.DataFrame, windows: tuple[int, ...] = (21, 63)) -> pd.DataFrame:
    """Create lagged market features without look-ahead leakage."""
    frames: list[pd.DataFrame] = []
    for window in windows:
        frames.append(returns.rolling(window).mean().add_suffix(f"_mom_{window}"))
        frames.append(returns.rolling(window).std().add_suffix(f"_vol_{window}"))
    cumulative = (1 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    frames.append(drawdown.add_suffix("_drawdown"))
    features = pd.concat(frames, axis=1).shift(1).dropna(how="any")
    return features


def align_returns_features(returns: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align returns and features on their common date index."""
    common = returns.index.intersection(features.index)
    return returns.loc[common], features.loc[common]
