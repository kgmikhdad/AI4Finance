from __future__ import annotations

import pandas as pd

from src.portfolio.backtester import cumulative_returns


def strategy_summary_frame(strategy_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Return cumulative wealth curves for multiple strategies."""
    return pd.DataFrame({name: cumulative_returns(returns) for name, returns in strategy_returns.items()})


def drawdown_frame(strategy_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Return drawdown curves for multiple strategies."""
    curves = strategy_summary_frame(strategy_returns)
    return curves / curves.cummax() - 1
