from __future__ import annotations

import pandas as pd

from src.portfolio.metrics import portfolio_turnover


def backtest_weights(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Backtest a portfolio using beginning-of-period weights.

    Weights are shifted by one day so today's realised return uses yesterday's decision.
    """
    aligned_weights = weights.reindex(returns.index).ffill().dropna(how="any")
    aligned_returns = returns.loc[aligned_weights.index]
    decision_weights = aligned_weights.shift(1).dropna(how="any")
    realised_returns = aligned_returns.loc[decision_weights.index]
    gross = (decision_weights * realised_returns).sum(axis=1)
    turnover = portfolio_turnover(aligned_weights).reindex(gross.index).fillna(0.0)
    cost = turnover * transaction_cost_bps / 10000
    net = gross - cost
    return pd.DataFrame({"gross_return": gross, "turnover": turnover, "cost": cost, "net_return": net})


def cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()
