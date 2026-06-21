from __future__ import annotations

import numpy as np
import pandas as pd

ArrayLike = pd.Series | np.ndarray | list[float]


def _to_series(returns: ArrayLike) -> pd.Series:
    if isinstance(returns, pd.Series):
        return returns.dropna()
    return pd.Series(returns, dtype="float64").dropna()


def annualized_return(returns: ArrayLike, periods_per_year: int = 252) -> float:
    r = _to_series(returns)
    if r.empty:
        return float("nan")
    return float((1 + r).prod() ** (periods_per_year / len(r)) - 1)


def annualized_volatility(returns: ArrayLike, periods_per_year: int = 252) -> float:
    r = _to_series(returns)
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: ArrayLike, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    r = _to_series(returns)
    excess = r - risk_free_rate / periods_per_year
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def sortino_ratio(returns: ArrayLike, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    r = _to_series(returns)
    excess = r - risk_free_rate / periods_per_year
    downside = excess[excess < 0].std(ddof=1)
    if downside == 0 or np.isnan(downside):
        return 0.0
    return float(excess.mean() / downside * np.sqrt(periods_per_year))


def max_drawdown(returns: ArrayLike) -> float:
    r = _to_series(returns)
    wealth = (1 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return float(drawdown.min())


def calmar_ratio(returns: ArrayLike, periods_per_year: int = 252) -> float:
    dd = abs(max_drawdown(returns))
    if dd == 0:
        return 0.0
    return annualized_return(returns, periods_per_year) / dd


def historical_var(returns: ArrayLike, alpha: float = 0.05) -> float:
    r = _to_series(returns)
    return float(r.quantile(alpha))


def expected_shortfall(returns: ArrayLike, alpha: float = 0.05) -> float:
    r = _to_series(returns)
    threshold = historical_var(r, alpha)
    tail = r[r <= threshold]
    return float(tail.mean()) if not tail.empty else threshold


def portfolio_turnover(weights: pd.DataFrame) -> pd.Series:
    return weights.diff().abs().sum(axis=1).fillna(0.0)


def metrics_table(strategy_returns: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, returns in strategy_returns.items():
        rows.append(
            {
                "strategy": name,
                "annualized_return": annualized_return(returns),
                "annualized_volatility": annualized_volatility(returns),
                "sharpe": sharpe_ratio(returns),
                "sortino": sortino_ratio(returns),
                "max_drawdown": max_drawdown(returns),
                "calmar": calmar_ratio(returns),
                "var_95": historical_var(returns),
                "expected_shortfall_95": expected_shortfall(returns),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")
