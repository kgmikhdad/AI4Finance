from __future__ import annotations

import pandas as pd


def apply_parallel_shock(returns: pd.DataFrame, shock: dict[str, float]) -> pd.DataFrame:
    """Apply one-day additive shocks to selected asset returns."""
    stressed = returns.copy()
    for asset, value in shock.items():
        if asset in stressed.columns:
            stressed[asset] = stressed[asset] + value
    return stressed


def scenario_library() -> dict[str, dict[str, float]]:
    """Stylised reserve-management stress scenarios."""
    return {
        "Long-duration bond selloff": {"TLT": -0.06, "IEF": -0.025, "SHY": -0.005},
        "Gold correction": {"GLD": -0.08},
        "USD appreciation proxy": {"FXE": -0.04, "FXY": -0.04},
        "USD depreciation proxy": {"FXE": 0.04, "FXY": 0.04},
        "Flight to quality": {"SHY": 0.005, "IEF": 0.018, "TLT": 0.035, "GLD": 0.025},
    }


def stress_strategy(weights: pd.Series, scenarios: dict[str, dict[str, float]]) -> pd.Series:
    """Estimate one-day strategy P/L under scenario shocks."""
    results = {}
    for name, shock in scenarios.items():
        pnl = 0.0
        for asset, value in shock.items():
            if asset in weights.index:
                pnl += weights[asset] * value
        results[name] = pnl
    return pd.Series(results, name="scenario_return")
