from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None
    spaces = None

from src.portfolio.constraints import normalise_weights


class FXReservePortfolioEnv(gym.Env if gym else object):
    """Constrained portfolio allocation environment for stylised FX reserves."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: pd.DataFrame,
        features: pd.DataFrame,
        max_weights: np.ndarray | None = None,
        transaction_cost_bps: float = 2.0,
        volatility_penalty: float = 0.25,
        drawdown_penalty: float = 0.75,
        turnover_penalty: float = 0.10,
        concentration_penalty: float = 0.05,
    ) -> None:
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required to use FXReservePortfolioEnv.")
        common = returns.index.intersection(features.index)
        self.returns = returns.loc[common].astype("float32")
        self.features = features.loc[common].astype("float32")
        self.asset_names = list(self.returns.columns)
        self.n_assets = len(self.asset_names)
        self.max_weights = np.ones(self.n_assets) if max_weights is None else np.asarray(max_weights, dtype="float64")
        self.transaction_cost_bps = transaction_cost_bps
        self.volatility_penalty = volatility_penalty
        self.drawdown_penalty = drawdown_penalty
        self.turnover_penalty = turnover_penalty
        self.concentration_penalty = concentration_penalty
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.n_assets,), dtype=np.float32)
        obs_size = self.features.shape[1] + self.n_assets + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.current_step = 0
        self.weights = np.repeat(1 / self.n_assets, self.n_assets)
        self.wealth = 1.0
        self.peak_wealth = 1.0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.weights = np.repeat(1 / self.n_assets, self.n_assets)
        self.wealth = 1.0
        self.peak_wealth = 1.0
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        new_weights = normalise_weights(action, self.max_weights)
        asset_returns = self.returns.iloc[self.current_step].to_numpy(dtype="float64")
        gross_return = float(np.dot(self.weights, asset_returns))
        turnover = float(np.abs(new_weights - self.weights).sum())
        cost = turnover * self.transaction_cost_bps / 10000
        net_return = gross_return - cost
        self.wealth *= 1 + net_return
        self.peak_wealth = max(self.peak_wealth, self.wealth)
        drawdown = min(self.wealth / self.peak_wealth - 1, 0.0)
        rolling_slice = self.returns.iloc[max(0, self.current_step - 20) : self.current_step + 1]
        volatility = float(rolling_slice.dot(self.weights).std()) if len(rolling_slice) > 1 else 0.0
        concentration = float(np.sum(new_weights**2))
        reward = (
            net_return
            - self.volatility_penalty * volatility
            - self.drawdown_penalty * abs(drawdown)
            - self.turnover_penalty * turnover
            - self.concentration_penalty * concentration
        )
        self.weights = new_weights
        self.current_step += 1
        terminated = self.current_step >= len(self.returns) - 1
        truncated = False
        info = {
            "gross_return": gross_return,
            "net_return": net_return,
            "turnover": turnover,
            "cost": cost,
            "drawdown": drawdown,
            "weights": self.weights.copy(),
            "wealth": self.wealth,
        }
        return self._get_observation(), float(reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        feature_vec = self.features.iloc[self.current_step].to_numpy(dtype="float32")
        drawdown = np.array([self.wealth / self.peak_wealth - 1], dtype="float32")
        wealth = np.array([self.wealth], dtype="float32")
        return np.concatenate([feature_vec, self.weights.astype("float32"), drawdown, wealth]).astype("float32")
