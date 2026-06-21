import numpy as np
import pandas as pd
import pytest

from src.data.features import compute_features, align_returns_features

pytest.importorskip("gymnasium")
from src.envs.fx_reserve_env import FXReservePortfolioEnv


def test_env_reset_and_step():
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(120, 3)), index=idx, columns=["A", "B", "C"])
    features = compute_features(returns, windows=(5, 10))
    returns, features = align_returns_features(returns, features)
    env = FXReservePortfolioEnv(returns, features, max_weights=np.array([0.6, 0.6, 0.6]))
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    obs, reward, terminated, truncated, info = env.step(np.array([0.1, 0.2, 0.3]))
    assert np.isfinite(reward)
    assert np.isclose(info["weights"].sum(), 1.0)
