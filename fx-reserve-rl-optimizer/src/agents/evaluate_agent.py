from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_model(model, env, asset_names: list[str]) -> tuple[pd.Series, pd.DataFrame]:
    """Evaluate a trained Stable-Baselines3 model on an environment."""
    obs, _ = env.reset()
    done = False
    returns = []
    weights = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        returns.append(info["net_return"])
        weights.append(info["weights"])
    index = env.returns.index[: len(returns)]
    return pd.Series(returns, index=index, name="PPO"), pd.DataFrame(weights, index=index, columns=asset_names)


def evaluate_random_policy(env, asset_names: list[str], seed: int = 42) -> tuple[pd.Series, pd.DataFrame]:
    """Evaluate a random policy for environment smoke testing."""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    done = False
    returns = []
    weights = []
    while not done:
        action = rng.normal(size=len(asset_names))
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        returns.append(info["net_return"])
        weights.append(info["weights"])
    index = env.returns.index[: len(returns)]
    return pd.Series(returns, index=index, name="Random"), pd.DataFrame(weights, index=index, columns=asset_names)
