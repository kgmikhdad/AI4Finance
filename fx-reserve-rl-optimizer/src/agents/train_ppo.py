from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.clean_data import clean_prices, compute_simple_returns
from src.data.download_data import download_prices
from src.data.features import align_returns_features, compute_features
from src.envs.fx_reserve_env import FXReservePortfolioEnv
from src.portfolio.constraints import weights_from_config
from src.utils.config import load_yaml
from src.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl_config.yaml")
    args = parser.parse_args()

    from stable_baselines3 import PPO

    root = Path(__file__).resolve().parents[2]
    assets_cfg = load_yaml(root / "configs" / "assets.yaml")
    constraints_cfg = load_yaml(root / "configs" / "constraints.yaml")
    rl_cfg = load_yaml(root / args.config)

    set_global_seed(int(rl_cfg.get("seed", 42)))
    tickers = list(assets_cfg["tickers"].keys())
    prices = clean_prices(download_prices(tickers, assets_cfg.get("start_date", "2007-01-01"), assets_cfg.get("end_date")))
    returns = compute_simple_returns(prices)
    features = compute_features(returns)
    returns, features = align_returns_features(returns, features)

    max_weights = weights_from_config(tickers, constraints_cfg.get("max_weight"))
    reward_cfg = rl_cfg.get("reward", {})
    env = FXReservePortfolioEnv(
        returns=returns,
        features=features,
        max_weights=max_weights,
        transaction_cost_bps=float(constraints_cfg.get("transaction_cost_bps", 2.0)),
        volatility_penalty=float(reward_cfg.get("volatility_penalty", 0.25)),
        drawdown_penalty=float(reward_cfg.get("drawdown_penalty", 0.75)),
        turnover_penalty=float(reward_cfg.get("turnover_penalty", 0.10)),
        concentration_penalty=float(reward_cfg.get("concentration_penalty", 0.05)),
    )

    model = PPO(
        rl_cfg.get("policy", "MlpPolicy"),
        env,
        learning_rate=float(rl_cfg.get("learning_rate", 0.0003)),
        gamma=float(rl_cfg.get("gamma", 0.99)),
        seed=int(rl_cfg.get("seed", 42)),
        verbose=1,
    )
    model.learn(total_timesteps=int(rl_cfg.get("total_timesteps", 50000)))

    model_dir = root / "models" / "ppo"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "ppo_fx_reserve")

    data_dir = root / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(data_dir / "prices.csv", index_label="date")
    returns.to_csv(data_dir / "returns.csv", index_label="date")
    features.to_csv(data_dir / "features.csv", index_label="date")

    print(f"Saved model to {model_dir / 'ppo_fx_reserve.zip'}")


if __name__ == "__main__":
    main()
