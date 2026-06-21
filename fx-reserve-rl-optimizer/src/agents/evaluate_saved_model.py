from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO

from src.agents.evaluate_agent import evaluate_model
from src.data.clean_data import clean_prices, compute_simple_returns
from src.data.download_data import download_prices
from src.data.features import align_returns_features, compute_features
from src.envs.fx_reserve_env import FXReservePortfolioEnv
from src.portfolio.constraints import weights_from_config
from src.utils.config import load_yaml


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    assets_cfg = load_yaml(root / "configs" / "assets.yaml")
    constraints_cfg = load_yaml(root / "configs" / "constraints.yaml")
    tickers = list(assets_cfg["tickers"].keys())
    prices = clean_prices(download_prices(tickers, assets_cfg.get("start_date", "2007-01-01"), assets_cfg.get("end_date")))
    returns = compute_simple_returns(prices)
    features = compute_features(returns)
    returns, features = align_returns_features(returns, features)
    max_weights = weights_from_config(tickers, constraints_cfg.get("max_weight"))
    env = FXReservePortfolioEnv(returns, features, max_weights=max_weights)
    model = PPO.load(root / "models" / "ppo" / "ppo_fx_reserve")
    ppo_returns, ppo_weights = evaluate_model(model, env, tickers)
    out = root / "reports" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    ppo_returns.to_csv(out / "ppo_returns.csv", index_label="date")
    ppo_weights.to_csv(out / "ppo_weights.csv", index_label="date")
    print(f"Saved PPO evaluation outputs to {out}")


if __name__ == "__main__":
    main()
