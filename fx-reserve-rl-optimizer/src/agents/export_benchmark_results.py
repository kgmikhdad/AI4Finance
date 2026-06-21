from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.clean_data import clean_prices, compute_simple_returns
from src.data.download_data import download_prices
from src.data.features import compute_features
from src.portfolio.backtester import backtest_weights
from src.portfolio.baselines import equal_weight, rolling_mean_variance, rolling_minimum_variance, static_reserve_benchmark
from src.portfolio.constraints import weights_from_config
from src.portfolio.metrics import metrics_table
from src.utils.config import load_yaml


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    assets_cfg = load_yaml(root / "configs" / "assets.yaml")
    constraints_cfg = load_yaml(root / "configs" / "constraints.yaml")
    tickers = list(assets_cfg["tickers"].keys())
    prices = clean_prices(download_prices(tickers, assets_cfg.get("start_date", "2007-01-01"), assets_cfg.get("end_date")))
    returns = compute_simple_returns(prices)
    _features = compute_features(returns)
    max_weights = weights_from_config(tickers, constraints_cfg.get("max_weight"))
    transaction_cost_bps = float(constraints_cfg.get("transaction_cost_bps", 2.0))

    strategy_weights = {
        "Equal weight": equal_weight(returns),
        "Static reserve benchmark": static_reserve_benchmark(returns),
        "Rolling minimum variance": rolling_minimum_variance(returns, max_weights=max_weights),
        "Rolling mean-variance": rolling_mean_variance(returns, max_weights=max_weights),
    }
    backtests = {name: backtest_weights(returns, weights, transaction_cost_bps) for name, weights in strategy_weights.items()}
    strategy_returns = {name: result["net_return"] for name, result in backtests.items()}
    report_dir = root / "reports" / "tables"
    report_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(root / "data" / "processed" / "prices.csv", index_label="date")
    returns.to_csv(root / "data" / "processed" / "returns.csv", index_label="date")
    metrics_table(strategy_returns).to_csv(report_dir / "benchmark_metrics.csv")
    pd.DataFrame(strategy_returns).to_csv(report_dir / "benchmark_returns.csv", index_label="date")
    for name, weights in strategy_weights.items():
        safe = name.lower().replace(" ", "_").replace("-", "_")
        weights.to_csv(report_dir / f"weights_{safe}.csv", index_label="date")


if __name__ == "__main__":
    main()
