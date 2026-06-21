from __future__ import annotations

from pathlib import Path

from src.data.clean_data import clean_prices, compute_simple_returns
from src.data.download_data import download_prices
from src.data.features import compute_features
from src.utils.config import load_yaml


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    assets_cfg = load_yaml(root / "configs" / "assets.yaml")
    tickers = list(assets_cfg["tickers"].keys())
    prices = clean_prices(download_prices(tickers, assets_cfg.get("start_date", "2007-01-01"), assets_cfg.get("end_date")))
    returns = compute_simple_returns(prices)
    features = compute_features(returns)
    out = root / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out / "prices.csv", index_label="date")
    returns.to_csv(out / "returns.csv", index_label="date")
    features.to_csv(out / "features.csv", index_label="date")
    print(f"Saved processed dataset to {out}")


if __name__ == "__main__":
    main()
