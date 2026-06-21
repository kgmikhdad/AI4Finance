from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def download_prices(
    tickers: list[str],
    start_date: str = "2007-01-01",
    end_date: str | None = None,
    fallback_if_error: bool = True,
) -> pd.DataFrame:
    """Download adjusted prices from yfinance, with a deterministic synthetic fallback.

    The fallback keeps the dashboard and tests usable when internet access is unavailable.
    """
    try:
        import yfinance as yf

        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"].copy()
        else:
            prices = data[["Close"]].copy()
            prices.columns = tickers
        prices = prices.dropna(how="all")
        if prices.empty:
            raise ValueError("No price data returned from yfinance.")
        return prices.ffill().dropna()
    except Exception:
        if not fallback_if_error:
            raise
        return generate_synthetic_prices(tickers=tickers, start_date=start_date, end_date=end_date)


def generate_synthetic_prices(
    tickers: list[str],
    start_date: str = "2007-01-01",
    end_date: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate stylised reserve-asset price paths for demos and tests."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize() if end_date is None else pd.Timestamp(end_date)
    dates = pd.bdate_range(start=start_date, end=end)
    n = len(dates)
    drift = np.array([0.012, 0.018, 0.022, 0.035, 0.006, 0.004])[: len(tickers)] / 252
    vol = np.array([0.025, 0.055, 0.13, 0.16, 0.09, 0.10])[: len(tickers)] / np.sqrt(252)
    shocks = rng.normal(drift, vol, size=(n, len(tickers)))
    prices = 100 * np.exp(np.cumsum(shocks, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def save_prices(prices: pd.DataFrame, path: str | Path) -> None:
    """Save price data to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index_label="date")
