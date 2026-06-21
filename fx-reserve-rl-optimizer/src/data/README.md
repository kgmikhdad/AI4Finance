# Data module

The data pipeline downloads public proxy assets through yfinance when available and falls back to deterministic synthetic prices if the download fails. This keeps the dashboard usable in offline or restricted environments.
