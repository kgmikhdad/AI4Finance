import numpy as np
import pandas as pd

from src.portfolio.baselines import equal_weight, rolling_minimum_variance, static_reserve_benchmark
from src.portfolio.constraints import validate_weights


def sample_returns():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.normal(0, 0.01, size=(100, 3)), index=idx, columns=["SHY", "IEF", "GLD"])


def test_equal_weight_valid():
    weights = equal_weight(sample_returns())
    assert validate_weights(weights)


def test_static_reserve_valid():
    weights = static_reserve_benchmark(sample_returns())
    assert validate_weights(weights)


def test_rolling_minimum_variance_valid():
    weights = rolling_minimum_variance(sample_returns(), window=20)
    assert validate_weights(weights)
