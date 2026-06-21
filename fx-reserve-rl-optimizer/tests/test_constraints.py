import numpy as np
import pandas as pd

from src.portfolio.constraints import normalise_weights, validate_weights


def test_normalise_weights_sum_to_one():
    weights = normalise_weights(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()


def test_normalise_weights_respects_max_weights():
    weights = normalise_weights(np.array([10.0, 1.0, 1.0]), np.array([0.5, 0.5, 0.5]))
    assert np.isclose(weights.sum(), 1.0)
    assert weights.max() <= 0.500001


def test_validate_weights():
    df = pd.DataFrame({"A": [0.5, 0.4], "B": [0.5, 0.6]})
    assert validate_weights(df)
