import pandas as pd

from src.portfolio.metrics import annualized_volatility, max_drawdown, portfolio_turnover, sharpe_ratio


def test_sharpe_ratio_is_finite():
    returns = pd.Series([0.01, -0.005, 0.002, 0.003])
    assert isinstance(sharpe_ratio(returns), float)


def test_max_drawdown_negative_or_zero():
    returns = pd.Series([0.1, -0.2, 0.05])
    assert max_drawdown(returns) <= 0


def test_annualized_volatility_non_negative():
    returns = pd.Series([0.01, -0.005, 0.002, 0.003])
    assert annualized_volatility(returns) >= 0


def test_portfolio_turnover():
    weights = pd.DataFrame({"A": [0.5, 0.6], "B": [0.5, 0.4]})
    turnover = portfolio_turnover(weights)
    assert turnover.iloc[0] == 0
    assert round(turnover.iloc[1], 6) == 0.2
