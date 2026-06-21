# Environment module

`FXReservePortfolioEnv` is a Gymnasium-compatible environment for stylised reserve portfolio allocation.

The action is a continuous vector that is projected into long-only constrained weights. The reward penalises volatility, drawdown, turnover, and concentration.
