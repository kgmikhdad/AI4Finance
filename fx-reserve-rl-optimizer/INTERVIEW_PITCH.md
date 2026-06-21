# Interview pitch

I built this project as a central-bank-style decision-support prototype rather than a trading bot. The model studies how dynamic allocation rules behave in a stylised FX reserve portfolio under reserve-management constraints.

## Core framing

The objective is not simply return maximisation. The allocation rule is evaluated against safety, liquidity, return, drawdown, turnover, concentration, and stress-test criteria.

## Technical components

- Data pipeline using public proxy assets
- Transparent benchmarks
- Custom Gymnasium environment
- PPO-based reinforcement-learning training script
- Portfolio-risk dashboard in Streamlit
- Stress-test engine

## Main explanation

The reinforcement-learning agent observes lagged market features and current portfolio state, chooses a continuous allocation vector, and receives a reward based on net return penalised by volatility, drawdown, turnover, and concentration. The final system compares this RL allocation framework against equal-weight, static reserve, minimum-variance, and mean-variance benchmarks.

## Caveat

The implementation is a public-data research prototype and does not represent actual BIS or central-bank reserve management.
