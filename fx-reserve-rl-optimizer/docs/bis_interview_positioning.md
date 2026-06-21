# BIS interview positioning

Use this framing:

> I implemented a public-data research prototype for stylised FX reserve portfolio optimisation. The goal was not to build a trading bot, but to study how dynamic allocation rules can be evaluated under institutional constraints such as safety, liquidity, diversification, turnover, drawdown, and stress resilience.

Strong points to mention:

- Transparent benchmarks are included before using RL.
- The custom Gymnasium environment makes the state-action-reward design explicit.
- PPO training is separated from the dashboard to keep deployment robust.
- The Streamlit app makes the project explainable and auditable.
