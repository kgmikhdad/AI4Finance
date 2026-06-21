# FX Reserve Portfolio Optimizer using Deep Reinforcement Learning

A research and educational prototype for **stylised foreign-exchange reserve portfolio optimisation** using constrained deep reinforcement learning and transparent portfolio benchmarks.

The project is designed as a central-bank-style decision-support application. It focuses on **safety, liquidity, return, drawdown control, turnover discipline, and stress resilience**.

## Project question

Can a constrained reinforcement-learning allocation rule dynamically allocate a stylised reserve portfolio across public market proxies while improving risk-adjusted performance relative to transparent benchmark strategies, without violating liquidity, concentration, turnover, and drawdown constraints?

## Implemented modules

- Public-data/synthetic fallback data pipeline
- Portfolio metrics: annualised return, volatility, Sharpe, Sortino, drawdown, VaR, expected shortfall, turnover
- Benchmarks: equal weight, static reserve benchmark, rolling minimum variance, rolling mean-variance
- Gymnasium-compatible `FXReservePortfolioEnv`
- Stable-Baselines3 PPO training and evaluation scripts
- Stress-scenario engine
- Streamlit dashboard
- Pytest smoke tests

## Local setup

```bash
cd fx-reserve-rl-optimizer
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run dashboard

```bash
streamlit run app/streamlit_app.py
```

## Train PPO

```bash
python -m src.agents.train_ppo --config configs/rl_config.yaml
```

## Streamlit deployment

Deploy from GitHub using this app file:

```text
fx-reserve-rl-optimizer/app/streamlit_app.py
```

If Streamlit Community Cloud does not detect nested dependencies, copy `fx-reserve-rl-optimizer/requirements.txt` to the repository root before deployment.

## Disclaimer

This project is a public-data research prototype. It does not represent the reserve portfolio, investment policy, internal systems, or recommendations of any central bank, the BIS, or any financial institution. It is not investment advice.
