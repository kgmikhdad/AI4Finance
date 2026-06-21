# Pull request summary

## Added

A self-contained implementation of `fx-reserve-rl-optimizer` inside the AI4Finance monorepo.

## Main components

- Streamlit dashboard at `app/streamlit_app.py`
- Config files for assets, constraints, backtesting, and PPO
- Data pipeline with yfinance download and synthetic fallback
- Portfolio metrics, constraints, baselines, and backtester
- Gymnasium-compatible `FXReservePortfolioEnv`
- Stable-Baselines3 PPO training script
- Agent evaluation helper
- Stress scenario engine
- Pytest tests
- Deployment notes

## Local validation commands

```bash
cd fx-reserve-rl-optimizer
pip install -r requirements.txt
pytest tests/
streamlit run app/streamlit_app.py
```

## Notes

The Streamlit dashboard currently displays benchmark and stress-test results directly. PPO training is implemented offline through `src.agents.train_ppo`; trained model result visualisation can be added after saving evaluation outputs.
