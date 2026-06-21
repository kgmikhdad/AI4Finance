from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.clean_data import clean_prices, compute_simple_returns
from src.data.download_data import download_prices
from src.data.features import compute_features
from src.portfolio.backtester import backtest_weights, cumulative_returns
from src.portfolio.baselines import equal_weight, rolling_mean_variance, rolling_minimum_variance, static_reserve_benchmark
from src.portfolio.constraints import weights_from_config
from src.portfolio.metrics import metrics_table, portfolio_turnover
from src.stress.stress_scenarios import scenario_library, stress_strategy
from src.utils.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="FX Reserve Portfolio Optimizer", layout="wide")
st.title("FX Reserve Portfolio Optimizer")
st.caption("Deep reinforcement learning decision-support prototype for stylised reserve allocation")

st.info(
    "This is a public-data research prototype. It is not investment advice and does not represent "
    "any central-bank, BIS, or institutional reserve portfolio."
)


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict]:
    assets_cfg = load_yaml(ROOT / "configs" / "assets.yaml")
    constraints_cfg = load_yaml(ROOT / "configs" / "constraints.yaml")
    tickers = list(assets_cfg["tickers"].keys())
    prices = clean_prices(download_prices(tickers, assets_cfg.get("start_date", "2007-01-01"), assets_cfg.get("end_date")))
    returns = compute_simple_returns(prices)
    features = compute_features(returns)
    return prices, returns, features, tickers, constraints_cfg


prices, returns, features, tickers, constraints_cfg = load_data()
max_weights = weights_from_config(tickers, constraints_cfg.get("max_weight"))
transaction_cost_bps = st.sidebar.slider("Transaction cost, bps", 0.0, 25.0, float(constraints_cfg.get("transaction_cost_bps", 2.0)), 0.5)
window = st.sidebar.slider("Rolling optimisation window", 21, 252, 63, 21)

weights = {
    "Equal weight": equal_weight(returns),
    "Static reserve benchmark": static_reserve_benchmark(returns),
    "Rolling minimum variance": rolling_minimum_variance(returns, window=window, max_weights=max_weights),
    "Rolling mean-variance": rolling_mean_variance(returns, window=window, max_weights=max_weights),
}
backtests = {name: backtest_weights(returns, w, transaction_cost_bps) for name, w in weights.items()}
strategy_returns = {name: bt["net_return"] for name, bt in backtests.items()}
metric_df = metrics_table(strategy_returns)

tab_overview, tab_data, tab_benchmarks, tab_risk, tab_stress, tab_model = st.tabs(
    ["Overview", "Data", "Benchmarks", "Risk", "Stress tests", "Model design"]
)

with tab_overview:
    st.subheader("Project objective")
    st.write(
        "The app compares transparent reserve-style portfolio benchmarks under constraints on "
        "concentration, turnover, drawdown, and risk. The reinforcement-learning layer is implemented "
        "in the repository and can be trained offline using Stable-Baselines3 PPO."
    )
    st.code("python -m src.agents.train_ppo --config configs/rl_config.yaml", language="bash")
    st.dataframe(metric_df.style.format("{:.4f}"))

with tab_data:
    st.subheader("Asset prices")
    st.plotly_chart(px.line(prices, title="Public proxy asset prices"), use_container_width=True)
    st.subheader("Return correlation")
    corr = returns.corr()
    st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", title="Daily return correlation"), use_container_width=True)

with tab_benchmarks:
    st.subheader("Cumulative performance")
    curve = pd.DataFrame({name: cumulative_returns(ret) for name, ret in strategy_returns.items()})
    st.plotly_chart(px.line(curve, title="Cumulative net returns"), use_container_width=True)
    selected = st.selectbox("Inspect benchmark weights", list(weights.keys()))
    st.plotly_chart(px.area(weights[selected], title=f"Portfolio weights: {selected}"), use_container_width=True)

with tab_risk:
    st.subheader("Risk metrics")
    st.dataframe(metric_df.style.format("{:.4f}"))
    dd = pd.DataFrame()
    for name, ret in strategy_returns.items():
        wealth = cumulative_returns(ret)
        dd[name] = wealth / wealth.cummax() - 1
    st.plotly_chart(px.line(dd, title="Drawdown comparison"), use_container_width=True)
    turnover = pd.DataFrame({name: portfolio_turnover(w) for name, w in weights.items()})
    st.plotly_chart(px.line(turnover.rolling(21).mean(), title="21-day rolling turnover"), use_container_width=True)

with tab_stress:
    st.subheader("Stylised one-day stress scenarios")
    scenarios = scenario_library()
    latest_weights = {name: w.iloc[-1] for name, w in weights.items()}
    stress_df = pd.DataFrame({name: stress_strategy(w, scenarios) for name, w in latest_weights.items()})
    st.dataframe(stress_df.style.format("{:.2%}"))
    st.plotly_chart(px.bar(stress_df, barmode="group", title="Scenario return by strategy"), use_container_width=True)

with tab_model:
    st.subheader("Reinforcement-learning environment")
    st.markdown(
        """
The implemented `FXReservePortfolioEnv` follows the standard Gymnasium pattern:

- **State**: lagged returns, rolling momentum, volatility, drawdown, current weights, wealth state.
- **Action**: continuous raw vector transformed into long-only portfolio weights.
- **Reward**: portfolio return minus penalties for volatility, drawdown, turnover, and concentration.
- **Constraints**: long-only allocation and maximum asset weights.

The app is intentionally separated from model training. Train PPO offline, save the model/results, and use the dashboard for analysis.
        """
    )
    st.code(
        "Reward = return - vol_penalty - drawdown_penalty - turnover_penalty - concentration_penalty",
        language="text",
    )
