# Architecture

```text
Data -> Features -> Benchmarks -> Backtests -> Metrics -> Streamlit
              |-> Gymnasium Environment -> PPO Agent -> Evaluation Outputs
```

The project separates model training from the dashboard. This avoids expensive training during Streamlit runtime and keeps deployment lightweight.
