# Reward function

The RL reward is designed for institutional reserve-management intuition:

```text
reward = net_return
       - volatility_penalty * rolling_volatility
       - drawdown_penalty * abs(drawdown)
       - turnover_penalty * turnover
       - concentration_penalty * sum(weights^2)
```

The goal is to avoid a return-only trading objective and instead reward stable, diversified, low-turnover behaviour.
