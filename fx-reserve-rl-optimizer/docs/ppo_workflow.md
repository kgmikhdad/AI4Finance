# PPO workflow

1. Build dataset.
2. Train PPO.
3. Evaluate saved PPO model.
4. Load exported PPO returns and weights in dashboard in a future enhancement.

Commands:

```bash
python -m src.data.build_dataset
python -m src.agents.train_ppo --config configs/rl_config.yaml
python -m src.agents.evaluate_saved_model
```
