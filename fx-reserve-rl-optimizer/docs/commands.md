# Command reference

```bash
cd fx-reserve-rl-optimizer
```

Install:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest tests/
```

Run dashboard:

```bash
streamlit run app/streamlit_app.py
```

Build processed data:

```bash
python -m src.data.build_dataset
```

Export benchmark results:

```bash
python -m src.agents.export_benchmark_results
```

Train PPO:

```bash
python -m src.agents.train_ppo --config configs/rl_config.yaml
```

Evaluate saved PPO:

```bash
python -m src.agents.evaluate_saved_model
```
