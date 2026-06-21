# Quickstart

```bash
cd fx-reserve-rl-optimizer
make install
make test
make app
```

Alternative without Make:

```bash
cd fx-reserve-rl-optimizer
pip install -r requirements.txt
pytest tests/
streamlit run app/streamlit_app.py
```

Train the PPO model:

```bash
python -m src.agents.train_ppo --config configs/rl_config.yaml
```

Export benchmark results:

```bash
python -m src.agents.export_benchmark_results
```
