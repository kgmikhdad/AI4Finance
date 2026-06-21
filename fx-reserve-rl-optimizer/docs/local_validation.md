# Local validation

Run:

```bash
cd fx-reserve-rl-optimizer
pip install -r requirements.txt
pytest tests/
streamlit run app/streamlit_app.py
```

Then train PPO only after the dashboard works:

```bash
python -m src.agents.train_ppo --config configs/rl_config.yaml
```
