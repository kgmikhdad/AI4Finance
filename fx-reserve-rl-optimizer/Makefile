.PHONY: install test app export train

install:
	pip install -r requirements.txt

test:
	pytest tests/

app:
	streamlit run app/streamlit_app.py

export:
	python -m src.agents.export_benchmark_results

train:
	python -m src.agents.train_ppo --config configs/rl_config.yaml
