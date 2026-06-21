# Codex continuation prompts

## 1. Replace README

Replace `README.md` with the full contents of `PROJECT_README.md`, preserving the project disclaimer and quickstart instructions.

## 2. Add PPO result loader

Modify the Streamlit dashboard so that if `reports/tables/ppo_returns.csv` and `reports/tables/ppo_weights.csv` exist, the app loads and displays PPO results alongside benchmark strategies.

## 3. Add CI

Create `.github/workflows/fx-reserve-optimizer-tests.yml` at the repository root to run tests only when `fx-reserve-rl-optimizer/**` changes.

## 4. Improve optimisation benchmarks

Replace the heuristic rolling mean-variance implementation with a constrained convex optimisation version using `scipy.optimize` or `cvxpy`.
