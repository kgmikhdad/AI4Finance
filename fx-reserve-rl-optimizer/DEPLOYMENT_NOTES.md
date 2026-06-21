# Deployment notes

## Streamlit Community Cloud

Use the following entrypoint when deploying from this monorepo:

```text
fx-reserve-rl-optimizer/app/streamlit_app.py
```

If Streamlit Community Cloud does not detect nested dependency files, copy this file to the repository root before deployment:

```text
fx-reserve-rl-optimizer/requirements.txt
```

## Local commands

```bash
cd fx-reserve-rl-optimizer
pip install -r requirements.txt
pytest tests/
streamlit run app/streamlit_app.py
```

## Optional CI workflow

The GitHub connector blocked direct creation of a nested workflow file during implementation. Add this manually later if desired:

```yaml
name: FX Reserve Optimizer Tests
on:
  push:
    paths:
      - 'fx-reserve-rl-optimizer/**'
  pull_request:
    paths:
      - 'fx-reserve-rl-optimizer/**'
jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: fx-reserve-rl-optimizer
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/
      - run: ruff check src app tests
```
