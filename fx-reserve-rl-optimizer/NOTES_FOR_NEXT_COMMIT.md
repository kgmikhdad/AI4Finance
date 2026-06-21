# Notes for next commit

This MVP is intentionally conservative and interview-oriented.

## Known caveats

1. `README.md` was initially created as a short stub. The complete documentation is in `PROJECT_README.md` because the connector could not replace the stub without a blob SHA.
2. The Streamlit app currently computes benchmark results live using public data or synthetic fallback data.
3. PPO training is implemented as an offline script. The dashboard does not yet auto-load trained PPO evaluation CSVs.
4. The GitHub Actions workflow file was blocked by the connector, so CI instructions are documented in `DEPLOYMENT_NOTES.md`.

## Recommended next improvements

- Replace the root project README stub with the contents of `PROJECT_README.md`.
- Run `pytest tests/` locally.
- Run the Streamlit dashboard locally.
- Train PPO and export evaluation results.
- Add trained-agent visualisation to the dashboard.
- Deploy with Streamlit Community Cloud.
