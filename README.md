# NFL Predictions V2 — same dashboard, current weekly data

This is a conservative refresh-only iteration of the original project. **The two saved LightGBM models, probability/spread scoring logic, Monte Carlo threshold, display, and SQLite schema remain unchanged.** V2 adds a completed-game stats refresh ahead of dashboard scoring and a Wednesday GitHub Actions refresh.

## Original model artifacts — compatibility resolved

The original `win_model_lgbm.joblib` has now been supplied and installed in `artifacts/`. It is a LightGBM classifier expecting **394** features, matching the supplied `features.joblib` (394 feature names) and the LightGBM spread model (394 features). `python check_artifacts.py` passes. The earlier 298-feature logistic-regression `win_model.joblib` is not used.

Optional files from the original project:

- `backtest_predictions.csv` — maintains empirical Monte Carlo residual sampling.
- `backtest_summary.txt` — maintains original fallback margin RMSE; otherwise existing code defaults to 10.5.

**Important:** Artifact input compatibility is verified, but the updated weekly feature calculations have not been validated against the unavailable original feature-training script. The updater assumes within-season trailing rolling *means* of nflreadpy weekly team statistics. Confirm that assumption before relying on predictions or deploying scheduled updates.

## Quick start on a Mac (JupyterLab)

Double-click `start_jupyter.command` (or run `./start_jupyter.command` in Terminal). It creates a local `.venv`, installs project dependencies plus JupyterLab, checks model artifacts, and starts JupyterLab in your browser. Open `NFL_V2_Local_Workbench.ipynb`; its cells let you check artifacts, refresh weekly stats, and start the **unchanged Streamlit dashboard** at `http://localhost:8501`. Stop the notebook server with Ctrl+C in Terminal. An alternative shell launcher is `./start_jupyter.sh`.

If macOS blocks the first double-click, right-click the `.command` file and choose **Open**, or run `bash start_jupyter.command` from Terminal. Python 3.13 is preferred. Internet is required for first-time package installation and new NFL statistics.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python check_artifacts.py
streamlit run ML_predict_2.py
```

Press **Refresh predictions** in the existing sidebar. For the current season, the app first fetches nflverse current weekly team stats and completed-game schedule, appends previously unseen team-week rolling-feature snapshots to `artifacts/team_feature_store.parquet`, and then scores using **the original models**. The dashboard appearance is unchanged.

CLI equivalent:

```bash
python weekly_refresh.py --season 2026
```

## Push to GitHub

Create a new V2 repository or branch and commit this project's contents, after passing `python check_artifacts.py` and validating the weekly feature-calculation methodology. GitHub Actions is configured for Wednesdays during the NFL season, plus manual `workflow_dispatch`. Enable **Settings → Actions → General → Workflow permissions → Read and write permissions** so it can commit the refreshed Parquet. On Streamlit Community Cloud, configure `ML_predict_2.py` as the main entry point. If the workflow modifies a large tracked artifact frequently, consider Git LFS or an object store instead.

The GitHub runner's local SQLite data is ephemeral. The dashboard maintains its own instance-local database as before. GitHub Actions commits only the refreshed team feature Parquet, not the prediction history database.

## Model behavior and caveats

- **Not retraining:** Models do not refit weekly; they receive updated team features. This is exactly the limited change requested.
- **No new information leakage:** New feature-store rows are restricted to completed games. Scoring selects rows strictly before the selected game week. Earlier historical predictions should be treated as retrospective re-scores when refreshed, not immutable pregame records.
- **Rolling-stat assumption:** The original training/feature builder was NOT provided. The new updater uses per-team, within-season trailing 1-, 3-, and 6-game rolling **means** of nflreadpy weekly team stats; this matches the feature names, but cannot be proven identical to the original training transform. Verify with the original training script before relying on V2 outputs. The updater fails loudly for missing fields or partially published stats instead of silently substituting zeros.
- **Season rollover:** If the historical training builder used cross-season rolling windows, extra verification is necessary. The current updater calculates rolling windows within each season.
- **During active weeks:** Some games may not yet have finalized statistics; the updater refuses the incomplete update rather than publish potentially inconsistent features.
- **Schedule access:** The dashboard continues to use the original latest-started-week restriction. V2 deliberately does not change which weeks appear in the sidebar.
- **No guarantees of profit:** Model correctness and calibration require out-of-sample evaluation.

## Included files

- `ML_predict_2.py` — original Streamlit app with a single refresh hook.
- `ML_Predict.py` — original individual game helper, unchanged.
- `weekly_update.py` — schema-checked weekly team-stats refresh.
- `weekly_refresh.py` — CLI orchestrator.
- `.github/workflows/weekly_refresh.yml` — scheduled update and commit.
- `check_artifacts.py` — checks for missing artifacts and mismatched model feature dimensions.
- `artifacts/` — compatible original winner and spread models, feature list, original team feature store, metadata, and supplied local results.

## Publish the local notebook/dashboard changes to GitHub

V2 includes `publish_github.py` and a new **Publish to GitHub** notebook cell. This was missing from the earlier local workbench. For the first push, create an **empty repository** on GitHub (without a README or starter commit), then in Jupyter set `repo_url` to the repository's HTTPS or SSH git URL and run the final cell. It initializes the local repository, stages only listed project files, commits, and pushes. Future pushes: leave `repo_url` blank and run the cell again. Alternatively:

```bash
python publish_github.py --repo-url https://github.com/YOUR_USERNAME/YOUR_NEW_V2_REPO.git
# Later: python publish_github.py
```

**Streamlit deployment is separate from a Git push:** In Streamlit Community Cloud choose the GitHub repository, the pushed branch, and `ML_predict_2.py` as the entry point. Once connected, new commits to that branch trigger redeployment. `.github/workflows/weekly_refresh.yml` already commits updated team feature data on Wednesdays (requires GitHub Actions read/write permissions).

The publisher does not upload local SQLite prediction history, temporary CSV outputs, or the virtual environment. Verify your repo destination before your first push. Model binaries are sizeable; if GitHub blocks uploads due to size limits, consider Git LFS or external artifact hosting.
