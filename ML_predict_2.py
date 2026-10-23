from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import nflreadpy as nfl
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'nflreadpy'. Install it in this environment before running ML_predict_2.py."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DB_FILE = ARTIFACT_DIR / "predictions_warehouse.sqlite"

WIN_MODEL_FILE = ARTIFACT_DIR / "win_model_lgbm.joblib"
SPREAD_MODEL_FILE = ARTIFACT_DIR / "spread_model_lgbm.joblib"
FEATURES_FILE = ARTIFACT_DIR / "features.joblib"
TEAM_STORE_FILE = ARTIFACT_DIR / "team_feature_store.parquet"
BACKTEST_PRED_FILE = ARTIFACT_DIR / "backtest_predictions.csv"
SUMMARY_FILE = ARTIFACT_DIR / "backtest_summary.txt"

MONTE_CARLO_THRESHOLD = 0.60
DEFAULT_N_SIMS = 200_000


TEAM_MAP = {
    "buf": "Buffalo Bills",
    "mia": "Miami Dolphins",
    "ne": "New England Patriots",
    "nwe": "New England Patriots",
    "nyj": "New York Jets",
    "bal": "Baltimore Ravens",
    "cin": "Cincinnati Bengals",
    "cle": "Cleveland Browns",
    "pit": "Pittsburgh Steelers",
    "hou": "Houston Texans",
    "ind": "Indianapolis Colts",
    "jax": "Jacksonville Jaguars",
    "ten": "Tennessee Titans",
    "den": "Denver Broncos",
    "kc": "Kansas City Chiefs",
    "kan": "Kansas City Chiefs",
    "lv": "Las Vegas Raiders",
    "oak": "Las Vegas Raiders",
    "sd": "Los Angeles Chargers",
    "lac": "Los Angeles Chargers",
    "dal": "Dallas Cowboys",
    "nyg": "New York Giants",
    "phi": "Philadelphia Eagles",
    "wsh": "Washington Commanders",
    "was": "Washington Commanders",
    "chi": "Chicago Bears",
    "det": "Detroit Lions",
    "gb": "Green Bay Packers",
    "gnb": "Green Bay Packers",
    "min": "Minnesota Vikings",
    "atl": "Atlanta Falcons",
    "car": "Carolina Panthers",
    "no": "New Orleans Saints",
    "nor": "New Orleans Saints",
    "tb": "Tampa Bay Buccaneers",
    "tam": "Tampa Bay Buccaneers",
    "ari": "Arizona Cardinals",
    "arz": "Arizona Cardinals",
    "lar": "Los Angeles Rams",
    "la": "Los Angeles Rams",
    "stl": "Los Angeles Rams",
    "sf": "San Francisco 49ers",
    "sfo": "San Francisco 49ers",
    "sea": "Seattle Seahawks",
}


def canonicalize_team(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip().lower().replace(".", "").replace(" ", "")
    if s in TEAM_MAP:
        return TEAM_MAP[s]
    raw = str(x).strip()
    if " " in raw and len(raw) > 6:
        return raw
    return None


def require_artifacts() -> None:
    missing = [
        str(path)
        for path in [WIN_MODEL_FILE, SPREAD_MODEL_FILE, FEATURES_FILE, TEAM_STORE_FILE]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required model artifacts:\n" + "\n".join(missing))


def load_artifacts() -> tuple[object, object, list[str], pd.DataFrame]:
    require_artifacts()
    win_model = joblib.load(WIN_MODEL_FILE)
    spread_model = joblib.load(SPREAD_MODEL_FILE)
    features = joblib.load(FEATURES_FILE)
    team_store = pd.read_parquet(TEAM_STORE_FILE)
    return win_model, spread_model, features, team_store


def default_dashboard_season() -> int:
    if SUMMARY_FILE.exists():
        try:
            meta_path = ARTIFACT_DIR / "training_meta.json"
            if meta_path.exists():
                return int(json.loads(meta_path.read_text()).get("end_season", datetime.now().year))
        except Exception:
            pass
    return datetime.now().year


def init_prediction_warehouse(db_path: Path = DB_FILE) -> None:
    db_path.parent.mkdir(exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weekly_predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                game_id TEXT,
                season_type TEXT,
                gameday TEXT,
                gametime TEXT,
                home_full TEXT NOT NULL,
                away_full TEXT NOT NULL,
                home_win_prob REAL NOT NULL,
                away_win_prob REAL NOT NULL,
                favorite_team TEXT NOT NULL,
                favorite_win_prob REAL NOT NULL,
                under_threshold INTEGER NOT NULL,
                pred_home_margin REAL NOT NULL,
                confidence REAL NOT NULL,
                monte_carlo_run INTEGER NOT NULL,
                monte_carlo_sims INTEGER,
                sim_home_win_prob REAL,
                sim_away_win_prob REAL,
                sim_home_margin_mean REAL,
                sim_home_margin_std REAL,
                sim_home_margin_p05 REAL,
                sim_home_margin_p50 REAL,
                sim_home_margin_p95 REAL,
                p_home_wins_by_7plus REAL,
                p_away_wins_by_7plus REAL,
                features_asof_home_season INTEGER NOT NULL,
                features_asof_home_week INTEGER NOT NULL,
                features_asof_away_season INTEGER NOT NULL,
                features_asof_away_week INTEGER NOT NULL,
                UNIQUE(season, week, home_full, away_full)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_weekly_predictions_week ON weekly_predictions(season, week)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_run_metrics (
                run_id TEXT PRIMARY KEY,
                created_at_utc TEXT NOT NULL,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                games_predicted INTEGER NOT NULL,
                games_final INTEGER NOT NULL,
                model_wins INTEGER NOT NULL,
                model_losses INTEGER NOT NULL,
                pushes INTEGER NOT NULL,
                pick_accuracy REAL,
                auc REAL,
                threshold REAL NOT NULL,
                monte_carlo_sims INTEGER NOT NULL,
                max_allowed_week INTEGER
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prediction_run_metrics_week ON prediction_run_metrics(season, week)"
        )
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(weekly_predictions)").fetchall()
        }
        if "run_id" not in existing_cols:
            conn.execute("ALTER TABLE weekly_predictions ADD COLUMN run_id TEXT")


def save_predictions_to_warehouse(predictions: pd.DataFrame, db_path: Path = DB_FILE) -> None:
    init_prediction_warehouse(db_path)
    records = predictions.to_dict(orient="records")
    columns = list(predictions.columns)
    placeholders = ", ".join(["?"] * len(columns))
    update_cols = [c for c in columns if c not in {"season", "week", "home_full", "away_full"}]
    update_clause = ", ".join([f"{c}=excluded.{c}" for c in update_cols])

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            f"""
            INSERT INTO weekly_predictions ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(season, week, home_full, away_full)
            DO UPDATE SET {update_clause}
            """,
            [[row.get(col) for col in columns] for row in records],
        )


def save_run_metrics(metrics: dict, db_path: Path = DB_FILE) -> None:
    init_prediction_warehouse(db_path)
    columns = list(metrics.keys())
    placeholders = ", ".join(["?"] * len(columns))
    update_cols = [c for c in columns if c != "run_id"]
    update_clause = ", ".join([f"{c}=excluded.{c}" for c in update_cols])

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"""
            INSERT INTO prediction_run_metrics ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(run_id)
            DO UPDATE SET {update_clause}
            """,
            [metrics.get(col) for col in columns],
        )


def load_predictions_from_warehouse(
    season: int, week: int, db_path: Path = DB_FILE
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT *
            FROM weekly_predictions
            WHERE season = ? AND week = ?
            ORDER BY gameday, gametime, home_full
            """,
            conn,
            params=(season, week),
        )


def load_latest_run_metrics(season: int, week: int, db_path: Path = DB_FILE) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    init_prediction_warehouse(db_path)
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT *
            FROM prediction_run_metrics
            WHERE season = ? AND week = ?
            ORDER BY created_at_utc DESC
            LIMIT 1
            """,
            conn,
            params=(season, week),
        )


def get_schedule(season: int) -> pd.DataFrame:
    sched = nfl.load_schedules(seasons=[season]).to_pandas()
    home_col = "home_team" if "home_team" in sched.columns else ("home" if "home" in sched.columns else None)
    away_col = "away_team" if "away_team" in sched.columns else ("away" if "away" in sched.columns else None)
    if home_col is None or away_col is None:
        raise ValueError("Schedule data must include home/away team columns.")
    if "week" not in sched.columns:
        raise ValueError("Schedule data must include a week column.")

    sched = sched.copy()
    sched["home_full"] = sched[home_col].apply(canonicalize_team)
    sched["away_full"] = sched[away_col].apply(canonicalize_team)
    sched = sched[sched["home_full"].notna() & sched["away_full"].notna()].copy()
    return sched


def max_predictable_week(sched: pd.DataFrame, season: int) -> int | None:
    today = datetime.now().date()
    prediction_cutoff = today + timedelta(days=1)
    if int(season) < today.year:
        return int(sched["week"].max()) if not sched.empty else None
    if int(season) > today.year:
        return None
    if "gameday" not in sched.columns:
        return int(sched["week"].max()) if not sched.empty else None

    sched_dates = sched.copy()
    sched_dates["gameday_dt"] = pd.to_datetime(sched_dates["gameday"], errors="coerce").dt.date
    started = sched_dates[
        sched_dates["gameday_dt"].notna() & (sched_dates["gameday_dt"] <= prediction_cutoff)
    ]
    if started.empty:
        return None
    return int(started["week"].max())


def allowed_weeks_for_dashboard(sched: pd.DataFrame, season: int) -> list[int]:
    weeks = sorted(int(w) for w in sched["week"].dropna().unique())
    max_week = max_predictable_week(sched, season)
    if max_week is None:
        return []
    return [w for w in weeks if w <= max_week]


def assert_week_is_predictable(sched: pd.DataFrame, season: int, week: int) -> int:
    max_week = max_predictable_week(sched, season)
    if max_week is None or int(week) > max_week:
        msg_week = "none yet" if max_week is None else str(max_week)
        raise ValueError(
            f"Week {week} is in the future for {season}. "
            f"Predictions are only allowed through week {msg_week}."
        )
    return max_week


def _latest_team_row_before(
    team_store: pd.DataFrame, team_full: str, season: int, week: int
) -> pd.Series:
    sub = team_store[team_store["team_full"] == team_full].copy()
    if sub.empty:
        raise ValueError(f"No feature-store rows found for {team_full}")

    sub = sub[(sub["season"] < season) | ((sub["season"] == season) & (sub["week"] < week))]
    if sub.empty:
        raise ValueError(f"No feature-store rows found for {team_full} before season={season}, week={week}")

    return sub.sort_values(["season", "week"]).iloc[-1]


def evaluate_week_predictions(
    predictions: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    run_id: str,
    created_at: str,
    threshold: float,
    n_sims: int,
    max_allowed_week: int | None,
) -> dict:
    if predictions.empty:
        return {
            "run_id": run_id,
            "created_at_utc": created_at,
            "season": 0,
            "week": 0,
            "games_predicted": 0,
            "games_final": 0,
            "model_wins": 0,
            "model_losses": 0,
            "pushes": 0,
            "pick_accuracy": None,
            "auc": None,
            "threshold": float(threshold),
            "monte_carlo_sims": int(n_sims),
            "max_allowed_week": max_allowed_week,
        }

    score_cols = ["season", "week", "home_full", "away_full", "home_score", "away_score"]
    if not {"home_score", "away_score"}.issubset(schedule.columns):
        completed = pd.DataFrame()
    else:
        completed = predictions.merge(
            schedule[score_cols].dropna(subset=["home_score", "away_score"]),
            on=["season", "week", "home_full", "away_full"],
            how="inner",
        )

    pushes = 0
    wins = 0
    losses = 0
    auc = None

    if not completed.empty:
        completed = completed.copy()
        completed["home_score"] = completed["home_score"].astype(float)
        completed["away_score"] = completed["away_score"].astype(float)
        completed["is_tie"] = completed["home_score"] == completed["away_score"]
        pushes = int(completed["is_tie"].sum())
        decided = completed[~completed["is_tie"]].copy()

        if not decided.empty:
            decided["actual_home_win"] = (decided["home_score"] > decided["away_score"]).astype(int)
            decided["winner"] = np.where(
                decided["actual_home_win"].eq(1), decided["home_full"], decided["away_full"]
            )
            wins = int((decided["favorite_team"] == decided["winner"]).sum())
            losses = int(len(decided) - wins)
            if decided["actual_home_win"].nunique() > 1:
                auc = float(roc_auc_score(decided["actual_home_win"], decided["home_win_prob"]))

    decided_games = wins + losses
    accuracy = float(wins / decided_games) if decided_games else None
    return {
        "run_id": run_id,
        "created_at_utc": created_at,
        "season": int(predictions["season"].iloc[0]),
        "week": int(predictions["week"].iloc[0]),
        "games_predicted": int(len(predictions)),
        "games_final": int(len(completed)),
        "model_wins": wins,
        "model_losses": losses,
        "pushes": pushes,
        "pick_accuracy": accuracy,
        "auc": auc,
        "threshold": float(threshold),
        "monte_carlo_sims": int(n_sims),
        "max_allowed_week": max_allowed_week,
    }


def make_feature_row(
    team_store: pd.DataFrame,
    features: list[str],
    home_full: str,
    away_full: str,
    season: int,
    week: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    roll_cols = [c for c in team_store.columns if c.endswith(("_L1", "_L3", "_L6"))]
    if not roll_cols:
        raise ValueError("team_feature_store.parquet has no rolling columns.")

    home_row = _latest_team_row_before(team_store, home_full, season, week)
    away_row = _latest_team_row_before(team_store, away_full, season, week)

    feat = {"is_home": 1.0}
    for col in roll_cols:
        feat[f"DIFF_{col}"] = float(home_row[col]) - float(away_row[col])

    for col in features:
        feat.setdefault(col, 0.0)

    X = pd.DataFrame([feat], columns=features).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, home_row, away_row


def load_margin_residuals() -> tuple[np.ndarray | None, float]:
    if BACKTEST_PRED_FILE.exists():
        bt = pd.read_csv(BACKTEST_PRED_FILE)
        if {"home_margin_true", "home_margin_pred"}.issubset(bt.columns):
            residuals = (bt["home_margin_true"] - bt["home_margin_pred"]).astype(float).dropna().values
            residuals = residuals[np.isfinite(residuals)]
            if residuals.size >= 200:
                return residuals, float(np.std(residuals))

    rmse = 10.5
    if SUMMARY_FILE.exists():
        try:
            text = SUMMARY_FILE.read_text()
            parsed = json.loads(text.split("\n", 1)[1])
            values = [v.get("margin_rmse") for v in parsed.values() if isinstance(v, dict)]
            values = [float(v) for v in values if v is not None]
            if values:
                rmse = float(np.mean(values))
        except Exception:
            pass
    return None, rmse


def run_margin_monte_carlo(
    pred_home_margin: float,
    *,
    n_sims: int = DEFAULT_N_SIMS,
    random_seed: int = 42,
) -> dict[str, float | int]:
    rng = np.random.default_rng(random_seed)
    residuals, rmse = load_margin_residuals()
    if residuals is not None:
        sims = pred_home_margin + rng.choice(residuals, size=n_sims, replace=True)
    else:
        sims = pred_home_margin + rng.normal(loc=0.0, scale=rmse, size=n_sims)

    q05, q50, q95 = np.quantile(sims, [0.05, 0.50, 0.95])
    return {
        "monte_carlo_run": 1,
        "monte_carlo_sims": int(n_sims),
        "sim_home_win_prob": float((sims > 0).mean()),
        "sim_away_win_prob": float((sims < 0).mean()),
        "sim_home_margin_mean": float(np.mean(sims)),
        "sim_home_margin_std": float(np.std(sims)),
        "sim_home_margin_p05": float(q05),
        "sim_home_margin_p50": float(q50),
        "sim_home_margin_p95": float(q95),
        "p_home_wins_by_7plus": float((sims >= 7).mean()),
        "p_away_wins_by_7plus": float((sims <= -7).mean()),
    }


def predict_week(
    season: int,
    week: int,
    *,
    n_sims: int = DEFAULT_N_SIMS,
    threshold: float = MONTE_CARLO_THRESHOLD,
    random_seed: int = 42,
) -> pd.DataFrame:
    win_model, spread_model, features, team_store = load_artifacts()
    sched = get_schedule(season)
    max_allowed_week = assert_week_is_predictable(sched, season, week)
    games = sched[sched["week"].astype(int) == int(week)].copy()
    if games.empty:
        return pd.DataFrame()

    rows = []
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = f"{season}_{week}_{created_at}"

    for idx, game in games.reset_index(drop=True).iterrows():
        home_full = game["home_full"]
        away_full = game["away_full"]
        X, home_row, away_row = make_feature_row(
            team_store, features, home_full, away_full, season, week
        )

        home_win_prob = float(win_model.predict_proba(X)[:, 1][0])
        away_win_prob = 1.0 - home_win_prob
        pred_home_margin = float(spread_model.predict(X)[0])
        favorite_team = home_full if home_win_prob >= away_win_prob else away_full
        favorite_win_prob = max(home_win_prob, away_win_prob)
        under_threshold = int(favorite_win_prob < threshold)

        row = {
            "run_id": run_id,
            "created_at_utc": created_at,
            "season": int(season),
            "week": int(week),
            "game_id": str(game.get("game_id", f"{season}_{week}_{idx + 1}")),
            "season_type": str(game.get("season_type", "")),
            "gameday": str(game.get("gameday", "")),
            "gametime": str(game.get("gametime", "")),
            "home_full": home_full,
            "away_full": away_full,
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
            "favorite_team": favorite_team,
            "favorite_win_prob": favorite_win_prob,
            "under_threshold": under_threshold,
            "pred_home_margin": pred_home_margin,
            "confidence": abs(home_win_prob - 0.5) * 2,
            "monte_carlo_run": 0,
            "monte_carlo_sims": None,
            "sim_home_win_prob": None,
            "sim_away_win_prob": None,
            "sim_home_margin_mean": None,
            "sim_home_margin_std": None,
            "sim_home_margin_p05": None,
            "sim_home_margin_p50": None,
            "sim_home_margin_p95": None,
            "p_home_wins_by_7plus": None,
            "p_away_wins_by_7plus": None,
            "features_asof_home_season": int(home_row["season"]),
            "features_asof_home_week": int(home_row["week"]),
            "features_asof_away_season": int(away_row["season"]),
            "features_asof_away_week": int(away_row["week"]),
        }

        if under_threshold:
            row.update(
                run_margin_monte_carlo(
                    pred_home_margin,
                    n_sims=n_sims,
                    random_seed=random_seed + idx,
                )
            )

        rows.append(row)

    out = pd.DataFrame(rows)
    save_predictions_to_warehouse(out)
    metrics = evaluate_week_predictions(
        out,
        sched,
        run_id=run_id,
        created_at=created_at,
        threshold=threshold,
        n_sims=n_sims,
        max_allowed_week=max_allowed_week,
    )
    save_run_metrics(metrics)
    return out


def render_dashboard() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'streamlit'. Install it, then run: streamlit run ML_predict_2.py"
        ) from exc

    st.set_page_config(page_title="NFL Weekly Predictions", layout="wide")
    st.title("NFL Weekly Predictions")

    default_season = default_dashboard_season()
    season = st.sidebar.number_input("Season", min_value=2015, max_value=2035, value=default_season, step=1)

    try:
        sched = get_schedule(int(season))
        available_weeks = allowed_weeks_for_dashboard(sched, int(season))
    except Exception as exc:
        st.error(f"Could not load schedule for {season}: {exc}")
        return

    if not available_weeks:
        st.warning(f"No weeks are available for {season} yet. Future-week predictions are disabled.")
        return

    default_week = available_weeks[-1]
    week = st.sidebar.selectbox("Week", available_weeks, index=available_weeks.index(default_week))
    n_sims = st.sidebar.number_input(
        "Monte Carlo simulations", min_value=10_000, max_value=1_000_000, value=DEFAULT_N_SIMS, step=10_000
    )
    threshold = st.sidebar.slider("Monte Carlo threshold", 0.50, 0.80, MONTE_CARLO_THRESHOLD, 0.01)

    refresh = st.sidebar.button("Refresh predictions")
    cached = load_predictions_from_warehouse(int(season), int(week))

    if refresh or cached.empty:
        with st.spinner("Scoring the week and updating SQLite warehouse..."):
            try:
                # On refresh, ingest completed games BEFORE generating predictions.
                # Historic replays remain unchanged; the existing saved model weights
                # are never retrained by this refresh operation.
                if int(season) == datetime.now().year:
                    from weekly_update import refresh_team_features
                    refresh_team_features(int(season))
                preds = predict_week(int(season), int(week), n_sims=int(n_sims), threshold=float(threshold))
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return
    else:
        preds = cached

    if preds.empty:
        st.warning("No games found for this season/week.")
        return

    total_games = len(preds)
    sim_games = int(preds["monte_carlo_run"].fillna(0).sum())
    avg_favorite = float(preds["favorite_win_prob"].mean())
    metrics = load_latest_run_metrics(int(season), int(week))
    metrics_row = metrics.iloc[0].to_dict() if not metrics.empty else {}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games", total_games)
    c2.metric("Monte Carlo Games", sim_games)
    c3.metric("Avg Favorite Win Prob", f"{avg_favorite:.1%}")
    if metrics_row:
        c4.metric(
            "Model W-L",
            f"{int(metrics_row['model_wins'])}-{int(metrics_row['model_losses'])}",
            f"{int(metrics_row['pushes'])} pushes",
        )
        auc_value = metrics_row.get("auc")
        c5.metric("Week AUC", "NA" if pd.isna(auc_value) else f"{float(auc_value):.3f}")
    else:
        c4.metric("Model W-L", "NA")
        c5.metric("Week AUC", "NA")

    view = preds.copy()
    percent_cols = [
        "home_win_prob",
        "away_win_prob",
        "favorite_win_prob",
        "sim_home_win_prob",
        "sim_away_win_prob",
        "p_home_wins_by_7plus",
        "p_away_wins_by_7plus",
    ]
    for col in percent_cols:
        if col in view.columns:
            view[col] = view[col].apply(lambda x: "" if pd.isna(x) else f"{float(x):.1%}")

    display_cols = [
        "gameday",
        "gametime",
        "away_full",
        "home_full",
        "favorite_team",
        "favorite_win_prob",
        "pred_home_margin",
        "monte_carlo_run",
        "sim_home_win_prob",
        "sim_home_margin_mean",
        "sim_home_margin_p05",
        "sim_home_margin_p95",
        "features_asof_home_week",
        "features_asof_away_week",
    ]
    display_cols = [c for c in display_cols if c in view.columns]
    st.dataframe(view[display_cols], width="stretch", hide_index=True)

    st.caption(f"SQLite warehouse: {DB_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NFL weekly prediction dashboard and SQLite warehouse.")
    parser.add_argument("--season", type=int, help="Season to score from the command line.")
    parser.add_argument("--week", type=int, help="Week to score from the command line.")
    parser.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS, help="Monte Carlo simulations per close game.")
    parser.add_argument("--threshold", type=float, default=MONTE_CARLO_THRESHOLD, help="Favorite win-prob threshold.")
    args, _ = parser.parse_known_args()

    if args.season is not None and args.week is not None:
        try:
            preds = predict_week(
                args.season,
                args.week,
                n_sims=args.n_sims,
                threshold=args.threshold,
            )
        except ValueError as exc:
            print(exc)
            return
        if preds.empty:
            print(f"No games found for season={args.season}, week={args.week}.")
            return
        print(preds.to_string(index=False))
        print(f"\nSaved {len(preds)} predictions to {DB_FILE}")
    else:
        render_dashboard()


if __name__ == "__main__":
    main()
