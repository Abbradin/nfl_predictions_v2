
# %%
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import nflreadpy as nfl

PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

WIN_MODEL_FILE    = ARTIFACT_DIR / "win_model_lgbm.joblib"
SPREAD_MODEL_FILE = ARTIFACT_DIR / "spread_model_lgbm.joblib"
FEATURES_FILE     = ARTIFACT_DIR / "features.joblib"
TEAM_STORE_FILE   = ARTIFACT_DIR / "team_feature_store.parquet"

TEAM_MAP = {
    "buf":"Buffalo Bills","mia":"Miami Dolphins","ne":"New England Patriots","nwe":"New England Patriots","nyj":"New York Jets",
    "bal":"Baltimore Ravens","cin":"Cincinnati Bengals","cle":"Cleveland Browns","pit":"Pittsburgh Steelers",
    "hou":"Houston Texans","ind":"Indianapolis Colts","jax":"Jacksonville Jaguars","ten":"Tennessee Titans",
    "den":"Denver Broncos","kc":"Kansas City Chiefs","kan":"Kansas City Chiefs","lv":"Las Vegas Raiders","oak":"Las Vegas Raiders",
    "sd":"Los Angeles Chargers","lac":"Los Angeles Chargers",
    "dal":"Dallas Cowboys","nyg":"New York Giants","phi":"Philadelphia Eagles","wsh":"Washington Commanders","was":"Washington Commanders",
    "chi":"Chicago Bears","det":"Detroit Lions","gb":"Green Bay Packers","gnb":"Green Bay Packers","min":"Minnesota Vikings",
    "atl":"Atlanta Falcons","car":"Carolina Panthers","no":"New Orleans Saints","nor":"New Orleans Saints","tb":"Tampa Bay Buccaneers","tam":"Tampa Bay Buccaneers",
    "ari":"Arizona Cardinals","arz":"Arizona Cardinals","lar":"Los Angeles Rams","la":"Los Angeles Rams","stl":"Los Angeles Rams",
    "sf":"San Francisco 49ers","sfo":"San Francisco 49ers","sea":"Seattle Seahawks",
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

def _latest_team_row_before(team_store: pd.DataFrame, team_full: str, season: int, week: int) -> pd.Series:
    """
    Return the most recent row for team_full strictly BEFORE (season, week).
    Works for playoffs because it will return the last regular-season week row.
    """
    sub = team_store[team_store["team_full"] == team_full].copy()
    if sub.empty:
        raise ValueError(f"No rows in team_feature_store for team_full={team_full}")

    # strictly before the target (season, week)
    sub = sub[(sub["season"] < season) | ((sub["season"] == season) & (sub["week"] < week))].copy()
    if sub.empty:
        raise ValueError(
            f"No feature rows found for {team_full} before season={season}, week={week}. "
            "This can happen if your store starts too late."
        )

    sub = sub.sort_values(["season", "week"])
    return sub.iloc[-1]

def predict_game_by_week(season: int, week: int, home_team: str, away_team: str) -> pd.DataFrame:
    """
    Predict ONE specific game for a specific (season, week).
    For playoffs: uses each team's most recent available rolling-feature row prior to that week.
    """
    for fp in [WIN_MODEL_FILE, SPREAD_MODEL_FILE, FEATURES_FILE, TEAM_STORE_FILE]:
        if not fp.exists():
            raise FileNotFoundError(f"Missing required artifact: {fp}")

    win_model = joblib.load(WIN_MODEL_FILE)
    spread_model = joblib.load(SPREAD_MODEL_FILE)
    FEATURES = joblib.load(FEATURES_FILE)
    team_store = pd.read_parquet(TEAM_STORE_FILE)

    home_full = canonicalize_team(home_team)
    away_full = canonicalize_team(away_team)
    if home_full is None or away_full is None:
        raise ValueError(f"Could not canonicalize teams. Got home={home_full}, away={away_full}")

    # --- verify the schedule row exists for that exact week + home/away (so you don't accidentally flip them) ---
    sched = nfl.load_schedules(seasons=[season]).to_pandas()
    home_col = "home_team" if "home_team" in sched.columns else ("home" if "home" in sched.columns else None)
    away_col = "away_team" if "away_team" in sched.columns else ("away" if "away" in sched.columns else None)
    if home_col is None or away_col is None:
        raise ValueError("Schedule missing home/away team columns.")
    if "week" not in sched.columns:
        raise ValueError("Schedule missing 'week' column.")

    sched["home_full"] = sched[home_col].apply(canonicalize_team)
    sched["away_full"] = sched[away_col].apply(canonicalize_team)

    row = sched[
        (sched["week"] == week) &
        (sched["home_full"] == home_full) &
        (sched["away_full"] == away_full)
    ].copy()

    if row.empty:
        alt = sched[
            (sched["week"] == week) &
            (
                ((sched["home_full"] == home_full) & (sched["away_full"] == away_full)) |
                ((sched["home_full"] == away_full) & (sched["away_full"] == home_full))
            )
        ][["season","week","home_full","away_full"]].drop_duplicates()

        raise ValueError(
            "No schedule row found for that exact (season, week, home, away).\n"
            f"Requested: season={season}, week={week}, home={home_full}, away={away_full}\n"
            f"Matches found in that week with either orientation:\n{alt.to_string(index=False)}"
        )

    # --- build features using LAST AVAILABLE team rows BEFORE the game week ---
    roll_cols = [c for c in team_store.columns if c.endswith("_L1") or c.endswith("_L3") or c.endswith("_L6")]
    if not roll_cols:
        raise ValueError("team_feature_store.parquet has no rolling columns (*_L1/*_L3/*_L6).")

    home_row = _latest_team_row_before(team_store, home_full, season, week)
    away_row = _latest_team_row_before(team_store, away_full, season, week)

    feat = {"is_home": 1.0}
    for c in roll_cols:
        feat[f"DIFF_{c}"] = float(home_row[c]) - float(away_row[c])

    # ensure all expected features exist
    for f in FEATURES:
        if f not in feat:
            feat[f] = 0.0

    X = pd.DataFrame([feat], columns=FEATURES).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    p_home = float(win_model.predict_proba(X)[:, 1][0])
    pred_margin = float(spread_model.predict(X)[0])

    out = pd.DataFrame([{
        "season": season,
        "week": week,
        "home_full": home_full,
        "away_full": away_full,
        "home_win_prob": p_home,
        "pred_home_margin": pred_margin,
        "confidence": abs(p_home - 0.5) * 2,
        "features_asof_home_season": int(home_row["season"]),
        "features_asof_home_week": int(home_row["week"]),
        "features_asof_away_season": int(away_row["season"]),
        "features_asof_away_week": int(away_row["week"]),
    }])

    return out


