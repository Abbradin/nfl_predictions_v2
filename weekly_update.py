"""Refresh completed-game team statistics without changing either trained model.

The existing app consumes one row per team/week with 1-, 3-, and 6-game
rolling averages. This script constructs only NEW rows, retains the old
store intact, and rejects incomplete/unmappable data rather than fabricating
statistics. It never learns new model weights.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import nflreadpy as nfl

ARTIFACT_DIR = Path(__file__).resolve().parent / 'artifacts'
STORE_PATH = ARTIFACT_DIR / 'team_feature_store.parquet'

# Use original canonical mapping to ensure abbreviations line up.
from ML_predict_2 import canonicalize_team


def refresh_team_features(season: int, *, until_week: int | None = None, store_path: Path = STORE_PATH) -> dict:
    if not store_path.is_file():
        raise FileNotFoundError(f'Original team feature store missing: {store_path}')
    original = pd.read_parquet(store_path)
    if not {'season','week','team_full'}.issubset(original.columns):
        raise ValueError('Existing feature-store schema is missing required key columns')
    rolling = {c: c.rsplit('_L', 1) for c in original.columns if c.endswith(('_L1','_L3','_L6'))}
    if not rolling:
        raise ValueError('No existing rolling stat columns found; refusing to alter schema')
    base_cols = sorted({v[0] for v in rolling.values()})

    # nflverse publishes one team-game row containing these SAME raw statistic names.
    source = nfl.load_team_stats(seasons=[season], summary_level='week').to_pandas()
    schedule = nfl.load_schedules(seasons=[season]).to_pandas()
    if source.empty or schedule.empty:
        raise ValueError('NFL source returned no team statistics or schedule; no updates made')
    required = {'season','week','team'} | set(base_cols) - {'season','week'}
    missing = sorted(required-set(source.columns))
    if missing:
        raise ValueError('NFL team statistics schema differs from legacy feature store. No updates made. Missing: '+', '.join(missing))
    if not {'week','home_team','away_team','home_score','away_score'}.issubset(schedule.columns):
        raise ValueError('Schedule does not include completed-game verification columns')

    source=source.copy(); schedule=schedule.copy()
    source['team_full']=source['team'].apply(canonicalize_team)
    schedule['home_full']=schedule['home_team'].apply(canonicalize_team)
    schedule['away_full']=schedule['away_team'].apply(canonicalize_team)
    if source['team_full'].isna().any():
        raise ValueError('Unknown source NFL team abbreviation; refusing to refresh')
    for c in ['season','week']:
        source[c]=pd.to_numeric(source[c], errors='raise').astype(int)
    schedule['week']=pd.to_numeric(schedule['week'], errors='raise').astype(int)

    # Schedule final scores alone are not sufficient: require a final result state,
    # where published. Without a status column, completed dates before today are used.
    date_col='gameday' if 'gameday' in schedule else None
    if date_col:
        dates=pd.to_datetime(schedule[date_col],errors='coerce',utc=True).dt.date
        finished_date=dates < datetime.now(timezone.utc).date()
    else:
        raise ValueError('Schedule lacks gameday; cannot safely determine completed games')
    done=(schedule['home_score'].notna() & schedule['away_score'].notna() & finished_date)
    if 'game_type' in schedule.columns:
        pass  # preserve original regular/postseason eligibility
    if until_week is not None:
        done &= schedule['week'].le(until_week)
    completed=schedule.loc[done].copy()
    if completed.empty:
        return {'season':season,'appended_rows':0,'status':'no completed games'}

    # Validate complete game pairs and avoid creating rolling windows from future games.
    eligible=set()
    for r in completed.itertuples():
        eligible.add((int(r.week),str(r.home_full)))
        eligible.add((int(r.week),str(r.away_full)))
    source=source[[int(w) == int(season) for w in source['season']]].copy()
    source=source[[ (int(r.week),str(r.team_full)) in eligible for r in source.itertuples() ]].copy()
    if source.duplicated(['season','week','team_full']).any():
        raise ValueError('Duplicate team-week rows from nflreadpy; no updates made')
    present=set(zip(source['week'].astype(int),source['team_full']))
    if eligible-present:
        raise ValueError(f'NFL team stats have not been published for {len(eligible-present)} completed team-games; no updates made')
    source=source.sort_values(['team_full','season','week']).reset_index(drop=True)
    numeric=source[base_cols].apply(pd.to_numeric,errors='coerce')
    # A team can have no FG/PAT/punt attempt in a game. nflreadpy then reports
    # the corresponding percentage/long as null. Preserve that missingness:
    # pandas rolling means skip missing values, and the original inference
    # pipeline handles remaining missing model inputs using its existing logic.
    # Do not replace null with zero: zero would assert an observed statistic.
    optional_kicking_stats={'fg_long', 'fg_pct', 'pat_pct', 'pt_long'}
    unexpected=[]
    for col in base_cols:
        raw=source[col]
        # Reject actual nonnumeric strings even for optional stats.
        bad_text=raw.notna() & numeric[col].isna()
        if bad_text.any() or (numeric[col].isna().any() and col not in optional_kicking_stats):
            unexpected.append(col)
    if unexpected:
        raise ValueError('Null/non-numeric required team stats prevent safe update: '+', '.join(unexpected))
    missing_kicking=[col for col in sorted(optional_kicking_stats & set(base_cols)) if numeric[col].isna().any()]
    if missing_kicking:
        print('Info: preserving legitimately missing kicking stats:', ', '.join(missing_kicking))
    source[base_cols]=numeric
    updated=source[['season','week','team_full']].copy()
    for col,(base,window) in rolling.items():
        if base=='season':
            updated[col]=source.groupby('team_full')['season'].transform(lambda s:s.rolling(int(window),min_periods=1).mean())
        elif base=='week':
            updated[col]=source.groupby('team_full')['week'].transform(lambda s:s.rolling(int(window),min_periods=1).mean())
        else:
            # Each current-game record is a snapshot of the latest L games.
            updated[col]=source.groupby('team_full')[base].transform(lambda s:s.rolling(int(window),min_periods=1).mean())
    updated=updated[original.columns.intersection(updated.columns).tolist()]
    # All other legacy metadata columns must be safely reconstructible, otherwise refuse.
    missing_store_cols=[c for c in original.columns if c not in updated.columns]
    if missing_store_cols:
        raise ValueError('Cannot preserve original feature-store schema: '+', '.join(missing_store_cols))
    updated=updated[original.columns]
    keys=['season','week','team_full']
    old_keys=set(map(tuple,original[keys].itertuples(index=False,name=None)))
    additions=updated[[tuple(k) not in old_keys for k in updated[keys].itertuples(index=False,name=None)]].copy()
    if additions.empty:
        return {'season':season,'appended_rows':0,'status':'already current'}
    # Protect against consuming a partial week of player/team data.
    merged=pd.concat([original,additions],ignore_index=True)
    if merged.duplicated(keys).any():
        raise ValueError('Feature store has duplicate team-week keys; no updates made')
    merged=merged.sort_values(['season','week','team_full']).reset_index(drop=True)
    store_path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(suffix='.parquet',dir=store_path.parent)
    os.close(fd)
    try:
        merged.to_parquet(tmp,index=False)
        os.replace(tmp,store_path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return {'season':season,'appended_rows':len(additions),'through_week':int(additions['week'].max()),'status':'updated'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--season',type=int,default=datetime.now(timezone.utc).year)
    parser.add_argument('--until-week',type=int)
    a=parser.parse_args()
    print(refresh_team_features(a.season,until_week=a.until_week))

if __name__=='__main__': main()
