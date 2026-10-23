"""CLI helper for GitHub Actions: refresh the store, then score eligible week."""
import argparse
from datetime import datetime, timezone
from weekly_update import refresh_team_features
from ML_predict_2 import get_schedule, allowed_weeks_for_dashboard, predict_week

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--season',type=int,default=datetime.now(timezone.utc).year)
    p.add_argument('--week',type=int)
    a=p.parse_args()
    print(refresh_team_features(a.season))
    sched=get_schedule(a.season)
    weeks=allowed_weeks_for_dashboard(sched,a.season)
    if not weeks: raise RuntimeError('No eligible weeks to score')
    week=a.week or weeks[-1]
    print(predict_week(a.season,week)[['home_full','away_full','favorite_team','favorite_win_prob','features_asof_home_week','features_asof_away_week']].to_string(index=False))
if __name__=='__main__': main()
