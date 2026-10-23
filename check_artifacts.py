"""Validate required artifacts and trained model input dimensions."""
from pathlib import Path
import joblib
base = Path(__file__).resolve().parent / 'artifacts'
required = ['win_model_lgbm.joblib', 'spread_model_lgbm.joblib', 'features.joblib', 'team_feature_store.parquet']
missing = [name for name in required if not (base / name).exists()]
if missing:
    raise SystemExit('Missing required files: ' + ', '.join(missing))
features = joblib.load(base / 'features.joblib')
for model_name in ('win_model_lgbm.joblib', 'spread_model_lgbm.joblib'):
    model = joblib.load(base / model_name)
    n = getattr(model, 'n_features_in_', None)
    if n is not None and n != len(features):
        raise SystemExit(f'INCOMPATIBLE: {model_name} expects {n} features but features.joblib has {len(features)}. Locate the original compatible winner model; do NOT retrain or truncate features if you intend to keep the original logic.')
print('All required artifacts present and feature dimensions match.')
