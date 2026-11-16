"""Quick local check of the saved demonstration artifacts.

Run only with trusted model files: joblib loading uses Python deserialization.
"""
from pathlib import Path
import json
import joblib
import pandas as pd

root = Path(__file__).resolve().parents[1]
with (root / 'models' / 'feature_names.json').open(encoding='utf8') as f:
    features = json.load(f)
data = pd.read_csv(root / 'prepared_data' / 'npp_prepared.csv')
model = joblib.load(root / 'models' / 'npp_xgboost_model.joblib')
assert len(features) == 96, len(features)
assert all(feature in data.columns for feature in features)
assert set(data.Label.unique()) == {0, 1}
assert len(data) == 12152, len(data)
pred = model.predict(data.loc[[0], features])
print('Model loaded. Test prediction:', int(pred[0]))
print('NOTE: this is a functionality check, NOT a held-out evaluation.')
