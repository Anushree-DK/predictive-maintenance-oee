"""Trains the RUL regressor on run-to-failure histories (real C-MAPSS data via
scripts/load_cmapss.py, or synthetic via scripts/generate_mock_data.py).

Uses a small bagged ensemble of gradient-boosted-tree regressors (bootstrap-
resampled, like a random forest of boosted trees): the spread across bag
predictions is what src/ml/predict.py turns into a confidence score.

HistGradientBoostingRegressor (sklearn) is used instead of XGBoost/LightGBM
because both of those need Apple's libomp runtime via Homebrew, which isn't
available in every environment; HGB has no native-library dependency and
gives comparable accuracy. Swap for Snowpark ML's XGBoost once training moves
into Snowflake, where libomp isn't a concern.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src import data_access
from src.feature_engineering import build_feature_table

FEATURE_PREFIXES = ("SENSOR_",)
N_BAGS = 10


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]


def build_training_set() -> tuple[pd.DataFrame, pd.Series]:
    raw = data_access.load_training_run_to_failure()

    # RUL at every cycle for every unit, not just the latest — this is what makes
    # it a regression training set instead of a single snapshot per unit.
    feature_rows = []
    for unit_id, group in raw.groupby("UNIT_ID"):
        group = group.sort_values("TIME_CYCLE")
        for i in range(10, len(group)):  # need enough history for rolling stats
            window = group.iloc[: i + 1]
            feats = build_feature_table(window.assign(MACHINE_ID=unit_id), group_col="MACHINE_ID").iloc[0]
            feats["RUL"] = group.iloc[i]["RUL"]
            feature_rows.append(feats)

    table = pd.DataFrame(feature_rows)
    X = table[_feature_columns(table)]
    y = table["RUL"]
    return X, y


def train_and_save():
    X, y = build_training_set()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    bags = []
    for i in range(N_BAGS):
        X_bag, y_bag = resample(X_train, y_train, random_state=i)
        model = HistGradientBoostingRegressor(
            max_iter=150, max_depth=6, learning_rate=0.08, random_state=i,
        )
        model.fit(X_bag, y_bag)
        bags.append(model)

    ensemble_pred = sum(m.predict(X_test) for m in bags) / N_BAGS
    ss_res = ((y_test - ensemble_pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    print(f"Validation R^2: {1 - ss_res / ss_tot:.3f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": bags, "feature_columns": list(X.columns)}, config.MODELS_DIR / "rul_model.joblib")
    print(f"saved -> {config.MODELS_DIR / 'rul_model.joblib'}")


if __name__ == "__main__":
    train_and_save()
