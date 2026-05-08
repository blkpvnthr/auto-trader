```python
"""
SIMPLIFIED WALK-FORWARD ANALYSIS (PORTFOLIO VERSION)

This is a sanitized version of a walk-forward trading model.

Purpose:
- Demonstrate ML workflow
- Show time-series validation discipline
- Avoid data leakage

NOTE:
All proprietary alpha logic, execution, and optimization layers
have been removed.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


# =========================
# CONFIG
# =========================

INPUT_PATH = Path("daily_system_outputs/daily_features.csv")

TRAIN_DAYS = 252
TEST_DAYS = 42
STEP_DAYS = 42

TARGET = "future_return_5d"


# =========================
# LOAD DATA
# =========================

def load_data():
    df = pd.read_csv(INPUT_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["timestamp", "symbol"])
    return df


# =========================
# FEATURE SELECTION (SAFE)
# =========================

def get_features(df):
    excluded = {
        "timestamp",
        "symbol",
        "future_return_1d",
        "future_return_3d",
        "future_return_5d",
    }

    features = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]

    return features


# =========================
# WALK-FORWARD SPLIT
# =========================

def make_windows(dates):
    windows = []
    start = 0

    while True:
        train_end = start + TRAIN_DAYS
        test_end = train_end + TEST_DAYS

        if test_end > len(dates):
            break

        train_dates = dates[start:train_end]
        test_dates = dates[train_end:test_end]

        windows.append((train_dates, test_dates))

        start += STEP_DAYS

    return windows


# =========================
# MODEL (SIMPLIFIED)
# =========================

def build_model():
    return HistGradientBoostingClassifier(max_depth=5)


# =========================
# WALK-FORWARD LOOP
# =========================

def run_walkforward(df):

    features = get_features(df)

    dates = sorted(df["timestamp"].unique())
    windows = make_windows(dates)

    all_results = []

    for i, (train_dates, test_dates) in enumerate(windows):

        train = df[df["timestamp"].isin(train_dates)].copy()
        test = df[df["timestamp"].isin(test_dates)].copy()

        if len(train) < 500:
            continue

        # Binary target (simplified)
        train["target"] = (train[TARGET] > 0).astype(int)
        test["target"] = (test[TARGET] > 0).astype(int)

        X_train = train[features].fillna(0)
        y_train = train["target"]

        X_test = test[features].fillna(0)
        y_test = test["target"]

        model = build_model()
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs)

        print(f"[window {i}] AUC = {auc:.4f}")

        result = test.copy()
        result["prediction"] = probs
        result["window"] = i

        all_results.append(result)

    return pd.concat(all_results, ignore_index=True)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    df = load_data()

    results = run_walkforward(df)

    print("\n===== SAMPLE OUTPUT =====")
    print(results.head())

