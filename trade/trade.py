"""
SANITIZED DAILY ML PIPELINE (PORTFOLIO VERSION)

This script demonstrates a simplified daily ML workflow for
financial time-series prediction.

IMPORTANT:
- All proprietary alpha logic has been removed
- No real execution or portfolio logic is included
- This is a research / demonstration pipeline only
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# =========================
# CONFIG
# =========================

LOOKBACK_DAYS = 365
SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

RAW_PATH = OUT_DIR / "raw.csv"
FEATURES_PATH = OUT_DIR / "features.csv"
PREDICTIONS_PATH = OUT_DIR / "predictions.csv"


# =========================
# ENV
# =========================

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")


# =========================
# DATA DOWNLOAD
# =========================

def download_data():

    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)

    frames = []

    for symbol in SYMBOLS:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )

            df = client.get_stock_bars(req).df.reset_index()

            if not df.empty:
                df["symbol"] = symbol
                frames.append(df)

        except Exception as e:
            print(f"[warning] {symbol} failed: {e}")

    data = pd.concat(frames, ignore_index=True)

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(["symbol", "timestamp"])

    return data


# =========================
# FEATURES (SIMPLIFIED)
# =========================

def build_features(df):

    df = df.copy()

    df["return_1d"] = df.groupby("symbol")["close"].pct_change()
    df["return_5d"] = df.groupby("symbol")["close"].pct_change(5)

    df["sma_20"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(20).mean())

    df["close_vs_sma"] = df["close"] / df["sma_20"] - 1

    df = df.fillna(0)

    return df


# =========================
# MODEL
# =========================

def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(max_depth=5))
    ])


# =========================
# TRAIN + PREDICT
# =========================

def run_model(df):

    latest_ts = df["timestamp"].max()

    train = df[df["timestamp"] < latest_ts].copy()
    test = df[df["timestamp"] == latest_ts].copy()

    train["target"] = (train["return_5d"] > 0).astype(int)

    features = [
        "return_1d",
        "return_5d",
        "close_vs_sma"
    ]

    X_train = train[features]
    y_train = train["target"]

    X_test = test[features]

    model = build_model()
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    test["prediction"] = probs

    return test


# =========================
# MAIN
# =========================

def main():

    print("[run] downloading data...")
    raw = download_data()
    raw.to_csv(RAW_PATH, index=False)

    print("[run] building features...")
    features = build_features(raw)
    features.to_csv(FEATURES_PATH, index=False)

    print("[run] running model...")
    predictions = run_model(features)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print("\n===== TOP PREDICTIONS =====")
    print(predictions.sort_values("prediction", ascending=False).head())


if __name__ == "__main__":
    main()

