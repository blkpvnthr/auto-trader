"""
SANITIZED PORTFOLIO VERSION

This file demonstrates a production-style data pipeline and feature engineering
framework for a quantitative trading system.

IMPORTANT:
- All alpha-generating logic has been removed or simplified
- Regime detection is a placeholder
- Signal generation is illustrative only
- This is NOT the live trading system

Purpose:
Showcase engineering, structure, and data workflows without exposing IP.
"""

import os
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# -----------------------------
# CONFIG
# -----------------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
ENV_PATH = PROJECT_ROOT / ".env"

OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RAW = OUT_DIR / "raw_data.csv"
OUT_FEATURES = OUT_DIR / "features.csv"

# Simplified universe (non-sensitive)
DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]


# -----------------------------
# HELPERS
# -----------------------------

def normalize_symbol(sym: str) -> str:
    return str(sym).replace(".", "-").upper().strip()


def get_headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0"
    }


# -----------------------------
# DATA FETCHING
# -----------------------------

def load_alpaca_client() -> StockHistoricalDataClient:
    load_dotenv(ENV_PATH)

    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError("Missing Alpaca API credentials")

    return StockHistoricalDataClient(api_key, secret_key)


def download_data(
    symbols: List[str],
    lookback_days: int = 365,
    feed: DataFeed = DataFeed.IEX,
) -> pd.DataFrame:

    client = load_alpaca_client()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    frames = []

    for symbol in symbols:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=feed,
            )

            bars = client.get_stock_bars(req).df.reset_index()

            if not bars.empty:
                bars["symbol"] = symbol
                frames.append(bars)

        except Exception as e:
            print(f"[warning] {symbol} failed: {e}")

    if not frames:
        raise RuntimeError("No data downloaded")

    df = pd.concat(frames, ignore_index=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    return df


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df = df.sort_values(["symbol", "timestamp"])

    feature_frames = []

    for symbol, g in df.groupby("symbol"):
        g = g.copy()

        # Returns
        g["return_1d"] = g["close"].pct_change()
        g["return_5d"] = g["close"].pct_change(5)

        # Moving averages
        g["sma_20"] = g["close"].rolling(20).mean()
        g["sma_50"] = g["close"].rolling(50).mean()

        # Relative positioning
        g["close_vs_sma20"] = g["close"] / g["sma_20"] - 1

        # Momentum indicators
        g["rsi_14"] = rsi(g["close"], 14)

        ema_12 = g["close"].ewm(span=12, adjust=False).mean()
        ema_26 = g["close"].ewm(span=26, adjust=False).mean()

        g["macd"] = ema_12 - ema_26
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()

        # Volatility
        g["volatility_20d"] = g["return_1d"].rolling(20).std()

        feature_frames.append(g)

    features = pd.concat(feature_frames, ignore_index=True)

    return features


# -----------------------------
# SIMPLIFIED SIGNAL (SAFE)
# -----------------------------

def generate_demo_signal(df: pd.DataFrame) -> pd.Series:
    """
    Placeholder signal logic.

    NOTE:
    The real system uses proprietary ML models and optimization.
    This is only for demonstration purposes.
    """
    return (
        df["return_5d"].fillna(0)
        + df["macd"].fillna(0)
    )


# -----------------------------
# SIMPLIFIED REGIME (SAFE)
# -----------------------------

def classify_market_regime(df: pd.DataFrame) -> pd.Series:
    """
    Placeholder regime classification.

    Real implementation uses statistical + ML-based modeling.
    """
    return np.where(df["return_5d"] > 0, "RISK_ON", "RISK_OFF")


# -----------------------------
# MAIN PIPELINE
# -----------------------------

def run_pipeline():

    print("[run] downloading data...")

    raw = download_data(DEFAULT_SYMBOLS)

    raw.to_csv(OUT_RAW, index=False)
    print(f"[saved] raw data -> {OUT_RAW}")

    print("[run] building features...")

    features = build_features(raw)

    print("[run] generating demo signals...")

    features["signal_score"] = generate_demo_signal(features)

    print("[run] classifying regime...")

    features["market_regime"] = classify_market_regime(features)

    features.to_csv(OUT_FEATURES, index=False)

    print(f"[saved] features -> {OUT_FEATURES}")

    print("\n[preview]")
    print(features.tail(10))


# -----------------------------
# ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    run_pipeline()
