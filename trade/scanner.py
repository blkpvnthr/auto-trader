#!/usr/bin/env python3
"""
daily_scanner.py

Daily Alpaca-powered breakout momentum scanner.

Purpose:
    1. Download a candidate universe from S&P 500 + NASDAQ-listed symbols.
    2. Download daily OHLCV bars from Alpaca for that universe.
    3. Engineer momentum, trend, volatility, RSI, MACD, Bollinger, liquidity,
       and market-regime features.
    4. Train a continuation model on MACD/RSI/StochRSI/Bollinger relationships, then filter for stocks with positive momentum that closed
       above the previous day's high.
    5. Rank symbols primarily by volume, then learned continuation probability, breakout strength, momentum,
       trend confirmation, liquidity, relative volume, and risk-on market context.
    6. Write outputs for downstream training/execution.

Core signal:
    close > previous_day_high
    daily change % > 0
    MACD/RSI/StochRSI positive momentum confirmation

Signal interpretation:
    - StochRSI %K cross above %D = buy setup
    - StochRSI %K above %D and both cross above 20 with RSI 30-40 = stronger buy setup
    - StochRSI %K cross below %D = liquidate / sell warning
    - StochRSI %K below %D, both above 80, and RSI > 69 = sell warning
    - RSI cross above 72 = liquidation warning
    - RSI cross below 30 = favorable buy condition
    - MACD value line cross above average line = buy condition
    - MACD value line cross below average line = liquidation warning
    - Bollinger Bands are used as overbought/oversold/trend context, not order signals

Outputs:
    live/latest_breakout_momentum_symbols.txt
    live/latest_breakout_momentum_symbols_csv.txt
    live/latest_breakout_momentum_candidates.csv
    live/latest_breakout_momentum_candidates.txt
    live/latest_breakout_momentum_score_chart.png
    live/latest_breakout_momentum_probability_chart.png
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# Paths / environment
# =============================================================================

THIS_FILE = Path(__file__).resolve()
LIVE_DIR = THIS_FILE.parent
ENV_PATH = LIVE_DIR / ".env"

load_dotenv(ENV_PATH)

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise RuntimeError(
        f"Missing Alpaca credentials. Expected APCA_API_KEY_ID and "
        f"APCA_API_SECRET_KEY in {ENV_PATH}"
    )

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


# =============================================================================
# Configuration
# =============================================================================

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

LOOKBACK_DAYS = 520
MAX_SYMBOLS = 1000
CHUNK_SIZE = 20
TOP_N = 30
MAX_TRAIN_EVENTS = 50_000
MIN_MODEL_PROB = 0.45

REGIME_SYMBOLS = ["SPY", "QQQ", "IWM", "VIXY"]
CORE_SYMBOLS = ["SPY", "QQQ", "DIA", "IWM", "VIXY", "IBIT", "TLT"]

MIN_PRICE = 5.00
MAX_PRICE = 100.00
MIN_VOLUME_1D = 1_000_000
MIN_AVG_VOLUME20 = 750_000
MIN_AVG_DOLLAR_VOLUME20 = 5_000_000
MIN_REL_VOLUME_1D = 1.0

MIN_DAILY_CHANGE_PCT = 0.0
MIN_BREAKOUT_PCT = 0.0
MIN_MOMENTUM_SCORE = 0.50
MIN_TREND_SCORE = 0.35
MIN_LIQUIDITY_SCORE = 0.50
MAX_VOLATILITY20 = 0.10
MIN_VOLATILITY20 = 0.003

REQUIRE_BULL_OR_NEUTRAL_REGIME = True
REQUIRE_NO_LIQUIDATION_SIGNAL = False
REQUIRE_SMA_BUY_SIGNAL = False

OUT_SYMBOLS = LIVE_DIR / "latest_breakout_momentum_symbols.txt"
OUT_SYMBOLS_CSV = LIVE_DIR / "latest_breakout_momentum_symbols_csv.txt"
OUT_CSV = LIVE_DIR / "latest_breakout_momentum_candidates.csv"
OUT_TXT = LIVE_DIR / "latest_breakout_momentum_candidates.txt"
OUT_PLOT_SCORE = LIVE_DIR / "latest_breakout_momentum_score_chart.png"
OUT_PLOT_PROB = LIVE_DIR / "latest_breakout_momentum_probability_chart.png"
OUT_UNIVERSE = LIVE_DIR / "alpaca_dynamic_universe.csv"
OUT_RAW_BARS = LIVE_DIR / "alpaca_daily_bars.csv"


# =============================================================================
# Utilities
# =============================================================================

def normalize_symbol(sym: str) -> str:
    return str(sym).strip().upper().replace("-", ".")


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def pct(a: pd.Series, periods: int = 1) -> pd.Series:
    return a.pct_change(periods)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def request_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }


# =============================================================================
# Universe loading
# =============================================================================

def get_sp500_symbols() -> list[str]:
    print("[setup] fetching S&P 500 symbols...")

    resp = requests.get(WIKI_SP500_URL, headers=request_headers(), timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise RuntimeError("Could not parse S&P 500 table from Wikipedia.")

    df = tables[0].copy()
    if "Symbol" not in df.columns:
        raise RuntimeError("S&P 500 table does not contain a Symbol column.")

    symbols = (
        df["Symbol"]
        .astype(str)
        .map(normalize_symbol)
        .drop_duplicates()
        .tolist()
    )

    symbols = sorted({s for s in symbols if s and s != "NAN"})
    print(f"[setup] S&P 500 symbols={len(symbols)}")
    return symbols


def get_nasdaq_symbols() -> list[str]:
    print("[setup] fetching NASDAQ-listed symbols...")

    resp = requests.get(NASDAQ_LISTED_URL, headers=request_headers(), timeout=30)
    resp.raise_for_status()

    lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
    data_lines = [line for line in lines if not line.startswith("File Creation Time")]

    df = pd.read_csv(StringIO("\n".join(data_lines)), sep="|")

    if "Symbol" not in df.columns:
        raise RuntimeError("NASDAQ listed file does not contain a Symbol column.")

    if "Test Issue" in df.columns:
        df = df[df["Test Issue"].astype(str).str.upper() == "N"].copy()

    symbols: list[str] = []

    for raw_sym in df["Symbol"].tolist():
        if pd.isna(raw_sym):
            continue

        sym = normalize_symbol(raw_sym)

        if not sym or sym == "NAN":
            continue

        # Skip preferreds / warrants / special issues that often fail Alpaca requests.
        if "$" in sym or "^" in sym or "/" in sym:
            continue

        symbols.append(sym)

    symbols = sorted(set(symbols))
    print(f"[setup] NASDAQ-listed symbols={len(symbols)}")
    return symbols


def get_universe() -> tuple[list[str], list[str]]:
    sp500 = get_sp500_symbols()
    nasdaq = get_nasdaq_symbols()

    symbols = list(dict.fromkeys(CORE_SYMBOLS + REGIME_SYMBOLS + sp500 + nasdaq))
    symbols = [normalize_symbol(s) for s in symbols if s and str(s).strip()]
    symbols = sorted(set(symbols))

    if MAX_SYMBOLS is not None:
        symbols = symbols[:MAX_SYMBOLS]

        for required in CORE_SYMBOLS + REGIME_SYMBOLS:
            required = normalize_symbol(required)
            if required not in symbols:
                symbols.append(required)

        symbols = sorted(set(symbols))

    pd.DataFrame({"symbol": symbols}).to_csv(OUT_UNIVERSE, index=False)

    analysis_symbols = [s for s in symbols if s not in set(REGIME_SYMBOLS)]

    print(f"[universe] analysis_symbols={len(analysis_symbols)}")
    print(f"[universe] total_with_regime_symbols={len(symbols)}")
    print(f"[saved] universe={OUT_UNIVERSE}")

    return analysis_symbols, symbols


# =============================================================================
# Alpaca download
# =============================================================================

def fetch_chunk(symbols: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )

    bars = client.get_stock_bars(req)
    df = bars.df.reset_index()

    if df.empty:
        return pd.DataFrame()

    needed = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise RuntimeError(f"Alpaca response missing columns: {missing}")

    df = df[needed].copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def download_daily_bars(symbols: list[str]) -> pd.DataFrame:
    end = now_utc()
    start = end - timedelta(days=LOOKBACK_DAYS)

    chunks = [symbols[i:i + CHUNK_SIZE] for i in range(0, len(symbols), CHUNK_SIZE)]
    frames: list[pd.DataFrame] = []

    print(f"[download] {start.date()} -> {end.date()} chunks={len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print(f"[download] chunk={i}/{len(chunks)} size={len(chunk)}")

        try:
            df = fetch_chunk(chunk, start, end)

            if not df.empty:
                frames.append(df)

        except Exception as exc:
            print(f"[warning] chunk failed size={len(chunk)} error={exc}")
            print("[warning] retrying symbols individually...")

            for sym in chunk:
                try:
                    df = fetch_chunk([sym], start, end)
                    if not df.empty:
                        frames.append(df)
                except Exception as sym_exc:
                    print(f"[skip] symbol={sym} error={sym_exc}")

    if not frames:
        raise RuntimeError("No data downloaded from Alpaca.")

    data = pd.concat(frames, ignore_index=True)

    data = (
        data.dropna(subset=["symbol", "timestamp", "open", "high", "low", "close"])
        .drop_duplicates(subset=["symbol", "timestamp"])
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )

    data.to_csv(OUT_RAW_BARS, index=False)

    print(f"[download] rows={len(data):,} symbols={data['symbol'].nunique():,}")
    print(f"[saved] raw bars={OUT_RAW_BARS}")

    return data


# =============================================================================
# Regime features
# =============================================================================

def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    market = df[df["symbol"].isin(REGIME_SYMBOLS)].copy()

    if market.empty:
        df = df.copy()
        for col in [
            "market_ret",
            "market_vol",
            "qqq_ret_5d",
            "spy_ret_5d",
            "iwm_ret_5d",
            "vixy_ret_5d",
            "bull",
            "bear",
            "risk_on_score",
        ]:
            df[col] = 0.0
        return df

    pivot = market.pivot(index="timestamp", columns="symbol", values="close")
    returns = pivot.pct_change()

    regime = pd.DataFrame(index=pivot.index)
    regime["market_ret"] = returns.mean(axis=1).fillna(0.0)
    regime["market_vol"] = returns.std(axis=1).fillna(0.0)

    regime["qqq_ret_5d"] = pivot["QQQ"].pct_change(5) if "QQQ" in pivot.columns else 0.0
    regime["spy_ret_5d"] = pivot["SPY"].pct_change(5) if "SPY" in pivot.columns else 0.0
    regime["iwm_ret_5d"] = pivot["IWM"].pct_change(5) if "IWM" in pivot.columns else 0.0
    regime["vixy_ret_5d"] = pivot["VIXY"].pct_change(5) if "VIXY" in pivot.columns else 0.0

    regime["bull"] = (
        (regime["market_ret"] > 0)
        & (regime["qqq_ret_5d"] > 0)
        & (regime["spy_ret_5d"] > 0)
    ).astype(int)

    regime["bear"] = (
        (regime["market_ret"] < 0)
        | (regime["vixy_ret_5d"] > 0)
    ).astype(int)

    regime["risk_on_score"] = (
        0.35 * (regime["qqq_ret_5d"] > 0).astype(int)
        + 0.35 * (regime["spy_ret_5d"] > 0).astype(int)
        + 0.15 * (regime["iwm_ret_5d"] > 0).astype(int)
        + 0.15 * (regime["vixy_ret_5d"] < 0).astype(int)
        - 0.25 * regime["bear"]
    ).clip(0.0, 1.0)

    regime = regime.reset_index()

    out = df.merge(regime, on="timestamp", how="left")

    for col in [
        "market_ret",
        "market_vol",
        "qqq_ret_5d",
        "spy_ret_5d",
        "iwm_ret_5d",
        "vixy_ret_5d",
        "bull",
        "bear",
        "risk_on_score",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out


# =============================================================================
# Feature engineering
# =============================================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "timestamp"]).copy()
    frames: list[pd.DataFrame] = []

    for sym, g in df.groupby("symbol", sort=False):
        g = g.copy()
        g["symbol"] = sym

        g["dailychg_pct"] = pct(g["close"], 1) * 100.0
        g["dailychg_positive"] = (g["dailychg_pct"] > 0).astype(int)

        g["prev_high"] = g["high"].shift(1)
        g["prev_close"] = g["close"].shift(1)
        g["breakout"] = (g["close"] > g["prev_high"]).astype(int)
        g["breakout_pct"] = (safe_div(g["close"], g["prev_high"]) - 1.0) * 100.0

        g["ret_1d"] = pct(g["close"], 1)
        g["ret_3d"] = pct(g["close"], 3)
        g["ret_5d"] = pct(g["close"], 5)
        g["ret_10d"] = pct(g["close"], 10)
        g["ret_21d"] = pct(g["close"], 21)
        g["ret_63d"] = pct(g["close"], 63)

        g["positive_momentum"] = (
            (g["ret_1d"] > 0)
            & (g["ret_3d"] > 0)
            & (
                (g["ret_5d"] > 0)
                | (g["ret_10d"] > 0)
                | (g["ret_21d"] > 0)
            )
        ).astype(int)

        g["volatility10"] = g["ret_1d"].rolling(10).std()
        g["volatility20"] = g["ret_1d"].rolling(20).std()

        g["sma5"] = g["close"].rolling(5).mean()
        g["sma10"] = g["close"].rolling(10).mean()
        g["sma20"] = g["close"].rolling(20).mean()
        g["sma50"] = g["close"].rolling(50).mean()
        g["sma200"] = g["close"].rolling(200).mean()

        g["dist_sma5"] = safe_div(g["close"], g["sma5"]) - 1.0
        g["dist_sma10"] = safe_div(g["close"], g["sma10"]) - 1.0
        g["dist_sma20"] = safe_div(g["close"], g["sma20"]) - 1.0
        g["dist_sma50"] = safe_div(g["close"], g["sma50"]) - 1.0
        g["dist_sma200"] = safe_div(g["close"], g["sma200"]) - 1.0

        g["sma5_above_sma20"] = (g["sma5"] > g["sma20"]).astype(int)
        g["sma20_above_sma50"] = (g["sma20"] > g["sma50"]).astype(int)
        g["sma50_above_sma200"] = (g["sma50"] > g["sma200"]).astype(int)

        g["sma_bull_stack"] = (
            (g["sma5"] > g["sma20"])
            & (g["sma20"] > g["sma50"])
            & (g["sma50"] > g["sma200"])
        ).astype(int)

        g["sma_buy_signal"] = (
            (g["sma5"] > g["sma20"])
            & (g["sma50"] > g["sma200"])
        ).astype(int)

        g["sma5_20_spread"] = safe_div(g["sma5"], g["sma20"]) - 1.0
        g["sma50_200_spread"] = safe_div(g["sma50"], g["sma200"]) - 1.0

        delta = g["close"].diff()
        gain = delta.clip(lower=0.0).rolling(14).mean()
        loss = (-delta.clip(upper=0.0)).rolling(14).mean()
        rs = safe_div(gain, loss)
        g["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        g["rsi_overbought"] = (g["rsi"] > 70).astype(int)
        g["rsi_oversold"] = (g["rsi"] < 30).astype(int)
        g["rsi_momentum_ok"] = ((g["rsi"] >= 40) & (g["rsi"] <= 72)).astype(int)
        g["rsi_buy_zone"] = ((g["rsi"] > 30) & (g["rsi"] < 40)).astype(int)
        g["rsi_cross_below_30"] = (
            (g["rsi"].shift(1) >= 30) & (g["rsi"] < 30)
        ).astype(int)
        g["rsi_cross_above_72"] = (
            (g["rsi"].shift(1) <= 72) & (g["rsi"] > 72)
        ).astype(int)

        ema12 = g["close"].ewm(span=12, adjust=False).mean()
        ema26 = g["close"].ewm(span=26, adjust=False).mean()

        g["macd"] = ema12 - ema26
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
        g["macd_hist"] = g["macd"] - g["macd_signal"]
        g["macd_bull"] = (g["macd"] > g["macd_signal"]).astype(int)
        g["macd_improving"] = (g["macd_hist"] > g["macd_hist"].shift(1)).astype(int)
        g["macd_cross_above_signal"] = (
            (g["macd"].shift(1) <= g["macd_signal"].shift(1))
            & (g["macd"] > g["macd_signal"])
        ).astype(int)
        g["macd_cross_below_signal"] = (
            (g["macd"].shift(1) >= g["macd_signal"].shift(1))
            & (g["macd"] < g["macd_signal"])
        ).astype(int)

        rsi_min = g["rsi"].rolling(14).min()
        rsi_max = g["rsi"].rolling(14).max()
        stoch_rsi = 100.0 * safe_div(g["rsi"] - rsi_min, rsi_max - rsi_min)
        g["stochrsi"] = stoch_rsi.clip(0.0, 100.0)
        g["stochrsi_k"] = g["stochrsi"].rolling(3).mean()
        g["stochrsi_d"] = g["stochrsi_k"].rolling(3).mean()
        g["stochrsi_k_above_d"] = (g["stochrsi_k"] > g["stochrsi_d"]).astype(int)
        g["stochrsi_k_below_d"] = (g["stochrsi_k"] < g["stochrsi_d"]).astype(int)
        g["stochrsi_cross_above_d"] = (
            (g["stochrsi_k"].shift(1) <= g["stochrsi_d"].shift(1))
            & (g["stochrsi_k"] > g["stochrsi_d"])
        ).astype(int)
        g["stochrsi_cross_below_d"] = (
            (g["stochrsi_k"].shift(1) >= g["stochrsi_d"].shift(1))
            & (g["stochrsi_k"] < g["stochrsi_d"])
        ).astype(int)
        g["stochrsi_cross_above_20"] = (
            (g["stochrsi_k"].shift(1) <= 20)
            & (g["stochrsi_d"].shift(1) <= 20)
            & (g["stochrsi_k"] > 20)
            & (g["stochrsi_d"] > 20)
        ).astype(int)
        g["stochrsi_cross_below_80"] = (
            (g["stochrsi_k"].shift(1) >= 80)
            & (g["stochrsi_d"].shift(1) >= 80)
            & (g["stochrsi_k"] < 80)
            & (g["stochrsi_d"] < 80)
        ).astype(int)

        g["stochrsi_buy_signal"] = (
            (g["stochrsi_k_above_d"] == 1)
            & (g["stochrsi_cross_above_20"] == 1)
            & (g["rsi"] > 30)
            & (g["rsi"] < 40)
        ).astype(int)
        g["stochrsi_buy_cross"] = (g["stochrsi_cross_above_d"] == 1).astype(int)
        g["stochrsi_sell_signal"] = (
            (g["stochrsi_k_below_d"] == 1)
            & (g["stochrsi_k"] > 80)
            & (g["stochrsi_d"] > 80)
            & (g["rsi"] > 69)
        ).astype(int)
        g["stochrsi_liquidate_cross"] = (g["stochrsi_cross_below_d"] == 1).astype(int)

        bb_mid = g["close"].rolling(20).mean()
        bb_std = g["close"].rolling(20).std()

        g["bb_mid"] = bb_mid
        g["bb_upper"] = bb_mid + 2.0 * bb_std
        g["bb_lower"] = bb_mid - 2.0 * bb_std
        g["bb_width"] = safe_div(g["bb_upper"] - g["bb_lower"], g["close"])
        g["bb_position"] = safe_div(g["close"] - g["bb_lower"], g["bb_upper"] - g["bb_lower"])
        g["bb_trend_confirm"] = (
            (g["close"] > g["bb_mid"])
            & (g["bb_mid"] > g["bb_mid"].shift(1))
        ).astype(int)
        g["bb_overbought_context"] = (g["bb_position"] >= 0.90).astype(int)
        g["bb_oversold_context"] = (g["bb_position"] <= 0.10).astype(int)
        g["bb_neutral_to_bull_context"] = (
            (g["bb_position"] >= 0.45)
            & (g["bb_position"] <= 0.90)
            & (g["close"] >= g["bb_mid"])
        ).astype(int)

        g["bb_cross_above_upper"] = (
            (g["close"].shift(1) <= g["bb_upper"].shift(1))
            & (g["close"] > g["bb_upper"])
        ).astype(int)

        g["bb_upper_reject"] = (
            (g["high"] > g["bb_upper"])
            & (g["close"] < g["bb_upper"])
        ).astype(int)

        g["buy_signal"] = (
            (g["macd_cross_above_signal"] == 1)
            | (g["stochrsi_buy_signal"] == 1)
            | (g["stochrsi_buy_cross"] == 1)
            | (g["rsi_cross_below_30"] == 1)
        ).astype(int)

        g["liquidation_signal"] = (
            (g["macd_cross_below_signal"] == 1)
            | (g["stochrsi_sell_signal"] == 1)
            | (g["stochrsi_liquidate_cross"] == 1)
            | (g["rsi_cross_above_72"] == 1)
        ).astype(int)

        g["confluence_reversal_risk"] = (
            (g["liquidation_signal"] == 1)
            | (g["bb_upper_reject"] == 1)
            | (
                (g["bb_overbought_context"] == 1)
                & (g["rsi"] > 69)
                & (g["macd_improving"] == 0)
            )
        ).astype(int)

        g["dollar_volume"] = g["close"] * g["volume"]
        g["avg_dollar_volume20"] = g["dollar_volume"].rolling(20).mean()
        g["avg_volume20"] = g["volume"].rolling(20).mean()
        g["volume_sma20"] = g["volume"].rolling(20).mean()
        g["volume_ratio20"] = safe_div(g["volume"], g["volume_sma20"])

        g["trend_score"] = (
            0.25 * g["sma5_above_sma20"]
            + 0.20 * g["sma20_above_sma50"]
            + 0.20 * g["sma50_above_sma200"]
            + 0.15 * (g["dist_sma20"] > 0).astype(int)
            + 0.10 * (g["dist_sma50"] > 0).astype(int)
            + 0.10 * (g["sma5_20_spread"] > 0).astype(int)
        ).clip(0.0, 1.0)

        g["momentum_score"] = (
            0.14 * (g["ret_1d"] > 0).astype(int)
            + 0.12 * (g["ret_3d"] > 0).astype(int)
            + 0.12 * (g["ret_5d"] > 0).astype(int)
            + 0.10 * (g["ret_10d"] > 0).astype(int)
            + 0.08 * (g["ret_21d"] > 0).astype(int)
            + 0.14 * g["macd_bull"]
            + 0.10 * g["macd_cross_above_signal"]
            + 0.08 * g["macd_improving"]
            + 0.08 * g["rsi_momentum_ok"]
            + 0.08 * g["stochrsi_k_above_d"]
            + 0.06 * g["stochrsi_buy_cross"]
        ).clip(0.0, 1.0)

        g["liquidity_score"] = (
            0.40 * (g["avg_dollar_volume20"] >= MIN_AVG_DOLLAR_VOLUME20).astype(int)
            + 0.25 * (g["avg_volume20"] >= MIN_AVG_VOLUME20).astype(int)
            + 0.20 * (g["volume_ratio20"] >= 1.0).astype(int)
            + 0.15 * (g["volume_ratio20"] >= 1.25).astype(int)
        ).clip(0.0, 1.0)

        g["breakout_momentum_score"] = (
            0.24 * g["momentum_score"]
            + 0.17 * g["trend_score"]
            + 0.12 * g["liquidity_score"]
            + 0.10 * g["risk_on_score"]
            + 0.08 * g["dailychg_positive"]
            + 0.08 * g["positive_momentum"]
            + 0.07 * g["breakout"]
            + 0.06 * g["buy_signal"]
            + 0.05 * g["bb_trend_confirm"]
            + 0.03 * g["bb_neutral_to_bull_context"]
            - 0.14 * g["liquidation_signal"]
            - 0.08 * g["confluence_reversal_risk"]
            - 0.08 * g["bear"]
        ).clip(0.0, 1.0)

        g["next_close"] = g["close"].shift(-1)
        g["next_ret_1d"] = safe_div(g["next_close"], g["close"]) - 1.0
        g["future_close_3d"] = g["close"].shift(-3)
        g["future_ret_3d"] = safe_div(g["future_close_3d"], g["close"]) - 1.0

        # The model learns whether these breakout + indicator relationships
        # tend to lead to positive follow-through, instead of treating every
        # indicator rule as a mandatory hard-coded order.
        g["label"] = (
            (g["next_ret_1d"] > 0)
            & (g["future_ret_3d"].fillna(0.0) > -0.01)
        ).astype(int)

        frames.append(g)

    out = pd.concat(frames, ignore_index=True)

    if "symbol" not in out.columns:
        raise RuntimeError("symbol column missing after feature engineering.")

    return out


# =============================================================================
# Model
# =============================================================================

FEATURES = [
    "dailychg_pct",
    "dailychg_positive",
    "breakout",
    "breakout_pct",
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_21d",
    "ret_63d",
    "positive_momentum",
    "volatility10",
    "volatility20",
    "market_ret",
    "market_vol",
    "qqq_ret_5d",
    "spy_ret_5d",
    "iwm_ret_5d",
    "vixy_ret_5d",
    "bull",
    "bear",
    "risk_on_score",
    "dist_sma5",
    "dist_sma10",
    "dist_sma20",
    "dist_sma50",
    "dist_sma200",
    "sma5_above_sma20",
    "sma20_above_sma50",
    "sma50_above_sma200",
    "sma_bull_stack",
    "sma_buy_signal",
    "sma5_20_spread",
    "sma50_200_spread",
    "rsi",
    "rsi_overbought",
    "rsi_oversold",
    "rsi_momentum_ok",
    "rsi_buy_zone",
    "rsi_cross_below_30",
    "rsi_cross_above_72",
    "macd",
    "macd_signal",
    "macd_hist",
    "macd_bull",
    "macd_improving",
    "macd_cross_above_signal",
    "macd_cross_below_signal",
    "stochrsi",
    "stochrsi_k",
    "stochrsi_d",
    "stochrsi_k_above_d",
    "stochrsi_k_below_d",
    "stochrsi_cross_above_d",
    "stochrsi_cross_below_d",
    "stochrsi_cross_above_20",
    "stochrsi_cross_below_80",
    "stochrsi_buy_signal",
    "stochrsi_buy_cross",
    "stochrsi_sell_signal",
    "stochrsi_liquidate_cross",
    "buy_signal",
    "liquidation_signal",
    "bb_width",
    "bb_position",
    "bb_trend_confirm",
    "bb_overbought_context",
    "bb_oversold_context",
    "bb_neutral_to_bull_context",
    "bb_cross_above_upper",
    "bb_upper_reject",
    "confluence_reversal_risk",
    "volume",
    "avg_volume20",
    "volume_ratio20",
    "avg_dollar_volume20",
    "trend_score",
    "momentum_score",
    "liquidity_score",
    "breakout_momentum_score",
]


def train_model(events: pd.DataFrame) -> Pipeline:
    events = events.dropna(subset=FEATURES + ["label"]).copy()
    events = events.sort_values("timestamp")

    if len(events) > MAX_TRAIN_EVENTS:
        events = events.tail(MAX_TRAIN_EVENTS)

    if events.empty:
        raise RuntimeError("No valid training events after feature filtering.")

    y = events["label"].astype(int)

    if y.nunique() < 2:
        raise RuntimeError(
            "Training labels contain only one class. Increase LOOKBACK_DAYS or relax filters."
        )

    print(f"[ml] train_rows={len(events):,} continuation_rate={y.mean():.4f}")
    print(
        "[ml] learning from MACD/RSI/StochRSI/Bollinger relationships instead of "
        "using those relationships only as hard-coded rules."
    )

    model = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.04,
                    max_leaf_nodes=31,
                    l2_regularization=0.08,
                    min_samples_leaf=25,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(events[FEATURES], y)
    return model


# =============================================================================
# Candidate selection
# =============================================================================

def get_latest_rows(data: pd.DataFrame, analysis_symbols: list[str]) -> pd.DataFrame:
    latest_rows = []

    for sym, g in data[data["symbol"].isin(analysis_symbols)].groupby("symbol"):
        g = g.sort_values("timestamp").copy()
        if not g.empty:
            latest_rows.append(g.tail(1))

    if not latest_rows:
        return pd.DataFrame()

    return pd.concat(latest_rows, ignore_index=True)


def rank_candidates(latest: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    latest = latest.copy()

    required = list(dict.fromkeys([
        "close",
        "prev_high",
        "dailychg_pct",
        "breakout",
        "breakout_pct",
        "positive_momentum",
        "momentum_score",
        "trend_score",
        "liquidity_score",
        "volume",
        "avg_volume20",
        "avg_dollar_volume20",
        "volume_ratio20",
        "volatility20",
        "breakout_momentum_score",
        "macd",
        "macd_signal",
        "macd_bull",
        "macd_cross_above_signal",
        "macd_cross_below_signal",
        "rsi",
        "rsi_cross_below_30",
        "rsi_cross_above_72",
        "stochrsi_k",
        "stochrsi_d",
        "stochrsi_k_above_d",
        "stochrsi_buy_signal",
        "stochrsi_sell_signal",
        "liquidation_signal",
        "bb_trend_confirm",
        *FEATURES,
    ]))

    missing = [c for c in required if c not in latest.columns]
    if missing:
        raise RuntimeError(f"Latest frame missing required columns: {missing}")

    latest = latest.dropna(subset=required).copy()

    if latest.empty:
        return latest

    latest["model_continuation_prob"] = model.predict_proba(latest[FEATURES])[:, 1]
    
    latest["rank_score"] = (
            0.35 * latest["model_continuation_prob"]
            + 0.20 * latest["momentum_score"]
            + 0.15 * latest["signal_confidence"]
            + 0.10 * latest["trend_score"]
            + 0.10 * latest["liquidity_score"]
            + 0.10 * latest["volume_ratio20"].clip(0, 3) / 3
        ).clip(0.0, 1.0)

    mask = (
        (latest["close"] >= MIN_PRICE)
        & (latest["close"] < MAX_PRICE)
        & (latest["volume"] >= MIN_VOLUME_1D)
        & (latest["avg_volume20"] >= MIN_AVG_VOLUME20)
        & (latest["avg_dollar_volume20"] >= MIN_AVG_DOLLAR_VOLUME20)
        & (latest["volume_ratio20"] >= MIN_REL_VOLUME_1D)
        & (latest["volatility20"] >= MIN_VOLATILITY20)
        & (latest["volatility20"] <= MAX_VOLATILITY20)
        & (latest["breakout"] == 1)
        & (latest["breakout_pct"] > MIN_BREAKOUT_PCT)
        & (latest["dailychg_pct"] > MIN_DAILY_CHANGE_PCT)
        & (latest["positive_momentum"] == 1)
        & (latest["momentum_score"] >= MIN_MOMENTUM_SCORE)
        & (latest["rsi"] < 72)
        & (latest["model_continuation_prob"] >= MIN_MODEL_PROB)
        & (latest["trend_score"] >= MIN_TREND_SCORE)
        & (latest["liquidity_score"] >= MIN_LIQUIDITY_SCORE)
    )

    if REQUIRE_BULL_OR_NEUTRAL_REGIME and "bear" in latest.columns:
        mask &= latest["bear"] == 0

    if REQUIRE_NO_LIQUIDATION_SIGNAL and "liquidation_signal" in latest.columns:
        mask &= latest["liquidation_signal"] == 0

    if REQUIRE_SMA_BUY_SIGNAL and "sma_buy_signal" in latest.columns:
        mask &= latest["sma_buy_signal"] == 1

    candidates = latest.loc[mask].copy()

    print(
        f"[filter] latest={len(latest)} candidates={len(candidates)} "
        f"signal=close>prev_high + learned MACD/RSI/StochRSI/Bollinger context"
    )

    if candidates.empty:
        print("[rank] no candidates passed breakout momentum ML filters.")
        return candidates

    candidates["rank_score"] = (
        0.45 * candidates["model_continuation_prob"]
        + 0.20 * candidates["breakout_momentum_score"]
        + 0.15 * candidates["momentum_score"]
        + 0.10 * candidates["trend_score"]
        + 0.05 * candidates["liquidity_score"]
        + 0.05 * candidates["risk_on_score"]
        - 0.15 * candidates["liquidation_signal"]
        - 0.10 * candidates["confluence_reversal_risk"]
    ).clip(0.0, 1.0)

    # User-requested final ordering: sort the selected list by volume.
    candidates = (
        candidates.sort_values(
            [
                "volume",
                "avg_dollar_volume20",
                "model_continuation_prob",
                "rank_score",
                "breakout_pct",
                "dailychg_pct",
            ],
            ascending=False,
        )
        .head(TOP_N)
        .reset_index(drop=True)
    )

    candidates["rank"] = np.arange(1, len(candidates) + 1)
    candidates["price"] = candidates["close"]

    return candidates


# =============================================================================
# Output
# =============================================================================

DISPLAY_COLS = [
    "rank",
    "symbol",
    "price",
    "prev_high",
    "dailychg_pct",
    "breakout_pct",
    "breakout_momentum_score",
    "momentum_score",
    "trend_score",
    "liquidity_score",
    "risk_on_score",
    "bull",
    "bear",
    "positive_momentum",
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_21d",
    "sma_buy_signal",
    "sma_bull_stack",
    "sma5_above_sma20",
    "sma50_above_sma200",
    "rsi",
    "rsi_overbought",
    "rsi_oversold",
    "rsi_momentum_ok",
    "rsi_buy_zone",
    "rsi_cross_below_30",
    "rsi_cross_above_72",
    "macd",
    "macd_signal",
    "macd_hist",
    "macd_bull",
    "macd_improving",
    "macd_cross_above_signal",
    "macd_cross_below_signal",
    "stochrsi_k",
    "stochrsi_d",
    "stochrsi_k_above_d",
    "stochrsi_buy_signal",
    "stochrsi_buy_cross",
    "stochrsi_sell_signal",
    "stochrsi_liquidate_cross",
    "buy_signal",
    "liquidation_signal",
    "bb_position",
    "bb_width",
    "bb_trend_confirm",
    "bb_overbought_context",
    "bb_oversold_context",
    "bb_neutral_to_bull_context",
    "bb_cross_above_upper",
    "bb_upper_reject",
    "confluence_reversal_risk",
    "volume",
    "avg_volume20",
    "volume_ratio20",
    "avg_dollar_volume20",
    "volatility20",
]


def plot_candidates(df_out: pd.DataFrame) -> None:
    if df_out.empty:
        return

    score_df = df_out.sort_values("breakout_momentum_score", ascending=True)

    plt.figure(figsize=(12, 7))
    plt.barh(score_df["symbol"], score_df["breakout_momentum_score"])
    plt.title("Daily Breakout Momentum Score")
    plt.xlabel("Breakout Momentum Score")
    plt.ylabel("Symbol")
    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.savefig(OUT_PLOT_SCORE, dpi=150)
    plt.close()

    prob_df = df_out.sort_values("model_continuation_prob", ascending=True) if "model_continuation_prob" in df_out.columns else pd.DataFrame()
    if not prob_df.empty:
        plt.figure(figsize=(12, 7))
        plt.barh(prob_df["symbol"], prob_df["model_continuation_prob"])
        plt.title("Learned Continuation Probability")
        plt.xlabel("Model Continuation Probability")
        plt.ylabel("Symbol")
        plt.grid(True, axis="x")
        plt.tight_layout()
        plt.savefig(OUT_PLOT_PROB, dpi=150)
        plt.close()
        print(f"[plot] saved={OUT_PLOT_PROB}")

    print(f"[plot] saved={OUT_PLOT_SCORE}")


def write_outputs(candidates: pd.DataFrame) -> None:
    if candidates.empty:
        print("[output] no candidates passed filters.")
        OUT_SYMBOLS.write_text("", encoding="utf-8")
        OUT_SYMBOLS_CSV.write_text("", encoding="utf-8")
        OUT_TXT.write_text("No candidates passed filters.", encoding="utf-8")
        pd.DataFrame().to_csv(OUT_CSV, index=False)
        return

    symbols = candidates["symbol"].astype(str).tolist()
    symbol_list_str = ", ".join(f'"{s}"' for s in symbols)

    cols = [c for c in DISPLAY_COLS if c in candidates.columns]
    df_out = candidates[cols].copy()

    percent_cols = [
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ret_10d",
        "ret_21d",
        "volatility20",
    ]

    for col in percent_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col] * 100.0

    rename_map = {
        "price": "price_usd",
        "dailychg_pct": "dailychg_%",
        "breakout_pct": "breakout_%",
        "ret_1d": "ret_1d_%",
        "ret_3d": "ret_3d_%",
        "ret_5d": "ret_5d_%",
        "ret_10d": "ret_10d_%",
        "ret_21d": "ret_21d_%",
        "volatility20": "volatility20_%",
    }

    df_out = df_out.rename(columns=rename_map)

    for col in ["price_usd", "prev_high"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].round(2)

    df_out = df_out.round(4)

    OUT_SYMBOLS.write_text(symbol_list_str, encoding="utf-8")
    OUT_SYMBOLS_CSV.write_text(symbol_list_str, encoding="utf-8")
    OUT_TXT.write_text(df_out.to_string(index=False), encoding="utf-8")

    candidates.to_csv(OUT_CSV, index=False)

    print("\n===== TOP DAILY BREAKOUT MOMENTUM SYMBOLS SORTED BY VOLUME =====")
    print(symbol_list_str)

    print("\n===== TOP DAILY BREAKOUT MOMENTUM DATAFRAME =====")
    print(df_out.to_string(index=False))

    print(f"\n[output] symbols={OUT_SYMBOLS}")
    print(f"[output] symbols_csv_list={OUT_SYMBOLS_CSV}")
    print(f"[output] candidates_csv={OUT_CSV}")
    print(f"[output] candidates_txt={OUT_TXT}")

    plot_candidates(candidates)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("===== DAILY ALPACA VOLUME-RANKED BREAKOUT MOMENTUM SCANNER =====")

    analysis_symbols, symbols = get_universe()

    data = download_daily_bars(symbols)
    data = add_regime(data)
    data = add_features(data)

    train_events = data[
        (data["symbol"].isin(analysis_symbols))
        & (data["breakout"] == 1)
        & (data["dailychg_positive"] == 1)
        & (data["avg_dollar_volume20"] >= MIN_AVG_DOLLAR_VOLUME20)
        & (data["momentum_score"] >= 0.40)
    ].copy()

    if train_events.empty:
        raise RuntimeError("No training events found for breakout momentum model.")

    model = train_model(train_events)

    latest = get_latest_rows(data, analysis_symbols)

    if latest.empty:
        print("[latest] no latest rows found.")
        write_outputs(pd.DataFrame())
        return

    latest_date = latest["timestamp"].max()
    print(f"[latest] date={latest_date} rows={len(latest)}")

    candidates = rank_candidates(latest, model)

    print(f"[latest] candidates_after_filters={len(candidates)}")

    write_outputs(candidates)


if __name__ == "__main__":
    main()
