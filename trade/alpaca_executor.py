#!/usr/bin/env python3
"""
alpaca_executor.py

Always-executing, market-hours guarded Alpaca executor.

There is intentionally NO dry-run mode in this file.

Default behavior:
    python alpaca_executor.py

Default input files:
    daily_system_outputs/daily_walkforward_emulated_rebalance_orders.csv
    daily_system_outputs/daily_walkforward_emulated_exit_orders.csv

Paper account is used by default.
Use --live only when you intentionally want live brokerage execution.

Production controls:
    1. Market-open guard.
    2. Daily-loss kill switch.
    3. Sell-before-buy execution.
    4. Cash-buffer protection.
    5. Confidence-based adaptive sizing.
    6. Intraday timing controls.
    7. Pre-trade sanity filter:
        - max spread %
        - min quote size
        - min estimated dollar volume
        - volatility cap
        - stale quote rejection
    8. Fill tracking and slippage logging.
    9. CSV / JSONL / TXT execution logs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


# =============================================================================
# Paths / environment
# =============================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

OUT_DIR = PROJECT_ROOT / "daily_system_outputs"
LOG_DIR = PROJECT_ROOT / "logs" / "execution"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_REBALANCE_ORDERS = OUT_DIR / "daily_walkforward_emulated_rebalance_orders.csv"
DEFAULT_EXIT_ORDERS = OUT_DIR / "daily_walkforward_emulated_exit_orders.csv"


# =============================================================================
# Config / data models
# =============================================================================

@dataclass
class ExecutorConfig:
    # Account mode.
    paper: bool = True

    # Capital / sizing controls.
    cash_buffer_pct: float = 0.08
    min_cash_dollars: float = 500.0
    min_trade_dollars: float = 100.0
    max_order_dollars: float = 15_000.0
    max_single_order_qty: int = 10_000

    # Execution behavior.
    sell_first: bool = True
    cancel_open_orders_first: bool = False
    sleep_between_orders_sec: float = 0.35

    # Market clock.
    require_market_open: bool = True
    wait_for_market_open: bool = False
    max_wait_for_open_minutes: int = 180
    market_clock_poll_sec: float = 30.0

    # Order details.
    allow_fractional: bool = False
    time_in_force: TimeInForce = TimeInForce.DAY
    order_tag_prefix: str = "explosive_alpha"

    # Kill switch.
    enable_kill_switch: bool = True
    max_daily_loss_pct: float = 0.02
    max_daily_loss_dollars: float = 2_000.0
    kill_switch_file: Path = LOG_DIR / "kill_switch_state.json"

    # Adaptive sizing.
    enable_adaptive_sizing: bool = True
    min_confidence_scale: float = 0.35
    max_confidence_scale: float = 1.25
    default_confidence: float = 0.50

    # Intraday timing.
    enable_intraday_timing: bool = True
    avoid_first_minutes: int = 5
    avoid_last_minutes: int = 10
    early_session_order_mult: float = 0.70
    midday_order_mult: float = 1.00
    late_session_order_mult: float = 0.50
    allow_late_sells: bool = True

    # Pre-trade sanity checks.
    enable_pretrade_sanity: bool = True
    quote_feed: DataFeed = DataFeed.IEX
    max_spread_pct: float = 0.015
    min_bid_ask_size: int = 1
    min_estimated_dollar_volume: float = 500_000.0
    max_quote_age_seconds: int = 600
    max_order_notional_vs_estimated_dv_pct: float = 0.05
    max_row_volatility20: float = 0.20
    max_row_atr_pct: float = 0.18
    sanity_check_sells: bool = False

    # Fill tracking.
    track_fills: bool = True
    fill_poll_attempts: int = 6
    fill_poll_sleep_sec: float = 2.0


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    timestamp: str


# =============================================================================
# Generic helpers
# =============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def run_id() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def safe_symbol(value: Any) -> str:
    return str(value).upper().strip().replace(".", "-")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    return str(value)


def write_event(event: str, payload: dict[str, Any] | None = None) -> None:
    path = LOG_DIR / "execution_events.jsonl"
    row = {
        "timestamp": utc_now().isoformat(),
        "event": event,
        "payload": payload or {},
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=json_default) + "\n")


def save_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"[saved] {path}")


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=json_default) + "\n", encoding="utf-8")
    print(f"[saved] {path}")


# =============================================================================
# Alpaca clients / snapshots
# =============================================================================

def get_api_credentials() -> tuple[str, str]:
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError(
            f"Missing Alpaca credentials. Expected APCA_API_KEY_ID and "
            f"APCA_API_SECRET_KEY in {ENV_PATH}"
        )

    return api_key, secret_key


def get_trading_client(config: ExecutorConfig | None = None) -> TradingClient:
    config = config or ExecutorConfig()
    api_key, secret_key = get_api_credentials()
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=config.paper)


def get_data_client() -> StockHistoricalDataClient:
    api_key, secret_key = get_api_credentials()
    return StockHistoricalDataClient(api_key, secret_key)


def get_account_snapshot(client: TradingClient) -> AccountSnapshot:
    account = client.get_account()
    return AccountSnapshot(
        equity=to_float(getattr(account, "equity", 0.0)),
        cash=to_float(getattr(account, "cash", 0.0)),
        buying_power=to_float(getattr(account, "buying_power", 0.0)),
        portfolio_value=to_float(getattr(account, "portfolio_value", 0.0)),
        timestamp=utc_now().isoformat(),
    )


def get_positions_df(client: TradingClient) -> pd.DataFrame:
    positions = client.get_all_positions()
    rows: list[dict[str, Any]] = []

    for p in positions:
        rows.append(
            {
                "symbol": safe_symbol(getattr(p, "symbol", "")),
                "qty": to_float(getattr(p, "qty", 0.0)),
                "avg_entry_price": to_float(getattr(p, "avg_entry_price", 0.0)),
                "current_price": to_float(getattr(p, "current_price", 0.0)),
                "market_value": to_float(getattr(p, "market_value", 0.0)),
                "cost_basis": to_float(getattr(p, "cost_basis", 0.0)),
                "unrealized_pl": to_float(getattr(p, "unrealized_pl", 0.0)),
                "unrealized_plpc": to_float(getattr(p, "unrealized_plpc", 0.0)),
            }
        )

    cols = [
        "symbol",
        "qty",
        "avg_entry_price",
        "current_price",
        "market_value",
        "cost_basis",
        "unrealized_pl",
        "unrealized_plpc",
    ]
    return pd.DataFrame(rows, columns=cols)


# =============================================================================
# Market clock / timing
# =============================================================================

def get_market_clock_info(client: TradingClient) -> dict[str, Any]:
    clock = client.get_clock()
    return {
        "is_open": bool(getattr(clock, "is_open", False)),
        "timestamp": str(getattr(clock, "timestamp", "")),
        "next_open": str(getattr(clock, "next_open", "")),
        "next_close": str(getattr(clock, "next_close", "")),
    }


def market_open_guard(client: TradingClient, config: ExecutorConfig) -> bool:
    if not config.require_market_open:
        print("[clock] require_market_open=False; market-open guard disabled.")
        write_event("market_clock_guard_disabled", {})
        return True

    info = get_market_clock_info(client)

    print(
        "[clock] "
        f"is_open={info['is_open']} "
        f"timestamp={info['timestamp']} "
        f"next_open={info['next_open']} "
        f"next_close={info['next_close']}"
    )
    write_event("market_clock_checked", info)

    if info["is_open"]:
        return True

    if not config.wait_for_market_open:
        print("[clock] market is closed. Skipping execution. Use --wait-for-open to wait.")
        write_event("market_closed_skip_execution", info)
        return False

    max_wait_sec = max(0, int(config.max_wait_for_open_minutes * 60))
    started = time.time()

    while time.time() - started < max_wait_sec:
        time.sleep(max(5.0, float(config.market_clock_poll_sec)))
        info = get_market_clock_info(client)

        print(
            "[clock] waiting | "
            f"is_open={info['is_open']} "
            f"next_open={info['next_open']}"
        )
        write_event("market_clock_wait_poll", info)

        if info["is_open"]:
            print("[clock] market is open. Continuing execution.")
            write_event("market_open_wait_complete", info)
            return True

    print("[clock] market did not open within wait limit. Skipping execution.")
    write_event(
        "market_open_wait_timeout",
        {**info, "max_wait_for_open_minutes": config.max_wait_for_open_minutes},
    )
    return False


def parse_clock_ts(value: str) -> pd.Timestamp | None:
    try:
        if not value:
            return None
        return pd.Timestamp(value)
    except Exception:
        return None


def session_timing_multiplier(
    client: TradingClient,
    side: str,
    config: ExecutorConfig,
) -> tuple[float, str]:
    if not config.enable_intraday_timing:
        return 1.0, "TIMING_DISABLED"

    info = get_market_clock_info(client)
    now_ts = parse_clock_ts(info["timestamp"])
    next_close = parse_clock_ts(info["next_close"])

    if now_ts is None or next_close is None:
        return 1.0, "TIMING_UNKNOWN"

    try:
        local_now = now_ts.tz_convert("America/New_York")
        local_close = next_close.tz_convert("America/New_York")
    except Exception:
        local_now = now_ts
        local_close = next_close

    open_dt = local_now.replace(hour=9, minute=30, second=0, microsecond=0)
    minutes_since_open = (local_now - open_dt).total_seconds() / 60.0
    minutes_to_close = (local_close - local_now).total_seconds() / 60.0

    side = side.lower()

    if side == "buy" and minutes_since_open < config.avoid_first_minutes:
        return 0.0, f"SKIP_BUY_FIRST_{config.avoid_first_minutes}_MIN"

    if side == "buy" and minutes_to_close < config.avoid_last_minutes:
        return 0.0, f"SKIP_BUY_LAST_{config.avoid_last_minutes}_MIN"

    if side == "sell" and minutes_to_close < config.avoid_last_minutes and config.allow_late_sells:
        return 1.0, "LATE_SELL_ALLOWED"

    if minutes_since_open < 30:
        return config.early_session_order_mult, "EARLY_SESSION_SIZE_REDUCTION"

    if minutes_to_close < 45:
        return config.late_session_order_mult, "LATE_SESSION_SIZE_REDUCTION"

    return config.midday_order_mult, "MIDDAY_NORMAL"


# =============================================================================
# Kill switch
# =============================================================================

def read_kill_switch_state(config: ExecutorConfig) -> dict[str, Any]:
    path = Path(config.kill_switch_file)
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_kill_switch_state(config: ExecutorConfig, state: dict[str, Any]) -> None:
    path = Path(config.kill_switch_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=json_default) + "\n", encoding="utf-8")


def enforce_daily_loss_kill_switch(
    account: AccountSnapshot,
    config: ExecutorConfig,
) -> bool:
    if not config.enable_kill_switch:
        write_event("kill_switch_disabled", {})
        return True

    today = utc_now().date().isoformat()
    state = read_kill_switch_state(config)

    if state.get("date") != today:
        state = {
            "date": today,
            "start_equity": account.equity,
            "kill_triggered": False,
            "triggered_at": "",
            "reason": "",
        }
        write_kill_switch_state(config, state)

    start_equity = to_float(state.get("start_equity"), account.equity)
    daily_pnl = account.equity - start_equity
    daily_loss_pct = -daily_pnl / max(start_equity, 1e-9)

    hard_dollar_loss = daily_pnl <= -abs(config.max_daily_loss_dollars)
    hard_pct_loss = daily_loss_pct >= abs(config.max_daily_loss_pct)

    print(
        "[kill-switch] "
        f"start_equity=${start_equity:,.2f} "
        f"current_equity=${account.equity:,.2f} "
        f"daily_pnl=${daily_pnl:,.2f} "
        f"daily_loss_pct={daily_loss_pct:.2%}"
    )

    payload = {
        "date": today,
        "start_equity": start_equity,
        "current_equity": account.equity,
        "daily_pnl": daily_pnl,
        "daily_loss_pct": daily_loss_pct,
        "max_daily_loss_pct": config.max_daily_loss_pct,
        "max_daily_loss_dollars": config.max_daily_loss_dollars,
    }
    write_event("kill_switch_checked", payload)

    if state.get("kill_triggered") or hard_dollar_loss or hard_pct_loss:
        reason = state.get("reason") or "daily loss limit exceeded"
        if hard_dollar_loss:
            reason = f"max daily dollar loss exceeded: ${daily_pnl:,.2f}"
        if hard_pct_loss:
            reason = f"max daily pct loss exceeded: {daily_loss_pct:.2%}"

        state.update(
            {
                "kill_triggered": True,
                "triggered_at": utc_now().isoformat(),
                "reason": reason,
                "current_equity": account.equity,
                "daily_pnl": daily_pnl,
                "daily_loss_pct": daily_loss_pct,
            }
        )
        write_kill_switch_state(config, state)

        print(f"[kill-switch] BLOCKING EXECUTION: {reason}")
        write_event("kill_switch_triggered", state)
        return False

    return True


# =============================================================================
# Order frame / adaptive sizing
# =============================================================================

def normalize_order_frame(orders: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["symbol", "side", "delta_shares", "delta_dollars", "price", "reason"]

    if orders is None or orders.empty:
        return pd.DataFrame(columns=cols)

    out = orders.copy()

    defaults = {
        "symbol": "",
        "side": "",
        "delta_shares": 0,
        "delta_dollars": 0.0,
        "price": 0.0,
        "reason": "",
    }

    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    out["symbol"] = out["symbol"].map(safe_symbol)
    out["side"] = out["side"].astype(str).str.lower().str.strip()
    out["delta_shares"] = pd.to_numeric(out["delta_shares"], errors="coerce").fillna(0.0)
    out["delta_dollars"] = pd.to_numeric(out["delta_dollars"], errors="coerce").fillna(0.0)
    out["price"] = pd.to_numeric(out["price"], errors="coerce").fillna(0.0)
    out["reason"] = out["reason"].astype(str)

    # Preserve optional columns for sanity/adaptive sizing.
    for col in [
        "trade_confidence",
        "confidence",
        "confidence_score",
        "continuation_prob",
        "continuation_probability",
        "explosive_alpha_probability",
        "model_score",
        "volatility20",
        "volatility_20d",
        "atr_pct",
        "avg_dollar_volume20",
        "dollar_volume",
        "volume",
        "relative_volume_20",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[out["symbol"] != ""].copy()
    out = out[out["side"].isin(["buy", "sell"])].copy()

    return out.reset_index(drop=True)


def row_confidence(row: pd.Series, config: ExecutorConfig) -> float:
    for col in [
        "trade_confidence",
        "confidence",
        "confidence_score",
        "continuation_prob",
        "continuation_probability",
        "explosive_alpha_probability",
        "model_score",
    ]:
        if col in row and pd.notna(row.get(col)):
            value = to_float(row.get(col), config.default_confidence)
            if value > 1.0:
                value = value / 100.0
            return float(max(0.0, min(1.0, value)))

    return float(config.default_confidence)


def confidence_multiplier(confidence: float, config: ExecutorConfig) -> float:
    raw = 0.50 + confidence
    return float(max(config.min_confidence_scale, min(config.max_confidence_scale, raw)))


def build_order_plan(
    rebalance_orders: pd.DataFrame | None,
    exit_orders: pd.DataFrame | None,
    config: ExecutorConfig,
) -> pd.DataFrame:
    rebalance = normalize_order_frame(rebalance_orders)
    exits = normalize_order_frame(exit_orders)

    if not exits.empty:
        exits["reason"] = exits["reason"].replace("", "exit_removed_symbol")

    plan = pd.concat([exits, rebalance], ignore_index=True)

    if plan.empty:
        return pd.DataFrame()

    plan["abs_delta_dollars"] = plan["delta_dollars"].abs()
    plan["abs_delta_shares"] = plan["delta_shares"].abs()

    plan = plan[
        (plan["side"] == "sell")
        | (plan["abs_delta_dollars"] >= config.min_trade_dollars)
    ].copy()

    if plan.empty:
        return pd.DataFrame()

    plan["confidence_used"] = plan.apply(lambda r: row_confidence(r, config), axis=1)
    plan["confidence_multiplier"] = plan["confidence_used"].map(
        lambda x: confidence_multiplier(x, config)
    )

    buy_mask = plan["side"] == "buy"

    if config.enable_adaptive_sizing:
        plan.loc[buy_mask, "abs_delta_dollars"] = (
            plan.loc[buy_mask, "abs_delta_dollars"]
            * plan.loc[buy_mask, "confidence_multiplier"]
        )

    plan["order_qty"] = plan["abs_delta_shares"].apply(to_int)

    missing_qty = (plan["order_qty"] <= 0) & (plan["price"] > 0)
    plan.loc[missing_qty, "order_qty"] = (
        plan.loc[missing_qty, "abs_delta_dollars"] / plan.loc[missing_qty, "price"]
    ).fillna(0).astype(int)

    plan.loc[buy_mask, "capped_dollars"] = plan.loc[buy_mask, "abs_delta_dollars"].clip(
        upper=config.max_order_dollars
    )
    plan.loc[~buy_mask, "capped_dollars"] = plan.loc[~buy_mask, "abs_delta_dollars"]

    plan.loc[buy_mask & (plan["price"] > 0), "order_qty"] = (
        plan.loc[buy_mask & (plan["price"] > 0), "capped_dollars"]
        / plan.loc[buy_mask & (plan["price"] > 0), "price"]
    ).fillna(0).astype(int)

    plan = plan[plan["order_qty"] > 0].copy()
    plan = plan[plan["order_qty"] <= config.max_single_order_qty].copy()

    if plan.empty:
        return pd.DataFrame()

    side_rank = {"sell": 0, "buy": 1} if config.sell_first else {"buy": 0, "sell": 1}
    plan["side_rank"] = plan["side"].map(side_rank).fillna(9)

    plan = plan.sort_values(
        ["side_rank", "abs_delta_dollars"],
        ascending=[True, False],
    ).reset_index(drop=True)

    stamp = run_id()
    plan["client_order_id"] = [
        f"{config.order_tag_prefix}_{stamp}_{idx}_{row.symbol}_{row.side}"[:48]
        for idx, row in plan.iterrows()
    ]
    plan["planned_at"] = utc_now().isoformat()

    return plan


# =============================================================================
# Pre-trade sanity filter
# =============================================================================

def quote_to_payload(symbol: str, quote: Any) -> dict[str, Any]:
    bid_price = to_float(getattr(quote, "bid_price", 0.0))
    ask_price = to_float(getattr(quote, "ask_price", 0.0))
    bid_size = to_float(getattr(quote, "bid_size", 0.0))
    ask_size = to_float(getattr(quote, "ask_size", 0.0))
    timestamp = str(getattr(quote, "timestamp", ""))

    mid = (bid_price + ask_price) / 2.0 if bid_price > 0 and ask_price > 0 else 0.0
    spread = ask_price - bid_price if ask_price > 0 and bid_price > 0 else 0.0
    spread_pct = spread / mid if mid > 0 else 999.0

    return {
        "symbol": symbol,
        "bid_price": bid_price,
        "ask_price": ask_price,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "mid_price": mid,
        "spread": spread,
        "spread_pct": spread_pct,
        "quote_timestamp": timestamp,
    }


def latest_quote_payload(symbol: str, config: ExecutorConfig) -> dict[str, Any]:
    client = get_data_client()

    try:
        req = StockLatestQuoteRequest(
            symbol_or_symbols=[symbol],
            feed=config.quote_feed,
        )
        quotes = client.get_stock_latest_quote(req)
        quote = quotes.get(symbol) if isinstance(quotes, dict) else None

        if quote is None:
            return {
                "symbol": symbol,
                "quote_error": "no_quote_returned",
                "spread_pct": 999.0,
                "bid_size": 0.0,
                "ask_size": 0.0,
                "mid_price": 0.0,
            }

        return quote_to_payload(symbol, quote)

    except Exception as exc:
        return {
            "symbol": symbol,
            "quote_error": str(exc),
            "spread_pct": 999.0,
            "bid_size": 0.0,
            "ask_size": 0.0,
            "mid_price": 0.0,
        }


def quote_age_seconds(quote_timestamp: str) -> float:
    try:
        ts = pd.Timestamp(quote_timestamp)
        now = pd.Timestamp.now(tz=ts.tz)
        return max(0.0, float((now - ts).total_seconds()))
    except Exception:
        return 999999.0


def row_estimated_dollar_volume(row: pd.Series) -> float:
    for col in ["avg_dollar_volume20", "dollar_volume"]:
        value = to_float(row.get(col, 0.0))
        if value > 0:
            return value

    volume = to_float(row.get("volume", 0.0))
    price = to_float(row.get("price", 0.0))
    if volume > 0 and price > 0:
        return volume * price

    return 0.0


def row_volatility(row: pd.Series) -> tuple[float, float]:
    vol20 = 0.0
    atr_pct = 0.0

    for col in ["volatility20", "volatility_20d"]:
        value = to_float(row.get(col, 0.0))
        if value > 0:
            vol20 = value
            break

    atr_pct = to_float(row.get("atr_pct", 0.0))

    return vol20, atr_pct


def sanity_check_order(
    row: pd.Series,
    config: ExecutorConfig,
) -> tuple[bool, str, dict[str, Any]]:
    if not config.enable_pretrade_sanity:
        return True, "SANITY_DISABLED", {}

    symbol = safe_symbol(row.get("symbol"))
    side = str(row.get("side")).lower()

    # Usually let liquidation/risk sells through.
    if side == "sell" and not config.sanity_check_sells:
        return True, "SELL_SANITY_BYPASS", {}

    payload: dict[str, Any] = {
        "symbol": symbol,
        "side": side,
    }

    quote_payload = latest_quote_payload(symbol, config)
    payload.update(quote_payload)

    spread_pct = to_float(quote_payload.get("spread_pct"), 999.0)
    bid_size = to_float(quote_payload.get("bid_size"), 0.0)
    ask_size = to_float(quote_payload.get("ask_size"), 0.0)
    mid_price = to_float(quote_payload.get("mid_price"), 0.0)
    age = quote_age_seconds(str(quote_payload.get("quote_timestamp", "")))

    payload["quote_age_seconds"] = age

    if "quote_error" in quote_payload:
        return False, f"QUOTE_ERROR: {quote_payload['quote_error']}", payload

    if mid_price <= 0:
        return False, "BAD_QUOTE_MID_PRICE", payload

    if spread_pct > config.max_spread_pct:
        return False, f"SPREAD_TOO_WIDE: {spread_pct:.2%}", payload

    if min(bid_size, ask_size) < config.min_bid_ask_size:
        return False, f"QUOTE_SIZE_TOO_SMALL: bid={bid_size} ask={ask_size}", payload

    if age > config.max_quote_age_seconds:
        return False, f"QUOTE_STALE: {age:.1f}s", payload

    estimated_dv = row_estimated_dollar_volume(row)
    payload["estimated_dollar_volume"] = estimated_dv

    if estimated_dv > 0 and estimated_dv < config.min_estimated_dollar_volume:
        return False, f"LOW_DOLLAR_VOLUME: ${estimated_dv:,.2f}", payload

    order_notional = estimate_order_dollars(row)
    payload["order_notional"] = order_notional

    if estimated_dv > 0:
        order_vs_dv = order_notional / estimated_dv
        payload["order_vs_estimated_dv_pct"] = order_vs_dv

        if order_vs_dv > config.max_order_notional_vs_estimated_dv_pct:
            return False, f"ORDER_TOO_LARGE_VS_DV: {order_vs_dv:.2%}", payload

    vol20, atr_pct = row_volatility(row)
    payload["volatility20"] = vol20
    payload["atr_pct"] = atr_pct

    if vol20 > config.max_row_volatility20:
        return False, f"VOLATILITY20_TOO_HIGH: {vol20:.2%}", payload

    if atr_pct > config.max_row_atr_pct:
        return False, f"ATR_PCT_TOO_HIGH: {atr_pct:.2%}", payload

    return True, "SANITY_OK", payload


# =============================================================================
# Execution
# =============================================================================

def required_cash_buffer(account: AccountSnapshot, config: ExecutorConfig) -> float:
    return max(config.min_cash_dollars, account.equity * config.cash_buffer_pct)


def estimate_order_dollars(row: pd.Series) -> float:
    capped = to_float(row.get("capped_dollars", 0.0))
    if capped > 0:
        return capped

    qty = to_float(row.get("order_qty", 0.0))
    price = to_float(row.get("price", 0.0))

    if qty > 0 and price > 0:
        return qty * price

    return to_float(row.get("abs_delta_dollars", 0.0))


def submit_market_order(
    client: TradingClient,
    symbol: str,
    side: str,
    qty: int,
    config: ExecutorConfig,
    client_order_id: str,
) -> Any:
    order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        type=OrderType.MARKET,
        time_in_force=config.time_in_force,
        client_order_id=client_order_id,
    )

    return client.submit_order(order)


def poll_order_fill(
    client: TradingClient,
    order_id: str,
    planned_price: float,
    config: ExecutorConfig,
) -> dict[str, Any]:
    if not config.track_fills or not order_id:
        return {
            "fill_status": "not_tracked",
            "filled_qty": 0.0,
            "filled_avg_price": 0.0,
            "slippage_pct": 0.0,
            "slippage_dollars": 0.0,
        }

    last_order = None

    for _ in range(max(1, config.fill_poll_attempts)):
        try:
            last_order = client.get_order_by_id(order_id)
        except Exception as exc:
            return {
                "fill_status": "fill_lookup_error",
                "fill_error": str(exc),
                "filled_qty": 0.0,
                "filled_avg_price": 0.0,
                "slippage_pct": 0.0,
                "slippage_dollars": 0.0,
            }

        status = str(getattr(last_order, "status", ""))
        if status.lower() in {"filled", "partially_filled", "canceled", "rejected", "expired"}:
            break

        time.sleep(max(0.25, float(config.fill_poll_sleep_sec)))

    if last_order is None:
        return {
            "fill_status": "unknown",
            "filled_qty": 0.0,
            "filled_avg_price": 0.0,
            "slippage_pct": 0.0,
            "slippage_dollars": 0.0,
        }

    status = str(getattr(last_order, "status", ""))
    filled_qty = to_float(getattr(last_order, "filled_qty", 0.0))
    filled_avg_price = to_float(getattr(last_order, "filled_avg_price", 0.0))

    if planned_price > 0 and filled_avg_price > 0:
        slippage_pct = filled_avg_price / planned_price - 1.0
        slippage_dollars = (filled_avg_price - planned_price) * filled_qty
    else:
        slippage_pct = 0.0
        slippage_dollars = 0.0

    return {
        "fill_status": status,
        "filled_qty": filled_qty,
        "filled_avg_price": filled_avg_price,
        "slippage_pct": slippage_pct,
        "slippage_dollars": slippage_dollars,
    }


def execute_order_plan(
    client: TradingClient,
    order_plan: pd.DataFrame,
    config: ExecutorConfig,
) -> pd.DataFrame:
    results: list[dict[str, Any]] = []

    if order_plan.empty:
        return pd.DataFrame()

    for _, row in order_plan.iterrows():
        symbol = safe_symbol(row.get("symbol"))
        side = str(row.get("side")).lower()
        qty = to_int(row.get("order_qty"))
        estimated_dollars = estimate_order_dollars(row)
        client_order_id = str(row.get("client_order_id"))
        reason = str(row.get("reason", ""))
        planned_price = to_float(row.get("price"))

        timing_mult, timing_reason = session_timing_multiplier(client, side, config)

        if timing_mult <= 0.0:
            result = {
                "timestamp": utc_now().isoformat(),
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "estimated_dollars": estimated_dollars,
                "price": planned_price,
                "reason": reason,
                "client_order_id": client_order_id,
                "status": "skipped_timing",
                "message": timing_reason,
                "order_id": "",
            }
            print(f"[skip] {side.upper()} {qty} {symbol}: {timing_reason}")
            results.append(result)
            write_event("order_skipped_timing", result)
            continue

        if side == "buy" and timing_mult != 1.0:
            original_qty = qty
            qty = int(max(0, qty * timing_mult))
            estimated_dollars = estimated_dollars * timing_mult
            print(
                f"[timing] BUY {symbol}: qty {original_qty} -> {qty} "
                f"mult={timing_mult:.2f} reason={timing_reason}"
            )

        result_base = {
            "timestamp": utc_now().isoformat(),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "estimated_dollars": estimated_dollars,
            "price": planned_price,
            "reason": reason,
            "client_order_id": client_order_id,
            "timing_multiplier": timing_mult,
            "timing_reason": timing_reason,
            "confidence_used": to_float(row.get("confidence_used")),
            "confidence_multiplier": to_float(row.get("confidence_multiplier")),
        }

        if qty <= 0:
            result = {**result_base, "status": "skipped", "message": "qty <= 0", "order_id": ""}
            results.append(result)
            write_event("order_skipped", result)
            continue

        sanity_ok, sanity_reason, sanity_payload = sanity_check_order(row, config)
        if not sanity_ok:
            result = {
                **result_base,
                "status": "skipped_sanity",
                "message": sanity_reason,
                "order_id": "",
                **{f"sanity_{k}": v for k, v in sanity_payload.items()},
            }
            print(f"[skip] {side.upper()} {qty} {symbol}: {sanity_reason}")
            results.append(result)
            write_event("order_skipped_sanity", result)
            continue

        if side == "buy":
            account = get_account_snapshot(client)
            projected_cash = account.cash - estimated_dollars
            buffer = required_cash_buffer(account, config)

            if projected_cash < buffer:
                message = (
                    f"cash buffer violation | cash={account.cash:.2f} "
                    f"order={estimated_dollars:.2f} projected_cash={projected_cash:.2f} "
                    f"required_buffer={buffer:.2f}"
                )
                print(f"[skip] BUY {qty} {symbol}: {message}")

                result = {
                    **result_base,
                    "status": "skipped_cash_buffer",
                    "message": message,
                    "order_id": "",
                    "cash_before": account.cash,
                    "projected_cash": projected_cash,
                    "required_cash_buffer": buffer,
                }
                results.append(result)
                write_event("order_skipped_cash_buffer", result)
                continue

        try:
            submitted = submit_market_order(
                client=client,
                symbol=symbol,
                side=side,
                qty=qty,
                config=config,
                client_order_id=client_order_id,
            )

            order_id = str(getattr(submitted, "id", ""))
            status = str(getattr(submitted, "status", "submitted"))

            print(f"[submitted] {side.upper()} {qty} {symbol} status={status} id={order_id}")

            fill_info = poll_order_fill(
                client=client,
                order_id=order_id,
                planned_price=planned_price,
                config=config,
            )

            result = {
                **result_base,
                "status": status,
                "message": "submitted",
                "order_id": order_id,
                **{f"sanity_{k}": v for k, v in sanity_payload.items()},
                **fill_info,
            }
            results.append(result)
            write_event("order_submitted", result)

            time.sleep(config.sleep_between_orders_sec)

        except APIError as exc:
            message = str(exc)
            print(f"[error] Alpaca APIError {side.upper()} {qty} {symbol}: {message}")
            result = {**result_base, "status": "api_error", "message": message, "order_id": ""}
            results.append(result)
            write_event("order_api_error", result)

        except Exception as exc:
            message = str(exc)
            print(f"[error] submit failed {side.upper()} {qty} {symbol}: {message}")
            result = {**result_base, "status": "error", "message": message, "order_id": ""}
            results.append(result)
            write_event("order_error", result)

    return pd.DataFrame(results)


# =============================================================================
# Logging compatibility
# =============================================================================

def save_execution_results(
    order_plan: pd.DataFrame,
    results: pd.DataFrame,
    output_dir: Path | str | None = None,
    run_stamp: str | None = None,
) -> tuple[Path, Path]:
    output_path = Path(output_dir) if output_dir is not None else LOG_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    stamp = run_stamp or run_id()

    order_plan_path = output_path / f"order_plan_{stamp}.csv"
    results_path = output_path / f"order_results_{stamp}.csv"

    save_df(order_plan if order_plan is not None else pd.DataFrame(), order_plan_path)
    save_df(results if results is not None else pd.DataFrame(), results_path)

    return order_plan_path, results_path


def save_execution_summary(
    run_stamp: str,
    config: ExecutorConfig,
    account_before: AccountSnapshot,
    account_after: AccountSnapshot | None,
    order_plan: pd.DataFrame,
    results: pd.DataFrame,
    positions_before: pd.DataFrame,
    positions_after: pd.DataFrame | None,
) -> None:
    submitted = 0
    skipped = 0
    errors = 0
    total_slippage_dollars = 0.0
    avg_slippage_pct = 0.0
    sanity_skips = 0

    if results is not None and not results.empty and "status" in results.columns:
        statuses = results["status"].astype(str).str.lower()
        submitted = int(statuses.isin(["submitted", "accepted", "new", "filled", "partially_filled"]).sum())
        skipped = int(statuses.str.contains("skip", regex=True).sum())
        errors = int(statuses.str.contains("error|rejected|canceled", regex=True).sum())
        sanity_skips = int(statuses.eq("skipped_sanity").sum())

        if "slippage_dollars" in results.columns:
            total_slippage_dollars = float(pd.to_numeric(results["slippage_dollars"], errors="coerce").fillna(0.0).sum())
        if "slippage_pct" in results.columns:
            slip = pd.to_numeric(results["slippage_pct"], errors="coerce").dropna()
            avg_slippage_pct = float(slip.mean()) if not slip.empty else 0.0

    text = f"""
===== ALPACA EXECUTION SUMMARY =====
run_id                  : {run_stamp}
timestamp_utc           : {utc_now().isoformat()}
paper                   : {config.paper}
mode                    : EXECUTION

account_before_equity   : ${account_before.equity:,.2f}
account_before_cash     : ${account_before.cash:,.2f}
account_before_bp       : ${account_before.buying_power:,.2f}
account_after_equity    : {f"${account_after.equity:,.2f}" if account_after else "unavailable"}
account_after_cash      : {f"${account_after.cash:,.2f}" if account_after else "unavailable"}
account_after_bp        : {f"${account_after.buying_power:,.2f}" if account_after else "unavailable"}

orders_planned          : {0 if order_plan is None else len(order_plan)}
orders_result_rows      : {0 if results is None else len(results)}
submitted_count         : {submitted}
skipped_count           : {skipped}
sanity_skipped_count    : {sanity_skips}
error_count             : {errors}

positions_before        : {0 if positions_before is None else len(positions_before)}
positions_after         : {0 if positions_after is None else len(positions_after)}

cash_buffer_pct         : {config.cash_buffer_pct:.2%}
min_cash_dollars        : ${config.min_cash_dollars:,.2f}
min_trade_dollars       : ${config.min_trade_dollars:,.2f}
max_order_dollars       : ${config.max_order_dollars:,.2f}

kill_switch_enabled     : {config.enable_kill_switch}
max_daily_loss_pct      : {config.max_daily_loss_pct:.2%}
max_daily_loss_dollars  : ${config.max_daily_loss_dollars:,.2f}

adaptive_sizing_enabled : {config.enable_adaptive_sizing}
intraday_timing_enabled : {config.enable_intraday_timing}

pretrade_sanity_enabled : {config.enable_pretrade_sanity}
max_spread_pct          : {config.max_spread_pct:.2%}
min_bid_ask_size        : {config.min_bid_ask_size}
min_est_dollar_volume   : ${config.min_estimated_dollar_volume:,.2f}
max_quote_age_seconds   : {config.max_quote_age_seconds}
max_volatility20        : {config.max_row_volatility20:.2%}
max_atr_pct             : {config.max_row_atr_pct:.2%}

fill_tracking_enabled   : {config.track_fills}
total_slippage_dollars  : ${total_slippage_dollars:,.2f}
avg_slippage_pct        : {avg_slippage_pct:.4%}
""".strip()

    path = LOG_DIR / f"execution_summary_{run_stamp}.txt"
    path.write_text(text + "\n", encoding="utf-8")
    print(f"[saved] {path}")


# =============================================================================
# Runner
# =============================================================================

def run_execution(
    rebalance_orders: pd.DataFrame,
    exit_orders: pd.DataFrame | None = None,
    config: ExecutorConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, AccountSnapshot, pd.DataFrame]:
    config = config or ExecutorConfig()
    stamp = run_id()
    client = get_trading_client(config)

    account_before = get_account_snapshot(client)
    positions_before = get_positions_df(client)

    print("===== ALPACA EXECUTOR =====")
    print(f"paper         : {config.paper}")
    print("mode          : EXECUTION")
    print(f"equity        : ${account_before.equity:,.2f}")
    print(f"cash          : ${account_before.cash:,.2f}")
    print(f"buying_power  : ${account_before.buying_power:,.2f}")
    print(f"positions     : {len(positions_before)}")

    save_json(asdict(account_before), LOG_DIR / f"account_before_{stamp}.json")
    save_df(positions_before, LOG_DIR / f"positions_before_{stamp}.csv")

    if not market_open_guard(client, config):
        results = pd.DataFrame(
            [
                {
                    "timestamp": utc_now().isoformat(),
                    "status": "skipped_market_closed",
                    "message": "market is closed",
                }
            ]
        )
        save_df(pd.DataFrame(), LOG_DIR / f"order_plan_{stamp}.csv")
        save_df(results, LOG_DIR / f"order_results_{stamp}.csv")
        save_execution_summary(
            run_stamp=stamp,
            config=config,
            account_before=account_before,
            account_after=None,
            order_plan=pd.DataFrame(),
            results=results,
            positions_before=positions_before,
            positions_after=None,
        )
        return pd.DataFrame(), results, account_before, positions_before

    if not enforce_daily_loss_kill_switch(account_before, config):
        results = pd.DataFrame(
            [
                {
                    "timestamp": utc_now().isoformat(),
                    "status": "skipped_kill_switch",
                    "message": "daily loss kill switch active",
                }
            ]
        )
        save_df(pd.DataFrame(), LOG_DIR / f"order_plan_{stamp}.csv")
        save_df(results, LOG_DIR / f"order_results_{stamp}.csv")
        save_execution_summary(
            run_stamp=stamp,
            config=config,
            account_before=account_before,
            account_after=None,
            order_plan=pd.DataFrame(),
            results=results,
            positions_before=positions_before,
            positions_after=None,
        )
        return pd.DataFrame(), results, account_before, positions_before

    if config.cancel_open_orders_first:
        print("[orders] cancel_open_orders requested")
        client.cancel_orders()
        print("[orders] open orders canceled.")

    order_plan = build_order_plan(
        rebalance_orders=rebalance_orders,
        exit_orders=exit_orders,
        config=config,
    )

    if order_plan.empty:
        print("[plan] no valid orders after filtering.")
        results = pd.DataFrame()
        save_df(order_plan, LOG_DIR / f"order_plan_{stamp}.csv")
        save_df(results, LOG_DIR / f"order_results_{stamp}.csv")
        save_execution_summary(
            run_stamp=stamp,
            config=config,
            account_before=account_before,
            account_after=None,
            order_plan=order_plan,
            results=results,
            positions_before=positions_before,
            positions_after=None,
        )
        return order_plan, results, account_before, positions_before

    print("\n===== ORDER PLAN =====")
    display_cols = [
        "symbol",
        "side",
        "order_qty",
        "price",
        "abs_delta_dollars",
        "capped_dollars",
        "confidence_used",
        "confidence_multiplier",
        "reason",
    ]
    print(order_plan[[c for c in display_cols if c in order_plan.columns]].to_string(index=False))

    save_df(order_plan, LOG_DIR / f"order_plan_{stamp}.csv")

    results = execute_order_plan(client, order_plan, config)

    save_df(results, LOG_DIR / f"order_results_{stamp}.csv")

    account_after = get_account_snapshot(client)
    positions_after = get_positions_df(client)

    save_json(asdict(account_after), LOG_DIR / f"account_after_{stamp}.json")
    save_df(positions_after, LOG_DIR / f"positions_after_{stamp}.csv")

    save_execution_summary(
        run_stamp=stamp,
        config=config,
        account_before=account_before,
        account_after=account_after,
        order_plan=order_plan,
        results=results,
        positions_before=positions_before,
        positions_after=positions_after,
    )

    return order_plan, results, account_before, positions_before


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute Alpaca orders for the trading system.")

    parser.add_argument("--rebalance-orders", default="")
    parser.add_argument("--exit-orders", default="")

    parser.add_argument("--live", action="store_true")

    parser.add_argument("--cash-buffer", type=float, default=0.08)
    parser.add_argument("--min-trade-dollars", type=float, default=100.0)
    parser.add_argument("--max-order-dollars", type=float, default=15_000.0)
    parser.add_argument("--cancel-open-orders-first", action="store_true")

    parser.add_argument("--allow-closed-market", action="store_true")
    parser.add_argument("--wait-for-open", action="store_true")
    parser.add_argument("--max-wait-for-open-minutes", type=int, default=180)
    parser.add_argument("--market-clock-poll-sec", type=float, default=30.0)

    parser.add_argument("--disable-kill-switch", action="store_true")
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.02)
    parser.add_argument("--max-daily-loss-dollars", type=float, default=2_000.0)

    parser.add_argument("--disable-adaptive-sizing", action="store_true")
    parser.add_argument("--min-confidence-scale", type=float, default=0.35)
    parser.add_argument("--max-confidence-scale", type=float, default=1.25)

    parser.add_argument("--disable-intraday-timing", action="store_true")
    parser.add_argument("--avoid-first-minutes", type=int, default=5)
    parser.add_argument("--avoid-last-minutes", type=int, default=10)

    parser.add_argument("--disable-pretrade-sanity", action="store_true")
    parser.add_argument("--max-spread-pct", type=float, default=0.015)
    parser.add_argument("--min-bid-ask-size", type=int, default=1)
    parser.add_argument("--min-estimated-dollar-volume", type=float, default=500_000.0)
    parser.add_argument("--max-quote-age-seconds", type=int, default=600)
    parser.add_argument("--max-order-vs-dollar-volume-pct", type=float, default=0.05)
    parser.add_argument("--max-volatility20", type=float, default=0.20)
    parser.add_argument("--max-atr-pct", type=float, default=0.18)
    parser.add_argument("--sanity-check-sells", action="store_true")

    parser.add_argument("--disable-fill-tracking", action="store_true")
    parser.add_argument("--fill-poll-attempts", type=int, default=6)
    parser.add_argument("--fill-poll-sleep-sec", type=float, default=2.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rebalance_path = Path(args.rebalance_orders) if args.rebalance_orders else DEFAULT_REBALANCE_ORDERS
    exit_path = Path(args.exit_orders) if args.exit_orders else DEFAULT_EXIT_ORDERS

    if not rebalance_path.exists():
        raise RuntimeError(f"Rebalance orders not found: {rebalance_path}. Run trade.py first.")

    print(f"[executor] using rebalance_orders={rebalance_path}")
    print(f"[executor] using exit_orders={exit_path if exit_path.exists() else 'NONE'}")

    rebalance = pd.read_csv(rebalance_path)
    exits = (
        pd.read_csv(exit_path)
        if exit_path.exists() and exit_path.stat().st_size > 0
        else pd.DataFrame()
    )

    config = ExecutorConfig(
        paper=not args.live,
        cash_buffer_pct=args.cash_buffer,
        min_trade_dollars=args.min_trade_dollars,
        max_order_dollars=args.max_order_dollars,
        cancel_open_orders_first=args.cancel_open_orders_first,
        require_market_open=not args.allow_closed_market,
        wait_for_market_open=args.wait_for_open,
        max_wait_for_open_minutes=args.max_wait_for_open_minutes,
        market_clock_poll_sec=args.market_clock_poll_sec,
        enable_kill_switch=not args.disable_kill_switch,
        max_daily_loss_pct=args.max_daily_loss_pct,
        max_daily_loss_dollars=args.max_daily_loss_dollars,
        enable_adaptive_sizing=not args.disable_adaptive_sizing,
        min_confidence_scale=args.min_confidence_scale,
        max_confidence_scale=args.max_confidence_scale,
        enable_intraday_timing=not args.disable_intraday_timing,
        avoid_first_minutes=args.avoid_first_minutes,
        avoid_last_minutes=args.avoid_last_minutes,
        enable_pretrade_sanity=not args.disable_pretrade_sanity,
        max_spread_pct=args.max_spread_pct,
        min_bid_ask_size=args.min_bid_ask_size,
        min_estimated_dollar_volume=args.min_estimated_dollar_volume,
        max_quote_age_seconds=args.max_quote_age_seconds,
        max_order_notional_vs_estimated_dv_pct=args.max_order_vs_dollar_volume_pct,
        max_row_volatility20=args.max_volatility20,
        max_row_atr_pct=args.max_atr_pct,
        sanity_check_sells=args.sanity_check_sells,
        track_fills=not args.disable_fill_tracking,
        fill_poll_attempts=args.fill_poll_attempts,
        fill_poll_sleep_sec=args.fill_poll_sleep_sec,
    )

    run_execution(
        rebalance_orders=rebalance,
        exit_orders=exits,
        config=config,
    )


if __name__ == "__main__":
    main()
