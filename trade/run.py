"""
SANITIZED EXECUTION ENGINE (PORTFOLIO VERSION)

This module demonstrates a simplified execution framework for
a quantitative trading system.

IMPORTANT:
- No real alpha logic included
- No production risk model included
- Execution logic is simplified / illustrative only
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
import pandas as pd
import numpy as np


# =========================
# CONFIG
# =========================

@dataclass
class ExecutorConfig:
    paper: bool = True
    poll_seconds: int = 30
    max_position_size: float = 0.20
    min_trade_dollars: float = 100.0


# =========================
# MOCK BROKER (SAFE)
# =========================

class MockBroker:
    """
    Simulated broker interface for demonstration.
    """

    def __init__(self):
        self.cash = 100000
        self.positions = {}

    def submit_order(self, symbol: str, side: str, qty: int, price: float):
        notional = qty * price

        if side == "buy":
            self.cash -= notional
            self.positions[symbol] = self.positions.get(symbol, 0) + qty
        else:
            self.cash += notional
            self.positions[symbol] = self.positions.get(symbol, 0) - qty

        return {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "timestamp": datetime.utcnow().isoformat()
        }


# =========================
# ORDER PROCESSING
# =========================

def normalize_orders(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["side"] = df["side"].astype(str).str.lower()

    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    df["delta_dollars"] = pd.to_numeric(df["delta_dollars"], errors="coerce").fillna(0)

    return df


def build_order_plan(df: pd.DataFrame, config: ExecutorConfig):

    df = normalize_orders(df)

    if df.empty:
        return df

    df["order_qty"] = (df["delta_dollars"] / df["price"]).fillna(0).astype(int)
    df = df[df["order_qty"] > 0]

    df = df[df["delta_dollars"] >= config.min_trade_dollars]

    return df.reset_index(drop=True)


# =========================
# EXECUTION LOOP
# =========================

class SimpleExecutor:

    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.broker = MockBroker()

    def execute_orders(self, orders: pd.DataFrame):

        if orders.empty:
            print("[executor] no orders to execute")
            return pd.DataFrame()

        results = []

        for _, row in orders.iterrows():

            symbol = row["symbol"]
            side = row["side"]
            qty = int(row["order_qty"])
            price = float(row["price"])

            if qty <= 0:
                continue

            result = self.broker.submit_order(symbol, side, qty, price)

            print(f"[executed] {side.upper()} {qty} {symbol} @ {price}")

            results.append(result)

        return pd.DataFrame(results)

    def run_once(self, orders: pd.DataFrame):

        plan = build_order_plan(orders, self.config)

        print("\n===== ORDER PLAN =====")
        print(plan.head())

        results = self.execute_orders(plan)

        print("\n===== EXECUTION RESULTS =====")
        print(results.head())

        return results


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    config = ExecutorConfig()

    # Example dummy orders
    sample_orders = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "NVDA"],
        "side": ["buy", "buy", "sell"],
        "delta_dollars": [5000, 3000, 2000],
        "price": [170, 320, 900],
    })

    executor = SimpleExecutor(config)
    executor.run_once(sample_orders)

