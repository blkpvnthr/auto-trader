```python
"""
SANITIZED PORTFOLIO ALLOCATION MODULE (PORTFOLIO VERSION)

This module demonstrates a simplified portfolio allocation framework.

IMPORTANT:
- All proprietary optimization logic has been removed
- No alpha-generating scoring is included
- This is a structural demonstration only
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================

@dataclass
class OptimizerConfig:
    starting_equity: float = 100_000.0
    cash_buffer_pct: float = 0.10
    max_gross_exposure: float = 0.90
    max_position_weight: float = 0.20
    min_position_weight: float = 0.02
    max_positions: int = 10
    min_trade_dollars: float = 100.0


# =========================
# BASIC SCORING (SAFE)
# =========================

def simple_score(df: pd.DataFrame) -> pd.Series:
    """
    Simplified scoring function.

    NOTE:
    Real system uses proprietary alpha signals.
    This is only for demonstration.
    """
    return (
        df["prediction"].fillna(0)
        + df.get("return_5d", 0)
    )


# =========================
# WEIGHT ALLOCATION
# =========================

def allocate_weights(
    candidates: pd.DataFrame,
    config: OptimizerConfig,
) -> pd.DataFrame:

    df = candidates.copy()

    if df.empty:
        return df

    df["score"] = simple_score(df)

    df = df.sort_values("score", ascending=False).head(config.max_positions)

    total_score = df["score"].sum()

    if total_score <= 0:
        df["weight"] = 1.0 / len(df)
    else:
        df["weight"] = df["score"] / total_score

    # Apply constraints
    max_weight = config.max_position_weight
    df["weight"] = df["weight"].clip(upper=max_weight)

    # Normalize to gross exposure
    gross_target = min(config.max_gross_exposure, 1.0 - config.cash_buffer_pct)

    total_weight = df["weight"].sum()

    if total_weight > 0:
        df["weight"] = df["weight"] / total_weight * gross_target

    # Dollar allocation
    df["target_dollars"] = df["weight"] * config.starting_equity

    df = df[df["target_dollars"] >= config.min_trade_dollars]

    return df.reset_index(drop=True)


# =========================
# REBALANCING
# =========================

def build_rebalance_orders(
    targets: pd.DataFrame,
    current_positions: pd.DataFrame,
    equity: float,
):

    if targets.empty:
        return pd.DataFrame()

    df = targets.copy()

    if current_positions is None or current_positions.empty:
        df["current_dollars"] = 0
    else:
        pos = current_positions.copy()
        pos["symbol"] = pos["symbol"].astype(str)

        df = df.merge(
            pos[["symbol", "market_value"]],
            on="symbol",
            how="left"
        )

        df["current_dollars"] = df["market_value"].fillna(0)

    df["delta_dollars"] = df["target_dollars"] - df["current_dollars"]
    df["side"] = np.where(df["delta_dollars"] > 0, "buy", "sell")

    return df


# =========================
# SUMMARY
# =========================

def summarize(targets: pd.DataFrame, equity: float):

    if targets.empty:
        return {
            "positions": 0,
            "gross_exposure": 0.0,
            "cash": 1.0
        }

    gross = targets["weight"].sum()

    return {
        "positions": len(targets),
        "gross_exposure": float(gross),
        "cash": float(1.0 - gross),
        "equity": equity
    }
```
