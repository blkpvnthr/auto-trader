#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "daily_system_outputs"
if not DATA_DIR.exists():
    DATA_DIR = ROOT

OUT_DIR = DATA_DIR / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "continuation_importance": DATA_DIR / "continuation_feature_importance.csv",
    "next_bar_importance": DATA_DIR / "next_bar_feature_importance.csv",
    "continuation_predictions": DATA_DIR / "continuation_predictions.csv",
    "next_bar_predictions": DATA_DIR / "next_bar_predictions.csv",
    "continuation_thresholds": DATA_DIR / "continuation_threshold_table.csv",
    "next_bar_thresholds": DATA_DIR / "next_bar_threshold_table.csv",
    "top_continuation": DATA_DIR / "top_continuation_trades.csv",
    "top_next_bar": DATA_DIR / "top_next_bar_trades.csv",
    "trade_actions": DATA_DIR / "next_bar_trade_actions.csv",
    "raw_bars": DATA_DIR / "daily_raw_bars.csv",
    "universe": DATA_DIR / "download_universe_symbols.csv",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[skip] missing: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    return df


def save_barh(df, x, y, title, xlabel, out_path, top_n=25):
    if df.empty or x not in df.columns or y not in df.columns:
        print(f"[skip] {title}")
        return

    plot_df = df.sort_values(x, ascending=False).head(top_n).copy()
    plot_df = plot_df.sort_values(x, ascending=True)

    plt.figure(figsize=(12, 8))
    plt.barh(plot_df[y].astype(str), plot_df[x])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[saved] {out_path}")


def plot_feature_importance():
    cont = read_csv(FILES["continuation_importance"])
    nxt = read_csv(FILES["next_bar_importance"])

    save_barh(
        cont,
        x="importance_auc_drop",
        y="feature",
        title="Continuation Model Feature Importance",
        xlabel="AUC Drop After Permutation",
        out_path=OUT_DIR / "continuation_feature_importance.png",
    )

    save_barh(
        nxt,
        x="importance_auc_drop",
        y="feature",
        title="Next-Bar Model Feature Importance",
        xlabel="AUC Drop After Permutation",
        out_path=OUT_DIR / "next_bar_feature_importance.png",
    )


def plot_threshold_table():
    for name, path in [
        ("continuation", FILES["continuation_thresholds"]),
        ("next_bar", FILES["next_bar_thresholds"]),
    ]:
        df = read_csv(path)

        if df.empty or "threshold" not in df.columns:
            continue

        plt.figure(figsize=(12, 7))

        for col in ["precision", "recall", "accuracy", "score"]:
            if col in df.columns:
                plt.plot(df["threshold"], df[col], label=col)

        plt.title(f"{name.title()} Threshold Optimization")
        plt.xlabel("Threshold")
        plt.ylabel("Metric")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out = OUT_DIR / f"{name}_threshold_optimization.png"
        plt.savefig(out, dpi=150)
        plt.close()

        print(f"[saved] {out}")

        if "trade_count" in df.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(df["threshold"], df["trade_count"])
            plt.title(f"{name.title()} Trade Count by Threshold")
            plt.xlabel("Threshold")
            plt.ylabel("Trade Count")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            out = OUT_DIR / f"{name}_threshold_trade_count.png"
            plt.savefig(out, dpi=150)
            plt.close()

            print(f"[saved] {out}")


def plot_prediction_distributions():
    configs = [
        (
            "continuation",
            FILES["continuation_predictions"],
            "continuation_probability",
            "Continuation Probability Distribution",
        ),
        (
            "next_bar",
            FILES["next_bar_predictions"],
            "next_bar_positive_probability",
            "Next-Bar Positive Probability Distribution",
        ),
    ]

    for name, path, prob_col, title in configs:
        df = read_csv(path)

        if df.empty or prob_col not in df.columns:
            continue

        plt.figure(figsize=(10, 6))
        plt.hist(df[prob_col].dropna(), bins=50)
        plt.title(title)
        plt.xlabel(prob_col)
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out = OUT_DIR / f"{name}_probability_distribution.png"
        plt.savefig(out, dpi=150)
        plt.close()

        print(f"[saved] {out}")


def plot_latest_top_trades():
    configs = [
        (
            "continuation",
            FILES["top_continuation"],
            "continuation_probability",
            "Top Continuation Trades - Latest Date",
        ),
        (
            "next_bar",
            FILES["top_next_bar"],
            "next_bar_positive_probability",
            "Top Next-Bar Trades - Latest Date",
        ),
    ]

    for name, path, prob_col, title in configs:
        df = read_csv(path)

        if df.empty or "timestamp" not in df.columns or prob_col not in df.columns:
            continue

        latest_date = df["timestamp"].max()
        latest = df[df["timestamp"] == latest_date].copy()

        latest = latest.sort_values(prob_col, ascending=True)

        plt.figure(figsize=(12, 7))
        plt.barh(latest["symbol"].astype(str), latest[prob_col])
        plt.title(f"{title} ({latest_date.date()})")
        plt.xlabel(prob_col)
        plt.ylabel("Symbol")
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        out = OUT_DIR / f"{name}_latest_top_trades.png"
        plt.savefig(out, dpi=150)
        plt.close()

        print(f"[saved] {out}")


def plot_trade_actions():
    df = read_csv(FILES["trade_actions"])

    if df.empty or "trade_action" not in df.columns:
        return

    counts = df["trade_action"].value_counts().sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(counts.index.astype(str), counts.values)
    plt.title("Trade Action Counts")
    plt.xlabel("Count")
    plt.ylabel("Trade Action")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    out = OUT_DIR / "trade_action_counts.png"
    plt.savefig(out, dpi=150)
    plt.close()

    print(f"[saved] {out}")

    if "timestamp" in df.columns:
        daily = (
            df.groupby([df["timestamp"].dt.date, "trade_action"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )

        plt.figure(figsize=(14, 7))
        for col in daily.columns:
            plt.plot(daily.index, daily[col], label=col)

        plt.title("Trade Actions Over Time")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out = OUT_DIR / "trade_actions_over_time.png"
        plt.savefig(out, dpi=150)
        plt.close()

        print(f"[saved] {out}")


def plot_top_symbol_price_history():
    raw = read_csv(FILES["raw_bars"])
    trades = read_csv(FILES["top_next_bar"])

    if raw.empty or trades.empty:
        return

    if "symbol" not in raw.columns or "close" not in raw.columns:
        return

    prob_col = "next_bar_positive_probability"
    if prob_col not in trades.columns:
        prob_col = "continuation_probability" if "continuation_probability" in trades.columns else None

    if prob_col is None:
        return

    latest_date = trades["timestamp"].max()
    top_symbols = (
        trades[trades["timestamp"] == latest_date]
        .sort_values(prob_col, ascending=False)
        .head(10)["symbol"]
        .astype(str)
        .tolist()
    )

    for sym in top_symbols:
        g = raw[raw["symbol"].astype(str) == sym].copy()

        if g.empty:
            continue

        g = g.sort_values("timestamp").tail(120)

        plt.figure(figsize=(12, 6))
        plt.plot(g["timestamp"], g["close"])
        plt.title(f"{sym} Close Price - Last 120 Bars")
        plt.xlabel("Date")
        plt.ylabel("Close")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out = OUT_DIR / f"price_history_{sym}.png"
        plt.savefig(out, dpi=150)
        plt.close()

        print(f"[saved] {out}")


def write_summary():
    rows = []

    for name, path in FILES.items():
        if path.exists():
            try:
                df = pd.read_csv(path)
                rows.append(
                    {
                        "file": path.name,
                        "rows": len(df),
                        "columns": len(df.columns),
                    }
                )
            except Exception:
                rows.append(
                    {
                        "file": path.name,
                        "rows": None,
                        "columns": None,
                    }
                )

    summary = pd.DataFrame(rows)
    out = OUT_DIR / "visualization_input_summary.csv"
    summary.to_csv(out, index=False)

    print(f"[saved] {out}")


def main():
    print("===== VISUALIZING MODEL OUTPUTS =====")
    print(f"[data_dir] {DATA_DIR}")
    print(f"[out_dir]  {OUT_DIR}")

    write_summary()
    plot_feature_importance()
    plot_threshold_table()
    plot_prediction_distributions()
    plot_latest_top_trades()
    plot_trade_actions()
    plot_top_symbol_price_history()

    print("\n[done] visualizations created.")


if __name__ == "__main__":
    main()