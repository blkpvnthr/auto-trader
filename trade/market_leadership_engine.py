from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class LeadershipConfig:
    benchmark: str = "QQQ"
    lags: tuple[int, ...] = (1, 2, 3, 5)
    lookback: int = 120
    min_obs: int = 60


class MarketLeadershipEngine:
    def __init__(self, config: LeadershipConfig = LeadershipConfig()):
        self.config = config

    def compute_returns(self, close_df: pd.DataFrame) -> pd.DataFrame:
        return close_df.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    def score_leaders(self, close_df: pd.DataFrame) -> pd.DataFrame:
        """
        close_df columns should be symbols, including benchmark.
        Example columns: QQQ, NVDA, MSFT, AAPL, SOXX, TQQQ, SQQQ
        """

        cfg = self.config

        if cfg.benchmark not in close_df.columns:
            raise ValueError(f"Benchmark {cfg.benchmark} missing from close_df")

        returns = self.compute_returns(close_df).tail(cfg.lookback)

        benchmark_ret = returns[cfg.benchmark]

        rows = []

        for symbol in returns.columns:
            if symbol == cfg.benchmark:
                continue

            symbol_ret = returns[symbol]

            lag_scores = []

            for lag in cfg.lags:
                # symbol now vs benchmark future
                future_benchmark = benchmark_ret.shift(-lag)

                aligned = pd.concat([symbol_ret, future_benchmark], axis=1).dropna()

                if len(aligned) < cfg.min_obs:
                    continue

                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

                if pd.notna(corr):
                    lag_scores.append(corr)

            if not lag_scores:
                continue

            recent_rs = close_df[symbol] / close_df[cfg.benchmark]
            rs_slope = recent_rs.tail(20).pct_change(fill_method=None).mean()

            volume_proxy = abs(symbol_ret.tail(10)).mean()

            lead_score = (
                0.55 * np.nanmax(lag_scores)
                + 0.30 * np.nan_to_num(rs_slope)
                + 0.15 * np.nan_to_num(volume_proxy)
            )

            rows.append({
                "symbol": symbol,
                "lead_score": float(lead_score),
                "best_lag_corr": float(np.nanmax(lag_scores)),
                "avg_lag_corr": float(np.nanmean(lag_scores)),
                "rs_slope": float(np.nan_to_num(rs_slope)),
                "volatility_impulse": float(np.nan_to_num(volume_proxy)),
            })

        if not rows:
            return pd.DataFrame()

        return (
            pd.DataFrame(rows)
            .sort_values("lead_score", ascending=False)
            .reset_index(drop=True)
        )