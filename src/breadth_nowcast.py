"""当公开宽度 CSV 落后于最新收盘时，用行业 ETF 外推当日宽度。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SECTOR_ETFS


def _participation(prices: pd.DataFrame) -> pd.Series:
    columns = [ticker for ticker in SECTOR_ETFS if ticker in prices.columns]
    if not columns:
        return pd.Series(dtype=float)
    flags = []
    for ticker in columns:
        close = pd.to_numeric(prices[ticker], errors="coerce")
        above = (
            (close > close.rolling(20, min_periods=20).mean()).astype(float)
            + (close > close.rolling(50, min_periods=50).mean()).astype(float)
            + (close > close.rolling(200, min_periods=200).mean()).astype(float)
        ) / 3.0
        flags.append(above)
    return pd.concat(flags, axis=1).mean(axis=1) * 100.0


def nowcast_breadth(
    prices: pd.DataFrame,
    breadth: pd.DataFrame,
    uptrend: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """把昨日 CSV 基准加上今日行业参与度和 RSP/SPY 超额收益的外推。"""
    market_day = pd.Timestamp(prices.index[-1])
    breadth_day = pd.Timestamp(breadth.dropna(how="all").index[-1])
    if breadth_day >= market_day:
        return breadth, uptrend, ""

    participation = _participation(prices)
    if len(participation.dropna()) < 2 or "RSP" not in prices.columns or "SPY" not in prices.columns:
        return breadth, uptrend, "宽度 CSV 落后，但行业或 RSP/SPY 不足以做 T+0 预估。"

    part_delta = float(participation.iloc[-1] - participation.iloc[-2])
    rsp_ret = float(pd.to_numeric(prices["RSP"], errors="coerce").pct_change().iloc[-1])
    spy_ret = float(pd.to_numeric(prices["SPY"], errors="coerce").pct_change().iloc[-1])
    excess = 0.0 if not np.isfinite(rsp_ret) or not np.isfinite(spy_ret) else (rsp_ret - spy_ret) * 100.0
    if not np.isfinite(part_delta):
        part_delta = 0.0
    delta = float(np.clip(0.7 * part_delta + 0.3 * excess, -15.0, 15.0))

    broadened = breadth.copy()
    last = broadened.iloc[-1].copy()
    for column in ("Breadth_Index_Raw", "Breadth_Index_8MA"):
        if column in last.index and np.isfinite(last[column]):
            last[column] = float(np.clip(float(last[column]) + delta, 0.0, 100.0))
    broadened.loc[market_day] = last

    updated = uptrend.copy()
    up_last = updated.iloc[-1].copy()
    if "ratio" in up_last.index and np.isfinite(up_last["ratio"]):
        up_last["ratio"] = float(np.clip(float(up_last["ratio"]) + delta, 0.0, 100.0))
    updated.loc[market_day] = up_last
    note = (
        f"T+0 宽度预估：CSV {breadth_day.date()} 早于行情 {market_day.date()}，"
        f"ΔBreadth={delta:+.2f} 个百分点（行业参与度变化与 RSP-SPY 超额收益）。"
    )
    return broadened, updated, note
