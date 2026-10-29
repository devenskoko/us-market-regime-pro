"""SMA50 的 ATR 迟滞带，以及偏离度的连续仓位。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ATR_BAND, ATR_WINDOW, SIGMOID_SLOPE


def average_true_range(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    window: int = ATR_WINDOW,
) -> pd.Series:
    """14 日平均真实波幅。没有高低价时退回收盘价绝对变动。"""
    close = pd.to_numeric(close, errors="coerce")
    if high is None or low is None or high.empty or low.empty:
        true_range = close.diff().abs()
    else:
        high = pd.to_numeric(high, errors="coerce").reindex(close.index)
        low = pd.to_numeric(low, errors="coerce").reindex(close.index)
        prev = close.shift(1)
        true_range = pd.concat(
            [(high - low).abs(), (high - prev).abs(), (low - prev).abs()],
            axis=1,
        ).max(axis=1)
        true_range = true_range.where(high.notna() & low.notna(), close.diff().abs())
    return true_range.rolling(window, min_periods=window).mean()


def sigmoid_exposure(close: float, sma50: float, atr: float, slope: float = SIGMOID_SLOPE) -> float:
    """把相对 SMA50 的 ATR 偏离映射到 20%–100%。"""
    if not np.isfinite(close) or not np.isfinite(sma50) or not np.isfinite(atr) or atr <= 0:
        return 0.50
    x = float(np.clip((close - sma50) / atr, -8.0, 8.0))
    raw = 1.0 / (1.0 + np.exp(-slope * x))
    return float(0.20 + 0.80 * raw)


def sma50_hysteresis(
    close: pd.Series,
    sma50: pd.Series,
    atr: pd.Series,
    band: float = ATR_BAND,
) -> pd.DataFrame:
    """升档线和降档线分开。缓冲带内保持上一交易日的状态和仓位。

    降级后，收回升档线，或连续两个交易日收在 SMA50 之上，才回到升档。
    """
    frame = pd.DataFrame({"close": close, "sma50": sma50, "atr": atr}).sort_index()
    states: list[str | None] = []
    exposures: list[float] = []
    state: str | None = None
    exposure = np.nan
    prev_close = np.nan
    prev_sma = np.nan
    for row in frame.itertuples(index=False):
        price = float(row.close) if np.isfinite(row.close) else np.nan
        mid = float(row.sma50) if np.isfinite(row.sma50) else np.nan
        width = float(row.atr) if np.isfinite(row.atr) else np.nan
        if not np.isfinite(price) or not np.isfinite(mid) or not np.isfinite(width) or width <= 0:
            states.append(state)
            exposures.append(float(exposure) if np.isfinite(exposure) else np.nan)
            prev_close, prev_sma = price, mid
            continue
        upper = mid + band * width
        lower = mid - band * width
        proposed = sigmoid_exposure(price, mid, width)
        if state is None:
            state = "ABOVE" if price > mid else "BELOW"
            exposure = proposed
        elif state == "ABOVE":
            if price < lower:
                state = "BELOW"
                exposure = proposed
            elif price > upper:
                exposure = proposed
        else:
            two_closes = np.isfinite(prev_close) and np.isfinite(prev_sma) and price > mid and prev_close > prev_sma
            if price > upper or two_closes:
                state = "ABOVE"
                exposure = proposed
            elif price < lower:
                exposure = proposed
        states.append(state)
        exposures.append(float(exposure))
        prev_close, prev_sma = price, mid
    frame["state"] = states
    frame["exposure"] = exposures
    frame["upper"] = frame["sma50"] + band * frame["atr"]
    frame["lower"] = frame["sma50"] - band * frame["atr"]
    return frame
