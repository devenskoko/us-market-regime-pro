"""引擎 C：底层基准均线排列与 HYG/IEF 信贷门控。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.benchmark_router import route_benchmark
from src.config import FFILL_LIMIT, HIGH_WINDOW, SMA_FAST, SMA_MID, SMA_SLOW
from src.hysteresis import average_true_range, sma50_hysteresis
from src.leverage_gate import realized_vol


@dataclass
class TrendCreditState:
    ticker: str
    benchmark: str
    asof: pd.Timestamp | None
    close: float | None
    sma20: float | None
    sma50: float | None
    sma200: float | None
    sma20_slope: float | None
    high_252: float | None
    ratio_to_high: float | None
    drawdown_252: float | None
    low_5: float | None
    trend_is_bull: bool
    stacked_bull: bool
    below_sma50: bool
    short_momentum_weak: bool
    hyg_ief: float | None
    hyg_ief_sma50: float | None
    credit_risk_off: bool
    ready: bool
    reason: str
    atr14: float | None = None
    sma50_upper: float | None = None
    sma50_lower: float | None = None
    hysteresis_state: str = ""
    confirmed_below_sma50: bool = False
    sticky_exposure: float | None = None
    hv20: float | None = None


def _last_sma(series: pd.Series, window: int) -> float:
    value = series.rolling(window, min_periods=window).mean().iloc[-1]
    return float(value) if np.isfinite(value) else np.nan


def assess_trend_credit(
    prices: pd.DataFrame,
    ticker: str,
    asof: pd.Timestamp | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
) -> TrendCreditState:
    benchmark = route_benchmark(ticker)
    if asof is None:
        asof = pd.Timestamp(prices.index[-1])
    history = prices.loc[:asof]
    empty = TrendCreditState(
        ticker=ticker,
        benchmark=benchmark,
        asof=asof,
        close=None,
        sma20=None,
        sma50=None,
        sma200=None,
        sma20_slope=None,
        high_252=None,
        ratio_to_high=None,
        drawdown_252=None,
        low_5=None,
        trend_is_bull=False,
        stacked_bull=False,
        below_sma50=False,
        short_momentum_weak=False,
        hyg_ief=None,
        hyg_ief_sma50=None,
        credit_risk_off=False,
        ready=False,
        reason="",
    )
    if benchmark not in history.columns:
        empty.reason = f"缺少底层基准 {benchmark}"
        return empty

    close = pd.to_numeric(history[benchmark], errors="coerce").dropna()
    if close.empty:
        empty.reason = f"{benchmark} 无有效收盘价"
        return empty

    sma20 = _last_sma(close, SMA_FAST)
    sma50 = _last_sma(close, SMA_MID)
    sma200 = _last_sma(close, SMA_SLOW)
    sma20_series = close.rolling(SMA_FAST, min_periods=SMA_FAST).mean()
    if len(sma20_series.dropna()) > SMA_FAST:
        previous = sma20_series.iloc[-1 - SMA_FAST]
        current = sma20_series.iloc[-1]
        slope = float(current / previous - 1.0) if np.isfinite(previous) and previous != 0 else np.nan
    else:
        slope = np.nan

    peak = float(close.iloc[-HIGH_WINDOW:].max()) if len(close) >= HIGH_WINDOW else float(close.max())
    last = float(close.iloc[-1])
    ratio_to_high = last / peak if peak else np.nan
    low_5 = float(close.iloc[-5:].min()) if len(close) >= 5 else last
    trend_is_bull = bool(np.isfinite(sma200) and last > sma200)
    stacked = bool(np.isfinite(sma50) and np.isfinite(sma200) and last > sma50 > sma200)
    below_mid = bool(np.isfinite(sma200) and np.isfinite(sma50) and last > sma200 and last < sma50)
    weak = bool(np.isfinite(sma20) and last < sma20)

    hyg_ief = np.nan
    hyg_sma = np.nan
    credit_off = False
    if {"HYG", "IEF"}.issubset(history.columns):
        ratio = pd.to_numeric(history["HYG"], errors="coerce") / pd.to_numeric(history["IEF"], errors="coerce")
        ratio = ratio.ffill(limit=FFILL_LIMIT)
        hyg_sma = _last_sma(ratio.dropna(), SMA_MID)
        if len(ratio.dropna()):
            hyg_ief = float(ratio.dropna().iloc[-1])
        credit_off = bool(np.isfinite(hyg_ief) and np.isfinite(hyg_sma) and hyg_ief < hyg_sma)

    high_series = None
    low_series = None
    if high is not None and benchmark in high.columns:
        high_series = pd.to_numeric(high.loc[:asof, benchmark], errors="coerce")
    if low is not None and benchmark in low.columns:
        low_series = pd.to_numeric(low.loc[:asof, benchmark], errors="coerce")
    atr_series = average_true_range(close, high_series, low_series)
    sma50_series = close.rolling(SMA_MID, min_periods=SMA_MID).mean()
    path = sma50_hysteresis(close, sma50_series, atr_series)
    atr14 = float(path["atr"].iloc[-1]) if np.isfinite(path["atr"].iloc[-1]) else np.nan
    upper = float(path["upper"].iloc[-1]) if np.isfinite(path["upper"].iloc[-1]) else np.nan
    lower = float(path["lower"].iloc[-1]) if np.isfinite(path["lower"].iloc[-1]) else np.nan
    hyst = path["state"].iloc[-1] or ""
    sticky = float(path["exposure"].iloc[-1]) if np.isfinite(path["exposure"].iloc[-1]) else np.nan
    hv_series = realized_vol(close)
    hv20 = float(hv_series.iloc[-1]) if np.isfinite(hv_series.iloc[-1]) else np.nan
    confirmed_below = hyst == "BELOW"

    ready = bool(np.isfinite(sma200) and np.isfinite(sma50) and np.isfinite(sma20))
    reason = "OK" if ready else f"{benchmark} 均线尚未预热"
    return TrendCreditState(
        ticker=ticker,
        benchmark=benchmark,
        asof=pd.Timestamp(close.index[-1]),
        close=last,
        sma20=None if not np.isfinite(sma20) else sma20,
        sma50=None if not np.isfinite(sma50) else sma50,
        sma200=None if not np.isfinite(sma200) else sma200,
        sma20_slope=None if not np.isfinite(slope) else slope,
        high_252=peak,
        ratio_to_high=float(ratio_to_high),
        drawdown_252=float(1.0 - ratio_to_high),
        low_5=low_5,
        trend_is_bull=trend_is_bull,
        stacked_bull=stacked,
        below_sma50=below_mid,
        short_momentum_weak=weak,
        hyg_ief=None if not np.isfinite(hyg_ief) else float(hyg_ief),
        hyg_ief_sma50=None if not np.isfinite(hyg_sma) else float(hyg_sma),
        credit_risk_off=credit_off,
        ready=ready,
        reason=reason,
        atr14=None if not np.isfinite(atr14) else atr14,
        sma50_upper=None if not np.isfinite(upper) else upper,
        sma50_lower=None if not np.isfinite(lower) else lower,
        hysteresis_state=str(hyst),
        confirmed_below_sma50=confirmed_below,
        sticky_exposure=None if not np.isfinite(sticky) else sticky,
        hv20=None if not np.isfinite(hv20) else hv20,
    )
