"""引擎 A：VIX 期限结构评分与正交趋势门控。

Score_vol 只由期权期限结构决定。趋势布尔值只改变展示状态，不改分数。
时间步 t 的滚动统计只使用 [t-W+1, t]。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from src.config import (
    FACTOR_TICKERS,
    PERCENTILE_MIN_VALID_FRACTION,
    PERCENTILE_WINDOW,
    READINESS_MIN_BARS,
    READINESS_TICKERS,
    Z_WINDOW,
)

CRASH_BUY = "🚨 CRASH BUY"
STRONG_BUY = "🟢 STRONG BUY"
BUY_DIP = "🟡 BUY DIP"
NEUTRAL = "⏸ NEUTRAL"
SELL_HEDGE = "🟠 SELL/HEDGE"
STRONG_SELL = "🔴 STRONG SELL"
EUPHORIA = "🔥 EUPHORIA"
WAIT_VOL = "WAIT Vol/波动"
WAIT_MOM = "WAIT Mom/动量"
NO_TRADE = "🚫 NO TRADE"
HOLD_CORE = "HOLD Core/核心"
HOLD_VOL = "HOLD Vol/波动"
BACKWARDATION_ALERT = "⚠️ BACKWARDATION DE-LEVERAGE（期权倒挂去杠杆警报）"
DATA_NA = "DATA N/A"


@dataclass
class VolAssessment:
    status: str
    score: float | None
    display_state: str | None = None
    raw_signal: str | None = None
    z_ts: float | None = None
    r_ts: float | None = None
    r_ts_prev: float | None = None
    r_basis: float | None = None
    vix: float | None = None
    p_vvix: float | None = None
    p_skew: float | None = None
    factors: dict[str, float] = field(default_factory=dict)
    is_backwardation_risk: bool = False
    backwardation_alert: str | None = None
    asof: pd.Timestamp | None = None
    reason: str = ""


def rolling_percentile_rank(series: pd.Series, window: int = PERCENTILE_WINDOW) -> pd.Series:
    """经验百分位秩，窗口闭区间含当日，取值 [0, 100]。"""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    min_valid = max(1, int(window * PERCENTILE_MIN_VALID_FRACTION))
    for end in range(window - 1, len(values)):
        sample = values[end - window + 1 : end + 1]
        current = sample[-1]
        if not np.isfinite(current):
            continue
        valid = sample[np.isfinite(sample)]
        if len(valid) < min_valid:
            continue
        less = float(np.sum(valid < current))
        equal = float(np.sum(valid == current))
        out[end] = 100.0 * (less + 0.5 * equal) / float(len(valid))
    return pd.Series(out, index=series.index)


def _factor_term_z(z_ts: float) -> float:
    if z_ts >= 2.5:
        return 3.0
    if z_ts >= 2.0:
        return 2.0
    if z_ts >= 1.2:
        return 1.0
    if z_ts > -1.2:
        return 0.0
    if z_ts > -2.0:
        return -1.5
    return -2.5


def _factor_term_level(r_ts: float) -> float:
    if r_ts >= 1.08:
        return 2.0
    if r_ts >= 1.00:
        return 1.0
    if r_ts >= 0.88:
        return 0.0
    return -1.5


def _factor_basis(r_basis: float, vix: float) -> float:
    if r_basis >= 1.10 and vix > 25.0:
        return 1.5
    if r_basis < 0.85 and vix < 15.0:
        return -1.5
    return 0.0


def _factor_skew(p_skew: float, r_ts: float) -> float:
    if p_skew >= 85.0 and r_ts < 0.92:
        return -1.5
    if p_skew <= 15.0 and r_ts > 1.00:
        return 1.0
    return 0.0


def _factor_vvix(p_vvix: float, z_ts: float) -> float:
    if p_vvix >= 90.0 and z_ts >= 1.5:
        return 1.0
    if p_vvix >= 85.0 and z_ts < 0.0:
        return -1.0
    return 0.0


def score_vol_components(
    z_ts: float,
    r_ts: float,
    r_basis: float,
    vix: float,
    p_vvix: float,
    p_skew: float,
) -> dict[str, float]:
    """五项结构分。函数签名里没有趋势，分数不可能被均线改写。"""
    factors = {
        "s1": _factor_term_z(z_ts),
        "s2": _factor_term_level(r_ts),
        "s3": _factor_basis(r_basis, vix),
        "s4": _factor_skew(p_skew, r_ts),
        "s5": _factor_vvix(p_vvix, z_ts),
    }
    factors["score"] = float(sum(factors.values()))
    return factors


def raw_signal_name(score: float) -> str:
    if score >= 6.0:
        return "CRASH_BUY"
    if score >= 5.0:
        return "STRONG_BUY"
    if score >= 3.0:
        return "BUY_DIP"
    if score > -2.0:
        return "NEUTRAL"
    if score > -5.0:
        return "SELL_HEDGE"
    if score > -6.0:
        return "STRONG_SELL"
    return "EUPHORIA"


def apply_orthogonal_gates(
    raw_score: float,
    trend_is_bull: bool,
    *,
    r_ts: float,
    r_ts_prev: float,
    z_ts: float,
    p_skew: float,
    close: float,
    low_5: float,
    short_momentum_weak: bool,
) -> str:
    """只返回展示状态。raw_score 是只读入参。"""
    signal = raw_signal_name(raw_score)
    if signal == "CRASH_BUY":
        if r_ts <= r_ts_prev:
            return CRASH_BUY
        return WAIT_VOL
    if signal == "STRONG_BUY":
        if z_ts >= 1.8 and close > low_5:
            return STRONG_BUY
        return WAIT_MOM
    if signal == "BUY_DIP":
        if trend_is_bull:
            return BUY_DIP
        return NO_TRADE
    if signal == "NEUTRAL":
        return NEUTRAL
    complacency = r_ts < 0.90 or p_skew >= 80.0
    if signal == "SELL_HEDGE":
        if complacency:
            return SELL_HEDGE
        return HOLD_CORE
    if signal == "STRONG_SELL":
        if complacency and short_momentum_weak:
            return STRONG_SELL
        return HOLD_VOL
    return EUPHORIA


def _finite_last(series: pd.Series) -> bool:
    if series.empty:
        return False
    value = pd.to_numeric(series.iloc[-1], errors="coerce")
    return bool(np.isfinite(value))


def _valid_count(series: pd.Series) -> int:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return int(np.isfinite(values).sum())


def _na(reason: str, asof: pd.Timestamp | None = None) -> VolAssessment:
    return VolAssessment(status=DATA_NA, score=None, asof=asof, reason=reason)


def _slice_asof(prices: pd.DataFrame, asof_index: int | None) -> pd.DataFrame:
    if asof_index is None:
        asof_index = len(prices) - 1
    if asof_index < 0:
        asof_index += len(prices)
    if asof_index < 0 or asof_index >= len(prices):
        raise IndexError(f"asof_index 越界: {asof_index}")
    return prices.iloc[: asof_index + 1].copy()


def _short_momentum_weak(close: pd.Series) -> bool:
    """短周期动量走弱：收盘价低于自身 20 日均线。"""
    if len(close) < 20:
        return False
    sma20 = close.rolling(20, min_periods=20).mean().iloc[-1]
    last = close.iloc[-1]
    if not np.isfinite(sma20) or not np.isfinite(last):
        return False
    return bool(last < sma20)


def compute_vol_snapshot(prices: pd.DataFrame, asof_index: int | None = None) -> VolAssessment:
    """计算截至 asof 的结构分。先截断到 asof，滚动窗口看不到未来。"""
    frame = _slice_asof(prices, asof_index)
    asof = pd.Timestamp(frame.index[-1])
    required = list(READINESS_TICKERS) + list(FACTOR_TICKERS)
    missing = [ticker for ticker in required if ticker not in frame.columns]
    if missing:
        return _na("缺少必选序列: " + ", ".join(missing), asof)

    for ticker in required:
        series = frame[ticker]
        if not _finite_last(series) or _valid_count(series) < READINESS_MIN_BARS:
            return _na(
                f"{ticker} 当日缺失或有效历史短于 {READINESS_MIN_BARS} 日",
                asof,
            )

    vix = pd.to_numeric(frame["^VIX"], errors="coerce")
    vix3m = pd.to_numeric(frame["^VIX3M"], errors="coerce")
    vix9d = pd.to_numeric(frame["^VIX9D"], errors="coerce")
    vvix = pd.to_numeric(frame["^VVIX"], errors="coerce")
    skew = pd.to_numeric(frame["^SKEW"], errors="coerce")

    r_ts = vix / vix3m
    mu = r_ts.rolling(Z_WINDOW, min_periods=Z_WINDOW).mean()
    sigma = r_ts.rolling(Z_WINDOW, min_periods=Z_WINDOW).std(ddof=1)
    z_ts = (r_ts - mu) / sigma
    r_basis = vix9d / vix
    p_vvix = rolling_percentile_rank(vvix, PERCENTILE_WINDOW)
    p_skew = rolling_percentile_rank(skew, PERCENTILE_WINDOW)

    latest = {
        "z_ts": z_ts.iloc[-1],
        "r_ts": r_ts.iloc[-1],
        "r_ts_prev": r_ts.iloc[-2] if len(r_ts) >= 2 else np.nan,
        "r_basis": r_basis.iloc[-1],
        "vix": vix.iloc[-1],
        "p_vvix": p_vvix.iloc[-1],
        "p_skew": p_skew.iloc[-1],
    }
    if not all(np.isfinite(value) for value in latest.values()):
        return _na("滚动 Z-Score 或百分位在当日不可计算", asof)

    factors = score_vol_components(
        z_ts=float(latest["z_ts"]),
        r_ts=float(latest["r_ts"]),
        r_basis=float(latest["r_basis"]),
        vix=float(latest["vix"]),
        p_vvix=float(latest["p_vvix"]),
        p_skew=float(latest["p_skew"]),
    )
    score = float(factors["score"])
    backwardation = bool(latest["r_ts"] > 1.00 and latest["z_ts"] > 1.0 and score < 5.0)
    return VolAssessment(
        status="OK",
        score=score,
        raw_signal=raw_signal_name(score),
        z_ts=float(latest["z_ts"]),
        r_ts=float(latest["r_ts"]),
        r_ts_prev=float(latest["r_ts_prev"]),
        r_basis=float(latest["r_basis"]),
        vix=float(latest["vix"]),
        p_vvix=float(latest["p_vvix"]),
        p_skew=float(latest["p_skew"]),
        factors=factors,
        is_backwardation_risk=backwardation,
        backwardation_alert=BACKWARDATION_ALERT if backwardation else None,
        asof=asof,
        reason="OK",
    )


def evaluate_vix_structure(
    prices: pd.DataFrame,
    trend_is_bull: bool,
    asof_index: int | None = None,
    benchmark_close: pd.Series | None = None,
) -> VolAssessment:
    """返回结构分与门控后的展示状态。trend_is_bull 不参与计分。"""
    assessment = compute_vol_snapshot(prices, asof_index=asof_index)
    if assessment.status != "OK" or assessment.score is None:
        return assessment

    frame = _slice_asof(prices, asof_index)
    if benchmark_close is None:
        bench = pd.to_numeric(frame["^GSPC"], errors="coerce")
    else:
        bench = pd.to_numeric(benchmark_close, errors="coerce").iloc[: len(frame)]
    bench = bench.dropna()
    close = float(bench.iloc[-1])
    low_5 = float(bench.iloc[-5:].min()) if len(bench) >= 5 else close
    state = apply_orthogonal_gates(
        assessment.score,
        trend_is_bull,
        r_ts=float(assessment.r_ts),
        r_ts_prev=float(assessment.r_ts_prev),
        z_ts=float(assessment.z_ts),
        p_skew=float(assessment.p_skew),
        close=close,
        low_5=low_5,
        short_momentum_weak=_short_momentum_weak(bench),
    )
    # 门控只替换展示字段。分数对象在入参处已经算完。
    return replace(assessment, display_state=state)
