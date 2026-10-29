"""引擎 B：市场宽度健康度与巨头掩护下的顶背离检测。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import DEFENSIVE_SECTORS, HIGH_WINDOW, OFFENSIVE_SECTORS, SECTOR_ALIASES


@dataclass
class BreadthScore:
    score: float | None
    components: dict[str, float] = field(default_factory=dict)
    fast: float | None = None
    slow: float | None = None
    fast_slope_5: float | None = None
    uptrend_ratio: float | None = None
    uptrend_ma10: float | None = None
    uptrend_slope: float | None = None
    uptrend_color: str | None = None
    rsp_spy: float | None = None
    off_def_spread: float | None = None
    status: str = "OK"
    reason: str = ""


@dataclass
class DivergenceResult:
    flag: bool
    reasons: list[str] = field(default_factory=list)
    ratio_to_high: float | None = None
    near_high: bool = False


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _scale_if_fraction(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if len(finite) and float(finite.max()) <= 1.0:
        return numeric * 100.0
    return numeric


def _canon_sector(name: object) -> str:
    key = " ".join(str(name).strip().casefold().replace("&", " and ").split())
    return SECTOR_ALIASES.get(key, str(name).strip())


def _level_score(level: float) -> float:
    if level >= 70.0:
        return 25.0
    if level <= 20.0:
        return 0.0
    return 25.0 * (level - 20.0) / 50.0


def _red_down_score(ratio: float) -> float:
    """空头趋势主导时把占比线性映射到 0～5 分。"""
    if ratio <= 0:
        return 0.0
    if ratio >= 30.0:
        return 5.0
    return 5.0 * ratio / 30.0


def _uptrend_component(ratio: float, slope: float, color: str) -> float:
    if color == "GREEN" and 25.0 <= ratio <= 38.0:
        return 25.0
    if color == "GREEN" and ratio > 38.0:
        return 18.0
    if color == "RED" and ratio < 12.0 and slope > 0:
        return 15.0
    if color == "RED" and slope <= 0:
        return _red_down_score(ratio)
    if color == "GREEN":
        return 12.0 if ratio >= 20.0 else 8.0
    if color == "RED" and slope > 0:
        return 8.0
    return 5.0


def _prepare_breadth(breadth: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    frame = breadth.copy()
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.set_index("Date")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
    frame = frame.sort_index().loc[:asof]
    for column in ("Breadth_Index_Raw", "Breadth_Index_8MA", "Breadth_Index_200MA"):
        if column in frame.columns:
            frame[column] = _scale_if_fraction(frame[column])
    return frame


def _prepare_uptrend(uptrend: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    frame = uptrend.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
    frame = frame.sort_index().loc[:asof]
    if "ratio" in frame.columns:
        anchor = pd.to_numeric(frame["ratio"], errors="coerce")
        finite = anchor[np.isfinite(anchor)]
        if len(finite) and float(finite.max()) <= 1.0:
            for column in ("ratio", "ma10", "slope"):
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce") * 100.0
    return frame


def _cross_component(breadth: pd.DataFrame) -> tuple[float, float, float, float]:
    fast_series = pd.to_numeric(breadth["Breadth_Index_8MA"], errors="coerce")
    slow_series = pd.to_numeric(breadth["Breadth_Index_200MA"], errors="coerce")
    fast = float(fast_series.iloc[-1])
    slow = float(slow_series.iloc[-1])
    slope_value = fast_series.diff(5).iloc[-1]
    slope = float(slope_value) if np.isfinite(slope_value) else 0.0
    recent = breadth["Is_Trough_8MA_Below_04"].iloc[:-1].tail(10) if "Is_Trough_8MA_Below_04" in breadth.columns else []
    trough_recovery = slope > 0 and any(_as_bool(value) for value in recent)
    if fast > slow and slope > 0:
        score = 20.0
    elif fast > slow:
        score = 12.0
    elif slope > 0 or trough_recovery:
        score = 10.0
    else:
        score = 0.0
    return score, fast, slow, slope


def _rsp_component(prices: pd.DataFrame, asof: pd.Timestamp) -> tuple[float, float | None, pd.Series]:
    if "RSP" not in prices.columns or "SPY" not in prices.columns:
        return 0.0, None, pd.Series(dtype=float)
    ratio = pd.to_numeric(prices.loc[:asof, "RSP"], errors="coerce") / pd.to_numeric(prices.loc[:asof, "SPY"], errors="coerce")
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratio) < 50:
        latest = float(ratio.iloc[-1]) if len(ratio) else None
        return 0.0, latest, ratio
    sma20 = ratio.rolling(20, min_periods=20).mean().iloc[-1]
    sma50 = ratio.rolling(50, min_periods=50).mean().iloc[-1]
    latest = float(ratio.iloc[-1])
    if latest > sma20 and sma20 > sma50:
        return 15.0, latest, ratio
    if latest > sma20:
        return 8.0, latest, ratio
    return 0.0, latest, ratio


def _sector_spread(sectors: pd.DataFrame | None) -> tuple[float, float | None]:
    if sectors is None or sectors.empty or "Ratio" not in sectors.columns or "Sector" not in sectors.columns:
        return 8.0, None
    frame = sectors.copy()
    frame["Sector"] = frame["Sector"].map(_canon_sector)
    frame["Ratio"] = _scale_if_fraction(frame["Ratio"])
    offensive = frame.loc[frame["Sector"].isin(OFFENSIVE_SECTORS), "Ratio"]
    defensive = frame.loc[frame["Sector"].isin(DEFENSIVE_SECTORS), "Ratio"]
    if offensive.empty or defensive.empty:
        return 8.0, None
    spread = float(offensive.mean() - defensive.mean())
    if spread > 5.0:
        return 15.0, spread
    if spread >= -5.0:
        return 8.0, spread
    return 0.0, spread


def score_breadth(
    prices: pd.DataFrame,
    breadth: pd.DataFrame | None,
    uptrend: pd.DataFrame | None,
    sectors: pd.DataFrame | None,
    asof: pd.Timestamp | None = None,
) -> BreadthScore:
    if asof is None:
        asof = pd.Timestamp(prices.index[-1])
    if breadth is None or uptrend is None or breadth.empty or uptrend.empty:
        return BreadthScore(score=None, status="DATA N/A", reason="宽度数据不可用")

    breadth_hist = _prepare_breadth(breadth, asof)
    uptrend_hist = _prepare_uptrend(uptrend, asof)
    if breadth_hist.empty or uptrend_hist.empty:
        return BreadthScore(score=None, status="DATA N/A", reason="截至 asof 没有宽度样本")

    cross, fast, slow, slope = _cross_component(breadth_hist)
    level = _level_score(slow)
    ratio = float(pd.to_numeric(uptrend_hist["ratio"], errors="coerce").iloc[-1])
    ma10_value = pd.to_numeric(uptrend_hist["ma10"], errors="coerce").iloc[-1] if "ma10" in uptrend_hist.columns else np.nan
    slope_value = pd.to_numeric(uptrend_hist["slope"], errors="coerce").iloc[-1] if "slope" in uptrend_hist.columns else 0.0
    up_slope = float(slope_value) if np.isfinite(slope_value) else 0.0
    color = str(uptrend_hist["color"].iloc[-1]).strip().upper() if "color" in uptrend_hist.columns else "RED"
    participation = _uptrend_component(ratio, up_slope, color)
    rsp_score, rsp_value, _ = _rsp_component(prices, asof)
    sector_score, spread = _sector_spread(sectors)
    total = float(np.clip(level + cross + participation + rsp_score + sector_score, 0.0, 100.0))
    return BreadthScore(
        score=total,
        components={
            "spx_level": level,
            "cross": cross,
            "uptrend": participation,
            "rsp_spy": rsp_score,
            "off_def": sector_score,
        },
        fast=fast,
        slow=slow,
        fast_slope_5=slope,
        uptrend_ratio=ratio,
        uptrend_ma10=None if not np.isfinite(ma10_value) else float(ma10_value),
        uptrend_slope=up_slope,
        uptrend_color=color,
        rsp_spy=rsp_value,
        off_def_spread=spread,
        status="OK",
        reason="OK",
    )


def detect_bearish_divergence(
    benchmark_close: pd.Series,
    breadth: pd.DataFrame,
    uptrend: pd.DataFrame,
    prices: pd.DataFrame,
    asof: pd.Timestamp | None = None,
) -> DivergenceResult:
    """指数贴近 252 日高点，且内部宽度至少两项走弱。"""
    close = pd.to_numeric(benchmark_close, errors="coerce").dropna()
    if asof is not None:
        close = close.loc[:asof]
    if close.empty:
        return DivergenceResult(False, ["NO_BENCHMARK"])

    window = close.iloc[-HIGH_WINDOW:] if len(close) >= HIGH_WINDOW else close
    high = float(window.max()) if len(close) >= HIGH_WINDOW else np.nan
    last = float(close.iloc[-1])
    ratio_to_high = last / high if np.isfinite(high) and high else np.nan
    near_high = bool(np.isfinite(ratio_to_high) and ratio_to_high >= 0.98)

    decision_asof = pd.Timestamp(close.index[-1])
    breadth_hist = _prepare_breadth(breadth, decision_asof)
    uptrend_hist = _prepare_uptrend(uptrend, decision_asof)
    internal: list[str] = []
    fast = float(pd.to_numeric(breadth_hist["Breadth_Index_8MA"], errors="coerce").iloc[-1])
    slow = float(pd.to_numeric(breadth_hist["Breadth_Index_200MA"], errors="coerce").iloc[-1])
    bearish = _as_bool(breadth_hist["Bearish_Signal"].iloc[-1]) if "Bearish_Signal" in breadth_hist.columns else False
    if fast < slow or bearish:
        internal.append("BREADTH_DEATH_CROSS_OR_BEARISH_SIGNAL")

    ratio = float(pd.to_numeric(uptrend_hist["ratio"], errors="coerce").iloc[-1])
    color = str(uptrend_hist["color"].iloc[-1]).strip().upper() if "color" in uptrend_hist.columns else ""
    if ratio < 25.0 or color == "RED":
        internal.append("UPTREND_RATIO_BELOW_25_OR_RED")

    if {"RSP", "SPY"}.issubset(prices.columns):
        q = pd.to_numeric(prices["RSP"], errors="coerce") / pd.to_numeric(prices["SPY"], errors="coerce")
        q = q.loc[:decision_asof].replace([np.inf, -np.inf], np.nan).dropna()
        if len(q) >= 20 and float(q.iloc[-1]) <= float(q.iloc[-20:-1].min()):
            internal.append("RSP_SPY_20D_LOW")

    flagged = bool(near_high and len(internal) >= 2)
    reasons = (["INDEX_NEAR_252D_HIGH"] + internal) if flagged else internal
    return DivergenceResult(flag=flagged, reasons=reasons, ratio_to_high=None if not np.isfinite(ratio_to_high) else float(ratio_to_high), near_high=near_high)
