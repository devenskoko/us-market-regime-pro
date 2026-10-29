"""分母端：利率、美元的 20 日变化率 Z-Score，以及信贷与宽度共振。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import MACRO_CHANGE_WINDOW, MACRO_Z_THRESHOLD, PERCENTILE_WINDOW


@dataclass(frozen=True)
class MacroState:
    active: bool
    z_tnx: float | None
    z_dxy: float | None
    rate_shock: bool
    dollar_shock: bool
    credit_breadth_resonance: bool
    reason: str


def rolling_change_z(
    series: pd.Series,
    change: int = MACRO_CHANGE_WINDOW,
    window: int = PERCENTILE_WINDOW,
) -> pd.Series:
    """20 日变化相对过去 252 日的 Z-Score，窗口含当日、不含未来。"""
    values = pd.to_numeric(series, errors="coerce")
    delta = values.diff(change)
    mu = delta.rolling(window, min_periods=window).mean()
    sigma = delta.rolling(window, min_periods=window).std(ddof=1)
    return (delta - mu) / sigma


def _last_finite(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = series.iloc[-1]
    if not np.isfinite(value):
        return None
    return float(value)


def assess_macro(
    prices: pd.DataFrame,
    *,
    credit_risk_off: bool,
    uptrend_ratio: float | None,
    asof: pd.Timestamp | None = None,
) -> MacroState:
    history = prices if asof is None else prices.loc[:asof]
    z_tnx = _last_finite(rolling_change_z(history["^TNX"])) if "^TNX" in history.columns else None
    z_dxy = _last_finite(rolling_change_z(history["DX-Y.NYB"])) if "DX-Y.NYB" in history.columns else None
    rate_shock = bool(z_tnx is not None and z_tnx > MACRO_Z_THRESHOLD)
    dollar_shock = bool(z_dxy is not None and z_dxy > MACRO_Z_THRESHOLD)
    resonance = bool(credit_risk_off and uptrend_ratio is not None and uptrend_ratio < 30.0)
    reasons: list[str] = []
    if rate_shock:
        reasons.append(f"10 年期收益率 20 日变化 Z={z_tnx:.2f}")
    if dollar_shock:
        reasons.append(f"美元指数 20 日变化 Z={z_dxy:.2f}")
    if resonance:
        reasons.append("HYG/IEF 低于 SMA50 且上升趋势占比低于 30%")
    active = bool(reasons)
    return MacroState(
        active=active,
        z_tnx=z_tnx,
        z_dxy=z_dxy,
        rate_shock=rate_shock,
        dollar_shock=dollar_shock,
        credit_breadth_resonance=resonance,
        reason="；".join(reasons) if reasons else "宏观分母端未触发",
    )
