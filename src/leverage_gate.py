"""用 20 日实现波动率限制日内重置杠杆倍数。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import HV_HIGH, HV_LOW, HV_WINDOW, LEVERAGE_LADDER, REGIME_1, REGIME_1B, REGIME_2, REGIME_3, REGIME_4, REGIME_5


def realized_vol(close: pd.Series, window: int = HV_WINDOW) -> pd.Series:
    log_return = np.log(pd.to_numeric(close, errors="coerce")).diff()
    return log_return.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252.0)


def hv_cap(hv20: float | None) -> tuple[float, str]:
    if hv20 is None or not np.isfinite(hv20):
        return 1.0, "HV20 不可用，杠杆上限降到 1×"
    if hv20 < HV_LOW:
        return 3.0, f"HV20 {hv20:.1%} < {HV_LOW:.0%}，方差损耗较低"
    if hv20 <= HV_HIGH:
        return 2.0, f"HV20 {hv20:.1%} 处于 {HV_LOW:.0%}–{HV_HIGH:.0%}，降为 2×"
    return 1.0, f"HV20 {hv20:.1%} > {HV_HIGH:.0%}，禁止高于 1× 的日内重置杠杆"


def regime_cap(regime_id: str) -> float:
    if regime_id == REGIME_1:
        return 3.0
    if regime_id in {REGIME_1B, REGIME_5}:
        return 2.0
    if regime_id == REGIME_2:
        return 1.0
    return 1.0


def resolve_leverage(hv20: float | None, regime_id: str, benchmark: str) -> tuple[float, str, str]:
    """返回 (倍数上限, 说明, 对应工具)。倍数取状态上限和波动率上限的较低者。"""
    vol_cap, vol_note = hv_cap(hv20)
    cap = min(regime_cap(regime_id), vol_cap)
    if regime_id in {REGIME_3, REGIME_4}:
        cap = 1.0
        vol_note = "修正或熊市状态本身不允许高于 1×；" + vol_note
    ladder = LEVERAGE_LADDER.get(benchmark, LEVERAGE_LADDER["^GSPC"])
    instrument = ladder[int(cap)]
    if regime_id == REGIME_2:
        cap = min(cap, 1.0)
        instrument = ladder[1]
        vol_note = "脆弱分化牛不加杠杆；" + vol_note
    return float(cap), vol_note, instrument
