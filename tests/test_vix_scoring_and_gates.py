"""Score_vol 正交性：趋势只改展示状态，不改分数。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.benchmark_router import route_benchmark
from src.vix_structure_engine import (
    BUY_DIP,
    CRASH_BUY,
    EUPHORIA,
    HOLD_CORE,
    HOLD_VOL,
    NEUTRAL,
    NO_TRADE,
    SELL_HEDGE,
    STRONG_BUY,
    STRONG_SELL,
    WAIT_MOM,
    WAIT_VOL,
    apply_orthogonal_gates,
    evaluate_vix_structure,
    score_vol_components,
)


def _index(rows: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2023-01-02", periods=rows)


def _buy_dip_market() -> pd.DataFrame:
    rows = 300
    ratio = np.full(rows, 0.95)
    ratio[-63:-1] = 0.90 + 0.08 * (np.arange(62) % 5) / 4.0
    ratio[-1] = 1.0
    vix = np.full(rows, 28.0)
    index = _index(rows)
    return pd.DataFrame(
        {
            "^VIX": vix,
            "^VIX3M": vix / ratio,
            "^VIX9D": vix * 1.12,
            "^VVIX": np.full(rows, 90.0),
            "^SKEW": np.full(rows, 125.0),
            "^GSPC": np.linspace(4000.0, 5200.0, rows),
            "^NDX": np.linspace(14000.0, 18000.0, rows),
        },
        index=index,
    )


def test_trend_does_not_change_score_and_bear_blocks_buy_dip() -> None:
    prices = _buy_dip_market()
    bull = evaluate_vix_structure(prices, trend_is_bull=True)
    bear = evaluate_vix_structure(prices, trend_is_bull=False)

    assert bull.status == "OK" and bear.status == "OK"
    assert bull.score is not None and bear.score is not None
    assert round(bull.score, 6) == round(bear.score, 6)
    assert bull.score == pytest.approx(bear.score, abs=1e-6)
    assert 3.0 <= bull.score < 5.0
    assert bull.display_state == BUY_DIP
    assert bear.display_state == NO_TRADE
    assert bull.factors == bear.factors


@pytest.mark.parametrize(
    ("z_ts", "expected"),
    [
        (2.5, 3.0),
        (2.0, 2.0),
        (1.2, 1.0),
        (0.0, 0.0),
        (-1.2, -1.5),
        (-2.0, -2.5),
        (1.199999, 0.0),
        (-1.199999, 0.0),
    ],
)
def test_term_z_factor_boundaries(z_ts: float, expected: float) -> None:
    factors = score_vol_components(z_ts, 0.95, 1.0, 20.0, 50.0, 50.0)
    assert factors["s1"] == expected


def _gate(score: float, trend_is_bull: bool = True, **overrides: float | bool) -> str:
    params = {
        "r_ts": 0.95,
        "r_ts_prev": 1.00,
        "z_ts": 2.0,
        "p_skew": 50.0,
        "close": 110.0,
        "low_5": 100.0,
        "short_momentum_weak": False,
    }
    params.update(overrides)
    return apply_orthogonal_gates(score, trend_is_bull, **params)  # type: ignore[arg-type]


def test_remaining_signal_gates() -> None:
    assert _gate(6.0, r_ts=1.10, r_ts_prev=1.20) == CRASH_BUY
    assert _gate(6.0, r_ts=1.30, r_ts_prev=1.10) == WAIT_VOL
    assert _gate(5.0, z_ts=1.8, close=110.0, low_5=100.0) == STRONG_BUY
    assert _gate(5.0, z_ts=1.7, close=110.0, low_5=100.0) == WAIT_MOM
    assert _gate(5.0, z_ts=2.0, close=100.0, low_5=100.0) == WAIT_MOM
    assert _gate(3.0, trend_is_bull=False) == NO_TRADE
    assert _gate(4.999, trend_is_bull=True) == BUY_DIP
    assert _gate(0.0, trend_is_bull=False) == NEUTRAL
    assert _gate(-2.0, r_ts=0.85, p_skew=10.0) == SELL_HEDGE
    assert _gate(-2.0, r_ts=0.95, p_skew=80.0) == SELL_HEDGE
    assert _gate(-2.0, r_ts=0.95, p_skew=50.0) == HOLD_CORE
    assert _gate(-5.0, r_ts=0.85, short_momentum_weak=True) == STRONG_SELL
    assert _gate(-5.0, r_ts=0.95, p_skew=50.0, short_momentum_weak=True) == HOLD_VOL
    assert _gate(-5.0, r_ts=0.85, short_momentum_weak=False) == HOLD_VOL
    assert _gate(-6.0, trend_is_bull=True) == EUPHORIA
    assert _gate(-6.0, trend_is_bull=False) == EUPHORIA


def test_benchmark_router() -> None:
    assert route_benchmark("TQQQ") == "^NDX"
    assert route_benchmark("QQQ") == "^NDX"
    assert route_benchmark("^NDX") == "^NDX"
    assert route_benchmark("IWM") == "^RUT"
    assert route_benchmark("SOXL") == "SMH"
    assert route_benchmark("SPY") == "^GSPC"
    assert route_benchmark("UPRO") == "^GSPC"
