"""数据缺失门控与滚动窗口零未来函数。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_fetcher import sanitize_prices
from src.vix_structure_engine import DATA_NA, evaluate_vix_structure


def _index(rows: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2023-01-02", periods=rows)


def _frame_from_columns(columns: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=_index(rows))


def _varying_market(rows: int = 300) -> pd.DataFrame:
    t = np.arange(rows, dtype=float)
    vix = 16.0 + 3.0 * np.sin(t / 5.0) + 0.01 * t
    vix3m = 19.0 + 2.0 * np.sin(t / 8.0 + 0.4) + 0.008 * t
    return _frame_from_columns(
        {
            "^VIX": vix,
            "^VIX3M": vix3m,
            "^VIX9D": vix * 1.02,
            "^VVIX": 85.0 + 8.0 * np.sin(t / 6.0),
            "^SKEW": 120.0 + 6.0 * np.cos(t / 9.0),
            "^GSPC": 4000.0 + t,
            "^NDX": 14000.0 + 2.0 * t,
        }
    )


def test_missing_required_series_is_data_na_not_zero() -> None:
    frame = _varying_market(300)
    frame.loc[frame.index[-1], "^VIX"] = np.nan
    result = evaluate_vix_structure(frame, trend_is_bull=True)
    assert result.status == DATA_NA
    assert result.score is None


def test_short_history_is_data_na_not_zero() -> None:
    result = evaluate_vix_structure(_varying_market(80), trend_is_bull=False)
    assert result.status == DATA_NA
    assert result.score is None


def test_sanitize_prices_forward_fills_at_most_two_days_and_never_backfills() -> None:
    frame = pd.DataFrame(
        {"^VIX": [1.0, np.nan, np.nan, np.nan, 5.0]},
        index=pd.bdate_range("2024-01-02", periods=5),
    )
    out = sanitize_prices(frame)
    assert out.iloc[1, 0] == 1.0
    assert out.iloc[2, 0] == 1.0
    assert np.isnan(out.iloc[3, 0])
    assert out.iloc[4, 0] == 5.0


def test_non_equity_print_does_not_extend_the_cash_session() -> None:
    index = pd.bdate_range("2024-01-02", periods=3)
    frame = pd.DataFrame(
        {
            "^GSPC": [100.0, 101.0, np.nan],
            "^VIX": [15.0, 16.0, np.nan],
            "^VIX3M": [18.0, 19.0, np.nan],
            "^NDX": [200.0, 201.0, np.nan],
            "DX-Y.NYB": [100.0, 101.0, 102.0],
        },
        index=index,
    )
    out = sanitize_prices(frame)
    assert out.index[-1] == index[1]
    assert out.loc[index[1], "^GSPC"] == 101.0
    assert index[2] not in out.index


def test_vix_only_print_does_not_extend_the_cash_session() -> None:
    index = pd.bdate_range("2024-01-02", periods=3)
    frame = pd.DataFrame(
        {
            "^GSPC": [100.0, 101.0, np.nan],
            "^VIX": [15.0, 16.0, 17.0],
            "^VIX3M": [18.0, 19.0, np.nan],
            "^NDX": [200.0, 201.0, np.nan],
        },
        index=index,
    )
    out = sanitize_prices(frame)
    assert out.index[-1] == index[1]
    assert out.loc[index[1], "^GSPC"] == 101.0
    assert out.loc[index[1], "^VIX"] == 16.0
    assert index[2] not in out.index


def test_trailing_empty_session_is_dropped_before_forward_fill() -> None:
    index = pd.bdate_range("2024-01-02", periods=3)
    frame = pd.DataFrame({"^VIX": [1.0, 2.0, np.nan]}, index=index)
    out = sanitize_prices(frame)
    assert len(out) == 2
    assert out.index[-1] == index[1]
    assert out.iloc[-1, 0] == 2.0


def test_skew_gap_one_hundred_days_ago_still_scores() -> None:
    frame = _varying_market(400)
    gap_end = len(frame) - 100
    frame.iloc[gap_end - 2 : gap_end + 1, frame.columns.get_loc("^SKEW")] = np.nan
    assert int(frame["^SKEW"].iloc[gap_end - 2 : gap_end + 1].isna().sum()) == 3
    assert frame["^SKEW"].iloc[-99:].notna().all()
    result = evaluate_vix_structure(frame, trend_is_bull=True)
    assert result.status == "OK"
    assert result.score is not None


def test_day_260_unchanged_after_future_is_rewritten() -> None:
    original = _varying_market(300)
    mutated = original.copy()
    mutated.iloc[260:] = mutated.iloc[260:] * 10.0

    before = evaluate_vix_structure(original, trend_is_bull=True, asof_index=259)
    after = evaluate_vix_structure(mutated, trend_is_bull=True, asof_index=259)

    assert before.status == "OK"
    assert after.status == "OK"
    assert before.score is not None and after.score is not None
    assert before.z_ts == pytest_approx(after.z_ts)
    assert before.score == pytest_approx(after.score)
    assert before.r_ts == pytest_approx(after.r_ts)
    assert before.p_vvix == pytest_approx(after.p_vvix)
    assert before.p_skew == pytest_approx(after.p_skew)
    for key in ("s1", "s2", "s3", "s4", "s5"):
        assert before.factors[key] == pytest_approx(after.factors[key])

    last_original = evaluate_vix_structure(original, trend_is_bull=True)
    last_mutated = evaluate_vix_structure(mutated, trend_is_bull=True)
    assert last_original.score is not None and last_mutated.score is not None
    assert last_original.score != pytest_approx(last_mutated.score)


def pytest_approx(value: float | None, abs_tol: float = 1e-12):
    import pytest

    return pytest.approx(value, abs=abs_tol)
