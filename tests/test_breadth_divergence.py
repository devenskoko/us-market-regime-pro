"""指数贴近高点但内部宽度走弱时，触发假突破并归入脆弱牛。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.breadth_engine import detect_bearish_divergence
from src.config import REGIME_2
from src.data_fetcher import normalize_breadth_frame
from src.regime_synthesizer import diagnose


def _market() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = 300
    index = pd.bdate_range("2023-01-02", periods=rows)
    ratio = np.full(rows, 0.93)
    ratio[-63:] = 0.93 + 0.008 * np.sin(np.linspace(0.0, 4.0 * np.pi, 63))
    ratio[-1] = 0.93
    vix = np.full(rows, 14.0)
    spy = np.linspace(400.0, 520.0, rows)
    prices = pd.DataFrame(
        {
            "^VIX": vix,
            "^VIX3M": vix / ratio,
            "^VIX9D": vix,
            "^VVIX": np.full(rows, 90.0),
            "^SKEW": np.full(rows, 125.0),
            "^GSPC": np.linspace(4000.0, 5600.0, rows),
            "^NDX": np.linspace(14000.0, 19000.0, rows),
            "SMH": np.linspace(120.0, 240.0, rows),
            "SPY": spy,
            "RSP": spy * np.linspace(1.0, 1.08, rows),
            "HYG": np.full(rows, 78.0),
            "IEF": np.full(rows, 95.0),
        },
        index=index,
    )
    breadth = pd.DataFrame(
        {
            "Breadth_Index_Raw": 55.0,
            "Breadth_Index_8MA": 45.0,
            "Breadth_Index_200MA": 62.0,
            "Bearish_Signal": True,
            "Is_Trough_8MA_Below_04": False,
        },
        index=index,
    )
    uptrend = pd.DataFrame(
        {
            "ratio": 18.0,
            "ma10": 19.0,
            "slope": -0.4,
            "trend": "down",
            "color": "RED",
        },
        index=index,
    )
    sectors = pd.DataFrame(
        {
            "Sector": [
                "Technology",
                "Consumer Discretionary",
                "Industrials",
                "Financials",
                "Communication Services",
                "Utilities",
                "Consumer Staples",
                "Health Care",
            ],
            "Ratio": [30, 30, 30, 30, 30, 20, 20, 20],
        }
    )
    return prices, breadth, uptrend, sectors


def test_new_high_with_weak_breadth_is_fragile_bull() -> None:
    prices, breadth, uptrend, sectors = _market()
    divergence = detect_bearish_divergence(prices["^GSPC"], breadth, uptrend, prices)
    assert divergence.near_high
    assert divergence.ratio_to_high is not None and divergence.ratio_to_high >= 0.98
    assert "BREADTH_DEATH_CROSS_OR_BEARISH_SIGNAL" in divergence.reasons
    assert "UPTREND_RATIO_BELOW_25_OR_RED" in divergence.reasons
    assert divergence.flag

    diagnosis = diagnose(["SPY"], prices, breadth, uptrend, sectors, breadth_source="test")
    row = diagnosis.tickers[0]
    assert row.divergence.flag
    assert row.regime.regime_id == REGIME_2
    assert diagnosis.anchor is not None
    assert diagnosis.anchor.regime.regime_id == REGIME_2


def test_fractional_breadth_columns_are_scaled_to_percent() -> None:
    raw = pd.DataFrame(
        {
            " Date ": ["2024-01-02", "2024-01-03"],
            "Breadth_Index_Raw": [0.40, 0.50],
            "Breadth_Index_200MA": [0.42, 0.45],
            "Breadth_Index_8MA": [0.30, 0.33],
            "Bearish_Signal": ["True", "False"],
        }
    )
    out = normalize_breadth_frame(raw)
    assert list(out.columns)[:3] == ["Breadth_Index_Raw", "Breadth_Index_200MA", "Breadth_Index_8MA"] or set(
        ["Breadth_Index_Raw", "Breadth_Index_200MA", "Breadth_Index_8MA"]
    ).issubset(out.columns)
    assert out["Breadth_Index_200MA"].max() == 45.0
    assert out["Breadth_Index_8MA"].iloc[-1] == 33.0
    assert bool(out["Bearish_Signal"].iloc[0]) is True
    assert bool(out["Bearish_Signal"].iloc[1]) is False
