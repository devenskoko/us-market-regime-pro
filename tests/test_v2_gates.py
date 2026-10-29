"""V2：迟滞带、宏观逆风、实现波动率杠杆、宽度预估和影子账本。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.breadth_nowcast import nowcast_breadth
from src.config import REGIME_1, REGIME_1B
from src.hysteresis import sma50_hysteresis
from src.ledger import backfill_outcomes, ledger_summary, record_diagnosis
from src.leverage_gate import resolve_leverage
from src.regime_synthesizer import classify_regime
from src.vix_structure_engine import merge_collinear_factors


def test_buffer_does_not_flip_and_two_closes_restore() -> None:
    index = pd.bdate_range("2020-01-01", periods=6)
    close = pd.Series([110.0, 99.0, 97.0, 101.0, 101.0, 103.0], index=index)
    sma = pd.Series(100.0, index=index)
    atr = pd.Series(10.0, index=index)
    path = sma50_hysteresis(close, sma, atr, band=0.25)
    assert list(path["state"]) == ["ABOVE", "ABOVE", "BELOW", "BELOW", "ABOVE", "ABOVE"]
    assert path.loc[index[1], "exposure"] == path.loc[index[0], "exposure"]


def test_macro_headwind_caps_strong_bull() -> None:
    decision = classify_regime(
        close=120.0,
        sma50=100.0,
        sma200=80.0,
        r_ts=0.90,
        r_ts_prev=0.91,
        score_vol=0.0,
        score_breadth=70.0,
        bearish_divergence=False,
        is_backwardation_risk=False,
        credit_risk_off=True,
        uptrend_ratio=25.0,
        uptrend_slope=0.2,
        confirmed_below_sma50=False,
        macro_headwind=True,
        macro_reason="信贷与宽度共振",
        sticky_exposure=0.92,
    )
    assert decision.regime_id == REGIME_1B
    assert decision.exposure_point is not None and decision.exposure_point <= 75.0


def test_hv20_downgrades_leverage() -> None:
    cap, _note, instrument = resolve_leverage(0.18, REGIME_1, "^NDX")
    assert cap == 3.0 and instrument == "TQQQ"
    cap, _note, instrument = resolve_leverage(0.25, REGIME_1, "^NDX")
    assert cap == 2.0 and instrument == "QLD"
    cap, _note, instrument = resolve_leverage(0.35, REGIME_1, "^NDX")
    assert cap == 1.0 and instrument == "QQQ"
    cap, _note, instrument = resolve_leverage(0.10, REGIME_1B, "^NDX")
    assert cap == 2.0 and instrument == "QLD"


def test_nowcast_adds_delta_when_csv_is_behind() -> None:
    index = pd.bdate_range("2024-01-01", periods=220)
    prices = pd.DataFrame(index=index)
    for ticker in ("XLK", "XLF", "XLV", "XLY", "XLI", "XLP", "XLE", "XLU", "XLB", "XLRE", "XLC"):
        prices[ticker] = np.linspace(100.0, 130.0, len(index))
    prices["RSP"] = np.linspace(50.0, 60.0, len(index))
    prices["SPY"] = np.linspace(400.0, 420.0, len(index))
    for ticker in ("XLU", "XLP", "XLE", "XLB", "XLRE", "XLC"):
        prices.loc[index[-1], ticker] = prices.loc[index[-2], ticker] * 0.5
    prices.loc[index[-1], "RSP"] = prices.loc[index[-2], "RSP"] * 0.98
    prices.loc[index[-1], "SPY"] = prices.loc[index[-2], "SPY"] * 1.01
    breadth = pd.DataFrame({"Breadth_Index_Raw": [40.0], "Breadth_Index_8MA": [40.0]}, index=index[:1])
    uptrend = pd.DataFrame({"ratio": [20.0], "ma10": [20.0], "slope": [0.1], "trend": ["up"], "color": ["GREEN"]}, index=index[:1])
    broadened, updated, note = nowcast_breadth(prices, breadth, uptrend)
    assert "T+0" in note
    assert broadened.index[-1] == index[-1]
    assert float(broadened["Breadth_Index_Raw"].iloc[-1]) != 40.0
    assert updated.index[-1] == index[-1]


def test_collinear_factors_are_not_double_counted() -> None:
    merged = merge_collinear_factors({"s1": 2.0, "s2": 1.0, "s3": 1.5, "s4": 0.0, "s5": 0.0, "score": 4.5}, 0.90)
    assert merged["s3"] == 0.0
    assert merged["score"] == 3.0
    untouched = merge_collinear_factors({"s1": 2.0, "s2": 1.0, "s3": 1.5, "s4": 0.0, "s5": 0.0, "score": 4.5}, 0.20)
    assert untouched["score"] == 4.5


def test_ledger_records_and_backfills(tmp_path) -> None:
    from src.regime_synthesizer import MarketDiagnosis, RegimeDecision, TickerDiagnosis
    from src.trend_credit_gate import TrendCreditState
    from src.vix_structure_engine import VolAssessment
    from src.breadth_engine import BreadthScore, DivergenceResult

    index = pd.bdate_range("2024-01-02", periods=30)
    prices = pd.DataFrame({"^GSPC": np.linspace(100.0, 130.0, len(index))}, index=index)
    trend = TrendCreditState(
        ticker="SPY",
        benchmark="^GSPC",
        asof=index[0],
        close=100.0,
        sma20=90.0,
        sma50=80.0,
        sma200=70.0,
        sma20_slope=0.01,
        high_252=100.0,
        ratio_to_high=1.0,
        drawdown_252=0.0,
        low_5=95.0,
        trend_is_bull=True,
        stacked_bull=True,
        below_sma50=False,
        short_momentum_weak=False,
        hyg_ief=0.8,
        hyg_ief_sma50=0.81,
        credit_risk_off=True,
        ready=True,
        reason="OK",
        hv20=0.2,
        atr14=10.0,
        hysteresis_state="ABOVE",
    )
    regime = RegimeDecision(REGIME_1B, "带逆风牛市", "上限 75%", 70.0, "", "", "测试", leverage_cap=2.0, instrument="SSO")
    diagnosis = MarketDiagnosis(
        asof=index[0],
        vol=VolAssessment(status="OK", score=-1.5, r_ts=0.88, z_ts=0.2),
        breadth=BreadthScore(score=70.0),
        anchor=None,
        tickers=[TickerDiagnosis("SPY", trend, "⏸ NEUTRAL", DivergenceResult(False), regime)],
    )
    path = tmp_path / "regime_ledger.db"
    assert record_diagnosis(diagnosis, path) == 1
    assert backfill_outcomes(prices, path) > 0
    assert "T+20" in ledger_summary(path)
