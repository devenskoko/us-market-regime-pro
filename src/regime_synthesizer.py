"""把趋势信贷、期权期限结构与市场宽度合成唯一的市场状态。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from src.breadth_engine import BreadthScore, DivergenceResult, detect_bearish_divergence, score_breadth
from src.config import DATA_NA, REGIME_1, REGIME_1B, REGIME_2, REGIME_3, REGIME_4, REGIME_5
from src.leverage_gate import resolve_leverage
from src.macro_gate import MacroState, assess_macro
from src.trend_credit_gate import TrendCreditState, assess_trend_credit
from src.vix_structure_engine import (
    VolAssessment,
    apply_orthogonal_gates,
    compute_vol_snapshot,
)


@dataclass(frozen=True)
class RegimeDecision:
    regime_id: str
    name_zh: str
    exposure_range: str
    exposure_point: float | None
    equity_action: str
    leverage_action: str
    reason: str
    leverage_cap: float = 1.0
    instrument: str = ""
    hv20: float | None = None
    hysteresis_state: str = ""


@dataclass
class TickerDiagnosis:
    ticker: str
    trend: TrendCreditState
    display_state: str | None
    divergence: DivergenceResult
    regime: RegimeDecision


@dataclass
class MarketDiagnosis:
    asof: pd.Timestamp | None
    vol: VolAssessment
    breadth: BreadthScore
    anchor: TickerDiagnosis | None
    tickers: list[TickerDiagnosis] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    breadth_source: str = ""


def _unavailable(reason: str) -> RegimeDecision:
    return RegimeDecision(
        regime_id=DATA_NA,
        name_zh="数据不足",
        exposure_range="—",
        exposure_point=None,
        equity_action="结构分或长均线不可用，不输出仓位。",
        leverage_action="禁止根据缺失数据推断杠杆方向。",
        reason=reason,
    )


def _clamp_exposure(raw: float | None, low: float, high: float, fallback: float) -> float:
    if raw is None:
        return fallback
    return float(min(high, max(low, raw)))


def _decision(
    regime_id: str,
    name_zh: str,
    exposure_range: str,
    exposure_point: float,
    equity_action: str,
    leverage_action: str,
    reason: str,
) -> RegimeDecision:
    return RegimeDecision(
        regime_id=regime_id,
        name_zh=name_zh,
        exposure_range=exposure_range,
        exposure_point=exposure_point,
        equity_action=equity_action,
        leverage_action=leverage_action,
        reason=reason,
    )


def classify_regime(
    *,
    close: float,
    sma50: float,
    sma200: float,
    r_ts: float,
    r_ts_prev: float,
    score_vol: float | None,
    score_breadth: float | None,
    bearish_divergence: bool,
    is_backwardation_risk: bool,
    credit_risk_off: bool,
    uptrend_ratio: float | None,
    uptrend_slope: float | None,
    confirmed_below_sma50: bool | None = None,
    macro_headwind: bool = False,
    macro_reason: str = "",
    sticky_exposure: float | None = None,
) -> RegimeDecision:
    """按风险优先级返回唯一 Regime。严格条件先匹配，剩余状态再归入最接近的象限。"""
    if score_vol is None or r_ts is None or sma50 is None or sma200 is None or close is None:
        return _unavailable("波动率结构分或均线缺失")
    if confirmed_below_sma50 is None:
        confirmed_below_sma50 = close < sma50
    exposure_unit = None if sticky_exposure is None else sticky_exposure * 100.0

    capitulation_vol = score_vol >= 5.0 and r_ts <= r_ts_prev
    capitulation_breadth = (
        uptrend_ratio is not None and uptrend_ratio < 10.0 and uptrend_slope is not None and uptrend_slope > 0
    )
    if capitulation_vol or capitulation_breadth:
        extra = " 底层基准仍在 SMA200 下方，只允许分批，并以拐点前低为止损。" if close <= sma200 else ""
        why = "期限结构恐慌停止发散" if capitulation_vol else "上升趋势占比跌破 10% 后拐头"
        return _decision(
            REGIME_5,
            "极端恐慌洗盘反转点",
            "40% ~ 70%",
            _clamp_exposure(exposure_unit, 40.0, 70.0, 55.0),
            "分批建立战术多头，止损放在拐点前低。",
            "赔率最高的右侧反转买点，杠杆仓位必须带硬止损。" + extra,
            why,
        )

    if close <= sma200 and credit_risk_off and score_breadth is not None and score_breadth < 35.0:
        return _decision(
            REGIME_4,
            "系统性熊市",
            "0% ~ 15%",
            _clamp_exposure(exposure_unit, 0.0, 15.0, 10.0),
            "权益降到短债（SGOV / BIL），保留极少风险预算。",
            "🚫 NO TRADE：严禁左侧死扛任何 2×/3× 杠杆多头 ETF。",
            "收盘价在 SMA200 下方，信贷比弱于自身 SMA50，且宽度低于 35",
        )

    correction = (close > sma200 and confirmed_below_sma50) or is_backwardation_risk
    if correction:
        why = "期限倒挂初期，尚未到达恐慌出清极值" if is_backwardation_risk else "收盘价确认跌破 SMA50 降档线，且尚未收回升档线"
        return _decision(
            REGIME_3,
            "中期修正 / 期权倒挂震荡",
            "20% ~ 40%",
            _clamp_exposure(exposure_unit, 20.0, 40.0, 30.0),
            "立刻降低总敞口，等待期限比率回到 0.98 以下或出现 STRONG BUY。",
            "立即去杠杆，3× ETF 退出，只留 1× 或现金。",
            why,
        )

    fragile = close > sma200 and (bearish_divergence or score_vol <= -2.0)
    if fragile:
        why = "指数贴近高位但内部宽度背离" if bearish_divergence else "波动率结构分进入 SELL/HEDGE 或 EUPHORIA"
        return _decision(
            REGIME_2,
            "脆弱分化牛（假突破高发区）",
            "50% ~ 65%",
            _clamp_exposure(exposure_unit, 50.0, 65.0, 60.0),
            "保留核心宽基，停止追高。",
            "严禁追高加仓；把 TQQQ / UPRO / SOXL 降维成 SPY / QQQ，或收紧止盈。",
            why,
        )

    strong = (
        close > sma200
        and sma50 > sma200
        and not confirmed_below_sma50
        and r_ts < 0.95
        and score_breadth is not None
        and score_breadth >= 60.0
        and not bearish_divergence
    )
    if strong and macro_headwind:
        return _decision(
            REGIME_1B,
            "带逆风牛市",
            "连续仓位，上限 75%",
            _clamp_exposure(exposure_unit, 20.0, 75.0, 75.0),
            "趋势仍在，但利率、美元或信贷宽度共振压制估值，权益上限锁定 75%。",
            "杠杆倍数上限 2×，不使用 3× 产品。",
            macro_reason or "宏观分母端逆风",
        )
    if strong:
        band_note = "收盘落在 SMA50 缓冲带内，维持上一交易日的升档状态。" if close < sma50 else "均线多头排列、期限结构无倒挂、宽度不少于 60 且没有顶背离。"
        return _decision(
            REGIME_1,
            "强劲全面牛市",
            "连续仓位，上限 100%",
            _clamp_exposure(exposure_unit, 20.0, 100.0, 95.0),
            "权益维持高仓，回撤参考底层 SMA20 或 SMA50。",
            "允许右侧顺势持有杠杆 ETF，追踪止损设在底层指数 SMA20 或 SMA50。",
            band_note,
        )

    if close <= sma200:
        return _decision(
            REGIME_4,
            "系统性熊市",
            "0% ~ 15%",
            _clamp_exposure(exposure_unit, 0.0, 15.0, 10.0),
            "长均线已经失守，权益降到短债，等待信贷与宽度同时修复。",
            "🚫 NO TRADE：严禁左侧死扛任何 2×/3× 杠杆多头 ETF。",
            "收盘价在 SMA200 上或下方；信贷或宽度尚未凑齐严格三项，仍按长均线失守处理",
        )
    if confirmed_below_sma50 or (score_vol >= 5.0 and r_ts > r_ts_prev):
        return _decision(
            REGIME_3,
            "中期修正 / 期权倒挂震荡",
            "20% ~ 40%",
            _clamp_exposure(exposure_unit, 20.0, 40.0, 30.0),
            "动量未修复，维持防守仓位。",
            "立即去杠杆，等待 STRONG BUY 或期限比率回落到 0.98 以下。",
            "价格或期限结构仍处于修正，未达到反转确认",
        )
    return _decision(
        REGIME_2,
        "脆弱分化牛（假突破高发区）",
        "50% ~ 65%",
        _clamp_exposure(exposure_unit, 50.0, 65.0, 60.0),
        "趋势还在，但不够格升入全面牛市。",
        "不加杠杆追价；杠杆 ETF 维持降维或不加仓。",
        "未同时满足强牛的期限结构、宽度与背离条件",
    )


def _with_leverage(regime: RegimeDecision, trend: TrendCreditState) -> RegimeDecision:
    cap, note, instrument = resolve_leverage(trend.hv20, regime.regime_id, trend.benchmark)
    return replace(
        regime,
        leverage_cap=cap,
        instrument=instrument,
        hv20=trend.hv20,
        hysteresis_state=trend.hysteresis_state,
        leverage_action=f"{regime.leverage_action} 实际上限 {cap:.0f}× → {instrument}。{note}",
    )


def _diagnose_one(
    ticker: str,
    prices: pd.DataFrame,
    breadth: pd.DataFrame | None,
    uptrend: pd.DataFrame | None,
    vol: VolAssessment,
    breadth_score: BreadthScore,
    asof: pd.Timestamp,
    macro: MacroState,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
) -> TickerDiagnosis:
    trend = assess_trend_credit(prices, ticker, asof=asof, high=high, low=low)
    divergence = DivergenceResult(False, ["NO_BREADTH"])
    if breadth is not None and uptrend is not None and trend.close is not None and trend.benchmark in prices.columns:
        divergence = detect_bearish_divergence(
            prices[trend.benchmark],
            breadth,
            uptrend,
            prices,
            asof=asof,
        )

    if vol.status != "OK" or vol.score is None or not trend.ready or trend.close is None or trend.sma50 is None or trend.sma200 is None:
        return TickerDiagnosis(
            ticker=ticker,
            trend=trend,
            display_state=None,
            divergence=divergence,
            regime=_unavailable(vol.reason or trend.reason or "数据不足"),
        )

    display = apply_orthogonal_gates(
        vol.score,
        trend.trend_is_bull,
        r_ts=float(vol.r_ts),
        r_ts_prev=float(vol.r_ts_prev),
        z_ts=float(vol.z_ts),
        p_skew=float(vol.p_skew),
        close=float(trend.close),
        low_5=float(trend.low_5 if trend.low_5 is not None else trend.close),
        short_momentum_weak=trend.short_momentum_weak,
    )
    regime = classify_regime(
        close=float(trend.close),
        sma50=float(trend.sma50),
        sma200=float(trend.sma200),
        r_ts=float(vol.r_ts),
        r_ts_prev=float(vol.r_ts_prev),
        score_vol=float(vol.score),
        score_breadth=breadth_score.score,
        bearish_divergence=divergence.flag,
        is_backwardation_risk=vol.is_backwardation_risk,
        credit_risk_off=trend.credit_risk_off,
        uptrend_ratio=breadth_score.uptrend_ratio,
        uptrend_slope=breadth_score.uptrend_slope,
        confirmed_below_sma50=trend.confirmed_below_sma50,
        macro_headwind=macro.active,
        macro_reason=macro.reason,
        sticky_exposure=trend.sticky_exposure,
    )
    regime = _with_leverage(regime, trend)
    return TickerDiagnosis(ticker=ticker, trend=trend, display_state=display, divergence=divergence, regime=regime)


def diagnose(
    tickers: list[str],
    prices: pd.DataFrame,
    breadth: pd.DataFrame | None,
    uptrend: pd.DataFrame | None,
    sectors: pd.DataFrame | None,
    warnings: list[str] | None = None,
    breadth_source: str = "",
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
) -> MarketDiagnosis:
    asof = pd.Timestamp(prices.index[-1])
    vol = compute_vol_snapshot(prices)
    breadth_score = score_breadth(prices, breadth, uptrend, sectors, asof=asof)
    anchor_trend = assess_trend_credit(prices, "^GSPC", asof=asof, high=high, low=low)
    macro = assess_macro(
        prices,
        credit_risk_off=anchor_trend.credit_risk_off,
        uptrend_ratio=breadth_score.uptrend_ratio,
        asof=asof,
    )
    rows = [
        _diagnose_one(ticker, prices, breadth, uptrend, vol, breadth_score, asof, macro, high, low)
        for ticker in tickers
    ]
    anchor = _diagnose_one("^GSPC", prices, breadth, uptrend, vol, breadth_score, asof, macro, high, low)
    return MarketDiagnosis(
        asof=vol.asof or asof,
        vol=vol,
        breadth=breadth_score,
        anchor=anchor,
        tickers=rows,
        warnings=list(warnings or []),
        breadth_source=breadth_source,
    )
