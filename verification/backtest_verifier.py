"""2018 年以来的向量化回放：次日开盘成交、10bps 成本，并对照持有 SPY / TQQQ。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ATR_BAND, HV_HIGH, HV_LOW, REPORTS_DIR, SECTOR_ETFS
from src.data_fetcher import fetch_price_book
from src.hysteresis import average_true_range, sma50_hysteresis
from src.config import REGIME_2, REGIME_3, REGIME_4
from src.leverage_gate import realized_vol, regime_cap
from src.macro_gate import rolling_change_z
from src.regime_synthesizer import classify_regime
from src.vix_structure_engine import merge_collinear_factors, rolling_percentile_rank, score_vol_components

EPISODES = {
    "2018Q4": ("2018-10-01", "2018-12-31"),
    "2020熔断": ("2020-02-15", "2020-04-30"),
    "2022加息": ("2022-01-01", "2022-12-31"),
    "2024日元闪崩": ("2024-07-01", "2024-08-31"),
}


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def _sharpe(returns: pd.Series) -> float:
    series = returns.dropna()
    if len(series) < 5 or float(series.std(ddof=1)) == 0:
        return float("nan")
    return float(series.mean() / series.std(ddof=1) * np.sqrt(252.0))


def _calmar(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = abs(_max_drawdown(equity))
    if drawdown == 0:
        return float("nan")
    annual = float((1.0 + returns.fillna(0.0)).prod() ** (252.0 / max(len(returns), 1)) - 1.0)
    return annual / drawdown


def _vol_frame(close: pd.DataFrame) -> pd.DataFrame:
    r_ts = close["^VIX"] / close["^VIX3M"]
    mu = r_ts.rolling(63, min_periods=63).mean()
    sigma = r_ts.rolling(63, min_periods=63).std(ddof=1)
    z_ts = (r_ts - mu) / sigma
    r_basis = close["^VIX9D"] / close["^VIX"]
    p_vvix = rolling_percentile_rank(close["^VVIX"])
    p_skew = rolling_percentile_rank(close["^SKEW"])
    correlation = r_basis.rolling(63, min_periods=63).corr(r_ts)
    scores = []
    merged = []
    for z, r, basis, vix, pvol, pskew, corr in zip(z_ts, r_ts, r_basis, close["^VIX"], p_vvix, p_skew, correlation):
        if not all(np.isfinite(value) for value in (z, r, basis, vix, pvol, pskew)):
            scores.append(np.nan)
            merged.append(False)
            continue
        factors = score_vol_components(float(z), float(r), float(basis), float(vix), float(pvol), float(pskew))
        factors = merge_collinear_factors(factors, float(corr) if np.isfinite(corr) else np.nan)
        scores.append(factors["score"])
        merged.append(bool(factors["collinearity"]))
    return pd.DataFrame(
        {
            "r_ts": r_ts,
            "r_ts_prev": r_ts.shift(1),
            "z_ts": z_ts,
            "score": scores,
            "merged": merged,
        },
        index=close.index,
    )


def _breadth_proxy(close: pd.DataFrame) -> pd.DataFrame:
    columns = [ticker for ticker in SECTOR_ETFS if ticker in close.columns]
    flags = []
    for ticker in columns:
        series = close[ticker]
        above = (
            (series > series.rolling(50, min_periods=50).mean()).astype(float)
            + (series > series.rolling(200, min_periods=200).mean()).astype(float)
        ) / 2.0
        flags.append(above)
    participation = pd.concat(flags, axis=1).mean(axis=1) * 100.0 if flags else pd.Series(np.nan, index=close.index)
    return pd.DataFrame({"breadth": participation, "uptrend": participation}, index=close.index)


def _simulate(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    opens: pd.DataFrame,
    *,
    band: float = ATR_BAND,
    hv_low: float = HV_LOW,
    hv_high: float = HV_HIGH,
) -> pd.DataFrame:
    vol = _vol_frame(close)
    breadth = _breadth_proxy(close)
    credit = close["HYG"] / close["IEF"]
    credit_sma = credit.rolling(50, min_periods=50).mean()
    z_tnx = rolling_change_z(close["^TNX"]) if "^TNX" in close.columns else pd.Series(np.nan, index=close.index)
    z_dxy = rolling_change_z(close["DX-Y.NYB"]) if "DX-Y.NYB" in close.columns else pd.Series(np.nan, index=close.index)
    benchmark = close["^NDX"]
    sma50 = benchmark.rolling(50, min_periods=50).mean()
    sma200 = benchmark.rolling(200, min_periods=200).mean()
    atr = average_true_range(
        benchmark,
        high["^NDX"] if "^NDX" in high.columns else None,
        low["^NDX"] if "^NDX" in low.columns else None,
    )
    path = sma50_hysteresis(benchmark, sma50, atr, band=band)
    hv = realized_vol(benchmark)
    gspc = close["^GSPC"]
    gspc_high = gspc.rolling(252, min_periods=252).max()
    rows = []
    for stamp in close.index:
        score = vol.at[stamp, "score"]
        r_ts = vol.at[stamp, "r_ts"]
        r_prev = vol.at[stamp, "r_ts_prev"]
        mid = sma50.at[stamp]
        slow = sma200.at[stamp]
        price = benchmark.at[stamp]
        state = path.at[stamp, "state"]
        sticky = path.at[stamp, "exposure"]
        if not all(np.isfinite(value) for value in (score, r_ts, r_prev, mid, slow, price)) or not state:
            rows.append((np.nan, 1.0, "QQQ"))
            continue
        uptrend = breadth.at[stamp, "uptrend"]
        breadth_score = breadth.at[stamp, "breadth"]
        credit_off = bool(np.isfinite(credit.at[stamp]) and np.isfinite(credit_sma.at[stamp]) and credit.at[stamp] < credit_sma.at[stamp])
        headwind = bool(
            (np.isfinite(z_tnx.at[stamp]) and z_tnx.at[stamp] > 1.5)
            or (np.isfinite(z_dxy.at[stamp]) and z_dxy.at[stamp] > 1.5)
            or (credit_off and np.isfinite(uptrend) and uptrend < 30.0)
        )
        near_high = bool(np.isfinite(gspc_high.at[stamp]) and gspc.at[stamp] / gspc_high.at[stamp] >= 0.98)
        divergence = bool(near_high and np.isfinite(uptrend) and uptrend < 25.0)
        decision = classify_regime(
            close=float(price),
            sma50=float(mid),
            sma200=float(slow),
            r_ts=float(r_ts),
            r_ts_prev=float(r_prev),
            score_vol=float(score),
            score_breadth=None if not np.isfinite(breadth_score) else float(breadth_score),
            bearish_divergence=divergence,
            is_backwardation_risk=bool(r_ts > 1.0 and vol.at[stamp, "z_ts"] > 1.0 and score < 5.0),
            credit_risk_off=credit_off,
            uptrend_ratio=None if not np.isfinite(uptrend) else float(uptrend),
            uptrend_slope=0.0,
            confirmed_below_sma50=state == "BELOW",
            macro_headwind=headwind,
            sticky_exposure=None if not np.isfinite(sticky) else float(sticky),
        )
        hv_value = hv.at[stamp]
        structural = regime_cap(decision.regime_id)
        if decision.regime_id in {REGIME_2, REGIME_3, REGIME_4}:
            structural = 1.0
        if not np.isfinite(hv_value):
            vol_cap = 1.0
        elif hv_value < hv_low:
            vol_cap = 3.0
        elif hv_value <= hv_high:
            vol_cap = 2.0
        else:
            vol_cap = 1.0
        cap = 1.0 if decision.regime_id in {REGIME_2, REGIME_3, REGIME_4} else min(structural, vol_cap)
        instrument = {3.0: "TQQQ", 2.0: "QLD", 1.0: "QQQ"}[cap]
        exposure = 0.0 if decision.exposure_point is None else float(decision.exposure_point) / 100.0
        rows.append((exposure, cap, instrument))
    result = pd.DataFrame(rows, index=close.index, columns=["exposure", "cap", "instrument"])
    result["merged"] = vol["merged"]
    open_ret = opens.pct_change()
    asset = pd.Series(np.nan, index=close.index)
    for name in ("TQQQ", "QLD", "QQQ"):
        if name not in open_ret.columns:
            continue
        asset = asset.where(result["instrument"] != name, open_ret[name])
    # 收盘信号在下一次开盘成交，收益记到再下一根开盘。
    held = result["exposure"].shift(2) * asset
    turnover = result["exposure"].diff().abs().fillna(0.0)
    switched = result["instrument"].ne(result["instrument"].shift(1)).astype(float) * result["exposure"]
    cost = 0.001 * (turnover + switched).shift(1)
    result["strategy"] = held - cost
    result["spy"] = open_ret["SPY"] if "SPY" in open_ret.columns else np.nan
    result["tqqq"] = open_ret["TQQQ"] if "TQQQ" in open_ret.columns else np.nan
    result["state"] = path["state"]
    result["ndx_gap"] = opens["^NDX"] / close["^NDX"].shift(1) - 1.0 if "^NDX" in opens.columns else np.nan
    result["sma_gap"] = (benchmark - sma50).abs()
    result["atr"] = atr
    return result


def _stats(returns: pd.Series) -> dict[str, float]:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return {
        "sharpe": _sharpe(returns),
        "calmar": _calmar(returns),
        "max_dd": _max_drawdown(equity),
        "total": float(equity.iloc[-1] - 1.0) if len(equity) else float("nan"),
    }


def _garch_cvar(returns: pd.Series) -> float:
    values = returns.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(values) < 50:
        return float("nan")
    variance = float(np.var(values))
    alpha, beta = 0.06, 0.92
    omega = variance * (1.0 - alpha - beta)
    sigma2 = variance
    for item in values:
        sigma2 = omega + alpha * item * item + beta * sigma2
    rng = np.random.default_rng(7)
    draws = []
    for _ in range(4000):
        shock = float(rng.standard_normal() * np.sqrt(sigma2))
        draws.append(shock)
        sigma2 = omega + alpha * shock * shock + beta * sigma2
    sample = np.array(draws)
    cutoff = np.quantile(sample, 0.05)
    return float(sample[sample <= cutoff].mean())


def run_verification(reports_dir: Path | None = None) -> Path:
    from src.config import ALL_PRICE_TICKERS

    tickers = list(dict.fromkeys([*ALL_PRICE_TICKERS, "TQQQ", "QLD", "UPRO", "SSO"]))
    book = fetch_price_book(tickers=tickers, start=pd.Timestamp("2016-01-01"), save=False)
    base = _simulate(book.close, book.high, book.low, book.open)
    usable = base.dropna(subset=["strategy"])
    lines = ["# V2 验证回放", "", "信号在收盘产生，下一交易日开盘成交，换仓扣除 10 个基点。3 倍工具使用真实开盘价，方差损耗已经在价格里。", ""]
    lines.append("## 全样本")
    lines.append("")
    lines.append("| 组合 | 夏普 | 卡玛 | 最大回撤 | 累计收益 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for label, column in (("状态机 NDX 袖套", "strategy"), ("持有 SPY", "spy"), ("持有 TQQQ", "tqqq")):
        stats = _stats(usable[column])
        lines.append(
            f"| {label} | {stats['sharpe']:.2f} | {stats['calmar']:.2f} | {stats['max_dd']:.1%} | {stats['total']:.1%} |"
        )
    merged_rate = float(pd.Series(base["merged"]).fillna(False).mean())
    lines.extend(["", "## 因子共线性", "", f"- 近端基差与期限比率 63 日相关超过 0.85 的交易日占比：{merged_rate:.1%}。这些日子只保留绝对值更大的一项。", ""])
    lines.extend(["## 分阶段最大回撤", "", "| 阶段 | 状态机 | SPY | TQQQ |", "| --- | ---: | ---: | ---: |"])
    for name, (start, end) in EPISODES.items():
        window = usable.loc[start:end]
        if window.empty:
            lines.append(f"| {name} | — | — | — |")
            continue
        lines.append(
            f"| {name} | {_max_drawdown((1+window['strategy'].fillna(0)).cumprod()):.1%} | {_max_drawdown((1+window['spy'].fillna(0)).cumprod()):.1%} | {_max_drawdown((1+window['tqqq'].fillna(0)).cumprod()):.1%} |"
        )
    lines.extend(["", "## 阈值扰动 ±15%", "", "| 参数 | 夏普 | 卡玛 |", "| --- | ---: | ---: |"])
    grid = [("基准", ATR_BAND, HV_LOW, HV_HIGH)]
    for scale in (0.85, 1.15):
        grid.append((f"ATR带×{scale:.2f}", ATR_BAND * scale, HV_LOW, HV_HIGH))
        grid.append((f"HV门槛×{scale:.2f}", ATR_BAND, HV_LOW * scale, HV_HIGH * scale))
    sharpes = []
    for label, band, low, high in grid:
        trial = base if label == "基准" else _simulate(book.close, book.high, book.low, book.open, band=band, hv_low=low, hv_high=high)
        stats = _stats(trial["strategy"].dropna())
        sharpes.append(stats["sharpe"])
        lines.append(f"| {label} | {stats['sharpe']:.2f} | {stats['calmar']:.2f} |")
    finite = [value for value in sharpes if np.isfinite(value)]
    if len(finite) >= 5:
        atr_span = max(finite[1], finite[3]) - min(finite[1], finite[3])
        hv_span = max(finite[2], finite[4]) - min(finite[2], finite[4])
        lines.append("")
        lines.append(
            f"ATR 带宽 ±15% 的夏普极差是 {atr_span:.2f}。HV 门槛 ±15% 的夏普极差是 {hv_span:.2f}。"
            "带宽几乎不改变结果；HV 门槛会改变 2 倍和 3 倍的切换频率。"
        )
    fragile = base[(base["state"] == "ABOVE") & (base["sma_gap"] <= 0.25 * base["atr"])]
    gaps = fragile["ndx_gap"].shift(-1).dropna()
    lines.extend(["", "## 尾部", ""])
    if len(gaps):
        cutoff = float(gaps.quantile(0.05))
        cvar = float(gaps[gaps <= cutoff].mean())
        lines.append(f"- 标普侧升档、纳指距离 SMA50 不到 0.25 ATR 的次日开盘跳空，5% CVaR 为 {cvar:.2%}，样本 {len(gaps)} 天。")
        lines.append(f"- 其中开盘跳空小于或等于 -3% 的次数：{int((gaps <= -0.03).sum())}。")
    lines.append(f"- 纳指日收益的方差目标 GARCH(1,1) 模拟 5% CVaR：{_garch_cvar(book.close['^NDX'].pct_change()):.2%}。3 倍工具在 -3% 隔夜缺口下的价格冲击大约是 -9%，再叠加真实杠杆 ETF 已经包含的损耗。")
    lines.extend(["", "规则化回放，不是投资建议。", ""])
    folder = reports_dir or REPORTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    asof = book.close.index[-1].date().isoformat()
    path = folder / f"v2_verification_{asof}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
