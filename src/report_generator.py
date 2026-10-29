"""Rich 终端看板与 Markdown / HTML 复盘报告。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config import REPORTS_DIR
from src.regime_synthesizer import MarketDiagnosis, TickerDiagnosis
from src.vix_structure_engine import BACKWARDATION_ALERT


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _trend_text(row: TickerDiagnosis) -> str:
    trend = row.trend
    if not trend.ready or trend.sma50 is None or trend.sma200 is None:
        return "均线未预热"
    if trend.stacked_bull:
        stack = "多头排列"
    elif trend.trend_is_bull and trend.below_sma50:
        stack = "200 上方 / 50 下方"
    elif trend.trend_is_bull:
        stack = "200 上方"
    else:
        stack = "200 下方"
    return f"{trend.benchmark} {stack}"


def _overview_text(diagnosis: MarketDiagnosis) -> Text:
    vol = diagnosis.vol
    breadth = diagnosis.breadth
    anchor = diagnosis.anchor
    body = Text()
    body.append(f"交易日  {diagnosis.asof.date() if diagnosis.asof is not None else '—'}\n")
    body.append(f"宽度来源  {diagnosis.breadth_source or '—'}\n")
    if anchor is not None:
        body.append("市场锚 ^GSPC  ", style="bold")
        body.append(f"{anchor.regime.regime_id}  {anchor.regime.name_zh}\n", style="bold")
        point = "—" if anchor.regime.exposure_point is None else f"{anchor.regime.exposure_point:.1f}%"
        body.append(f"建议权益敞口  {anchor.regime.exposure_range}（当前 {point}）\n")
        body.append(f"判定依据  {anchor.regime.reason}\n")
        trend = anchor.trend
        body.append(
            f"SMA50 迟滞  {trend.hysteresis_state or '—'}   "
            f"ATR14 { _fmt(trend.atr14) }   "
            f"降档线 { _fmt(trend.sma50_lower) } / 升档线 { _fmt(trend.sma50_upper) }   "
            f"HV20 { _fmt(None if trend.hv20 is None else trend.hv20 * 100) }%\n"
        )
    if vol.status != "OK":
        body.append(f"波动率引擎  {vol.status}  {vol.reason}\n", style="bold red")
    else:
        body.append(
            f"Score_vol { _fmt(vol.score) }   Z_TS { _fmt(vol.z_ts) }   "
            f"R_TS { _fmt(vol.r_ts, 4) }   R_Basis { _fmt(vol.r_basis, 4) }\n"
        )
        body.append(
            f"P_VVIX { _fmt(vol.p_vvix, 1) }   P_SKEW { _fmt(vol.p_skew, 1) }   "
            f"VIX { _fmt(vol.vix) }\n"
        )
        factors = vol.factors
        body.append(
            "因子  "
            + "  ".join(f"{key}={_fmt(factors.get(key), 1)}" for key in ("s1", "s2", "s3", "s4", "s5"))
            + "\n"
        )
    if vol.backwardation_alert:
        body.append(BACKWARDATION_ALERT + "\n", style="bold yellow")
    if breadth.score is None:
        body.append(f"宽度引擎  {breadth.status}  {breadth.reason}\n", style="bold red")
    else:
        relation = "快线高于慢线" if (breadth.fast or 0) > (breadth.slow or 0) else "快线低于或等于慢线"
        body.append(
            f"Score_breadth { _fmt(breadth.score, 1) }   {relation}   "
            f"8MA { _fmt(breadth.fast, 1) } / 200MA { _fmt(breadth.slow, 1) }   "
            f"5日斜率 { _fmt(breadth.fast_slope_5, 2) }\n"
        )
        body.append(
            f"Uptrend { _fmt(breadth.uptrend_ratio, 1) }%  {breadth.uptrend_color or '—'}   "
            f"RSP/SPY { _fmt(breadth.rsp_spy, 4) }   "
            f"进攻-防御 { _fmt(breadth.off_def_spread, 1) } pct\n"
        )
    if anchor is not None and anchor.divergence.flag:
        body.append("顶背离  BEARISH_BREADTH_DIVERGENCE\n", style="bold red")
        body.append("触发  " + ", ".join(anchor.divergence.reasons) + "\n")
    credit = anchor.trend if anchor is not None else None
    if credit is not None and credit.hyg_ief is not None:
        state = "低于 SMA50" if credit.credit_risk_off else "不低于 SMA50"
        body.append(f"HYG/IEF { _fmt(credit.hyg_ief, 4) }  相对 SMA50 {state}\n")
    for warning in diagnosis.warnings:
        body.append(warning + "\n", style="yellow")
    return body


def _ticker_table(diagnosis: MarketDiagnosis) -> Table:
    table = Table(title="标的路由与操作指令", show_lines=False)
    table.add_column("标的")
    table.add_column("路由趋势")
    table.add_column("基准收盘")
    table.add_column("Regime")
    table.add_column("敞口")
    table.add_column("波动率状态")
    table.add_column("顶背离")
    table.add_column("杠杆指令")
    for row in diagnosis.tickers:
        table.add_row(
            row.ticker,
            _trend_text(row),
            _fmt(row.trend.close),
            f"{row.regime.regime_id}\n{row.regime.name_zh}",
            f"{row.regime.exposure_range}\n{_fmt(row.regime.exposure_point, 1)}% / {row.regime.leverage_cap:.0f}× {row.regime.instrument}",
            row.display_state or row.regime.regime_id,
            "是" if row.divergence.flag else "否",
            row.regime.leverage_action,
        )
    return table


def render_console(diagnosis: MarketDiagnosis, console: Console | None = None) -> None:
    out = console or Console()
    out.print(Panel(_overview_text(diagnosis), title="美股趋势与市场状态", border_style="cyan"))
    out.print(_ticker_table(diagnosis))
    out.print("规则化监测结果，不是投资建议。滚动统计只使用截至当日的历史窗口。")


def _markdown(diagnosis: MarketDiagnosis) -> str:
    vol = diagnosis.vol
    breadth = diagnosis.breadth
    anchor = diagnosis.anchor
    asof = diagnosis.asof.date().isoformat() if diagnosis.asof is not None else date.today().isoformat()
    lines = [
        f"# 美股市场状态复盘 {asof}",
        "",
        f"- 宽度来源：`{diagnosis.breadth_source or '—'}`",
        f"- 波动率引擎：`{vol.status}`",
    ]
    if anchor is not None:
        lines.extend(
            [
                f"- 市场锚：`{anchor.regime.regime_id}` {anchor.regime.name_zh}",
                f"- 建议权益敞口：{anchor.regime.exposure_range}（当前 {_fmt(anchor.regime.exposure_point, 1)}%）",
                f"- SMA50 迟滞：{anchor.trend.hysteresis_state or '—'}，ATR14 {_fmt(anchor.trend.atr14)}，降档线 {_fmt(anchor.trend.sma50_lower)}，升档线 {_fmt(anchor.trend.sma50_upper)}，HV20 {_fmt(None if anchor.trend.hv20 is None else anchor.trend.hv20 * 100)}%",
                f"- 杠杆工具：{anchor.regime.leverage_cap:.0f}× {anchor.regime.instrument}",
                f"- 判定依据：{anchor.regime.reason}",
                f"- 宽基指令：{anchor.regime.equity_action}",
                f"- 杠杆指令：{anchor.regime.leverage_action}",
            ]
        )
    lines.extend(["", "## 期权期限结构", ""])
    if vol.status != "OK":
        lines.append(f"状态 `{vol.status}`：{vol.reason}")
    else:
        lines.extend(
            [
                f"- Score_vol：{_fmt(vol.score)}",
                f"- Z_TS(63)：{_fmt(vol.z_ts)}",
                f"- R_TS：{_fmt(vol.r_ts, 4)}（前一日 {_fmt(vol.r_ts_prev, 4)}）",
                f"- R_Basis：{_fmt(vol.r_basis, 4)}",
                f"- P_VVIX / P_SKEW：{_fmt(vol.p_vvix, 1)} / {_fmt(vol.p_skew, 1)}",
                f"- 因子：s1={_fmt(vol.factors.get('s1'), 1)}，s2={_fmt(vol.factors.get('s2'), 1)}，s3={_fmt(vol.factors.get('s3'), 1)}，s4={_fmt(vol.factors.get('s4'), 1)}，s5={_fmt(vol.factors.get('s5'), 1)}",
                f"- 倒挂去杠杆：{'是' if vol.is_backwardation_risk else '否'}",
            ]
        )
        if vol.backwardation_alert:
            lines.append(f"- {vol.backwardation_alert}")
    lines.extend(["", "## 市场宽度", ""])
    if breadth.score is None:
        lines.append(f"状态 `{breadth.status}`：{breadth.reason}")
    else:
        lines.extend(
            [
                f"- Score_breadth：{_fmt(breadth.score, 1)}",
                f"- 8MA / 200MA：{_fmt(breadth.fast, 1)} / {_fmt(breadth.slow, 1)}，5 日斜率 {_fmt(breadth.fast_slope_5, 2)}",
                f"- Uptrend Ratio：{_fmt(breadth.uptrend_ratio, 1)}%（{breadth.uptrend_color}，斜率 {_fmt(breadth.uptrend_slope, 2)}）",
                f"- RSP/SPY：{_fmt(breadth.rsp_spy, 4)}",
                f"- 进攻减防御：{_fmt(breadth.off_def_spread, 1)} 个百分点",
                "- 组件：" + "，".join(f"{key}={_fmt(value, 1)}" for key, value in breadth.components.items()),
            ]
        )
    if anchor is not None:
        lines.extend(["", "## 锚点顶背离", ""])
        lines.append(f"- BEARISH_BREADTH_DIVERGENCE：`{anchor.divergence.flag}`")
        if anchor.divergence.reasons:
            lines.append("- 条件：" + "，".join(anchor.divergence.reasons))
    lines.extend(
        [
            "",
            "## 分标的指令",
            "",
            "| 标的 | 基准 | 基准收盘 | SMA50 | SMA200 | Regime | 敞口 | 波动率状态 | 顶背离 | 杠杆指令 |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for row in diagnosis.tickers:
        lines.append(
            "| {ticker} | {benchmark} | {close} | {sma50} | {sma200} | {regime} | {exposure} | {state} | {div} | {action} |".format(
                ticker=row.ticker,
                benchmark=row.trend.benchmark,
                close=_fmt(row.trend.close),
                sma50=_fmt(row.trend.sma50),
                sma200=_fmt(row.trend.sma200),
                regime=f"{row.regime.regime_id} {row.regime.name_zh}",
                exposure=row.regime.exposure_range,
                state=row.display_state or "—",
                div="是" if row.divergence.flag else "否",
                action=row.regime.leverage_action.replace("|", "/"),
            )
        )
    if diagnosis.warnings:
        lines.extend(["", "## 数据警告", ""])
        lines.extend(f"- {warning}" for warning in diagnosis.warnings)
    lines.extend(
        [
            "",
            "## 约束",
            "",
            "- Score_vol 只来自 VIX 期限结构、近端基差、SKEW 与 VVIX 的滚动统计。",
            "- 价格趋势只作为执行门控：BUY DIP 在基准跌破 SMA200 时显示为 NO TRADE，分数不变。",
            "- Z-Score、百分位和均线都使用截至当日的滚动窗口，没有全样本拟合，也没有向后填充。",
            "",
            "规则化监测结果，不是投资建议。",
            "",
        ]
    )
    return "\n".join(lines)


def _html(markdown_text: str, asof: str) -> str:
    rows = []
    for line in markdown_text.splitlines():
        escaped = (
            line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        rows.append(f"<p>{escaped or '&nbsp;'}</p>")
    body = "\n".join(rows)
    return (
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>市场状态 {asof}</title>"
        "<style>body{font-family:Segoe UI,sans-serif;max-width:980px;margin:32px auto;line-height:1.45}"
        "p{margin:4px 0}</style></head><body>"
        f"{body}</body></html>"
    )


def export_reports(diagnosis: MarketDiagnosis, reports_dir: Path | None = None) -> tuple[Path, Path]:
    folder = reports_dir or REPORTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    asof = diagnosis.asof.date().isoformat() if diagnosis.asof is not None else date.today().isoformat()
    markdown_text = _markdown(diagnosis)
    markdown_path = folder / f"regime_report_{asof}.md"
    html_path = folder / f"regime_report_{asof}.html"
    markdown_path.write_text(markdown_text, encoding="utf-8")
    html_path.write_text(_html(markdown_text, asof), encoding="utf-8")
    return markdown_path, html_path
