"""命令行入口：拉取数据、判定市场状态并输出看板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data_fetcher import load_market_data
from src.ledger import backfill_outcomes, ledger_summary, record_diagnosis
from src.regime_synthesizer import diagnose
from src.report_generator import export_reports, render_console


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="美股趋势与市场状态双引擎监测")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["SPY", "QQQ", "TQQQ", "SOXL"],
        help="需要路由到底层基准的标的，例如 SPY QQQ TQQQ SOXL",
    )
    parser.add_argument("--export-md", action="store_true", help="把复盘写入 reports/regime_report_YYYY-MM-DD.md")
    parser.add_argument("--reports-dir", type=Path, default=None, help="报告输出目录，默认 reports/")
    parser.add_argument("--backtest", action="store_true", help="跑 2018 年以来的执行回放、敏感性网格和尾部压力")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bundle = load_market_data()
    except RuntimeError as exc:
        print(f"行情不可用：{exc}", file=sys.stderr)
        return 1
    diagnosis = diagnose(
        tickers=args.tickers,
        prices=bundle.prices,
        breadth=bundle.breadth,
        uptrend=bundle.uptrend,
        sectors=bundle.sectors,
        warnings=bundle.warnings,
        breadth_source=bundle.breadth_source,
        high=bundle.high,
        low=bundle.low,
    )
    render_console(diagnosis)
    record_diagnosis(diagnosis)
    backfill_outcomes(bundle.prices)
    print(ledger_summary())
    if args.export_md:
        markdown_path, html_path = export_reports(diagnosis, args.reports_dir)
        print(f"Markdown: {markdown_path}")
        print(f"HTML: {html_path}")
    if args.backtest:
        from verification.backtest_verifier import run_verification

        report_path = run_verification(args.reports_dir)
        print(f"Verification: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
