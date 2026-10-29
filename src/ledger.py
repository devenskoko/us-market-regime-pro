"""本地影子账本：记录每次诊断，并在后续行情里回填 T+1 / T+5 / T+20。"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import LEDGER_PATH
from src.regime_synthesizer import MarketDiagnosis

HORIZONS = (1, 5, 20)


def _connect(path=LEDGER_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
            asof TEXT NOT NULL,
            ticker TEXT NOT NULL,
            benchmark TEXT,
            regime_id TEXT,
            score_vol REAL,
            score_breadth REAL,
            r_ts REAL,
            z_ts REAL,
            hv20 REAL,
            atr14 REAL,
            exposure REAL,
            leverage_cap REAL,
            hysteresis_state TEXT,
            macro_headwind INTEGER,
            close REAL,
            run_at TEXT,
            PRIMARY KEY (asof, ticker)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS outcomes (
            asof TEXT NOT NULL,
            ticker TEXT NOT NULL,
            horizon INTEGER NOT NULL,
            fwd_return REAL,
            mae REAL,
            mfe REAL,
            PRIMARY KEY (asof, ticker, horizon)
        )
        """
    )
    return connection


def record_diagnosis(diagnosis: MarketDiagnosis, path=LEDGER_PATH) -> int:
    if diagnosis.asof is None:
        return 0
    asof = diagnosis.asof.date().isoformat()
    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for item in diagnosis.tickers:
        trend = item.trend
        regime = item.regime
        rows.append(
            (
                asof,
                item.ticker,
                trend.benchmark,
                regime.regime_id,
                diagnosis.vol.score,
                diagnosis.breadth.score,
                diagnosis.vol.r_ts,
                diagnosis.vol.z_ts,
                trend.hv20,
                trend.atr14,
                None if regime.exposure_point is None else float(regime.exposure_point) / 100.0,
                regime.leverage_cap,
                trend.hysteresis_state,
                int(regime.regime_id == "REGIME_1B_BULL_WITH_MACRO_HEADWIND"),
                trend.close,
                run_at,
            )
        )
    with _connect(path) as connection:
        connection.executemany(
            """
            INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(asof, ticker) DO UPDATE SET
                regime_id=excluded.regime_id,
                score_vol=excluded.score_vol,
                score_breadth=excluded.score_breadth,
                r_ts=excluded.r_ts,
                z_ts=excluded.z_ts,
                hv20=excluded.hv20,
                atr14=excluded.atr14,
                exposure=excluded.exposure,
                leverage_cap=excluded.leverage_cap,
                hysteresis_state=excluded.hysteresis_state,
                macro_headwind=excluded.macro_headwind,
                close=excluded.close,
                run_at=excluded.run_at
            """,
            rows,
        )
    return len(rows)


def backfill_outcomes(prices: pd.DataFrame, path=LEDGER_PATH) -> int:
    """用基准收盘价回填信号之后的收益、最大不利和最大有利偏移。"""
    if prices.empty:
        return 0
    filled = 0
    with _connect(path) as connection:
        snapshots = connection.execute(
            "SELECT asof, ticker, benchmark, close FROM snapshots"
        ).fetchall()
        for asof, _ticker, benchmark, entry in snapshots:
            if benchmark not in prices.columns or entry is None:
                continue
            series = pd.to_numeric(prices[benchmark], errors="coerce").dropna()
            start = pd.Timestamp(asof)
            future = series.loc[series.index > start]
            for horizon in HORIZONS:
                if len(future) < horizon:
                    continue
                window = future.iloc[:horizon]
                fwd = float(window.iloc[-1] / entry - 1.0)
                path_ret = window.to_numpy(dtype=float) / float(entry) - 1.0
                mae = float(np.nanmin(path_ret))
                mfe = float(np.nanmax(path_ret))
                connection.execute(
                    """
                    INSERT INTO outcomes VALUES (?,?,?,?,?,?)
                    ON CONFLICT(asof, ticker, horizon) DO UPDATE SET
                        fwd_return=excluded.fwd_return,
                        mae=excluded.mae,
                        mfe=excluded.mfe
                    """,
                    (asof, _ticker, horizon, fwd, mae, mfe),
                )
                filled += 1
    return filled


def ledger_summary(path=LEDGER_PATH) -> str:
    if not path.exists():
        return "影子账本还没有记录。"
    with _connect(path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
        matured = connection.execute(
            "SELECT COUNT(*), AVG(fwd_return) FROM outcomes WHERE horizon = 20"
        ).fetchone()
    matured_n = int(matured[0] or 0)
    if matured_n == 0:
        return f"影子账本已记录 {count} 条快照，T+20 尚未成熟。"
    wins = 0
    with _connect(path) as connection:
        wins = connection.execute(
            "SELECT SUM(CASE WHEN fwd_return > 0 THEN 1 ELSE 0 END) FROM outcomes WHERE horizon = 20"
        ).fetchone()[0]
    hit = 100.0 * float(wins or 0) / matured_n
    avg = float(matured[1] or 0.0)
    return f"影子账本 {count} 条快照；T+20 已成熟 {matured_n} 条，上涨占比 {hit:.1f}%，平均收益 {avg:.2%}。"
