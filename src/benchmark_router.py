"""杠杆 ETF / 宽基标的到底层无损耗指数的路由。"""

from __future__ import annotations

from src.config import NDX_UNDERLYINGS, RUT_UNDERLYINGS, SMH_UNDERLYINGS


def _normalize_symbol(ticker: str) -> str:
    raw = ticker.strip().upper()
    if raw.startswith("^"):
        return raw[1:]
    return raw


def route_benchmark(ticker: str) -> str:
    """返回趋势与胜率评估必须使用的底层基准。

    QQQ 一族 → ^NDX；IWM 一族 → ^RUT；半导体杠杆 → SMH；其余 → ^GSPC。
    """
    key = _normalize_symbol(ticker)
    if key in NDX_UNDERLYINGS:
        return "^NDX"
    if key in RUT_UNDERLYINGS:
        return "^RUT"
    if key in SMH_UNDERLYINGS:
        return "SMH"
    return "^GSPC"
