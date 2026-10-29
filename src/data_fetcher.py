"""TraderMonty CSV 与 yfinance 行情拉取、缓存、新鲜度与降级。"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from src.breadth_nowcast import nowcast_breadth
from src.config import (
    ALL_PRICE_TICKERS,
    BREADTH_BACKUP_URL,
    BREADTH_PRIMARY_URL,
    CACHE_DIR,
    DATA_STALE_WARNING,
    FFILL_LIMIT,
    READINESS_TICKERS,
    HISTORY_CALENDAR_DAYS,
    HTTP_RETRIES,
    HTTP_TIMEOUT_SECONDS,
    SECTOR_ETF_NAMES,
    SECTOR_ETFS,
    SECTOR_URL,
    STALE_CALENDAR_DAYS,
    UPTREND_URL,
    USER_AGENT,
)

HEADERS = {"User-Agent": USER_AGENT}

_BREADTH_PERCENT_COLUMNS = (
    "Breadth_Index_Raw",
    "Breadth_Index_200MA",
    "Breadth_Index_8MA",
)
_BOOL_COLUMNS = (
    "Bearish_Signal",
    "Is_Peak",
    "Is_Trough",
    "Is_Trough_8MA_Below_04",
)


@dataclass
class PriceBook:
    close: pd.DataFrame
    high: pd.DataFrame = field(default_factory=pd.DataFrame)
    low: pd.DataFrame = field(default_factory=pd.DataFrame)
    open: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class MarketDataBundle:
    prices: pd.DataFrame
    breadth: pd.DataFrame | None
    uptrend: pd.DataFrame | None
    sectors: pd.DataFrame | None
    breadth_source: str
    warnings: list[str] = field(default_factory=list)
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    open: pd.DataFrame | None = None


def sanitize_prices(close: pd.DataFrame) -> pd.DataFrame:
    """对齐交易日。只允许向前填充最多 2 日，禁止 bfill。"""
    out = close.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.apply(pd.to_numeric, errors="coerce")
    clock = [ticker for ticker in READINESS_TICKERS if ticker in out.columns]
    if clock:
        has_cash_close = out[clock].notna().any(axis=1)
        if bool(has_cash_close.any()):
            # 美元等近 24 小时品种不能把日历推到美股收盘之后。
            out = out.loc[: has_cash_close[has_cash_close].index[-1]]
    while len(out) and bool(out.iloc[-1].isna().all()):
        out = out.iloc[:-1]
    out = out.ffill(limit=FFILL_LIMIT)
    return out


def _request_bytes(url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(HTTP_RETRIES):
        try:
            response = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(1.2 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _read_csv_bytes(payload: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload))
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().lower(): column for column in frame.columns}


def _rename_first(frame: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    lookup = _column_lookup(frame)
    rename: dict[str, str] = {}
    for canonical, options in aliases.items():
        for option in options:
            actual = lookup.get(option.lower())
            if actual is not None and actual not in rename:
                rename[actual] = canonical
                break
    return frame.rename(columns=rename)


def _as_bool_series(series: pd.Series) -> pd.Series:
    def _one(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if pd.isna(value):
            return False
        return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}

    return series.map(_one)


def _scale_family_to_percent(frame: pd.DataFrame, columns: Iterable[str], anchor: str) -> pd.DataFrame:
    """若锚列最大值 ≤ 1，则把同族比例列从 [0, 1] 转为 [0, 100]。"""
    out = frame.copy()
    if anchor not in out.columns:
        return out
    anchor_values = pd.to_numeric(out[anchor], errors="coerce")
    finite = anchor_values[np.isfinite(anchor_values)]
    if len(finite) == 0 or float(finite.max()) > 1.0:
        return out
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce") * 100.0
    return out


def normalize_breadth_frame(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = _rename_first(
        frame,
        {
            "Date": ["date"],
            "Breadth_Index_Raw": ["breadth_index_raw"],
            "Breadth_Index_200MA": ["breadth_index_200ma"],
            "Breadth_Index_8MA": ["breadth_index_8ma"],
            "Breadth_200MA_Trend": ["breadth_200ma_trend"],
            "Bearish_Signal": ["bearish_signal"],
            "Is_Peak": ["is_peak"],
            "Is_Trough": ["is_trough"],
            "Is_Trough_8MA_Below_04": ["is_trough_8ma_below_04"],
        },
    )
    renamed = _scale_family_to_percent(renamed, _BREADTH_PERCENT_COLUMNS, "Breadth_Index_200MA")
    if "Date" not in renamed.columns:
        raise ValueError("市场宽度 CSV 缺少 Date 列")
    renamed["Date"] = pd.to_datetime(renamed["Date"]).dt.tz_localize(None).dt.normalize()
    for column in _BOOL_COLUMNS:
        if column in renamed.columns:
            renamed[column] = _as_bool_series(renamed[column])
    renamed = renamed.dropna(subset=["Date"]).sort_values("Date")
    renamed = renamed.drop_duplicates("Date", keep="last").set_index("Date")
    return renamed


def _derive_color(trend: object, slope: float, ratio: float) -> str:
    text = str(trend).strip().upper()
    if text in {"UP", "GREEN"}:
        return "GREEN"
    if text in {"DOWN", "RED"}:
        return "RED"
    if np.isfinite(slope) and slope > 0:
        return "GREEN"
    if np.isfinite(slope) and slope < 0:
        return "RED"
    return "GREEN" if ratio >= 25 else "RED"


def normalize_uptrend_frame(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = _rename_first(
        frame,
        {
            "worksheet": ["worksheet"],
            "date": ["date"],
            "ratio": ["ratio"],
            "ma10": ["ma10", "ma_10", "10ma"],
            "slope": ["slope"],
            "trend": ["trend"],
            "color": ["color"],
        },
    )
    if "worksheet" in renamed.columns:
        mask = renamed["worksheet"].astype(str).str.strip().str.lower().eq("all")
        renamed = renamed.loc[mask].copy()
    if renamed.empty or "date" not in renamed.columns or "ratio" not in renamed.columns:
        raise ValueError("上升趋势占比 CSV 缺少全市场 all 序列")
    renamed = _scale_family_to_percent(renamed, ["ratio", "ma10", "slope"], "ratio")
    renamed["date"] = pd.to_datetime(renamed["date"]).dt.tz_localize(None).dt.normalize()
    renamed["ratio"] = pd.to_numeric(renamed["ratio"], errors="coerce")
    if "ma10" in renamed.columns:
        renamed["ma10"] = pd.to_numeric(renamed["ma10"], errors="coerce")
    else:
        renamed["ma10"] = np.nan
    if "slope" in renamed.columns:
        renamed["slope"] = pd.to_numeric(renamed["slope"], errors="coerce")
    else:
        renamed["slope"] = np.nan
    if "trend" not in renamed.columns:
        renamed["trend"] = np.where(renamed["slope"] > 0, "up", "down")
    if "color" not in renamed.columns:
        renamed["color"] = [
            _derive_color(trend, slope, ratio)
            for trend, slope, ratio in zip(renamed["trend"], renamed["slope"], renamed["ratio"])
        ]
    else:
        renamed["color"] = renamed["color"].astype(str).str.strip().str.upper()
    renamed = renamed.dropna(subset=["date", "ratio"]).sort_values("date")
    renamed = renamed.drop_duplicates("date", keep="last").set_index("date")
    return renamed[["ratio", "ma10", "slope", "trend", "color"]]


def normalize_sector_frame(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = _rename_first(
        frame,
        {
            "Sector": ["sector", "name"],
            "Ratio": ["ratio"],
            "10MA": ["10ma", "ma10", "ma_10"],
            "Slope": ["slope"],
            "Trend": ["trend"],
            "Status": ["status"],
        },
    )
    if "Sector" not in renamed.columns or "Ratio" not in renamed.columns:
        raise ValueError("行业宽度 CSV 缺少 Sector 或 Ratio 列")
    renamed = _scale_family_to_percent(renamed, ["Ratio", "10MA"], "Ratio")
    renamed["Sector"] = renamed["Sector"].astype(str).str.strip()
    renamed["Ratio"] = pd.to_numeric(renamed["Ratio"], errors="coerce")
    return renamed.dropna(subset=["Sector", "Ratio"]).reset_index(drop=True)


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def _write_frame(frame: pd.DataFrame, name: str) -> None:
    path = _cache_path(name)
    frame.to_csv(path, index=True)


def _read_frame(name: str) -> pd.DataFrame | None:
    path = CACHE_DIR / name
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
    return frame.sort_index()


def _read_sectors(name: str) -> pd.DataFrame | None:
    path = CACHE_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def fetch_tradermonty_csvs() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, list[str], str]:
    """拉取三份公开 CSV。网络失败时读本地缓存；缓存也没有则返回空并交给降级计算器。"""
    warnings: list[str] = []
    breadth: pd.DataFrame | None = None
    uptrend: pd.DataFrame | None = None
    sectors: pd.DataFrame | None = None
    source = "tradermonty"

    try:
        try:
            payload = _request_bytes(BREADTH_PRIMARY_URL)
        except requests.RequestException:
            payload = _request_bytes(BREADTH_BACKUP_URL)
        breadth = normalize_breadth_frame(_read_csv_bytes(payload))
        _write_frame(breadth, "breadth_normalized.csv")
    except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
        breadth = _read_frame("breadth_normalized.csv")
        warnings.append(f"市场宽度 CSV 不可达，已尝试本地缓存：{exc}")

    try:
        uptrend = normalize_uptrend_frame(_read_csv_bytes(_request_bytes(UPTREND_URL)))
        _write_frame(uptrend, "uptrend_normalized.csv")
    except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
        uptrend = _read_frame("uptrend_normalized.csv")
        warnings.append(f"上升趋势 CSV 不可达，已尝试本地缓存：{exc}")

    try:
        sectors = normalize_sector_frame(_read_csv_bytes(_request_bytes(SECTOR_URL)))
        sectors.to_csv(_cache_path("sectors_normalized.csv"), index=False)
    except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
        sectors = _read_sectors("sectors_normalized.csv")
        warnings.append(f"行业宽度 CSV 不可达，已尝试本地缓存：{exc}")

    if breadth is None or uptrend is None or sectors is None:
        source = "unavailable"
    return breadth, uptrend, sectors, warnings, source


def _stamp_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(frame)
    out.columns = [str(column) for column in out.columns]
    index = pd.DatetimeIndex(out.index)
    if index.tz is None:
        index = index.tz_localize("America/New_York")
    else:
        index = index.tz_convert("America/New_York")
    out.index = index.tz_localize(None).normalize()
    return out


def _extract_ohlc(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if raw is None or raw.empty:
        return {}
    extracted: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = {str(value).lower(): value for value in raw.columns.get_level_values(0)}
        level1 = {str(value).lower(): value for value in raw.columns.get_level_values(1)}
        for field in ("Open", "High", "Low", "Close"):
            if field.lower() in level0:
                extracted[field] = _stamp_index(raw.xs(level0[field.lower()], axis=1, level=0))
            elif field.lower() in level1:
                extracted[field] = _stamp_index(raw.xs(level1[field.lower()], axis=1, level=1))
    elif "Close" in raw.columns:
        extracted["Close"] = _stamp_index(raw[["Close"]])
    if "Close" not in extracted:
        raise ValueError("yfinance 返回结果中没有 Close 列")
    return extracted


def _download_book(tickers: list[str], start: pd.Timestamp) -> PriceBook:
    import yfinance as yf

    raw = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    fields = _extract_ohlc(raw)
    close = sanitize_prices(fields["Close"])
    if close.empty:
        raise RuntimeError("yfinance 未返回任何收盘价")
    last = close.index[-1]

    def _side(name: str) -> pd.DataFrame:
        if name not in fields:
            return pd.DataFrame()
        side = sanitize_prices(fields[name])
        return side.loc[:last]

    return PriceBook(close=close, high=_side("High"), low=_side("Low"), open=_side("Open"))


def fetch_price_book(
    tickers: list[str] | None = None,
    start: pd.Timestamp | None = None,
    save: bool = True,
) -> PriceBook:
    """下载开高低收。失败时回退到本地收盘缓存。"""
    universe = list(tickers or ALL_PRICE_TICKERS)
    start = start or (pd.Timestamp.today().normalize() - pd.Timedelta(days=HISTORY_CALENDAR_DAYS))
    try:
        book = _download_book(universe, start)
        if save:
            _write_frame(book.close, "prices.csv")
            if not book.high.empty:
                _write_frame(book.high, "prices_high.csv")
            if not book.low.empty:
                _write_frame(book.low, "prices_low.csv")
            if not book.open.empty:
                _write_frame(book.open, "prices_open.csv")
        return book
    except Exception as exc:
        cached = _read_frame("prices.csv")
        if cached is None or cached.empty:
            raise RuntimeError(f"行情下载失败且无本地缓存：{exc}") from exc
        cached.attrs["cache_warning"] = f"行情下载失败，已使用本地缓存：{exc}"
        return PriceBook(
            close=cached,
            high=_read_frame("prices_high.csv") or pd.DataFrame(),
            low=_read_frame("prices_low.csv") or pd.DataFrame(),
            open=_read_frame("prices_open.csv") or pd.DataFrame(),
        )


def fetch_market_timeseries(tickers: list[str] | None = None) -> pd.DataFrame:
    """兼容旧调用：只返回收盘价。"""
    return fetch_price_book(tickers).close


def assess_freshness(source_last: pd.Timestamp, market_last: pd.Timestamp) -> str | None:
    delta_days = int((pd.Timestamp(market_last).normalize() - pd.Timestamp(source_last).normalize()).days)
    if delta_days > STALE_CALENDAR_DAYS:
        return DATA_STALE_WARNING
    return None


def build_fallback_datasets(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """用 RSP/SPY 与 11 个行业 ETF 的 50/200 日均线合成替代宽度。"""
    available = [ticker for ticker in SECTOR_ETFS if ticker in prices.columns]
    if not available:
        raise RuntimeError("无法构建宽度降级指标：缺少行业 ETF 行情")

    above_200 = []
    above_both = []
    latest_rows = []
    for ticker in available:
        close = pd.to_numeric(prices[ticker], errors="coerce")
        sma50 = close.rolling(50, min_periods=50).mean()
        sma200 = close.rolling(200, min_periods=200).mean()
        above_200.append(close > sma200)
        above_both.append((close > sma50) & (close > sma200))
        last_close = close.iloc[-1]
        last_sma50 = sma50.iloc[-1]
        last_sma200 = sma200.iloc[-1]
        ratio = 0.0
        if np.isfinite(last_close) and np.isfinite(last_sma50) and np.isfinite(last_sma200):
            ratio = 100.0 * ((last_close > last_sma50) + (last_close > last_sma200)) / 2.0
        latest_rows.append({"Sector": SECTOR_ETF_NAMES[ticker], "Ratio": ratio})

    raw = 100.0 * pd.concat(above_200, axis=1).mean(axis=1)
    participation = 100.0 * pd.concat(above_both, axis=1).mean(axis=1)
    breadth = pd.DataFrame(index=prices.index)
    breadth["Breadth_Index_Raw"] = raw
    breadth["Breadth_Index_8MA"] = raw.rolling(8, min_periods=8).mean()
    breadth["Breadth_Index_200MA"] = raw.rolling(200, min_periods=200).mean()
    trend = np.sign(breadth["Breadth_Index_200MA"].diff()).replace(0, np.nan).ffill(limit=FFILL_LIMIT)
    breadth["Breadth_200MA_Trend"] = trend.fillna(0.0)
    breadth["Bearish_Signal"] = breadth["Breadth_Index_8MA"] < breadth["Breadth_Index_200MA"]
    breadth["Is_Peak"] = False
    breadth["Is_Trough"] = False
    breadth["Is_Trough_8MA_Below_04"] = breadth["Breadth_Index_8MA"] < 40.0

    uptrend = pd.DataFrame(index=prices.index)
    uptrend["ratio"] = participation
    uptrend["ma10"] = participation.rolling(10, min_periods=10).mean()
    uptrend["slope"] = participation.diff(5)
    uptrend["trend"] = np.where(uptrend["slope"] > 0, "up", "down")
    uptrend["color"] = np.where(uptrend["slope"] > 0, "GREEN", "RED")

    sectors = pd.DataFrame(latest_rows)
    return breadth, uptrend, sectors


def load_market_data() -> MarketDataBundle:
    warnings: list[str] = []
    book = fetch_price_book()
    prices = book.close
    cache_warning = prices.attrs.get("cache_warning")
    if cache_warning:
        warnings.append(str(cache_warning))

    breadth, uptrend, sectors, csv_warnings, source = fetch_tradermonty_csvs()
    warnings.extend(csv_warnings)

    if breadth is None or uptrend is None or sectors is None:
        breadth, uptrend, sectors = build_fallback_datasets(prices)
        source = "yfinance-fallback"
        warnings.append("远程宽度 CSV 不完整，已切换到 RSP/SPY 与行业 ETF 均线降级指标。")

    market_last = pd.Timestamp(prices.index[-1])
    breadth_last = pd.Timestamp(breadth.dropna(how="all").index[-1])
    stale = assess_freshness(breadth_last, market_last)
    if stale:
        warnings.append(
            f"{DATA_STALE_WARNING}: 宽度数据最新日期 {breadth_last.date()} 落后行情 {market_last.date()} 超过 {STALE_CALENDAR_DAYS} 个自然日。"
        )
    if breadth_last < market_last and source == "tradermonty":
        breadth, uptrend, note = nowcast_breadth(prices, breadth, uptrend)
        if note:
            source = "tradermonty+nowcast"
            warnings.append(note)

    return MarketDataBundle(
        prices=prices,
        breadth=breadth,
        uptrend=uptrend,
        sectors=sectors,
        breadth_source=source,
        warnings=warnings,
        high=book.high,
        low=book.low,
        open=book.open,
    )
