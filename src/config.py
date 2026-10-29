"""集中管理数据契约、阈值与标的路由表。"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data_cache"
REPORTS_DIR = PROJECT_ROOT / "reports"

USER_AGENT = "us-market-regime-pro/2.0 (research; market-regime-monitor)"
HTTP_TIMEOUT_SECONDS = 25
HTTP_RETRIES = 3
HISTORY_CALENDAR_DAYS = 365 * 4 + 30
FFILL_LIMIT = 2
STALE_CALENDAR_DAYS = 5

Z_WINDOW = 63
PERCENTILE_WINDOW = 252
READINESS_MIN_BARS = 252
HIGH_WINDOW = 252
SMA_FAST = 20
SMA_MID = 50
SMA_SLOW = 200
ATR_WINDOW = 14
ATR_BAND = 0.25
HV_WINDOW = 20
HV_LOW = 0.22
HV_HIGH = 0.30
MACRO_Z_THRESHOLD = 1.5
MACRO_CHANGE_WINDOW = 20
COLLINEARITY_WINDOW = 63
COLLINEARITY_THRESHOLD = 0.85
SIGMOID_SLOPE = 1.25
LEDGER_PATH = CACHE_DIR / "regime_ledger.db"

# 严格就绪门控：当日缺失或历史不足时必须返回 DATA N/A，禁止记 0 分。
READINESS_TICKERS = ("^VIX", "^VIX3M", "^GSPC", "^NDX")
# 五项结构因子都依赖这些序列；缺失时同样拒绝打分，避免把缺失伪装成 0。
FACTOR_TICKERS = ("^VIX9D", "^VVIX", "^SKEW")

BREADTH_PRIMARY_URL = "https://tradermonty.github.io/market-breadth-analysis/market_breadth_data.csv"
BREADTH_BACKUP_URL = "https://raw.githubusercontent.com/tradermonty/market-breadth-analysis/main/market_breadth_data.csv"
UPTREND_URL = "https://raw.githubusercontent.com/tradermonty/uptrend-dashboard/main/data/uptrend_ratio_timeseries.csv"
SECTOR_URL = "https://raw.githubusercontent.com/tradermonty/uptrend-dashboard/main/data/sector_summary.csv"

VOL_TICKERS = ["^VIX9D", "^VIX", "^VIX3M", "^VIX6M", "^VVIX", "^SKEW"]
BENCHMARK_TICKERS = ["^GSPC", "^NDX", "^RUT", "SMH"]
TRADABLE_TICKERS = ["SPY", "QQQ", "IWM", "RSP"]
MACRO_TICKERS = ["HYG", "IEF", "^TNX", "DX-Y.NYB"]
SECTOR_ETFS = ["XLK", "XLY", "XLI", "XLF", "XLC", "XLU", "XLP", "XLV", "XLE", "XLB", "XLRE"]

ALL_PRICE_TICKERS = (
    VOL_TICKERS + BENCHMARK_TICKERS + TRADABLE_TICKERS + MACRO_TICKERS + SECTOR_ETFS
)

# 杠杆 ETF 不直接计算 200 日均线，路由到无损耗底层。
NDX_UNDERLYINGS = frozenset(
    {"QQQ", "QQQM", "TQQQ", "SQQQ", "QLD", "QID", "NDX", "NQ=F", "MNQ=F"}
)
RUT_UNDERLYINGS = frozenset({"IWM", "TNA", "TZA", "UWM", "TWM", "RUT", "RTY=F"})
SMH_UNDERLYINGS = frozenset({"SOXL", "SOXS", "SMH", "SOXX"})

LEVERAGED_LONG_TICKERS = frozenset(
    {"TQQQ", "QLD", "UPRO", "SSO", "SOXL", "TNA", "UWM", "TECL", "FAS", "SPXL"}
)

SECTOR_ALIASES = {
    "technology": "Technology",
    "consumer discretionary": "Consumer Discretionary",
    "consumer cyclical": "Consumer Discretionary",
    "industrials": "Industrials",
    "financials": "Financials",
    "financial": "Financials",
    "communication services": "Communication Services",
    "utilities": "Utilities",
    "consumer staples": "Consumer Staples",
    "consumer defensive": "Consumer Staples",
    "health care": "Health Care",
    "healthcare": "Health Care",
    "energy": "Energy",
    "basic materials": "Basic Materials",
    "materials": "Basic Materials",
    "real estate": "Real Estate",
}

OFFENSIVE_SECTORS = frozenset(
    {
        "Technology",
        "Consumer Discretionary",
        "Industrials",
        "Financials",
        "Communication Services",
    }
)
DEFENSIVE_SECTORS = frozenset({"Utilities", "Consumer Staples", "Health Care"})

SECTOR_ETF_NAMES = {
    "XLK": "Technology",
    "XLY": "Consumer Discretionary",
    "XLI": "Industrials",
    "XLF": "Financials",
    "XLC": "Communication Services",
    "XLU": "Utilities",
    "XLP": "Consumer Staples",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLB": "Basic Materials",
    "XLRE": "Real Estate",
}

REGIME_1 = "REGIME_1_STRONG_BULL"
REGIME_1B = "REGIME_1B_BULL_WITH_MACRO_HEADWIND"
REGIME_2 = "REGIME_2_NARROW_FRAGILE_BULL"
REGIME_3 = "REGIME_3_CORRECTION_HIGH_VOL"
REGIME_4 = "REGIME_4_SYSTEMIC_BEAR"
REGIME_5 = "REGIME_5_CAPITULATION_REVERSAL"

LEVERAGE_LADDER = {
    "^NDX": {3: "TQQQ", 2: "QLD", 1: "QQQ"},
    "^GSPC": {3: "UPRO", 2: "SSO", 1: "SPY"},
    "^RUT": {3: "TNA", 2: "UWM", 1: "IWM"},
    "SMH": {3: "SOXL", 2: "USD", 1: "SMH"},
}
DATA_NA = "DATA N/A"
DATA_STALE_WARNING = "DATA_STALE_WARNING"
