# 美股趋势与市场状态双引擎监测

零付费 API、严格滚动窗口的美股状态判定系统。期权期限结构、全市场宽度和底层指数均线分开计算：波动率分数不被价格趋势改写，趋势只决定能不能执行。

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python main.py --tickers SPY QQQ TQQQ SOXL --export-md
```

终端打印当前 Regime、VIX 期限结构分、宽度分、顶背离和每个标的的路由指令。加上 `--export-md` 后，报告写到 `reports/regime_report_YYYY-MM-DD.md`。

## 测试

```bash
pytest -v
```

覆盖三件事：趋势不改变 `Score_vol`，且熊市里的 `BUY DIP` 变成 `NO TRADE`；篡改未来行情不影响当日分数；指数贴着 252 日高点但宽度走弱时触发假突破，并归入 `REGIME_2_NARROW_FRAGILE_BULL`。

## 数据

宽度来自 TraderMonty 公开 CSV。行情来自 Yahoo Finance，本地缓存在 `data_cache/`。远程 CSV 不可达时，用 `RSP/SPY` 和 11 个 SPDR 行业 ETF 的均线占比做降级，程序不会因为宽度源中断而退出。

杠杆 ETF 会路由到底层指数：`TQQQ` → `^NDX`，`SOXL` → `SMH`，其余宽基默认 `^GSPC`。

## 约束

Z-Score、百分位和均线只用截至当日的窗口。缺失的必选序列返回 `DATA N/A`，不会记成 0 分。价格只做向前填充，且最多 2 日。

这是规则化监测，不是投资建议。

## V2

50 日均线使用 `±0.25 × ATR14` 迟滞带，缓冲带内不切换状态。仓位按偏离度连续计算。`HYG/IEF` 转弱且上升趋势占比低于 30%，或美债、美元 20 日变化的 252 日 Z-Score 高于 1.5 时，强牛降为带逆风牛市，权益上限 75%，杠杆不超过 2 倍。3 倍杠杆还要看基准的 20 日实现波动率：低于 22% 才允许 3 倍，22%–30% 降为 2 倍，高于 30% 只留 1 倍。

宽度 CSV 落后于最新收盘时，用 11 个行业 ETF 和 `RSP/SPY` 外推当日宽度。每次运行写入 `data_cache/regime_ledger.db`。历史对照用：

```bash
python main.py --tickers SPY QQQ TQQQ SOXL --export-md --backtest
```
