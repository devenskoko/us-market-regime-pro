# 美股趋势与市场状态（Market Regime）双引擎量化监测系统 — 从 0 到 1 AI 开发规范文档

> **给 AI 编程助手（Cursor / Claude Code / Windsurf / LLM Agent）的系统指令：**
> 你现在是一位资深华尔街量化系统架构师与全栈研发工程师。请严格遵循本规范文档（PRD & Technical Specification），从 0 到 1 实现一套**零付费 API 依赖、严格防未来函数（Zero Look-Ahead Bias）、因子正交解耦**的美股趋势与市场状态自动判定系统。
> 
> **执行要求：**
> 1. 先阅读全文，理解双引擎（期权波动率期限结构引擎 + 全市场宽度背离引擎）与正交趋势门控的数学逻辑及工程约束。
> 2. 严格按照第 6 节的目录结构创建项目文件，并按第 7 节的阶段顺序逐步实现、编写单元测试并验证运行。
> 3. 严禁使用任何需要全样本拟合的黑盒模型（如全历史 `fit` 的 HMM），所有统计量（如 $Z$-Score、分位数）必须采用严格的历史滚动窗口（Rolling Window）。

---

## 1. 项目背景与核心设计哲学

单一技术指标无法在美股市场同时做到“高灵敏度”与“低误报率”。本系统融合开源界实盘准确率最高的两大核心架构，构建**“期权微观结构防崩盘 + 市场宽度防假突破 + 均线信贷定牛熊”**的三维正交体系：

### 1.1 引擎 A：期权波动率期限结构引擎（源自 `aaajiao/VIX-Term-Structure-Pro v7.14`）
* **解决痛点（克服均线滞后性）：** 传统的 200 日简单移动平均线（$200\text{D SMA}$）具有严重的滞后性，往往等指数下跌 $10\% \sim 15\%$ 才触发死叉；若持有三倍杠杆 ETF（如 `TQQQ`、`UPRO`、`SOXL`），此时净值回撤已达 $30\% \sim 45\%$。
* **核心机理（期权微观结构先行）：** 美股历史上每一次系统性危机或流动性闪崩前，机构交易台都会在期权市场不计成本地抢购近月保护（Put 保护与近月波动率期货），导致短期隐含波动率（`^VIX9D`、`^VIX` 或近月期货 `VX1`）飙升速度远超中期隐含波动率（`^VIX3M` 或次月期货 `VX2`），引发**期限结构倒挂（Backwardation）**与 $Z$-Score 极值突破。
* **正交解耦铁律（Orthogonal Rule）：** **波动率结构得分（$\text{Score}_{\text{vol}}$）是纯粹的衍生品微观结构指标；价格趋势（Trend）绝不改变 $\text{Score}_{\text{vol}}$ 的数值，仅作为最终交易信号的展示与执行门控（Execution Gate）。**

### 1.2 引擎 B：全市场宽度与背离监控引擎（源自 `tradermonty/uptrend-dashboard` & `market-breadth-analysis`）
* **解决痛点（戳破巨头掩护下的脆弱假牛市）：** 美股大盘（`SPY`、`QQQ`）为市值加权指数，极易受头部权重股（Magnificent 7）操控。常出现“指数创新高，但底层 $60\% \sim 70\%$ 的成分股已在阴跌”的顶背离状态。
* **核心机理（真实参与度透视）：** 直接拉取 `tradermonty` 每日通过 GitHub Actions 自动计算的公开 CSV 数据源（标普 500 成分股高于 200 日均线比例及 8 日均值、全市场上升趋势股票占比 `Uptrend Ratio`、11 大行业板块趋势斜率），结合等权重与市值加权比值（$\text{RSP}/\text{SPY}$），精准识别**“指数创新高但宽度走弱”的中期见顶信号**与**“宽度极度超卖后的共振反转信号”**。

### 1.3 引擎 C：底层基准自动路由与信贷利差门控（Benchmark Auto-Routing & Credit Gate）
* **解决杠杆损耗失真：** 杠杆 ETF（如 `TQQQ`、`SQQQ`、`UPRO`、`SOXL`）存在严重的波动率损耗（Volatility Decay），直接在杠杆标的上计算 $200\text{D SMA}$ 会导致均线严重失真。系统必须将标的自动路由至对应的无损耗底层指数（如 `TQQQ/QQQ` $\rightarrow$ `^NDX`，`UPRO/SPY` $\rightarrow$ `^GSPC`，`TNA/IWM` $\rightarrow$ `^RUT`）。
* **跨资产信贷验证：** 引入高收益企业债与中长期国债比值（$\text{HYG}/\text{IEF}$），验证实体信贷市场是否出现违约利差走阔。

---

## 2. 零 API Key 数据源契约（Data Provider Contracts）

本系统所有数据均来自免费公开接口，无需注册任何付费 API Key。数据拉取模块必须实现本地缓存（SQLite 或 Parquet/CSV 缓存）与超时重试机制。

### 2.1 TraderMonty 公开市场宽度 CSV 数据源（GitHub 每日自动更新）

系统需通过 HTTP GET 拉取以下 3 个核心 CSV 数据流（需设置 `User-Agent` 并做字段名大小写兼容解析）：

1. **标普 500 市场宽度序列（`market-breadth-analysis`）**
   * **主 URL：** `https://tradermonty.github.io/market-breadth-analysis/market_breadth_data.csv`
   * **备用 Raw URL：** `https://raw.githubusercontent.com/tradermonty/market-breadth-analysis/main/market_breadth_data.csv`
   * **核心字段契约：**
     * `Date`: 交易日期（`YYYY-MM-DD`）
     * `S&P500_Price`: 标普 500 收盘价
     * `Breadth_Index_Raw`: 标普 500 成分股高于 $200\text{D SMA}$ 的原始比例（取值范围 $[0, 1]$ 或 $[0, 100]$，解析时需统一归一化为百分比 $[0, 100]$）
     * `Breadth_Index_200MA`: 宽度指数的 200 日均线（慢线）
     * `Breadth_Index_8MA`: 宽度指数的 8 日均线（快线）
     * `Breadth_200MA_Trend`: 长期宽度趋势（$1$ 为上升，$-1$ 或 $0$ 为下降）
     * `Bearish_Signal`: 空头死叉/顶部背离预警布尔标志
     * `Is_Peak` / `Is_Trough` / `Is_Trough_8MA_Below_04`: 局部波峰与极端超卖波谷标志

2. **全市场上升趋势股票占比序列（`uptrend-dashboard`）**
   * **主 URL：** `https://raw.githubusercontent.com/tradermonty/uptrend-dashboard/main/data/uptrend_ratio_timeseries.csv`
   * **核心字段契约（注意兼容大小写）：**
     * `date` (或 `Date`): 交易日期
     * `ratio` (或 `Ratio`): 当前处于上升趋势的全市场股票占比（通常归一化为百分比，其中 $< 10\%$ 为极端超卖区，$10\% \sim 20\%$ 为弱势空头区，$20\% \sim 30\%$ 为中性过渡区，$30\% \sim 37\%$ 为健康多头区，$> 37\% \sim 40\%$ 为短期过热区）
     * `ma10` (或 `10MA`): 上升趋势占比的 10 日移动平均
     * `slope`: 趋势斜率（正值代表扩散，负值代表收缩）
     * `trend`: 趋势方向（`UP` / `DOWN`）
     * `color`: 状态颜色标识（`GREEN` 代表多头主导，`RED` 代表空头主导）

3. **行业板块宽度汇总表（`sector_summary`）**
   * **主 URL：** `https://raw.githubusercontent.com/tradermonty/uptrend-dashboard/main/data/sector_summary.csv`
   * **核心用途：** 提取 11 大行业板块的上升趋势占比，计算**进攻型周期/成长板块（Technology, Consumer Discretionary, Industrials, Financials, Communication Services）**相对于**防御型板块（Utilities, Consumer Staples, Health Care）**的宽度差值（$\Delta \text{Breadth}_{\text{Off-Def}}$）。

> **数据新鲜度与降级契约（Freshness & Fallback Gate）：**
> 每次拉取 CSV 后，检查最新一行数据的 `Date` 与最新美股交易日的时间差 $\Delta t$。若 $\Delta t > 5$ 个自然日，触发 `DATA_STALE_WARNING` 警告；若网络异常或远程 CSV 无法访问，自动触发本地 `yfinance` 降级计算器（通过 `RSP/SPY` 动量比率与 11 个 SPDR 行业 ETF `XLK, XLY, XLI, XLF, XLC, XLU, XLP, XLV, XLE, XLB, XLRE` 高于各自 $50\text{D}$ 与 $200\text{D SMA}$ 的占比合成替代宽度指标）。

### 2.2 Yahoo Finance (`yfinance`) / CBOE 每日行情与波动率序列

通过 `yfinance` 批量拉取过去至少 $750$ 个交易日（约 3 年，确保 $252\text{D}$ 滚动窗口与 $200\text{D SMA}$ 完全预热）的日线 OHLCV 数据，时区统一对齐至美东时间 `America/New_York`：

| 维度类别 | Ticker 列表 | 用途说明 |
| :--- | :--- | :--- |
| **波动率期限结构** | `^VIX9D`, `^VIX`, `^VIX3M`, `^VIX6M` | 9天、30天、3个月、6个月隐含波动率，构建完整期限曲面 |
| **尾部风险与二阶波动** | `^VVIX`, `^SKEW` | 波动率的波动率（VVIX）与黑天鹅尾部偏度指数（SKEW） |
| **核心宽基指数基准** | `^GSPC` (SPX), `^NDX` (NDX), `^RUT` (RUT) | 无损耗底层指数基准，用于趋势门控与回测统计 |
| **可交易宽基与等权 ETF** | `SPY`, `QQQ`, `IWM`, `RSP` | 计算 $\text{RSP}/\text{SPY}$ 市值分化度及量价确认 |
| **跨资产信贷与宏观** | `HYG`, `IEF`, `^TNX`, `DX-Y.NYB` | 计算 $\text{HYG}/\text{IEF}$ 信贷风险偏好及美债/美元变化率 |

---

## 3. 核心算法与数学模型规范

AI 编写代码时必须严格按照以下数学公式实现，所有滚动统计量在时间步 $t$ 仅允许使用 $[t - W + 1, t]$ 的历史闭合数据。

### 3.1 标的自动路由映射（Underlying Benchmark Auto-Router）

当用户输入任何目标标的 `ticker` 时，趋势计算与胜率评估强制路由至底层无损耗指数 $B(\text{ticker})$：
* 若 $\text{ticker} \in \{\text{QQQ}, \text{QQQM}, \text{TQQQ}, \text{SQQQ}, \text{QLD}, \text{QID}, \text{NDX}, \text{NQ=F}, \text{MNQ=F}\}$，则 $B = \text{\textasciicircum NDX}$；
* 若 $\text{ticker} \in \{\text{IWM}, \text{TNA}, \text{TZA}, \text{UWM}, \text{TWM}, \text{RUT}, \text{RTY=F}\}$，则 $B = \text{\textasciicircum RUT}$；
* 若 $\text{ticker} \in \{\text{SOXL}, \text{SOXS}, \text{SMH}, \text{SOXX}\}$，则 $B = \text{SMH}$（或 $\text{\textasciicircum NDX}$）；
* 其余情况默认 $B = \text{\textasciicircum GSPC}$。

### 3.2 引擎 A：VIX 期限结构与期权微观结构评分模型（$\text{Score}_{\text{vol}}$）

#### (1) 数据就绪门控（Strict Readiness Gate）
在交易日 $t$，检查 `^VIX`、`^VIX3M`、`^GSPC`、`^NDX` 是否存在有效收盘价且有效历史长度 $L \ge 252$。若任一必选序列缺失或为 `NaN`，**严禁将该因子记为 $0$ 分**，必须直接中断打分并输出状态码 `DATA N/A`。

#### (2) 核心特征计算
1. **中期期限结构比率（Term Structure Ratio）与滚动 $Z$-Score：**
   $$R_{\text{TS}}(t) = \frac{\text{VIX}(t)}{\text{VIX3M}(t)}$$
   计算 63 日（季度）滚动均值 $\mu_{\text{TS}, 63}(t)$ 与标准差 $\sigma_{\text{TS}, 63}(t)$，得到期限结构 $Z$-Score：
   $$Z_{\text{TS}}(t) = \frac{R_{\text{TS}}(t) - \mu_{\text{TS}, 63}(t)}{\sigma_{\text{TS}, 63}(t)}$$
   *注：当 $R_{\text{TS}}(t) > 1.00$ 时，市场处于 **Backwardation（期限倒挂/急性恐慌）**；当 $R_{\text{TS}}(t) < 0.90$ 时，市场处于深度的 **Contango（正向升水/平静期）**。*

2. **短期恐慌基差比率（Short-End Basis）：**
   $$R_{\text{Basis}}(t) = \frac{\text{VIX9D}(t)}{\text{VIX}(t)}$$
   当 $R_{\text{Basis}}(t) > 1.05$ 时，表明超短期对冲需求极度拥挤；当 $R_{\text{Basis}}(t) < 0.88$ 时，表明近端情绪极度自满。

3. **VVIX 与 SKEW 动态分位数（Adaptive Rolling Percentiles）：**
   在过去 $W = 252$ 个交易日的滚动窗口内，计算当日 $\text{VVIX}(t)$ 与 $\text{SKEW}(t)$ 所处的经验百分位秩（Percentile Rank）：
   $$P_{\text{VVIX}}(t) = \text{PercentileRank}_{252}(\text{VVIX}_t) \in [0, 100]$$
   $$P_{\text{SKEW}}(t) = \text{PercentileRank}_{252}(\text{SKEW}_t) \in [0, 100]$$

#### (3) 纯结构分（$\text{Score}_{\text{vol}}$）合成规则
遵循 `VIX-Term-Structure-Pro` 的分值方向定义：**正分（$>0$）代表恐慌释放与超卖反转做多机会，负分（$<0$）代表情绪自满、期权昂贵或顶部风险对冲信号**。总分 $\text{Score}_{\text{vol}}(t) = \sum_{k=1}^{5} s_k(t)$：

1. **因子 1：期限结构 $Z$-Score（$s_1 \in [-3, +3]$）**
   * 若 $Z_{\text{TS}}(t) \ge 2.5$：$s_1 = +3.0$（极端恐慌极值）
   * 若 $2.0 \le Z_{\text{TS}}(t) < 2.5$：$s_1 = +2.0$
   * 若 $1.2 \le Z_{\text{TS}}(t) < 2.0$：$s_1 = +1.0$
   * 若 $-1.2 < Z_{\text{TS}}(t) < 1.2$：$s_1 = 0.0$
   * 若 $-2.0 < Z_{\text{TS}}(t) \le -1.2$：$s_1 = -1.5$
   * 若 $Z_{\text{TS}}(t) \le -2.0$：$s_1 = -2.5$（极度自满）

2. **因子 2：Contango / Backwardation 绝对水位（$s_2 \in [-2, +2]$）**
   * 若 $R_{\text{TS}}(t) \ge 1.08$：$s_2 = +2.0$（深度倒挂，流动性坍塌级恐慌）
   * 若 $1.00 \le R_{\text{TS}}(t) < 1.08$：$s_2 = +1.0$（轻度倒挂）
   * 若 $0.88 \le R_{\text{TS}}(t) < 1.00$：$s_2 = 0.0$（健康常态升水）
   * 若 $R_{\text{TS}}(t) < 0.88$：$s_2 = -1.5$（过度升水，市场严重低估波动风险）

3. **因子 3：近端基差确认（$s_3 \in [-1.5, +1.5]$）**
   * 若 $R_{\text{Basis}}(t) \ge 1.10$ 且 $\text{VIX}(t) > 25$：$s_3 = +1.5$（近端恐慌高潮）
   * 若 $R_{\text{Basis}}(t) < 0.85$ 且 $\text{VIX}(t) < 15$：$s_3 = -1.5$（近端极度贪婪自满）
   * 其余情况：$s_3 = 0.0$

4. **因子 4：SKEW 尾部暗流（$s_4 \in [-1.5, +1.0]$）**
   * 若 $P_{\text{SKEW}}(t) \ge 85$ 且 $R_{\text{TS}}(t) < 0.92$：$s_4 = -1.5$（指数平静但机构疯狂暗买虚值 Put 防黑天鹅，强烈顶部预警）
   * 若 $P_{\text{SKEW}}(t) \le 15$ 且 $R_{\text{TS}}(t) > 1.00$：$s_4 = +1.0$（尾部保护已被完全兑现，恐慌出清）
   * 其余情况：$s_4 = 0.0$

5. **因子 5：VVIX 波动率异动（$s_5 \in [-1.0, +1.0]$）**
   * 若 $P_{\text{VVIX}}(t) \ge 90$ 且 $Z_{\text{TS}}(t) \ge 1.5$：$s_5 = +1.0$（恐慌洗盘极值共振）
   * 若 $P_{\text{VVIX}}(t) \ge 85$ 且 $Z_{\text{TS}}(t) < 0$（即指数高位但 VVIX 异动拉升）：$s_5 = -1.0$（暴风雨前的宁静，机构抢筹波动率）
   * 其余情况：$s_5 = 0.0$

#### (4) 信号分级与正交门控过滤（Signal Tiers & Filtered States）
根据 $\text{Score}_{\text{vol}}(t)$ 映射原始信号等级，并施加**正交门控**：

| 原始信号等级 | 分值区间 | 触发门控条件（不满足则转入右侧等待/拦截状态） | 门控拦截后显示状态 |
| :--- | :--- | :--- | :--- |
| `🚨 CRASH BUY` | $\text{Score}_{\text{vol}} \ge 6.0$ | 极端恐慌反转，可豁免长期均线多空限制，但要求当日 $R_{\text{TS}}(t) \le R_{\text{TS}}(t-1)$（恐慌停止发散） | 若恐慌仍在加速发散 $\rightarrow$ `WAIT Vol/波动` |
| `🟢 STRONG BUY` | $5.0 \le \text{Score}_{\text{vol}} < 6.0$ | 要求 $Z_{\text{TS}}(t) \ge 1.8$ 且底层基准收盘价高于 5 日低点 | 若动量未企稳 $\rightarrow$ `WAIT Mom/动量` |
| `🟡 BUY DIP` | $3.0 \le \text{Score}_{\text{vol}} < 5.0$ | **强制检查长期趋势门控**：要求底层基准 $B(t) > \text{SMA}_{200}(B, t)$ | **若 $B(t) \le \text{SMA}_{200}(B, t)$ $\rightarrow$ `🚫 NO TRADE`（熊市严禁左侧抄底）** |
| `⏸ NEUTRAL` | $-2.0 < \text{Score}_{\text{vol}} < 3.0$ | 无极端结构偏向，由趋势与市场宽度主导仓位 | `⏸ NEUTRAL` |
| `🟠 SELL/HEDGE` | $-5.0 < \text{Score}_{\text{vol}} \le -2.0$ | **高胜率模式（High Win-Rate）核心自满确认**：要求满足 $R_{\text{TS}} < 0.90$ 或 $P_{\text{SKEW}} \ge 80$ | 若未满足核心自满确认 $\rightarrow$ `HOLD Core/核心` |
| `🔴 STRONG SELL` | $-6.0 < \text{Score}_{\text{vol}} \le -5.0$ | 要求核心自满确认成立且短周期动量走弱 | 若波动率结构不支持做空 $\rightarrow$ `HOLD Vol/波动` |
| `🔥 EUPHORIA` | $\text{Score}_{\text{vol}} \le -6.0$ | 极度贪婪与期权高度脆弱，立即缩减多头敞口或买入对冲保护 | `🔥 EUPHORIA` |

> **关键防坑规则（Acute Crash Override）：**
> 除了看 $\text{Score}_{\text{vol}}$ 极值外，必须单独输出一个布尔标志 `is_backwardation_risk`：当 $R_{\text{TS}}(t) > 1.00$ 且 $Z_{\text{TS}}(t) > 1.0$、但 $\text{Score}_{\text{vol}} < 5.0$（即**刚刚进入期限倒挂初期，尚未达到恐慌出清极值**）时，此时是**最危险的主跌浪初期**，系统必须立即发出 `⚠️ BACKWARDATION DE-LEVERAGE（期权倒挂去杠杆警报）`。

---

### 3.3 引擎 B：市场宽度健康度评分（$\text{Score}_{\text{breadth}} \in [0, 100]$）与背离检测器

读取 TraderMonty 数据源与 `RSP/SPY` 序列，计算 $0 \sim 100$ 分的综合市场宽度健康度得分 $\text{Score}_{\text{breadth}}(t)$：

1. **组件 1：标普 500 长期宽度水位（权重 $25\%$）**
   读取 `Breadth_Index_200MA`（归一化为 $0 \sim 100$）。若 $\ge 70\%$ 得满分 $25$ 分；若 $\le 20\%$ 得 $0$ 分；中间线性插值。
2. **组件 2：标普 500 宽度快慢线交叉与斜率（权重 $20\%$）**
   比较快线 `Breadth_Index_8MA` 与慢线 `Breadth_Index_200MA`：
   * 快线 $>$ 慢线 且快线 5 日斜率 $> 0$：得 $20$ 分（金叉共振扩张）
   * 快线 $>$ 慢线 但快线 5 日斜率 $\le 0$：得 $12$ 分（高位走平）
   * 快线 $\le$ 慢线 但快线 5 日斜率 $> 0$（或触发 `Is_Trough_8MA_Below_04` 后回升）：得 $10$ 分（超跌底部修复）
   * 快线 $\le$ 慢线 且快线 5 日斜率 $\le 0$（或 `Bearish_Signal == True`）：得 $0$ 分（死叉下行）
3. **组件 3：全市场上升趋势占比 `Uptrend Ratio`（权重 $25\%$）**
   读取 `uptrend_ratio_timeseries.csv` 最新一行的 `ratio`、`ma10`、`slope` 与 `color`：
   * 若 `color == 'GREEN'` 且 $25\% \le \text{ratio} \le 38\%$：得 $25$ 分（主升浪健康区间）
   * 若 `color == 'GREEN'` 且 $\text{ratio} > 38\%$：得 $18$ 分（短期过热预警）
   * 若 `color == 'RED'` 且 $\text{ratio} < 12\%$ 且 $\text{slope} > 0$：得 $15$ 分（极端超卖拐点）
   * 若 `color == 'RED'` 且 $\text{slope} \le 0$：得 $0 \sim 5$ 分（空头趋势主导）
4. **组件 4：等权重与市值加权相对强弱 $\text{RSP}/\text{SPY}$（权重 $15\%$）**
   令 $Q(t) = \frac{\text{RSP}(t)}{\text{SPY}(t)}$。若 $Q(t) > \text{SMA}_{20}(Q, t)$ 且 $\text{SMA}_{20}(Q, t) > \text{SMA}_{50}(Q, t)$，得 $15$ 分；若仅高于 $\text{SMA}_{20}$，得 $8$ 分；若均跌破，得 $0$ 分。
5. **组件 5：行业板块进攻/防御轮动差 $\Delta \text{Breadth}_{\text{Off-Def}}$（权重 $15\%$）**
   由 `sector_summary.csv` 计算进攻板块平均 `ratio` 减去防御板块平均 `ratio`。差值 $> +5\%$ 得 $15$ 分；差值在 $[-5\%, +5\%]$ 得 $8$ 分；差值 $< -5\%$（资金全面涌入公用事业/必选消费避险）得 $0$ 分。

#### 核心算法：巨头掩护下的“假突破/顶背离”检测器（Bearish Breadth Divergence Detector）
在交易日 $t$，若同时满足以下条件，则判定为 **`🚨 BEARISH_BREADTH_DIVERGENCE = True`（脆弱分化牛 / 假突破预警）**：
1. **指数处于高位：** 底层指数 $B(t)$ 距离过去 252 日最高点 $H_{252}(t) = \max_{i \in [0, 251]} B(t-i)$ 的回撤小于 $2.0\%$，即：
   $$\frac{B(t)}{H_{252}(t)} \ge 0.98$$
2. **内部宽度严重背离（满足以下任意两项）：**
   * `Breadth_Index_8MA` $<$ `Breadth_Index_200MA`（或 `Bearish_Signal == True`）；
   * 全市场 `Uptrend Ratio` $< 25\%$ 或 `color == 'RED'`；
   * 等权重比率 $Q(t) = \frac{\text{RSP}(t)}{\text{SPY}(t)}$ 创出过去 20 日新低。

---

## 4. 终极市场状态（Regime）五大象限综合判定矩阵

系统将 **趋势与信贷基准（Engine C）**、**期权期限结构（Engine A）** 与 **市场宽度（Engine B）** 汇总，输出唯一确定的当前美股市场状态（Regime）与仓位指令：

| 状态代码 (Regime ID) | 中文名称 | 触发判定逻辑 | 推荐总仓位敞口 | 杠杆 ETF (`TQQQ`/`UPRO`) 策略指令 |
| :--- | :--- | :--- | :--- | :--- |
| **`REGIME_1_STRONG_BULL`** | **强劲全面牛市** | $B(t) > \text{SMA}_{50} > \text{SMA}_{200}$，且 $R_{\text{TS}} < 0.95$（无倒挂），且 $\text{Score}_{\text{breadth}} \ge 60$，且无宽度顶背离 | $90\% \sim 100\%$ | 允许右侧顺势持有，追踪止损设于底层指数 $\text{SMA}_{20}$ 或 $\text{SMA}_{50}$ |
| **`REGIME_2_NARROW_FRAGILE_BULL`** | **脆弱分化牛（假突破高发区）** | $B(t) > \text{SMA}_{200}$，但触发 `BEARISH_BREADTH_DIVERGENCE == True` 或 $\text{Score}_{\text{vol}} \le -2.0$（`SELL/HEDGE` 或 `EUPHORIA`） | $50\% \sim 65\%$ | **严禁追高加仓**；将杠杆 ETF 降维切换为 $1\times$ 宽基（`SPY`/`QQQ`）或收紧止盈，警惕巨头补跌 |
| **`REGIME_3_CORRECTION_HIGH_VOL`** | **中期修正 / 期权倒挂震荡** | $B(t) > \text{SMA}_{200}$ 但跌破 $\text{SMA}_{50}$，或触发 `is_backwardation_risk == True`（$R_{\text{TS}} > 1.00$ 且未达恐慌极值） | $20\% \sim 40\%$ | **立即去杠杆**；等待 $R_{\text{TS}}$ 重新回落至 $0.98$ 以下或出现 `🟢 STRONG BUY` 确认信号再回补 |
| **`REGIME_4_SYSTEMIC_BEAR`** | **系统性熊市** | $B(t) \le \text{SMA}_{200}$ 且 $\text{HYG}/\text{IEF} < \text{SMA}_{50}(\text{HYG}/\text{IEF})$，且 $\text{Score}_{\text{breadth}} < 35$ | $0\% \sim 15\%$（转入 `SGOV`/`BIL` 短债） | **一票否决（`🚫 NO TRADE`）**：严禁左侧死扛任何 $2\times/3\times$ 杠杆多头 ETF |
| **`REGIME_5_CAPITULATION_REVERSAL`** | **极端恐慌洗盘反转点** | $\text{Score}_{\text{vol}} \ge 5.0$（触发 `🚨 CRASH BUY` 或 `🟢 STRONG BUY`）且 $R_{\text{TS}}(t) \le R_{\text{TS}}(t-1)$，或宽度触发极端超卖拐点（`Uptrend Ratio` $< 10\%$ 后拐头向上） | 分批建仓 $40\% \sim 70\%$ | 赔率最高的战术级右侧反转买点，以拐点前低作为严格止损位 |

---

## 5. 技术栈与项目目录结构

请使用 **Python 3.10+** 开发，依赖项保持极简与高可靠：
* 核心依赖：`pandas`, `numpy`, `yfinance`, `requests`, `rich`（用于终端高颜值仪表盘输出）, `pytest`（用于回归测试）。

```text
us-market-regime-pro/
├── README.md                       # 项目使用说明与快速启动指南
├── requirements.txt                # Python 依赖包列表
├── main.py                         # CLI 主入口（支持指定标的、输出终端看板与导出 Markdown/JSON 报告）
├── data_cache/                     # 本地 CSV / Parquet 缓存目录（自动创建，加入 .gitignore）
├── reports/                        # 每日自动生成的 Markdown 与 HTML 诊断报告目录
├── src/
│   ├── __init__.py
│   ├── config.py                   # 集中管理 URL 契约、阈值参数、标的路由映射表
│   ├── data_fetcher.py             # 负责拉取 TraderMonty CSV 与 yfinance 数据，含缓存、新鲜度校验与降级回退
│   ├── benchmark_router.py         # 杠杆 ETF 与宽基标的到底层无损耗指数的精确路由模块
│   ├── vix_structure_engine.py     # 引擎 A：复刻 aaajiao/VIX-Term-Structure-Pro 核心评分、Z-Score 与门控逻辑
│   ├── breadth_engine.py           # 引擎 B：解析 tradermonty 宽度 CSV、6/5 维打分与巨头顶背离检测器
│   ├── trend_credit_gate.py        # 引擎 C：计算 20D/50D/200D 均线排列、斜率及 HYG/IEF 信贷风险利差
│   ├── regime_synthesizer.py       # 顶层状态合成器：映射五大 Regime 象限并生成具体仓位与操作指令
│   └── report_generator.py         # 生成 Rich 终端彩色面板与结构化 Markdown 每日复盘报告
└── tests/
    ├── __init__.py
    ├── test_readiness_and_no_lookahead.py  # 验证数据缺失门控（DATA N/A）与滚动窗口零未来函数
    ├── test_vix_scoring_and_gates.py       # 验证 Score 正交性（趋势不改分，仅过滤 BUY DIP 为 NO TRADE）
    └── test_breadth_divergence.py          # 验证指数创新高但宽度走弱时的假突破报警逻辑
```

---

## 6. 分阶段开发实施路线图（AI Step-by-Step Implementation Plan）

请 AI 严格按以下 5 个阶段顺序编码，每完成一个阶段进行自检：

### Phase 1：基础设施、配置契约与防脆弱数据层（`config.py` & `data_fetcher.py`）
1. 在 `config.py` 中定义所有常量、TraderMonty CSV URLs、`yfinance` 标的池以及底层基准路由字典。
2. 在 `data_fetcher.py` 中实现：
   * `fetch_tradermonty_csvs()`：使用 `requests` 拉取 `market_breadth_data.csv`、`uptrend_ratio_timeseries.csv` 和 `sector_summary.csv`。**注意：** 必须对 CSV 列名进行规范化处理（`df.columns = [c.strip() for c in df.columns]`），自动识别百分比格式（若 `Breadth_Index_200MA` 最大值 $\le 1.0$，自动乘以 $100$ 转为百分比 $[0, 100]$）。
   * `fetch_market_timeseries()`：使用 `yfinance` 下载过去 3 年日线收盘价，按交易日索引对齐，并处理偶发的单日空值（仅允许最多向前填充 `ffill(limit=2)` 处理节假日跨市场错位，严禁向后填充 `bfill` 以防泄露未来数据）。
   * 实现 Fallback 机制：当 GitHub CSV 因网络问题不可达时，自动使用 `RSP/SPY` 与 11 大行业 ETF 均线宽度计算替代指标，确保程序永不崩溃。

### Phase 2：引擎 A — VIX 期限结构与正交门控实现（`vix_structure_engine.py`）
1. 实现 `Strict Readiness Gate`：若关键波动率序列有效样本不足 $252$ 天或当日数据缺失，返回 `status="DATA N/A", score=None`。
2. 严格按照第 3.2 节公式计算 $R_{\text{TS}}(t)$、$Z_{\text{TS}}(t)$、$R_{\text{Basis}}(t)$、$P_{\text{VVIX}}(t)$ 与 $P_{\text{SKEW}}(t)$。
3. 计算 $\text{Score}_{\text{vol}}(t)$，并实现正交趋势门控函数 `apply_orthogonal_gates(raw_score, trend_is_bull, ...)`：
   * 确保 `trend_is_bull` 无论为 `True` 还是 `False`，绝不修改 `raw_score` 的数值；
   * 当 `3.0 <= raw_score < 5.0`（`🟡 BUY DIP`）且 `trend_is_bull == False` 时，输出展示状态 `🚫 NO TRADE`。
   * 单独计算 `is_backwardation_risk`（当 $R_{\text{TS}}(t) > 1.00$ 且未达反转极值时置为 `True`）。

### Phase 3：引擎 B & C — 市场宽度背离引擎与趋势信贷门控（`breadth_engine.py` & `trend_credit_gate.py`）
1. 在 `trend_credit_gate.py` 中，通过 `benchmark_router.py` 将用户查询的标的（如 `TQQQ`, `QQQ`, `SPY`, `SOXL`）路由至底层基准，计算 $\text{SMA}_{20}$、$\text{SMA}_{50}$、$\text{SMA}_{200}$、20 日均线斜率、距 252 日高点回撤比例，以及信贷利差 $\text{HYG}/\text{IEF}$ 相对于其 50 日均线的状态。
2. 在 `breadth_engine.py` 中，计算五大宽度组件得分并汇总为 $\text{Score}_{\text{breadth}} \in [0, 100]$。
3. 实现 `detect_bearish_divergence()`，输出是否存在“巨头掩护下的假突破顶背离”及其具体触发原因列表。

### Phase 4：状态合成器、CLI 看板与自动化报告（`regime_synthesizer.py`, `report_generator.py`, `main.py`）
1. 在 `regime_synthesizer.py` 中整合三大引擎输出，判定当前所处的 `Regime ID`（五大象限之一），生成目标权益仓位百分比及针对普通宽基与 $3\times$ 杠杆标的的具体操作指引。
2. 在 `report_generator.py` 与 `main.py` 中：
   * 支持命令行参数：`python main.py --tickers SPY QQQ TQQQ SOXL IWM --export-md`。
   * 在终端使用 `rich.table` 与 `rich.panel` 打印结构化仪表盘（包含：当前市场 Regime、VIX 期限结构分与 $Z$-Score、市场宽度总分与快慢线状态、假突破背离警报、各标的底层路由趋势及操作指令）。
   * 在 `reports/` 目录下生成带日期戳的 Markdown 复盘报告（`reports/regime_report_YYYY-MM-DD.md`）。

### Phase 5：自动化单元测试与防未来函数校验（`tests/`）
编写并运行 `pytest` 测试套件，必须覆盖以下 3 个核心断言：
1. **正交性断言（Orthogonality Assertion）：** 固定一组 VIX 结构输入，分别传入 `trend_is_bull=True` 和 `trend_is_bull=False`，断言两者算出的 `Score_vol` 完全相等（精确到小数点后 6 位），且當 `3.0 <= Score_vol < 5.0` 时，后者状态严格变为 `🚫 NO TRADE`。
2. **零未来函数断言（No Look-Ahead Bias Assertion）：** 构造一段长度为 $300$ 天的模拟行情 $X_{1:300}$，先计算第 $260$ 天的 $Z_{\text{TS}}(260)$ 与 $\text{Score}_{\text{vol}}(260)$；随后将第 $261 \sim 300$ 天的数据人为篡改为极端暴涨或暴跌 10 倍，再次重算第 $260$ 天的所有指标，断言第 $260$ 天的输出分毫不变。
3. **顶背离检测断言（Breadth Divergence Assertion）：** 构造指数创 252 日新高、但 `Breadth_Index_8MA < Breadth_Index_200MA` 且 `Uptrend Ratio = 18%` 的测试样本，断言系统准确触发 `BEARISH_BREADTH_DIVERGENCE == True` 并归类为 `REGIME_2_NARROW_FRAGILE_BULL`。

---

## 7. 交付验收标准（Definition of Done）

当 AI 完成全部代码编写后，请自动在终端依次执行以下命令完成自验：
1. 运行测试套件：`pytest -v`（确保所有测试用例 $100\%$ 通过）。
2. 运行实盘拉取与诊断：`python main.py --tickers SPY QQQ TQQQ SOXL --export-md`。
3. 展示终端实际运行输出的今日美股真实趋势诊断结果。