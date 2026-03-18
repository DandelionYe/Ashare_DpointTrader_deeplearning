# A股 Dpoint 机器学习交易信号系统

<p align="center">
  <a href="README.md">
    <img src="https://img.shields.io/badge/Docs-English-blue?style=for-the-badge&logo=github" alt="English Docs"/>
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python"/>
  &nbsp;
  <img src="https://img.shields.io/badge/PyTorch-支持-ee4c2c?style=for-the-badge&logo=pytorch"/>
  &nbsp;
  <img src="https://img.shields.io/badge/市场-A股-gold?style=for-the-badge"/>
</p>

> **[📖 点击查看英文版 → README.md](README.md)**

---

本项目是一个面向**中国 A 股市场**的机器学习交易信号生成管线。核心信号 **Dpoint** 表示"明日收盘价高于今日"的预测概率，系统通过 Walk-Forward 时序验证搜索最优特征组合、模型架构与交易参数，最终输出包含完整回测结果的 Excel 报告。

> ⚠️ **免责声明** — 本项目仅供学习和研究使用。历史回测结果（尤其是样本内结果）**不代表**未来实盘表现。本项目内容不构成任何投资建议。

---

## 目录

- [核心概念](#核心概念)
- [系统架构](#系统架构)
- [特征工程](#特征工程)
- [支持的模型](#支持的模型)
- [回测规则](#回测规则)
- [项目结构](#项目结构)
- [安装部署](#安装部署)
- [使用说明](#使用说明)
- [输出文件说明](#输出文件说明)
- [核心设计说明](#核心设计说明)
- [已知局限](#已知局限)

---

## 核心概念

**Dpoint** 的定义：

```
Dpoint_t = P(close_{t+1} > close_t | X_t)
```

特征矩阵 `X_t` 中所有特征**仅使用 t 日及之前**的数据构建，严格杜绝前向偏差。模型本质上是一个二分类器，预测明日收盘价是否高于今日；输出的预测概率作为连续信号，驱动买卖决策。

---

## 系统架构

```
main_cli.py  ──────────────────────────────────────────────────────►  Excel + JSON
     │
     ├── data_loader.py          加载并清洗 A 股 OHLCV Excel 数据
     │
     ├── search_engine.py        随机搜索（探索 / 精细化 / 池采样 三模式轮次）
     │       ├── feature_dpoint.py   构建特征矩阵 X 和标签 y
     │       ├── model_builder.py    sklearn 模型（LogReg、SGD、XGB）
     │       ├── dl_model_builder.py PyTorch 模型（MLP、LSTM、GRU、CNN、Transformer）
     │       ├── splitter.py         Walk-Forward 时序切分
     │       ├── metrics.py          几何均值净值比率 + 交易次数惩罚项
     │       └── persistence.py      best_so_far.json / best_pool.json 持久化
     │
     ├── trainer_optimizer.py    全样本最终模型拟合
     ├── backtester_engine.py    事件驱动回测（含 A 股约束）
     └── reporter.py             Excel 工作簿 + 运行配置 JSON

dpoint_updater.py               独立工具：在新数据上重训并导出 Dpoint
```

---

## 特征工程

特征在 `feature_dpoint.py` 中计算，每个特征族均可通过搜索配置独立开关，为优化器提供宽广的组合搜索空间。

| 特征族 | 主要特征 | 配置开关 |
|---|---|---|
| **动量** | 多窗口收益率、均线偏离比 | `use_momentum` |
| **波动率** | 高低价幅、真实振幅、滚动标准差 / MAD | `use_volatility` |
| **成交量与流动性** | 对数成交量/金额、成交量 MA 比或 z-score | `use_volume` |
| **K 线形态** | 实体、上影线、下影线 | `use_candle` |
| **换手率** | 原始换手率、滚动均值 / 标准差 / z-score | `use_turnover` |
| **技术指标** | RSI、MACD（线 + 柱）、布林带宽、OBV | `use_ta_indicators` |

所有特征仅使用 t 日及之前的数据，不引入任何前向偏差。

---

## 支持的模型

| 类型 | 依赖库 | 说明 |
|---|---|---|
| `logreg` | scikit-learn | L1/L2 正则，含 StandardScaler Pipeline |
| `sgd` | scikit-learn | log-loss SGD，含 StandardScaler Pipeline |
| `xgb` | XGBoost（可选） | 自动检测 CUDA |
| `mlp` | PyTorch | 多层感知机 |
| `lstm` | PyTorch | 单/双向，1–2 层 |
| `gru` | PyTorch | 单向，1–2 层 |
| `cnn` | PyTorch | 多尺度一维卷积 |
| `transformer` | PyTorch | 仅编码器，含位置编码 |

---

## 回测规则

`backtester_engine.py` 严格模拟 A 股市场规则：

- **仅做多** — 不支持做空
- **T+1 近似** — 信号在 t 日收盘后生成，委托在 **t+1 日以开盘价（open_qfq）成交**
- **最小交易单位** — 100 股
- **交易成本** — 买入：0.03% 佣金；卖出：0.03% 佣金 + 0.10% 印花税（可配置）
- **持仓天数** — 以**交易日**计算，不含周末和节假日
- **信号确认** — 需连续 N 天满足条件才触发委托，平滑噪声
- **止盈 / 止损** — 可选，按比例触发
- **Buy & Hold 基准** — 同期买入持有净值，用于估算 alpha

### 执行层 (P0 功能)
- **滑点模型**: 固定 20 bps (0.2%) 滑点
- **涨跌停处理**: 涨停不能买，跌停不能卖
- **停牌处理**: 停牌时订单被拒绝
- **ST 股过滤**: 可选过滤 ST 股票
- **上市天数过滤**: 最少 60 个交易日上市要求
- **成交量过滤**: 最少日成交额要求（默认 100 万元）
- **执行统计**: 记录订单提交/成交/拒绝原因/滑点成本

---

## 项目结构

```
.
├── main_cli.py             入口 — 搜索 + 回测 + 报告生成
├── dpoint_updater.py       独立工具 — 在新数据上重训并导出 Dpoint 到 Excel
│
├── data_loader.py          Excel 加载器，含 OHLCV 数据验证
├── feature_dpoint.py       特征工程（全特征族 + 技术指标）
├── model_builder.py        sklearn 模型工厂
├── dl_model_builder.py     PyTorch 模型工厂（MLP/LSTM/GRU/CNN/Transformer）
│
├── search_engine.py        随机搜索：探索 / 精细化 / 池采样 三模式轮次
├── trainer_optimizer.py    公开训练 API；全样本最终拟合
├── splitter.py             Walk-Forward 切分 + 自适应折数 + Final Holdout
├── metrics.py              几何均值比率；交易次数惩罚项 + 完整风险指标
├── backtester_engine.py   A 股事件驱动回测引擎 + 执行层
├── calibration.py          概率校准（Platt、Isotonic、Brier Score、ECE/MCE）
├── explainer.py            特征重要性（树模型、排列重要性、SHAP）& 使用跟踪
├── regime.py               市场状态检测与分层分析
├── rolling_trainer.py      滚动再训练调度器（扩展/滚动窗口）
├── persistence.py          best_so_far.json / best_pool.json 读写
├── reporter.py             Excel 工作簿 + JSON + HTML 报告
├── html_reporter.py       HTML 仪表盘（含净值曲线、校准图）
├── run_manifest.py         实验清单管理 & 重放
├── repro.py                可复现性工具（种子、环境锁定）
├── compare_runs.py        运行结果比对工具
│
├── constants.py            全局常量（惩罚权重、文件名）
│
├── tests/                 自动化测试套件
│   ├── test_no_leakage.py    时序泄露测试
│   ├── test_splitter.py     Walk-Forward 切分测试
│   ├── test_execution.py    执行层测试
│   ├── test_fee_lot.py      费用和手数测试
│   ├── test_metrics.py      风险指标测试
│   ├── test_smoke.py       冒烟测试
│   ├── test_cli.py         CLI 参数测试
│   ├── test_reproducibility.py 可复现性测试
│   ├── test_rejection.py    订单拒绝逻辑测试
│   └── conftest.py         测试 fixtures
```

---

## 安装部署

### 1. 创建 conda 环境

```bash
conda create -n ashare_dpoint python=3.10
conda activate ashare_dpoint
```

### 2. 安装依赖

```bash
pip install pandas numpy scikit-learn joblib openpyxl xlsxwriter torch
# 可选：XGBoost 支持
pip install xgboost
```

> GPU 加速自动检测。若系统存在 CUDA 可用的 GPU，PyTorch 模型和 XGBoost 将自动启用 GPU 加速。

### 3. 准备数据文件

准备一个 Excel 文件，列名须严格与下表一致：

| 列名 | 说明 |
|---|---|
| `date` | 交易日期（支持多种格式自动解析） |
| `open_qfq` | 前复权开盘价 |
| `high_qfq` | 前复权最高价 |
| `low_qfq` | 前复权最低价 |
| `close_qfq` | 前复权收盘价 |
| `volume` | 成交量（股） |
| `amount` | 成交金额（元） |
| `turnover_rate` | 换手率（%） |

建议数据长度不少于约 300 个交易日，以保证机器学习训练稳定性。

---

## 使用说明

### 全新搜索

```bash
python main_cli.py --data_path /path/to/stock_data.xlsx --output_dir ./output --runs 200 --initial_cash 100000
```

或通过环境变量指定数据路径：

```bash
export ASHARE_DATA_PATH=/path/to/stock_data.xlsx
python main_cli.py --runs 200
```

### 理解 --mode 和 --seed 参数

#### --mode 参数

- **`first`（默认）**：从头开始全新搜索。系统会随机采样模型配置并进行评估。适用于：
  - 首次在数据集上运行
  - 想探索全新搜索空间
  - 想改变搜索策略

- **`continue`**：从之前找到的最佳配置继续搜索。系统会从 `best_so_far.json` 加载最佳结果并在此基础上继续搜索。适用于：
  - 延长之前未收敛的搜索
  - 想在已有最佳结果基础上寻找更好的配置
  - 搜索仍会随机探索，但会以已知最佳结果为起点

**使用示例：**
```bash
# 第一次运行：全新搜索 200 次迭代
python main_cli.py --data_path /path/to/stock_data.xlsx --runs 200

# 第二次运行：从最佳结果继续，再运行 100 次迭代
python main_cli.py --data_path /path/to/stock_data.xlsx --mode continue --runs 100
```

#### --seed 参数

随机种子控制搜索的可复现性。不同的种子会产生不同的搜索轨迹和可能不同的最终结果。

- 使用**相同种子**配合**相同数据**将始终产生相同结果
- 使用**不同种子**可以探索搜索空间的不同区域

**使用示例：**
```bash
# 运行1：使用种子 42
python main_cli.py --data_path /path/to/stock_data.xlsx --runs 200 --seed 42

# 运行2：使用种子 123（不同的探索路径）
python main_cli.py --data_path /path/to/stock_data.xlsx --runs 200 --seed 123
```

> **提示**：如果想要更稳健的结果，可以使用不同种子多次运行搜索，比较最佳配置的差异。

### 在已有结果上继续搜索

```bash
python main_cli.py --data_path /path/to/stock_data.xlsx --output_dir ./output --mode continue --runs 100
```

### 用最新行情数据更新 Dpoint

```bash
python dpoint_updater.py --output_dir ./output
```

工具会交互式询问使用哪次历史运行的配置，然后弹出文件选择窗口供你选择最新数据文件。

### 所有 CLI 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_path` | 环境变量 `ASHARE_DATA_PATH` | 输入 Excel 路径 |
| `--output_dir` | `./output` | 结果输出目录 |
| `--runs` | `200` | 搜索迭代次数 |
| `--mode` | `first` | `first`（全新）或 `continue`（继续） |
| `--initial_cash` | `100000` | 初始资金（元） |
| `--n_folds` | `auto` | Walk-forward 折数（0 = 自动推算） |
| `--n_jobs` | `1` | 并行进程数（CUDA 启用时自动限制） |
| `--seed` | `42` | 随机种子 |
| `--eval_tickers` | `` | 多标的泛化评估文件路径（逗号分隔） |
| `--use_holdout` | `1` | 启用 Final Holdout 测试 (1=是, 0=否) |
| `--holdout_ratio` | `0.15` | Holdout 比例（默认 15%） |
| `--use_embargo` | `0` | 启用 Embargo Gap 防止时序泄露 |
| `--embargo_days` | `5` | Embargo 天数 |
| `--use_sensitivity_analysis` | `1` | 启用参数敏感性分析 |
| `--use_regime_analysis` | `0` | 启用市场状态分层分析 |
| `--experiment_dir` | `auto` | 实验独立输出目录 |
| `--replay` | `` | 从历史实验重放 |
| `--rolling_mode` | `` | 滚动再训练模式: expanding, rolling |
| `--rolling_window_length` | `None` | 滚动窗口长度（天） |
| `--retrain_frequency` | `monthly` | 再训练频率: daily, weekly, monthly, quarterly |
| `--export_lock` | `` | 导出环境锁定文件 |

---

## 输出文件说明

每次运行在 `--output_dir` 目录下产生以下文件：

| 文件 | 说明 |
|---|---|
| `run_NNN.xlsx` | 多 Sheet Excel 工作簿（主要报告） |
| `run_NNN_config.json` | 完整配置与元数据 |
| `best_so_far.json` | 历史所有运行中的全局最优配置 |
| `best_pool.json` | Top-10 配置候选池 |

### Excel 各 Sheet 说明

| Sheet | 内容 |
|---|---|
| **Trades** | 每笔交易：买入/卖出日期、价格、盈亏、收益率、状态 |
| **EquityCurve** | 每日净值、现金、持仓市值、最大回撤、日收益率、Buy & Hold 基准曲线 |
| **Config** | 本次运行的全部特征 / 模型 / 交易参数 |
| **Log** | 数据加载报告、训练摘要、每轮搜索日志 |
| **ModelParams** | 特征系数与 Scaler 参数（仅 LogReg/SGD 模型输出） |
| **RiskMetrics** | 完整风险指标：夏普、索提诺、卡玛、最大回撤等 |
| **RegimeAnalysis** | 市场状态分层表现（高/低波动、牛/熊市） |
| **RegimeStratified** | 各市场状态下的详细指标 |
| **TradeDistribution** | 交易分布统计（盈亏、持仓天数） |
| **CalibrationMetrics** | 概率校准结果（Brier Score、ECE、MCE） |
| **FeatureUsage** | 搜索过程中特征组使用频率 |
| **FeatureImportance** | 最佳模型特征重要性（树模型、排列重要性、SHAP） |

---

## 核心设计说明

### Walk-Forward 时序验证 + Final Holdout
优化器使用不重叠的样本外验证窗口评估每个候选配置。训练集采用**扩展窗口**（expanding window），每折验证集严格位于训练集之后。目标指标为**各折净值比率的几何均值**，天然惩罚不稳定或高方差的策略。

**多阶段验证流程：**
1. **Search OOS**: 在搜索数据上进行 Walk-Forward 交叉验证
2. **Selection OOS**: Top-K 候选在搜索数据上重新验证
3. **Final Holdout OOS**: 最佳配置在完全留出的 holdout 数据上评估（默认 15%）

### 多重防过拟合机制
- **Final Holdout Split**: 留出 15% 数据作为最终测试集，搜索过程完全不接触
- **Nested Walk-Forward**: 内层 CV 用于模型选择，外层 CV 用于性能评估
- **Embargo Gap**: 训练集和验证集之间留出 5 天间隔，防止滚动窗口特征泄露
- **参数敏感性分析**: 检查最优解是否过于"尖锐"
- **多种子稳定性评估**: Top-N 候选使用多个随机种子重新评估
- **惩罚项**: 最差折惩罚、折方差惩罚、交易次数过少惩罚

### 交易次数惩罚项
软惩罚项抑制每折交易次数偏离目标值（`TARGET_CLOSED_TRADES_PER_FOLD`）的配置，防止优化器收敛到退化解（如从不交易或每日交易）。

### 探索 / 精细化 / 池采样 三模式搜索
随机搜索分轮进行，每轮候选由以下三种模式之一生成：
- **探索（Explore，约 30%）** — 在完整搜索空间中随机采样，用于发现新区域
- **精细化（Exploit，约 70%）** — 在当前最优配置（incumbent）附近施加小扰动，用于局部优化
- **池采样（Pool-Exploit）** — 从 Top-K 历史候选池中随机加权采样，避免单点收敛

每轮结束后立即更新 incumbent，使后续轮次的精细化候选能够基于最新最优配置生成。

### 自适应折数推算
`recommend_n_folds()` 根据有效数据量自动选择 Walk-Forward 折数，目标是每折验证期内期望交易次数不低于统计置信度所需的最低样本量。

---

## 已知局限

- **样本内最终净值曲线** — Excel 报告中的全样本回测在同一批数据上训练并预测，必然存在前向偏差，数值偏乐观。请以 Log Sheet 中各折样本外指标作为真实性能评估依据。
- **无实盘接入** — 本项目仅为研究工具，不含委托管理、券商接口或实时行情对接。
- **印花税税率** — 默认卖出成本使用 2023 年 8 月前的 0.10% 印花税。如需使用调整后的 0.05%，请在调用时传入 `commission_rate_sell=0.0008`。
- **跨标的泛化评估** — `--eval_tickers` 参数采用超参迁移方式（配置不变，在目标标的上从头训练），**不迁移**模型权重。

---

## 高级功能

### 概率校准
系统支持概率校准以提高预测可靠性：
- **方法**: 无、Platt Scaling、Isotonic Regression
- **指标**: Brier Score、期望校准误差 (ECE)、最大校准误差 (MCE)
- **验证**: 校准仅在验证集上拟合，不在测试数据上拟合

### 特征重要性与可解释性
- **特征使用跟踪**: 记录搜索过程中各特征组的使用情况
- **树模型重要性**: XGBoost 原生特征重要性
- **排列重要性**: 模型无关的重要性估计
- **SHAP 值**: 用于树模型和线性模型（需安装 SHAP）

### 市场状态分层分析
- **趋势检测**: 基于均线交叉（短期 vs 长期均线）
- **波动率状态**: 基于滚动波动率的高/中/低波动
- **组合状态**: 趋势 × 波动率矩阵
- **分层指标**: 按市场状态分解的性能指标

### 滚动再训练
- **窗口类型**: 扩展窗口（随时间增长）或滚动窗口（固定长度）
- **再训练频率**: 日、周、月、季
- **快照管理**: 跟踪模型快照以支持回滚

### 可复现性
- **全局种子**: 为 Python、NumPy、PyTorch、TensorFlow 设置种子
- **环境锁定**: 导出 `requirements-lock.txt` 实现环境可复现
- **实验清单**: 每次运行生成 `manifest.json` 包含完整元数据
- **CLI 重放**: 从历史清单重放实验
