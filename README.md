# A-Share Dpoint ML Trading Signal System

<p align="center">
  <a href="README_Chinese_ver.md">
    <img src="https://img.shields.io/badge/文档-中文版-red?style=for-the-badge&logo=github" alt="中文文档"/>
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python"/>
  &nbsp;
  <img src="https://img.shields.io/badge/PyTorch-supported-ee4c2c?style=for-the-badge&logo=pytorch"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Market-A--Share-gold?style=for-the-badge"/>
</p>

> **[📖 Click here to read the Chinese version → README_Chinese_ver.md](README_Chinese_ver.md)**

---

A machine-learning pipeline that generates next-day directional signals (**Dpoint**) for Chinese A-share stocks. The system searches for the best combination of feature engineering, model architecture, and trading parameters through walk-forward cross-validation, then outputs a full backtest report with an Excel workbook.

> ⚠️ **Disclaimer** — This project is for research and educational purposes only. Past backtest results, especially in-sample ones, do **not** guarantee future performance. Nothing here constitutes financial advice.

---

## Table of Contents

- [Core Concept](#core-concept)
- [Architecture Overview](#architecture-overview)
- [Feature Engineering](#feature-engineering)
- [Supported Models](#supported-models)
- [Backtesting Rules](#backtesting-rules)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Output Files](#output-files)
- [Key Design Decisions](#key-design-decisions)
- [Known Limitations](#known-limitations)

---

## Core Concept

**Dpoint** is defined as:

```
Dpoint_t = P(close_{t+1} > close_t | X_t)
```

All features in `X_t` are built exclusively from data available on or before day `t` — there is no forward leakage. The model is a binary classifier that predicts whether tomorrow's close will be higher than today's. The predicted probability is used as a continuous signal to drive buy and sell decisions.

---

## Architecture Overview

```
main_cli.py  ──────────────────────────────────────────────────────►  Excel + JSON
     │
     ├── data_loader.py          Load & clean A-share OHLCV Excel
     │
     ├── search_engine.py        Random search with explore/exploit rounds
     │       ├── feature_dpoint.py   Build feature matrix X and label y
     │       ├── model_builder.py    sklearn models (LogReg, SGD, XGB)
     │       ├── dl_model_builder.py PyTorch models (MLP, LSTM, GRU, CNN, Transformer)
     │       ├── splitter.py         Walk-forward time-series splits
     │       ├── metrics.py          Geometric mean equity ratio + trade penalty
     │       └── persistence.py      best_so_far.json / best_pool.json
     │
     ├── trainer_optimizer.py    Final full-sample model fit
     ├── backtester_engine.py    Event-driven backtest with A-share constraints
     └── reporter.py             Excel workbook + run config JSON

dpoint_updater.py               Standalone tool: retrain on fresh data and export Dpoint
```

---

## Feature Engineering

Features are computed in `feature_dpoint.py`. Each group can be switched on or off independently through the search configuration, giving the optimizer a wide combinatorial space to explore.

| Group | Key Features | Config Flag |
|---|---|---|
| **Momentum** | Multi-window returns, MA ratio | `use_momentum` |
| **Volatility** | HL range, True Range, rolling std / MAD | `use_volatility` |
| **Volume & Liquidity** | Log volume/amount, volume MA ratio or z-score | `use_volume` |
| **Candlestick** | Body, upper shadow, lower shadow | `use_candle` |
| **Turnover Rate** | Raw turnover, rolling mean / std / z-score | `use_turnover` |
| **TA Indicators** | RSI, MACD (line + histogram), Bollinger Band Width, OBV | `use_ta_indicators` |

All features use only information up to and including day `t`. No forward leakage is introduced.

---

## Supported Models

| Type | Library | Notes |
|---|---|---|
| `logreg` | scikit-learn | L1/L2, with StandardScaler pipeline |
| `sgd` | scikit-learn | log-loss SGD, with StandardScaler pipeline |
| `xgb` | XGBoost | Optional; auto-detects CUDA |
| `mlp` | PyTorch | Multi-layer perceptron |
| `lstm` | PyTorch | Uni- or bidirectional, 1–2 layers |
| `gru` | PyTorch | Uni-directional, 1–2 layers |
| `cnn` | PyTorch | Multi-scale 1D convolution |
| `transformer` | PyTorch | Encoder-only with positional encoding |

---

## Backtesting Rules

The backtester in `backtester_engine.py` faithfully models A-share market constraints:

- **Long-only** — no short selling
- **T+1 approximation** — signal generated at close of day `t`; order executes at the **open price of day `t+1`**
- **Minimum lot size** — 100 shares
- **Transaction costs** — buy: 0.03% commission; sell: 0.03% commission + 0.10% stamp duty (configurable)
- **Hold-day counting** — in **trading days**, not calendar days
- **Confirm days** — signal must persist for N consecutive days before triggering
- **Take-profit / Stop-loss** — optional, threshold-based
- **Buy & Hold benchmark** — computed alongside the strategy for alpha estimation

---

## Project Structure

```
.
├── main_cli.py             Entry point — search + backtest + report
├── dpoint_updater.py       Retrain on new data and export Dpoint to Excel
│
├── data_loader.py          Excel loader with OHLCV validation
├── feature_dpoint.py       Feature engineering (all groups + TA indicators)
├── model_builder.py        sklearn model factory
├── dl_model_builder.py     PyTorch model factory (MLP/LSTM/GRU/CNN/Transformer)
│
├── search_engine.py        Random search: explore / exploit / pool-exploit rounds
├── trainer_optimizer.py    Public training API; full-sample final fit
├── splitter.py             Walk-forward splits + adaptive fold count
├── metrics.py              Geometric mean ratio; trade-count penalty
├── backtester_engine.py    A-share event-driven backtest engine
├── persistence.py          best_so_far.json / best_pool.json I/O
├── reporter.py             Excel workbook + run config JSON writer
└── constants.py            Global constants (penalty weights, filenames)
```

---

## Installation

### 1. Create the conda environment

```bash
conda create -n ashare_dpoint python=3.10
conda activate ashare_dpoint
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn joblib openpyxl xlsxwriter torch
# Optional: for XGBoost support
pip install xgboost
```

> GPU acceleration is detected automatically. If a CUDA-capable GPU is present, PyTorch models and XGBoost will use it.

### 3. Prepare your data

Create an Excel file with these columns (column names must match exactly):

| Column | Description |
|---|---|
| `date` | Trading date (any parseable format) |
| `open_qfq` | Adjusted open price |
| `high_qfq` | Adjusted high price |
| `low_qfq` | Adjusted low price |
| `close_qfq` | Adjusted close price |
| `volume` | Trading volume (shares) |
| `amount` | Turnover amount (yuan) |
| `turnover_rate` | Turnover rate (%) |

A minimum of ~300 trading days is recommended for stable ML training.

---

## Usage

### Run a new search

```bash
python main_cli.py --data_path /path/to/stock_data.xlsx --output_dir ./output --runs 200 --initial_cash 100000
```

Or set the data path via environment variable:

```bash
export ASHARE_DATA_PATH=/path/to/stock_data.xlsx
python main_cli.py --runs 200
```

### Understanding --mode and --seed

#### --mode

- **`first` (default)**: Start a completely new search. The system will randomly sample model configurations and evaluate them using walk-forward cross-validation. This is recommended for:
  - First-time runs on a new dataset
  - When you want to explore a fresh search space
  - When you want to change the search strategy

- **`continue`**: Resume from the best configuration found in previous runs. The system loads the best result from `best_so_far.json` and continues searching from there. This is recommended for:
  - Extending a previous search to find better configurations
  - Running more iterations when the previous search didn't converge
  - The search will still explore randomly but uses the best known result as a starting point

**Example workflow:**
```bash
# First run: start a new search with 200 iterations
python main_cli.py --data_path /path/to/stock_data.xlsx --runs 200

# Second run: continue from the best result found, run 100 more iterations
python main_cli.py --data_path /path/to/stock_data.xlsx --mode continue --runs 100
```

#### --seed

The random seed controls the reproducibility of the search. Different seeds will produce different search trajectories and potentially different final results.

- Using the **same seed** with the **same data** will always produce identical results
- Using **different seeds** allows you to explore different parts of the search space

**Example:**
```bash
# Run 1: use seed 42
python main_cli.py --data_path /path/to/stock_data.xlsx --runs 200 --seed 42

# Run 2: use seed 123 (different exploration path)
python main_cli.py --data_path /path/to/stock_data.xlsx --runs 200 --seed 123
```

> **Tip**: If you want more robust results, you can run multiple searches with different seeds and compare the best configurations.

### Continue from a previous run

```bash
python main_cli.py --data_path /path/to/stock_data.xlsx --output_dir ./output --mode continue --runs 100
```

### Update Dpoint with new market data

```bash
python dpoint_updater.py --output_dir ./output
```

The tool will interactively ask which run to use, then open a file picker for the new data file.

### All CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--data_path` | (env `ASHARE_DATA_PATH`) | Path to input Excel |
| `--output_dir` | `./output` | Directory for results |
| `--runs` | `200` | Number of search iterations |
| `--mode` | `first` | `first` (fresh) or `continue` |
| `--initial_cash` | `100000` | Starting capital (yuan) |
| `--n_folds` | `auto` | Walk-forward folds (0 = auto-detect) |
| `--n_jobs` | `1` | Parallel jobs (auto-limited when CUDA is active) |
| `--seed` | `42` | Random seed |
| `--eval_tickers` | `` | Comma-separated paths for cross-ticker evaluation |

---

## Output Files

Each run produces three files in `--output_dir`:

| File | Description |
|---|---|
| `run_NNN.xlsx` | Multi-sheet Excel workbook |
| `run_NNN_config.json` | Full configuration and metadata |
| `best_so_far.json` | Global best configuration across all runs |
| `best_pool.json` | Top-10 configurations pool |

### Excel sheets

| Sheet | Contents |
|---|---|
| **Trades** | Every trade: entry/exit date, price, PnL, return, status |
| **EquityCurve** | Daily equity, cash, market value, drawdown, Buy & Hold benchmark |
| **Config** | All feature / model / trade parameters for this run |
| **Log** | Data loader report, training summary, search log per iteration |
| **ModelParams** | Feature coefficients and scaler parameters (LogReg/SGD only) |

---

## Key Design Decisions

### Walk-forward validation
The optimizer evaluates each candidate using non-overlapping out-of-sample validation windows. The training set expands (expanding window), while each validation fold is strictly after the training data. The objective metric is the **geometric mean of per-fold equity ratios**, which naturally penalizes inconsistent or high-variance strategies.

### Trade-count penalty
A soft penalty discourages configurations that generate too few or too many trades per fold. This prevents the optimizer from converging on degenerate solutions (e.g., never trading or trading every day).

### Explore / exploit / pool-exploit
The random search runs in rounds. In each round, candidates are generated by one of three modes:
- **Explore** (~30%) — fully random sampling from the search space
- **Exploit** (~70%) — small perturbations around the current best (incumbent)
- **Pool-exploit** — random draw from the Top-K pool, avoiding single-point convergence

The incumbent is updated after every round, so each subsequent round's exploit candidates benefit from the latest improvements.

### Adaptive fold count
`recommend_n_folds()` automatically selects the number of walk-forward folds based on the available data length, targeting a minimum number of expected trades per fold for statistical reliability.

---

## Known Limitations

- **In-sample final equity curve** — The full-sample backtest shown in the Excel report trains and predicts on the same data. It will overstate real performance. Use the per-fold out-of-sample metrics in the Log sheet for honest evaluation.
- **No live trading integration** — This is a research tool. There is no order management, broker connectivity, or real-time data feed.
- **Stamp duty rate** — The default sell-side cost uses 0.10% stamp duty (pre-2023 rate). Pass `commission_rate_sell=0.0008` to use the post-August-2023 rate of 0.05%.
- **Cross-ticker generalization** — The `--eval_tickers` flag uses hyperparameter transfer (same config, retrain from scratch on new ticker). It does **not** transfer model weights.
