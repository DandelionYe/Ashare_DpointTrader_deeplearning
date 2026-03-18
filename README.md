# A-Share Dpoint ML Trading Signal System

<p align="center">
  <a href="README_Chinese_ver.md">
    <img src="https://img.shields.io/badge/文档 - 中文版-red?style=for-the-badge&logo=github" alt="中文文档"/>
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python"/>
  &nbsp;
  <img src="https://img.shields.io/badge/PyTorch-2.10.0-ee4c2c?style=for-the-badge&logo=pytorch"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Market-A--Share-gold?style=for-the-badge"/>
</p>

> **[📖 Click here to read the Chinese version → README_Chinese_ver.md](README_Chinese_ver.md)**

---

A machine-learning pipeline that generates next-day directional signals (**Dpoint**) for Chinese A-share stocks. The system searches for the best combination of feature engineering, model architecture, and trading parameters through walk-forward cross-validation, then outputs a full backtest report with an Excel workbook.

> ⚠️ **Disclaimer** — This project is for research and educational purposes only. Past backtest results, especially in-sample ones, do **not** guarantee future performance. Nothing here constitutes financial advice.

> 📝 **Changelog** — For detailed version history and changes, see [CHANGELOG.md](CHANGELOG.md).

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
main_cli.py          Orchestrates the end-to-end pipeline
data_loader.py      Load, validate, clean, and split A-share OHLCV data
feature_dpoint.py   Build feature matrix X and label y
models.py           sklearn + PyTorch model factory
trainer.py          Search, CV scoring, final training, calibration
backtester.py       Backtest engine, buy&hold baseline, risk metrics
reporter.py         Excel / JSON / HTML reporting
utils.py            Reproducibility, manifests, environment helpers
dpoint_updater.py   Retrain and export fresh Dpoint
rolling_trainer.py  Rolling or expanding retraining utility
compare_runs.py     Compare historical run outputs
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
| **Technical Indicators** | RSI, MACD (line + histogram), Bollinger Band Width, OBV | `use_ta_indicators` |

All features use only information up to and including day `t`. No forward leakage is introduced.

### Technical Indicators (P3-19)

When `use_ta_indicators=True`, the following classic technical indicators are added:

- **RSI (Relative Strength Index)**: Computed for each window in `ta_windows`, normalized to [0, 1]
- **MACD**: Fixed parameters (12, 26, 9), outputs normalized MACD line and histogram
- **Bollinger Band Width**: Computed for each window in `ta_windows`, measures relative volatility
- **OBV (On-Balance Volume)**: Rolling z-score normalized energy volume indicator

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

All PyTorch models support automatic mixed precision (AMP) training on CUDA devices and include early stopping with patience-based checkpointing.

---

## Backtesting Rules

The backtester in `evaluation.py` faithfully models A-share market constraints:

- **Long-only** — no short selling
- **T+1 approximation** — signal generated at close of day `t`; order executes at the **open price of day `t+1`**
- **Minimum lot size** — 100 shares
- **Transaction costs** — buy: 0.03% commission; sell: 0.03% commission + 0.10% stamp duty (configurable)
- **Hold-day counting** — in **trading days**, not calendar days
- **Confirm days** — signal must persist for N consecutive days before triggering
- **Take-profit / Stop-loss** — optional, threshold-based
- **Buy & Hold benchmark** — computed alongside the strategy for alpha estimation

### Execution Layer (P0/P1 Features)
- **Slippage model**: Fixed 20 bps (0.2%) slippage on execution price, with optional layered slippage for large orders
- **Limit-up/down handling**: Cannot buy on limit-up, cannot sell on limit-down
- **Suspension handling**: Orders rejected when stock is suspended
- **ST stock filtering**: Optional filtering of ST stocks
- **Listing days filter**: Minimum 60 trading days listing requirement
- **Volume filter**: Minimum daily turnover requirement (default 1M CNY)
- **Execution statistics**: Tracks order submission, fill, rejection reasons, and slippage costs

---

## Project Structure

```
.
├── main_cli.py             Entry point — search + backtest + report
├── dpoint_updater.py       Retrain on new data and export Dpoint to Excel
│
├── data_loader.py          Excel loader with OHLCV validation + Walk-forward splits
├── feature_dpoint.py       Feature engineering (all groups + TA indicators)
├── models.py               Model factory (sklearn + PyTorch unified)
│
├── trainer.py              Merged module: random search + calibration + explainer + persistence
├── backtester.py           Merged module: backtest engine + risk metrics + regime analysis
├── reporter.py             Excel workbook + JSON + HTML report
├── rolling_trainer.py      Rolling retrain scheduler (expanding/rolling window)
├── utils.py                Reproducibility tools + experiment manifest
├── constants.py            Global constants (penalty weights, filenames)
├── compare_runs.py         Compare multiple runs
│
├── tests/                  Automated test suite
│   ├── test_no_leakage.py    Temporal leakage tests
│   ├── test_splitter.py      Walk-forward splitter tests
│   ├── test_execution.py     Execution layer tests
│   ├── test_fee_lot.py       Fee and lot size tests
│   ├── test_metrics.py       Risk metrics tests
│   ├── test_smoke.py        Smoke tests
│   ├── test_cli.py          CLI argument tests
│   ├── test_reproducibility.py  Reproducibility tests
│   ├── test_rejection.py    Order rejection logic tests
│   └── conftest.py          Test fixtures
│
├── .github/workflows/
│   └── ci.yml              GitHub Actions CI (pytest, flake8, black, isort, mypy)
│
├── requirements.txt        Python dependencies
├── pytest.ini              Pytest configuration
└── README.md               This file
```

**Merged Modules (Ver3.0):**
- `trainer.py` = calibration.py + explainer.py + persistence.py + search_engine.py + trainer_optimizer.py
- `backtester.py` = backtester_engine.py + metrics.py + regime.py

---

## Installation

### 1. Create the conda environment

```bash
conda create -n ashare_dpoint python=3.11
conda activate ashare_dpoint
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy scikit-learn joblib openpyxl xlsxwriter torch xgboost tabulate
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
| `--n_folds` | `-1` | Walk-forward folds (-1 = auto-detect based on data size) |
| `--n_jobs` | `-1` | Parallel jobs (-1 = auto-detect; CUDA active → 1, else 4) |
| `--seed` | `42` | Random seed |
| `--eval_tickers` | `` | Comma-separated paths for cross-ticker evaluation |
| `--use_holdout` | `1` | Enable final holdout test (1=yes, 0=no) |
| `--holdout_ratio` | `0.15` | Holdout ratio (15% default) |
| `--use_embargo` | `0` | Enable embargo gap to prevent temporal leakage |
| `--embargo_days` | `5` | Embargo days between train/val |
| `--use_sensitivity_analysis` | `1` | Enable parameter sensitivity analysis |
| `--use_regime_analysis` | `0` | Enable market regime stratified analysis |
| `--experiment_dir` | `auto` | Experiment-specific output directory |
| `--replay` | `` | Replay from historical experiment |
| `--rolling_mode` | `` | Rolling retrain mode: expanding, rolling |
| `--rolling_window_length` | `None` | Rolling window length (days) |
| `--retrain_frequency` | `monthly` | Retrain frequency: daily, weekly, monthly, quarterly |
| `--export_lock` | `` | Export environment lock file |

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
| **EquityCurve** | Daily equity, cash, market value, drawdown, daily returns, Buy & Hold benchmark |
| **Config** | All feature / model / trade parameters for this run |
| **Log** | Data loader report, training summary, search log per iteration |
| **ModelParams** | Feature coefficients and scaler parameters (LogReg/SGD only) |
| **RiskMetrics** | Complete risk metrics: Sharpe, Sortino, Calmar, Max Drawdown, etc. |
| **RegimeAnalysis** | Regime-based performance breakdown (high/low volatility, trend/non-trend) |
| **RegimeStratified** | Detailed metrics by market regime |
| **TradeDistribution** | Trade distribution statistics (PnL, holding days) |
| **CalibrationMetrics** | Probability calibration results (Brier score, ECE, MCE) |
| **FeatureUsage** | Feature group usage frequency during search |
| **FeatureImportance** | Best model feature importance (tree, permutation, SHAP) |

---

## Key Design Decisions

### Walk-forward validation with Final Holdout
The optimizer evaluates each candidate using non-overlapping out-of-sample validation windows. The training set expands (expanding window), while each validation fold is strictly after the training data. The objective metric is the **geometric mean of per-fold equity ratios**, which naturally penalizes inconsistent or high-variance strategies.

**Multi-stage validation:**
1. **Search OOS**: Walk-forward cross-validation on search data
2. **Selection OOS**: Top-K candidates re-validated on search data
3. **Final Holdout OOS**: Best configuration evaluated on completely held-out data (15% by default)

### Anti-overfitting Mechanisms
- **Final Holdout Split**: 15% of data held out from search, never touched until final evaluation
- **Nested Walk-Forward**: Inner CV for model selection within each outer fold
- **Embargo Gap**: 5-day gap between training and validation to prevent look-ahead bias
- **Parameter Sensitivity Analysis**: Checks if optimal solution is "too sharp"
- **Multi-seed Stability**: Top-N candidates re-evaluated with multiple seeds
- **Penalty Terms**: Worst-fold penalty, fold-variance penalty, too-few-trades penalty

### Trade-count penalty
A soft penalty discourages configurations that generate too few or too many trades per fold. This prevents the optimizer from converging on degenerate solutions (e.g., never trading or trading every day).

### Explore / exploit / pool-exploit
The random search runs in rounds. In each round, candidates are generated by one of three modes:
- **Explore** (~30%) — fully random sampling from the search space
- **Exploit** (~70%) — small perturbations around the current best (incumbent)
- **Pool-exploit** — random draw from the Top-K pool, avoiding single-point convergence

The incumbent is updated after every round, so each subsequent round's exploit candidates benefit from the latest improvements.

### Adaptive fold count
`recommend_n_folds()` automatically selects the number of walk-forward folds based on the available data length, targeting a minimum number of expected trades per fold for statistical reliability. The function applies a 0.88 shrinkage coefficient to compensate for rows lost during feature engineering (rolling window NaNs).

### CUDA-aware parallelism
The system automatically detects CUDA availability and adjusts `n_jobs` accordingly:
- **CUDA available**: Forces `n_jobs=1` to avoid joblib fork conflicts with CUDA context
- **CPU only**: Uses `n_jobs=4` as a conservative default for safe sklearn parallelization

---

## Known Limitations

- **In-sample final equity curve** — The full-sample backtest shown in the Excel report trains and predicts on the same data. It will overstate real performance. Use the per-fold out-of-sample metrics in the Log sheet for honest evaluation.
- **No live trading integration** — This is a research tool. There is no order management, broker connectivity, or real-time data feed.
- **Stamp duty rate** — The default sell-side cost uses 0.10% stamp duty (pre-2023 rate). Pass `commission_rate_sell=0.0008` to use the post-August-2023 rate of 0.05%.
- **Cross-ticker generalization** — The `--eval_tickers` flag uses hyperparameter transfer (same config, retrain from scratch on new ticker). It does **not** transfer model weights.
- **Deep learning model randomness** — MLP/LSTM/GRU/CNN/Transformer models contain stochastic initialization; results may vary slightly between runs even with the same seed due to CUDA non-determinism.
- **CI/CD Testing** — The project includes automated tests via GitHub Actions (pytest, flake8, black, isort, mypy). Tests run on Python 3.11 and 3.12 only.

---

## Advanced Features

### Probability Calibration
The system supports probability calibration to improve prediction reliability:
- **Methods**: None, Platt Scaling, Isotonic Regression
- **Metrics**: Brier Score, Expected Calibration Error (ECE), Maximum Calibration Error (MCE)
- **Validation**: Calibration fitted only on validation set, not on test data

### Feature Importance & Explainability
- **Feature Usage Tracking**: Records which feature groups are used during search
- **Tree Model Importance**: Native feature importance for XGBoost
- **Permutation Importance**: Model-agnostic importance estimation
- **SHAP Values**: For tree models and linear models (if SHAP is installed)

### Market Regime Analysis
- **Trend Detection**: Based on MA crossover (short vs long MA)
- **Volatility Regime**: High/Low/Medium volatility based on rolling volatility
- **Combined Regime**: Trend × Volatility matrix
- **Stratified Metrics**: Performance metrics broken down by regime

### Rolling Retrain
- **Window Types**: Expanding window (grows over time) or Rolling window (fixed length)
- **Retrain Frequencies**: Daily, Weekly, Monthly, Quarterly
- **Snapshot Management**: Keeps track of model snapshots for rollback
- **Calibration Drift Monitoring**: Detects calibration degradation over time

### Reproducibility
- **Global Seed**: Sets seeds for Python, NumPy, PyTorch, TensorFlow
- **Environment Lock**: Exports `requirements-lock.txt` for environment reproducibility
- **Experiment Manifest**: Each run generates `manifest.json` with full metadata
- **CLI Replay**: Re-run experiments from historical manifest

### Execution Layer Enhancements
- **Layered Slippage**: Order-size dependent slippage (10/20/30 bps for small/medium/large orders)
- **Partial Fill Simulation**: Volume-constrained partial execution modeling
- **Position Sizing**: Dynamic position calculation based on cash and risk limits
- **Reject Reason Tracking**: Detailed order rejection categorization

### HTML Reports
- **Interactive Dashboard**: Equity curves, drawdown plots, trade distributions
- **Calibration Visualizations**: Calibration curves with bin statistics
- **Feature Importance Charts**: Bar charts for top features
- **Monthly/Yearly Return Tables**: Period-based performance breakdown
