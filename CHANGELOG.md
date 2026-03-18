# Changelog

## [Ver3.0] - 2026-03-19

### 🚀 Major Refactor - Module Naming Unification

* **Unified module naming across codebase, README, CLI, and tests**
* Refactored core module names for clarity:

  * `data.py` → `data_loader.py` (Excel loader + walk-forward splits)
  * `training.py` → `trainer.py` (random search + calibration + persistence)
  * `evaluation.py` → `backtester.py` (backtest engine + metrics + regime analysis)
  * `reporting.py` → `reporter.py` (Excel/JSON/HTML reporting)
* Updated all imports in:

  * `main_cli.py`
  * `dpoint_updater.py`
  * All test files
* README Architecture Overview now shows flat module structure
* Removed all legacy module references from documentation

### 🧪 Test Quality Improvements

* Fixed fake tests using `try: ... except: pass; assert True` pattern
* All 110 tests now properly assert conditions
* Smoke tests now reliably catch regressions

### 📦 Previous Changes (Ver3.0 - 2026-03-18)

* Refactored project structure to improve modularity and maintainability
* Reorganized core modules:

  * `training.py` replaces `trainer_optimizer.py`
  * `evaluation.py` replaces `backtester_engine.py` and parts of `metrics.py`
  * `utils.py` consolidates run manifest utilities
* Removed legacy modules and unified logic into clearer functional boundaries

### 🧠 Core Logic Improvements

* Fixed execution feasibility logic:

  * Correct handling of volume-based liquidity filtering
  * Proper handling of ST filter toggle and listing days constraint
* Improved backtesting robustness:

  * Graceful handling of missing columns like `amount`
  * Better default fallbacks for incomplete datasets

### 📊 Metrics & Evaluation

* Reworked `trade_penalty` logic:

  * Penalty is now zero at target trades
  * Monotonic increase as deviation grows
* Improved consistency between evaluation metrics and test expectations

### 🧪 Testing & CI

* Updated all tests to align with new module structure
* Removed reliance on legacy module names (`trainer_optimizer`, `backtester_engine`, `run_manifest`)
* Fixed CI issues:

  * Removed invalid dependencies (e.g. `types-pandas`)
  * Fixed Python version compatibility
  * Prevented `conda` restart in CI environment

### ⚙️ CLI Improvements

* Fixed import-side effects in `main_cli.py`
* Ensured CLI logic only executes under `__main__`
* Improved compatibility with CI and testing environments

### 🧹 Cleanup

* Removed deprecated files:

  * `trainer_optimizer.py`
  * `backtester_engine.py`
  * `run_manifest.py`
* Simplified project structure and reduced duplication

---

## [Ver2.0] - Previous Release

* Initial structured version of the backtesting and training pipeline
* Introduced CI, testing framework, and modular components
* Included early implementations of execution constraints and evaluation metrics
