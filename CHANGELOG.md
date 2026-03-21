## [Ver3.5] - 2026-03-21

### Structural cleanup and module renaming
- Unified core module names for better clarity and maintainability:
  - `data.py` → `data_loader.py`
  - `training.py` → `trainer.py`
  - `evaluation.py` → `backtester.py`
  - `reporting.py` → `reporter.py`
- Updated module references across the main pipeline, tests, and documentation to match the current repository layout.

### CLI and conda environment handling
- Refined conda environment behavior:
  - the CLI no longer auto-relaunches by default;
  - relaunch only happens when `--use-conda-env` is explicitly provided;
  - added `--target-conda-env` for warning and expectation handling.
- Improved safety and control of environment switching in IDE, CI, and scheduled execution scenarios.

### Expanded test coverage
- Added coverage for:
  - backtester market-state behavior,
  - conda environment handling,
  - optional torch runtime,
  - trainer calibration,
  - trainer split modes.
- Strengthened regression coverage across training, backtesting, runtime environment, and compatibility behaviors.

### Dependencies and CI
- Added `requirements-dev.txt` to separate development/test dependencies from runtime dependencies.
- Updated CI to install from `requirements-dev.txt` for more consistent development and automation environments.

### Documentation
- Updated both README files to reflect the current repository structure, module naming, and test layout.
- Cleaned up version notes to make future tag-based release tracking clearer.