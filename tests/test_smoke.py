# test_smoke.py
"""
Smoke tests for main flow.
Ensures minimal configuration can run end-to-end.
"""
import sys
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import real modules only
from trainer import train_final_model_and_dpoint
from backtester import backtest_from_dpoint
from data_loader import walkforward_splits, load_stock_excel
from feature_dpoint import build_features_and_labels
from reporter import save_run_outputs
from models import make_model
from utils import set_global_seed, get_package_versions


# =============================================================================
# Helper functions
# =============================================================================

def make_minimal_ohlcv_df(n=120, seed=42):
    """Create minimal OHLCV dataframe for smoke tests."""
    np.random.seed(seed)
    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")
    
    prices = 10 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "date": dates,
        "close_qfq": prices,
        "open_qfq": prices,
        "high_qfq": prices * 1.01,
        "low_qfq": prices * 0.99,
        "volume": np.random.uniform(1e6, 1e7, n),
        "amount": np.random.uniform(1e7, 1e8, n),
        "turnover_rate": np.random.uniform(0.5, 5.0, n),
    })


def minimal_config():
    """Return minimal valid config for training."""
    return {
        "feature_config": {},
        "model_config": {"model_type": "logreg", "C": 1.0, "penalty": "l2", "solver": "lbfgs"},
        "trade_config": {
            "initial_cash": 100000.0,
            "buy_threshold": 0.6,
            "sell_threshold": 0.4,
            "confirm_days": 2,
            "min_hold_days": 1,
            "max_hold_days": 10,
        },
    }


# =============================================================================
# Smoke Test 1: Import verification
# =============================================================================

@pytest.mark.smoke
def test_imports_smoke():
    """
    Verify that all key modules can be successfully imported.
    
    This test ensures:
    - No import errors from real module names
    - All dependencies are available
    """
    import main_cli
    import trainer
    import backtester
    import data_loader
    import reporter
    import utils
    import models
    import feature_dpoint
    
    # Verify modules are loaded (not None)
    assert main_cli is not None
    assert trainer is not None
    assert backtester is not None
    assert data_loader is not None
    assert reporter is not None
    assert utils is not None
    assert models is not None
    assert feature_dpoint is not None


# =============================================================================
# Smoke Test 2: Minimal train and backtest
# =============================================================================

@pytest.mark.smoke
def test_minimal_train_and_backtest_smoke():
    """
    Verify that minimal training and backtest pipeline runs successfully.
    
    This test:
    - Creates minimal OHLCV data
    - Calls train_final_model_and_dpoint
    - Calls backtest_from_dpoint
    - Asserts core outputs exist and are non-empty
    """
    df = make_minimal_ohlcv_df(n=150)
    config = minimal_config()
    
    # Train
    dpoint, artifacts = train_final_model_and_dpoint(df, config, seed=42)
    
    # Assert training outputs
    assert dpoint is not None, "dpoint should not be None"
    assert len(dpoint) > 0, "dpoint should have data"
    assert artifacts is not None, "artifacts should not be None"
    
    # Backtest
    bt_result = backtest_from_dpoint(
        df,
        dpoint,
        buy_threshold=0.6,
        sell_threshold=0.4,
        confirm_days=2,
        initial_cash=100000.0,
    )
    
    # Assert backtest outputs
    assert bt_result is not None, "backtest result should not be None"
    assert hasattr(bt_result, "equity_curve") or "equity_curve" in bt_result, \
        "backtest result should have equity_curve"
    
    # Check metrics exist
    if hasattr(bt_result, "metrics"):
        assert bt_result.metrics is not None, "metrics should exist"
    elif isinstance(bt_result, dict) and "metrics" in bt_result:
        assert bt_result["metrics"] is not None, "metrics should exist"


# =============================================================================
# Smoke Test 3: Report generation
# =============================================================================

@pytest.mark.smoke
def test_report_generation_smoke(temp_output_dir):
    """
    Verify that reports can be generated and written to disk.
    
    This test:
    - Creates minimal backtest results
    - Calls save_run_outputs
    - Verifies output files exist and are non-empty
    """
    df = make_minimal_ohlcv_df(n=120)
    dpoint = pd.Series(0.5, index=df.index)
    
    bt_result = backtest_from_dpoint(
        df,
        dpoint,
        buy_threshold=0.6,
        sell_threshold=0.4,
        confirm_days=2,
        initial_cash=100000.0,
    )
    
    excel_path, config_path, run_id = save_run_outputs(
        output_dir=temp_output_dir,
        df_clean=df,
        log_notes=["Smoke test"],
        trades=bt_result.trades,
        equity_curve=bt_result.equity_curve,
        config=minimal_config(),
        feature_meta={"test": "smoke"},
        search_log=pd.DataFrame(),
    )
    
    # Assert files exist
    assert os.path.exists(excel_path), f"Excel report should exist: {excel_path}"
    assert os.path.exists(config_path), f"Config should exist: {config_path}"
    
    # Assert files are non-empty
    assert os.path.getsize(excel_path) > 0, "Excel report should not be empty"
    assert os.path.getsize(config_path) > 0, "Config should not be empty"
    
    # Assert run_id is valid
    assert run_id >= 1, "run_id should be positive"


# =============================================================================
# Smoke Test 4: End-to-end CLI
# =============================================================================

@pytest.mark.smoke
@pytest.mark.slow
def test_main_cli_end_to_end_smoke(tmp_path):
    """
    Verify that main_cli.py can run end-to-end from command line.
    
    This test:
    - Creates minimal Excel data file
    - Runs main_cli.py via subprocess
    - Sets CI=true and SKIP_CONDA=1 to bypass environment checks
    - Asserts returncode == 0 and output files are generated
    """
    data_path = tmp_path / "mini_data.xlsx"
    out_dir = tmp_path / "output"
    
    # Create minimal data
    df = make_minimal_ohlcv_df(n=120)
    df.to_excel(data_path, index=False)
    
    # Run CLI with minimal settings
    env = {**os.environ, "CI": "true", "SKIP_CONDA": "1"}
    
    result = subprocess.run(
        [
            sys.executable,
            "main_cli.py",
            "--data_path", str(data_path),
            "--runs", "1",
            "--seed", "42",
            "--output_dir", str(out_dir),
            "--n_folds", "2",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    
    # Assert success
    assert result.returncode == 0, f"CLI failed with:\n{result.stderr}\n\nstdout:\n{result.stdout}"
    
    # Assert output directory exists
    assert out_dir.exists(), f"Output directory should exist: {out_dir}"
    
    # Assert at least one result file was generated
    xlsx_files = list(out_dir.rglob("*.xlsx"))
    json_files = list(out_dir.rglob("*.json"))
    assert len(xlsx_files) > 0 or len(json_files) > 0, \
        "Should generate at least one output file (.xlsx or .json)"
