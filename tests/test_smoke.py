# test_smoke.py
"""
Smoke tests for main flow.
Ensures minimal configuration can run end-to-end.
"""
import numpy as np
import pandas as pd
import pytest
import os
import tempfile
from trainer import train_final_model_and_dpoint
from backtester import backtest_from_dpoint
from data_loader import walkforward_splits
from feature_dpoint import build_features_and_labels
from reporter import save_run_outputs
from models import make_model
from utils import set_global_seed, get_package_versions


@pytest.mark.smoke
class TestSmokeMinimal:
    """Smoke tests for minimal configuration."""
    
    def test_seed_setting(self):
        """Test that seed setting works."""
        result = set_global_seed(42)
        
        assert result["seed"] == 42
        assert "torch_deterministic" in result
    
    def test_package_versions(self):
        """Test that package versions can be retrieved."""
        versions = get_package_versions()
        
        assert "python" in versions
        assert "pandas" in versions
    
    def test_feature_engineering(self):
        """Test that feature engineering works."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

        df = pd.DataFrame({
            "date": dates,
            "close_qfq": 10 + np.cumsum(np.random.randn(n)),
            "open_qfq": 10 + np.cumsum(np.random.randn(n)),
            "high_qfq": 10 + np.cumsum(np.random.randn(n)),
            "low_qfq": 10 + np.cumsum(np.random.randn(n)),
            "volume": np.random.uniform(1e6, 1e7, n),
            "amount": np.random.uniform(1e7, 1e8, n),
            "turnover_rate": np.random.uniform(0.5, 5.0, n),
        })

        feature_config = {}

        X, y, meta = build_features_and_labels(df, feature_config)
        assert len(X) > 0
        assert len(y) > 0
        assert hasattr(meta, "feature_names") or isinstance(meta, dict)
    
    def test_splitter(self):
        """Test that splitter works."""
        X = pd.DataFrame({"x": range(100)}, index=range(100))
        y = pd.Series(range(100), index=range(100))
        
        splits = walkforward_splits(X, y, n_folds=2, min_rows=10)
        
        assert len(splits) >= 0
    
    def test_model_creation(self):
        """Test that model can be created."""
        config = {
            "model_type": "logreg",
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
        }
        model = make_model({"model_config": config}, seed=42)
        assert model is not None
    
    def test_backtest_basic(self):
        """Test that backtest runs with minimal data."""
        np.random.seed(42)
        n = 50
        dates = pd.date_range(start="2020-01-01", periods=n, freq="B")
        
        prices = 10 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "date": dates,
            "close_qfq": prices,
            "open_qfq": prices,
            "high_qfq": prices * 1.01,
            "low_qfq": prices * 0.99,
            "volume": np.random.uniform(1e6, 1e7, n),
        })
        
        dpoint = pd.Series(0.5, index=dates)
        
        result = backtest_from_dpoint(
            df,
            dpoint,
            buy_threshold=0.6,
            sell_threshold=0.4,
            confirm_days=2,
            initial_cash=100000.0,
        )
        
        assert result is not None
        assert hasattr(result, "trades")
        assert hasattr(result, "equity_curve")
    
    def test_full_pipeline_sklearn(self, tmp_path):
        """Test full pipeline with sklearn model."""
        np.random.seed(42)
        n = 150
        dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

        prices = 10 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "date": dates,
            "close_qfq": prices,
            "open_qfq": prices,
            "high_qfq": prices * 1.01,
            "low_qfq": prices * 0.99,
            "volume": np.random.uniform(1e6, 1e7, n),
            "amount": np.random.uniform(1e7, 1e8, n),
            "turnover_rate": np.random.uniform(0.5, 5.0, n),
        })

        feature_config = {}
        X, y, meta = build_features_and_labels(df, feature_config)

        config = {
            "feature_config": feature_config,
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

        dpoint, artifacts = train_final_model_and_dpoint(df, config, seed=42)
        assert "feature_meta" in artifacts
    
    def test_report_generation(self, minimal_price_data, temp_output_dir):
        """Test that report generation works."""
        np.random.seed(42)

        dpoint = pd.Series(0.5, index=minimal_price_data.index)

        result = backtest_from_dpoint(
            minimal_price_data,
            dpoint,
            buy_threshold=0.6,
            sell_threshold=0.4,
            confirm_days=2,
        )

        log_notes = ["Test log note"]

        excel_path, config_path, run_id = save_run_outputs(
            output_dir=temp_output_dir,
            df_clean=minimal_price_data,
            log_notes=log_notes,
            trades=result.trades,
            equity_curve=result.equity_curve,
            config={
                "feature_config": {},
                "model_config": {},
                "trade_config": {},
            },
            feature_meta={"test": "meta"},
            search_log=pd.DataFrame(),
        )

        assert os.path.exists(excel_path)
        assert os.path.exists(config_path)
        assert run_id >= 1


@pytest.mark.smoke
class TestSmokeCLI:
    """Smoke tests for CLI components."""

    def test_imports(self):
        """Test that all main modules can be imported."""
        import main_cli
        import reporter
        import trainer
        import backtester
        import data_loader
        import utils
        import models
        import feature_dpoint

        assert True
