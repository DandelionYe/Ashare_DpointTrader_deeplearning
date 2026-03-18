# test_reproducibility.py
"""
Tests for reproducibility.
Ensures deterministic results with fixed seeds.
"""
import numpy as np
import pandas as pd
import pytest
from repro import set_global_seed, get_git_commit_hash, get_package_versions
from backtester_engine import backtest_from_dpoint
from trainer_optimizer import train_final_model_and_dpoint
from feature_dpoint import build_features_and_labels


class TestReproducibility:
    """Test reproducibility with fixed seeds."""
    
    def test_numpy_reproducibility(self):
        """Test that NumPy produces same results with same seed."""
        set_global_seed(42)
        arr1 = np.random.randn(100)
        
        set_global_seed(42)
        arr2 = np.random.randn(100)
        
        assert np.allclose(arr1, arr2)
    
    def test_random_reproducibility(self):
        """Test that Python random produces same results with same seed."""
        import random
        
        set_global_seed(42)
        list1 = [random.random() for _ in range(10)]
        
        set_global_seed(42)
        list2 = [random.random() for _ in range(10)]
        
        assert list1 == list2
    
    def test_backtest_reproducibility(self):
        """Test that backtest produces same results with same seed."""
        np.random.seed(42)
        n = 100
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
        
        dpoint = pd.Series(np.random.uniform(0.3, 0.7, n), index=dates)
        
        set_global_seed(42)
        result1 = backtest_from_dpoint(
            df, dpoint,
            buy_threshold=0.6,
            sell_threshold=0.4,
            confirm_days=2,
            initial_cash=100000.0,
        )
        
        set_global_seed(42)
        result2 = backtest_from_dpoint(
            df, dpoint,
            buy_threshold=0.6,
            sell_threshold=0.4,
            confirm_days=2,
            initial_cash=100000.0,
        )
        
        if len(result1.trades) > 0 and len(result2.trades) > 0:
            assert result1.trades["buy_price"].iloc[0] == pytest.approx(
                result2.trades["buy_price"].iloc[0], rel=0.001
            )
    
    def test_git_hash_available(self):
        """Test that git commit hash can be retrieved."""
        git_hash = get_git_commit_hash()
        
        assert isinstance(git_hash, str)
        assert len(git_hash) > 0
    
    def test_package_versions_available(self):
        """Test that package versions can be retrieved."""
        versions = get_package_versions()
        
        assert "python" in versions
        assert isinstance(versions["python"], str)


class TestDeterministicTorch:
    """Test PyTorch determinism (if available)."""
    
    @pytest.fixture
    def torch_available(self):
        """Check if torch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def test_torch_seed(self, torch_available):
        """Test that torch seed can be set."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        
        result = set_global_seed(42)
        
        assert result.get("torch_deterministic") is True
    
    def test_torch_reproducibility(self, torch_available):
        """Test that torch produces same results with same seed."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        
        import torch
        
        set_global_seed(42)
        t1 = torch.randn(10)
        
        set_global_seed(42)
        t2 = torch.randn(10)
        
        assert torch.allclose(t1, t2)
