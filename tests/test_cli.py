# test_cli.py
"""
Tests for CLI parameter parsing and replay functionality.
"""
import argparse
import json
import os
import sys
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


class TestCLIArguments:
    """Test CLI argument parsing."""
    
    def test_basic_arguments(self):
        """Test that basic arguments are parsed correctly."""
        test_args = [
            "main_cli.py",
            "--data_path", "test.xlsx",
            "--seed", "123",
            "--runs", "50",
        ]
        
        with patch.object(sys, "argv", test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument("--data_path", type=str)
            parser.add_argument("--seed", type=int, default=42)
            parser.add_argument("--runs", type=int, default=100)
            parser.add_argument("--output_dir", type=str, default="./output")
            parser.add_argument("--mode", choices=["first", "continue"], default="first")
            
            args = parser.parse_args()
            
            assert args.data_path == "test.xlsx"
            assert args.seed == 123
            assert args.runs == 50
    
    def test_replay_argument(self):
        """Test that replay argument is parsed."""
        test_args = [
            "main_cli.py",
            "--replay", "latest",
            "--data_path", "test.xlsx",
        ]
        
        with patch.object(sys, "argv", test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument("--replay", type=str, default="")
            parser.add_argument("--data_path", type=str)
            
            args = parser.parse_args()
            
            assert args.replay == "latest"
    
    def test_experiment_dir_argument(self):
        """Test that experiment_dir argument is parsed."""
        test_args = [
            "main_cli.py",
            "--experiment_dir", "./exp_test",
            "--data_path", "test.xlsx",
        ]
        
        with patch.object(sys, "argv", test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument("--experiment_dir", type=str, default=None)
            parser.add_argument("--data_path", type=str)
            
            args = parser.parse_args()
            
            assert args.experiment_dir == "./exp_test"
    
    def test_export_lock_argument(self):
        """Test that export_lock argument is parsed."""
        test_args = [
            "main_cli.py",
            "--export_lock", "requirements-lock.txt",
        ]
        
        with patch.object(sys, "argv", test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument("--export_lock", type=str, default="")
            
            args = parser.parse_args()
            
            assert args.export_lock == "requirements-lock.txt"


class TestRunManifest:
    """Test run_manifest module functionality."""
    
    def test_get_ticker_list_from_dataframe(self):
        """Test ticker extraction from DataFrame."""
        from run_manifest import get_ticker_list
        
        df = pd.DataFrame({
            "ticker": ["600000", "600001", "600000"],
            "close": [10, 11, 12],
        })
        
        tickers = get_ticker_list(df, "test.xlsx")
        
        assert len(tickers) == 2
        assert "600000" in tickers
    
    def test_get_ticker_list_from_filename(self):
        """Test ticker extraction from filename."""
        from run_manifest import get_ticker_list
        
        df = pd.DataFrame({"close": [10, 11, 12]})
        
        tickers = get_ticker_list(df, "600000_5Y.xlsx")
        
        assert "600000_5Y" in tickers
    
    def test_create_manifest(self, tmp_path):
        """Test manifest creation."""
        from run_manifest import create_manifest
        import os
        exp_dir = str(tmp_path / "exp_001")
        os.makedirs(exp_dir, exist_ok=True)
        
        manifest = create_manifest(
            experiment_dir=exp_dir,
            run_id=1,
            timestamp="2024-01-01T00:00:00",
            git_commit_hash="abc123",
            package_versions={"pandas": "2.0.0"},
            seed=42,
            data_info={"data_path": "test.xlsx", "data_hash": "xyz", "n_rows": 100},
            cli_args={"runs": 50},
        )
        
        assert manifest["run_id"] == 1
        assert manifest["experiment_id"] == 1
        assert manifest["seed"] == 42
        assert manifest["git_commit_hash"] == "abc123"
        
        manifest_path = tmp_path / "exp_001" / "manifest.json"
        assert manifest_path.exists()
    
    def test_load_manifest(self, tmp_path):
        """Test manifest loading."""
        from run_manifest import create_manifest, load_manifest
        import os
        exp_dir = str(tmp_path / "exp_001")
        os.makedirs(exp_dir, exist_ok=True)
        
        create_manifest(
            experiment_dir=exp_dir,
            run_id=1,
            timestamp="2024-01-01T00:00:00",
            git_commit_hash="abc123",
            package_versions={"pandas": "2.0.0"},
            seed=42,
            data_info={},
            cli_args={},
        )
        
        loaded = load_manifest(exp_dir)
        
        assert loaded is not None
        assert loaded["run_id"] == 1
    
    def test_list_experiments(self, tmp_path):
        """Test listing experiments."""
        from run_manifest import create_manifest, list_experiments
        import os
        exp_dir = str(tmp_path / "exp_001")
        os.makedirs(exp_dir, exist_ok=True)
        
        create_manifest(
            experiment_dir=exp_dir,
            run_id=1,
            timestamp="2024-01-01T00:00:00",
            git_commit_hash="abc123",
            package_versions={},
            seed=42,
            data_info={},
            cli_args={},
        )
        
        experiments = list_experiments(str(tmp_path))
        
        assert len(experiments) == 1
        assert experiments[0]["experiment_id"] == 1


class TestCompareRuns:
    """Test compare_runs functionality."""
    
    def test_compare_configs(self):
        """Test config comparison."""
        from compare_runs import compare_configs
        
        config1 = {
            "best_config": {
                "model_config": {"model_type": "LogReg"},
                "feature_config": {"return_lag_1": True},
            }
        }
        config2 = {
            "best_config": {
                "model_config": {"model_type": "XGBoost"},
                "feature_config": {"return_lag_1": True},
            }
        }
        
        diff = compare_configs(config1, config2)
        
        assert len(diff) > 0
        assert any("model_type" in str(d) for d in diff)
    
    def test_load_experiment_data(self, tmp_path):
        """Test loading experiment data."""
        from run_manifest import create_manifest
        from compare_runs import load_experiment_data
        
        exp_dir = tmp_path / "exp_001"
        exp_dir.mkdir()
        
        create_manifest(
            experiment_dir=str(exp_dir),
            run_id=1,
            timestamp="2024-01-01T00:00:00",
            git_commit_hash="abc123",
            package_versions={},
            seed=42,
            data_info={},
            cli_args={},
        )
        
        data = load_experiment_data(str(exp_dir))
        
        assert data is not None
        assert "manifest" in data
