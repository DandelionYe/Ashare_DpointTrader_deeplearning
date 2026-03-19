# test_report.py
"""
Tests for report generation functionality.

P2 修复：删除所有 try/except pass + assert True 模式，
改为真实断言，确保测试失败时能正确暴露问题。
"""
import os
import json
import pandas as pd
import pytest
from reporter import (
    save_run_outputs,
    find_latest_run,
    escape_excel_formulas,
)
from backtester import calculate_risk_metrics


class _DFPrepMixin:
    """Shared mixin for tests that need _prepare_df helper."""
    def _prepare_df(self, df):
        """Prepare dataframe with proper date column."""
        df = df.copy()
        if df.index.name == "date":
            df = df.reset_index()
        if "date" not in df.columns:
            df = df.reset_index()
            if "index" in df.columns:
                df = df.rename(columns={"index": "date"})
        if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])
        return df


class TestReportGeneration:
    """Test report generation functionality."""

    def _prepare_df(self, df):
        """Prepare dataframe with proper date column."""
        df = df.copy()
        if df.index.name == "date":
            df = df.reset_index()
        if "date" not in df.columns:
            df = df.reset_index()
            if "index" in df.columns:
                df = df.rename(columns={"index": "date"})
        if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])
        return df

    def test_save_run_outputs_basic(self, minimal_price_data, temp_output_dir):
        """Test basic report generation."""
        df = self._prepare_df(minimal_price_data)
        dpoint = pd.Series(0.5, index=df.index)

        from backtester import backtest_from_dpoint
        result = backtest_from_dpoint(
            df,
            dpoint,
            buy_threshold=0.6,
            sell_threshold=0.4,
            confirm_days=2,
        )

        log_notes = ["Test note 1", "Test note 2"]
        config = {
            "feature_config": {"return_lag_1": True},
            "model_config": {"model_type": "LogReg"},
            "trade_config": {
                "initial_cash": 100000.0,
                "buy_threshold": 0.6,
                "sell_threshold": 0.4,
                "confirm_days": 2,
                "min_hold_days": 1,
            },
        }

        excel_path, config_path, run_id = save_run_outputs(
            output_dir=temp_output_dir,
            df_clean=df,
            log_notes=log_notes,
            trades=result.trades,
            equity_curve=result.equity_curve,
            config=config,
            feature_meta={"test_meta": "value"},
            search_log=pd.DataFrame({"fold": [1], "metric": [0.5]}),
        )

        # 断言文件存在
        assert os.path.exists(excel_path), f"Excel file not created: {excel_path}"
        assert os.path.exists(config_path), f"Config file not created: {config_path}"
        assert run_id >= 1, f"Invalid run_id: {run_id}"

    def test_config_json_structure(self, minimal_price_data, temp_output_dir):
        """Test that config JSON has correct structure."""
        df = self._prepare_df(minimal_price_data)
        dpoint = pd.Series(0.5, index=df.index)

        from backtester import backtest_from_dpoint
        result = backtest_from_dpoint(
            df,
            dpoint,
            buy_threshold=0.6,
            sell_threshold=0.4,
            confirm_days=2,
        )

        config = {
            "feature_config": {},
            "model_config": {},
            "trade_config": {"initial_cash": 100000.0},
        }

        _, config_path, run_id = save_run_outputs(
            output_dir=temp_output_dir,
            df_clean=df,
            log_notes=[],
            trades=result.trades,
            equity_curve=result.equity_curve,
            config=config,
            feature_meta={},
            search_log=pd.DataFrame(),
        )

        with open(config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        # 断言必需字段存在
        assert "run_id" in saved_config, "Missing run_id in config"
        assert "created_at" in saved_config, "Missing created_at in config"
        assert "data_hash" in saved_config, "Missing data_hash in config"
        assert "best_config" in saved_config, "Missing best_config in config"
        assert "feature_meta" in saved_config, "Missing feature_meta in config"

    def test_escape_excel_formulas(self):
        """Test Excel formula escaping."""
        df = pd.DataFrame({
            "col1": ["=SUM(A1)", "normal", "+B2", "-C3", "@D4"],
            "col2": [1, 2, 3, 4, 5],
        })

        result = escape_excel_formulas(df)

        assert result["col1"].iloc[0].startswith("'"), "Formula '=SUM(A1)' should be escaped"
        assert result["col1"].iloc[1] == "normal", "Normal text should not be modified"
        assert result["col1"].iloc[2].startswith("'"), "Formula '+B2' should be escaped"
        assert result["col1"].iloc[3].startswith("'"), "Formula '-C3' should be escaped"
        assert result["col1"].iloc[4].startswith("'"), "Formula '@D4' should be escaped"


class TestFindLatestRun:
    """Test find_latest_run functionality."""

    def test_find_latest_run_empty(self, temp_output_dir):
        """Test finding latest run in empty directory."""
        result = find_latest_run(temp_output_dir)
        assert result is None, "Should return None for empty directory"

    def test_find_latest_run_exists(self, temp_output_dir):
        """Test finding latest run when runs exist."""
        config_path = os.path.join(temp_output_dir, "run_001_config.json")
        xlsx_path = os.path.join(temp_output_dir, "run_001.xlsx")

        with open(config_path, "w") as f:
            f.write('{"run_id": 1}')

        import pandas as pd
        pd.DataFrame({"a": [1]}).to_excel(xlsx_path, index=False)

        result = find_latest_run(temp_output_dir)

        assert result is not None, "Should find the run"
        assert result[0] == 1, f"Wrong run_id: expected 1, got {result[0]}"


class TestRiskMetricsInReport:
    """Test risk metrics in report."""

    def _prepare_df(self, df):
        """Prepare dataframe with proper date column."""
        df = df.copy()
        if df.index.name == "date":
            df = df.reset_index()
        if "date" not in df.columns:
            df = df.reset_index()
            if "index" in df.columns:
                df = df.rename(columns={"index": "date"})
        if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])
        return df

    def test_metrics_included(self, minimal_price_data, temp_output_dir):
        """Test that risk metrics are included in report."""
        df = self._prepare_df(minimal_price_data)
        dpoint = pd.Series(0.5, index=df.index)

        from backtester import backtest_from_dpoint
        result = backtest_from_dpoint(
            df,
            dpoint,
            buy_threshold=0.6,
            sell_threshold=0.4,
            confirm_days=2,
            initial_cash=100000.0,
        )

        risk_metrics = calculate_risk_metrics(
            equity_curve=result.equity_curve,
            trades=result.trades,
            initial_cash=100000.0,
        )

        # 断言指标存在
        assert "total_return" in risk_metrics, "Missing total_return"
        assert "max_drawdown" in risk_metrics, "Missing max_drawdown"

        _, _, run_id = save_run_outputs(
            output_dir=temp_output_dir,
            df_clean=df,
            log_notes=[],
            trades=result.trades,
            equity_curve=result.equity_curve,
            config={},
            feature_meta={},
            search_log=pd.DataFrame(),
        )

        files = os.listdir(temp_output_dir)
        xlsx_files = [f for f in files if f.endswith(".xlsx")]

        assert len(xlsx_files) > 0, "No Excel files created"


# =========================================================
# Phase 3: 新增回归测试 - Holdout 和 config JSON
# =========================================================

class TestHoldoutMetricHandling(_DFPrepMixin):
    """Test holdout metric handling in report generation."""

    def test_save_run_outputs_with_holdout_in_config_json(self, minimal_price_data, temp_output_dir):
        """Test that holdout_metric/holdout_equity are written to config JSON."""
        df = self._prepare_df(minimal_price_data)
        dpoint = pd.Series(0.5, index=df.index)

        from backtester import backtest_from_dpoint
        result = backtest_from_dpoint(
            df,
            dpoint,
            buy_threshold=0.6,
            sell_threshold=0.4,
            confirm_days=2,
        )

        config = {
            "feature_config": {},
            "model_config": {},
            "trade_config": {"initial_cash": 100000.0},
        }

        # 测试显式传入 holdout 参数
        _, config_path, run_id = save_run_outputs(
            output_dir=temp_output_dir,
            df_clean=df,
            log_notes=[],
            trades=result.trades,
            equity_curve=result.equity_curve,
            config=config,
            feature_meta={},
            search_log=pd.DataFrame(),
            holdout_metric=0.15,
            holdout_equity=115000.0,
            holdout_calibration_comparison={"brier_score_raw": 0.2, "ece_raw": 0.1},
        )

        # 断言 config JSON 包含 holdout 字段
        with open(config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        assert saved_config.get("holdout_metric") == 0.15, \
            "holdout_metric should be written to config JSON"
        assert saved_config.get("holdout_equity") == 115000.0, \
            "holdout_equity should be written to config JSON"
        assert saved_config.get("holdout_calibration_comparison") is not None, \
            "holdout_calibration_comparison should be written to config JSON"

    def test_save_run_outputs_calibration_metrics_sheet(self, minimal_price_data, temp_output_dir):
        """Test that CalibrationMetrics sheet includes holdout_calibration_comparison."""
        df = self._prepare_df(minimal_price_data)
        dpoint = pd.Series(0.5, index=df.index)

        from backtester import backtest_from_dpoint
        result = backtest_from_dpoint(
            df,
            dpoint,
            buy_threshold=0.6,
            sell_threshold=0.4,
            confirm_days=2,
        )

        config = {
            "feature_config": {},
            "model_config": {},
            "trade_config": {"initial_cash": 100000.0},
            "calibration_config": {"method": "platt"},
        }

        holdout_cal = {
            "calibration_method": "platt",
            "brier_score_raw": 0.25,
            "brier_score_calibrated": 0.20,
            "ece_raw": 0.05,
            "ece_calibrated": 0.03,
        }

        excel_path, _, _ = save_run_outputs(
            output_dir=temp_output_dir,
            df_clean=df,
            log_notes=[],
            trades=result.trades,
            equity_curve=result.equity_curve,
            config=config,
            feature_meta={},
            search_log=pd.DataFrame(),
            holdout_metric=0.15,
            holdout_equity=115000.0,
            holdout_calibration_comparison=holdout_cal,
        )

        # 读取 Excel 的 CalibrationMetrics sheet
        xls = pd.ExcelFile(excel_path)
        assert "CalibrationMetrics" in xls.sheet_names, \
            "CalibrationMetrics sheet should exist"

        cal_df = pd.read_excel(excel_path, sheet_name="CalibrationMetrics")
        
        # 断言包含 holdout 对比信息
        metrics = cal_df["metric"].tolist()
        assert any("holdout" in str(m) for m in metrics), \
            "CalibrationMetrics should include holdout comparison data"

    def test_save_run_outputs_without_holdout(self, minimal_price_data, temp_output_dir):
        """Test save_run_outputs without holdout parameters (default None)."""
        df = self._prepare_df(minimal_price_data)
        dpoint = pd.Series(0.5, index=df.index)

        from backtester import backtest_from_dpoint
        result = backtest_from_dpoint(
            df,
            dpoint,
            buy_threshold=0.6,
            sell_threshold=0.4,
            confirm_days=2,
        )

        config = {
            "feature_config": {},
            "model_config": {},
            "trade_config": {"initial_cash": 100000.0},
        }

        # 测试不传入 holdout 参数（默认 None）
        _, config_path, _ = save_run_outputs(
            output_dir=temp_output_dir,
            df_clean=df,
            log_notes=[],
            trades=result.trades,
            equity_curve=result.equity_curve,
            config=config,
            feature_meta={},
            search_log=pd.DataFrame(),
        )

        # 断言 config JSON 中 holdout 字段为 None 或空
        with open(config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        assert saved_config.get("holdout_metric") is None, \
            "holdout_metric should be None when not provided"
        assert saved_config.get("holdout_equity") is None, \
            "holdout_equity should be None when not provided"


class TestSplitModeInConfig(_DFPrepMixin):
    """Test split_mode is correctly written to config."""

    def test_split_mode_walkforward(self, minimal_price_data, temp_output_dir):
        """Test split_mode='walkforward' is saved."""
        df = self._prepare_df(minimal_price_data)
        dpoint = pd.Series(0.5, index=df.index)

        from backtester import backtest_from_dpoint
        result = backtest_from_dpoint(
            df,
            dpoint,
            buy_threshold=0.6,
            sell_threshold=0.4,
            confirm_days=2,
        )

        config = {
            "feature_config": {},
            "model_config": {},
            "trade_config": {"initial_cash": 100000.0},
            "split_mode": "walkforward",
        }

        _, config_path, _ = save_run_outputs(
            output_dir=temp_output_dir,
            df_clean=df,
            log_notes=[],
            trades=result.trades,
            equity_curve=result.equity_curve,
            config=config,
            feature_meta={},
            search_log=pd.DataFrame(),
        )

        with open(config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        assert saved_config.get("best_config", {}).get("split_mode") == "walkforward"
