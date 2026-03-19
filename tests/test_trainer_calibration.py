# test_trainer_calibration.py
"""
Phase 3: 回归测试 - Trainer 校准逻辑（测试生产主链）

测试目标：
1. _eval_candidate 尊重 use_for_threshold=False（交易用 raw）
2. _eval_candidate 尊重 use_for_threshold=True（交易用 calibrated）
3. _eval_on_holdout 校准拟合使用 validation 预测，且汇总逻辑正确
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from trainer import (
    _eval_candidate,
    _eval_on_holdout,
    build_features_and_labels,
    ProbabilityCalibrator,
)


class TestEvalCandidateCalibration:
    """Test _eval_candidate respects calibration config."""

    def _create_test_data(self, n_samples=150):
        """创建最小测试数据。"""
        np.random.seed(42)
        return pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=n_samples),
            "open_qfq": 10.0 + np.random.rand(n_samples),
            "high_qfq": 11.0 + np.random.rand(n_samples),
            "low_qfq": 9.0 + np.random.rand(n_samples),
            "close_qfq": 10.5 + np.random.rand(n_samples),
            "volume": 1000000 + np.random.rand(n_samples) * 100000,
            "amount": 10000000 + np.random.rand(n_samples) * 1000000,
            "turnover_rate": 0.01 + np.random.rand(n_samples) * 0.01,
        }).set_index("date")

    def _create_base_candidate(self, use_for_threshold):
        """创建基础候选配置。"""
        return {
            "feature_config": {
                "windows": [5, 10],
                "use_momentum": True,
                "use_volatility": False,
                "use_volume": False,
                "use_candle": False,
                "use_turnover": False,
                "vol_metric": "std",
                "liq_transform": "ratio",
            },
            "model_config": {
                "model_type": "logreg",
                "penalty": "l2",
                "C": 1.0,
            },
            "trade_config": {
                "initial_cash": 100000.0,
                "buy_threshold": 0.55,
                "sell_threshold": 0.45,
                "confirm_days": 1,
                "min_hold_days": 1,
                "max_hold_days": 20,
            },
            "calibration_config": {
                "method": "platt",
                "use_for_threshold": use_for_threshold,
            },
            "candidate_seed": 42,
        }

    def test_eval_candidate_respects_use_for_threshold_false(self):
        """
        验证 use_for_threshold=False 时，backtest_fold_stats 收到的是 raw predictions。
        """
        df_clean = self._create_test_data()
        candidate = self._create_base_candidate(use_for_threshold=False)

        # 记录 transform 输入和 backtest_fold_stats 收到的 dpoint_val
        transform_inputs = []
        received_dpoint = []

        def mock_transform(self, y_prob):
            arr = np.asarray(y_prob, dtype=float)
            transform_inputs.append(arr.copy())
            # 返回明显不同的值（+0.12345），方便验证
            return np.clip(arr + 0.12345, 0.0, 1.0)

        def mock_backtest_fold_stats(df_full, X_val, dpoint_val, trade_cfg):
            received_dpoint.append(np.asarray(dpoint_val, dtype=float).copy())
            return {
                "equity_end": 110000.0,
                "n_closed": 5,
                "n_total": 5,
            }

        with patch('trainer.backtest_fold_stats', side_effect=mock_backtest_fold_stats):
            with patch('trainer.walkforward_splits') as mock_splits:
                with patch.object(ProbabilityCalibrator, 'transform', new=mock_transform):
                    X, y, meta = build_features_and_labels(df_clean, candidate["feature_config"])
                    if X is None or len(X) < 50:
                        pytest.skip("Feature engineering failed or too few samples")

                    split_point = len(X) // 2
                    mock_splits.return_value = [
                        ((X.iloc[:split_point], y.iloc[:split_point]),
                         (X.iloc[split_point:], y.iloc[split_point:]))
                    ]

                    metric, equity, info, fold_details = _eval_candidate(
                        candidate, df_clean, max_features=100, n_folds=1,
                        train_start_ratio=0.5, wf_min_rows=20,
                        computed_feats=(X, y, meta),
                        use_embargo=False, embargo_days=0, use_nested_wf=False,
                    )

        # 断言：transform 被调用了
        assert len(transform_inputs) > 0, "ProbabilityCalibrator.transform should be called"
        assert len(received_dpoint) > 0, "backtest_fold_stats should be called"

        # 核心断言：use_for_threshold=False 时，交易用 raw
        raw_pred = transform_inputs[0]
        expected_calibrated = np.clip(raw_pred + 0.12345, 0.0, 1.0)
        actual_trade_pred = received_dpoint[0]

        # 断言交易用的是 raw，不是 calibrated
        np.testing.assert_allclose(actual_trade_pred, raw_pred, rtol=1e-7, atol=1e-7)
        assert not np.allclose(actual_trade_pred, expected_calibrated, rtol=1e-7, atol=1e-7), \
            "Should use raw predictions when use_for_threshold=False"

        # 断言 calibration_summary 存在
        assert "calibration_summary" in info
        cal_summary = info["calibration_summary"]
        assert cal_summary.get("calibration_method") == "platt"
        assert "brier_score_raw" in cal_summary or len(cal_summary) == 1  # 至少有 method 或有指标

    def test_eval_candidate_respects_use_for_threshold_true(self):
        """
        验证 use_for_threshold=True 时，backtest_fold_stats 收到的是 calibrated predictions。
        """
        df_clean = self._create_test_data()
        candidate = self._create_base_candidate(use_for_threshold=True)

        # 记录 transform 输入和 backtest_fold_stats 收到的 dpoint_val
        transform_inputs = []
        received_dpoint = []

        def mock_transform(self, y_prob):
            arr = np.asarray(y_prob, dtype=float)
            transform_inputs.append(arr.copy())
            # 返回明显不同的值（+0.12345），方便验证
            return np.clip(arr + 0.12345, 0.0, 1.0)

        def mock_backtest_fold_stats(df_full, X_val, dpoint_val, trade_cfg):
            received_dpoint.append(np.asarray(dpoint_val, dtype=float).copy())
            return {
                "equity_end": 110000.0,
                "n_closed": 5,
                "n_total": 5,
            }

        with patch('trainer.backtest_fold_stats', side_effect=mock_backtest_fold_stats):
            with patch('trainer.walkforward_splits') as mock_splits:
                with patch.object(ProbabilityCalibrator, 'transform', new=mock_transform):
                    X, y, meta = build_features_and_labels(df_clean, candidate["feature_config"])
                    if X is None or len(X) < 50:
                        pytest.skip("Feature engineering failed or too few samples")

                    split_point = len(X) // 2
                    mock_splits.return_value = [
                        ((X.iloc[:split_point], y.iloc[:split_point]),
                         (X.iloc[split_point:], y.iloc[split_point:]))
                    ]

                    metric, equity, info, fold_details = _eval_candidate(
                        candidate, df_clean, max_features=100, n_folds=1,
                        train_start_ratio=0.5, wf_min_rows=20,
                        computed_feats=(X, y, meta),
                        use_embargo=False, embargo_days=0, use_nested_wf=False,
                    )

        # 断言：transform 被调用了
        assert len(transform_inputs) > 0, "ProbabilityCalibrator.transform should be called"
        assert len(received_dpoint) > 0, "backtest_fold_stats should be called"

        # 核心断言：use_for_threshold=True 时，交易用 calibrated
        raw_pred = transform_inputs[0]
        expected_calibrated = np.clip(raw_pred + 0.12345, 0.0, 1.0)
        actual_trade_pred = received_dpoint[0]

        # 断言交易用的是 calibrated，不是 raw
        np.testing.assert_allclose(actual_trade_pred, expected_calibrated, rtol=1e-7, atol=1e-7)
        assert not np.allclose(actual_trade_pred, raw_pred, rtol=1e-7, atol=1e-7), \
            "Should use calibrated predictions when use_for_threshold=True"

        # 断言 calibration_summary 存在
        assert "calibration_summary" in info
        cal_summary = info["calibration_summary"]
        assert cal_summary.get("calibration_method") == "platt"


class TestEvalOnHoldoutCalibration:
    """Test _eval_on_holdout calibration uses validation set for fitting."""

    def _create_test_data(self, n_search=150, n_holdout=50):
        """创建搜索集和 holdout 集。"""
        np.random.seed(42)
        search_df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=n_search),
            "open_qfq": 10.0 + np.random.rand(n_search),
            "high_qfq": 11.0 + np.random.rand(n_search),
            "low_qfq": 9.0 + np.random.rand(n_search),
            "close_qfq": 10.5 + np.random.rand(n_search),
            "volume": 1000000 + np.random.rand(n_search) * 100000,
            "amount": 10000000 + np.random.rand(n_search) * 1000000,
            "turnover_rate": 0.01 + np.random.rand(n_search) * 0.01,
        }).set_index("date")

        holdout_df = pd.DataFrame({
            "date": pd.date_range("2023-06-01", periods=n_holdout),
            "open_qfq": 10.0 + np.random.rand(n_holdout),
            "high_qfq": 11.0 + np.random.rand(n_holdout),
            "low_qfq": 9.0 + np.random.rand(n_holdout),
            "close_qfq": 10.5 + np.random.rand(n_holdout),
            "volume": 1000000 + np.random.rand(n_holdout) * 100000,
            "amount": 10000000 + np.random.rand(n_holdout) * 1000000,
            "turnover_rate": 0.01 + np.random.rand(n_holdout) * 0.01,
        }).set_index("date")

        return search_df, holdout_df

    def _create_base_candidate(self):
        """创建基础候选配置。"""
        return {
            "feature_config": {
                "windows": [5, 10],
                "use_momentum": True,
                "use_volatility": False,
                "use_volume": False,
                "use_candle": False,
                "use_turnover": False,
                "vol_metric": "std",
                "liq_transform": "ratio",
            },
            "model_config": {
                "model_type": "logreg",
                "penalty": "l2",
                "C": 1.0,
            },
            "trade_config": {
                "initial_cash": 100000.0,
                "buy_threshold": 0.55,
                "sell_threshold": 0.45,
                "confirm_days": 1,
                "min_hold_days": 1,
                "max_hold_days": 20,
            },
            "calibration_config": {
                "method": "platt",
                "use_for_threshold": True,
            },
            "candidate_seed": 42,
        }

    def test_holdout_calibration_uses_validation_prediction_length(self):
        """
        断言 _eval_on_holdout 中 calibrator.fit() 收到的是 validation 集长度，
        且 holdout_calibration_comparison 包含正确的汇总指标。
        """
        search_df, holdout_df = self._create_test_data()
        candidate = self._create_base_candidate()

        # 记录 calibrator.fit() 的调用参数
        # 在 patch 前保存原始方法
        original_fit = ProbabilityCalibrator.fit
        fit_calls = []

        def mock_fit(self, y_true, y_prob):
            fit_calls.append({
                "y_true_len": len(y_true),
                "y_prob_len": len(y_prob),
            })
            # 调用原始 fit 方法
            return original_fit(self, y_true, y_prob)

        with patch.object(ProbabilityCalibrator, "fit", new=mock_fit):
            with patch('trainer.backtest_fold_stats') as mock_backtest:
                mock_backtest.return_value = {
                    "equity_end": 110000.0,
                    "n_closed": 5,
                    "n_total": 5,
                }

                with patch('trainer.walkforward_splits') as mock_splits:
                    X_search, y_search, meta = build_features_and_labels(
                        search_df, candidate["feature_config"]
                    )
                    if X_search is None or len(X_search) < 50:
                        pytest.skip("Feature engineering failed or too few samples")

                    split_point = len(X_search) // 2
                    mock_splits.return_value = [
                        ((X_search.iloc[:split_point], y_search.iloc[:split_point]),
                         (X_search.iloc[split_point:], y_search.iloc[split_point:]))
                    ]

                    metric, equity, info, fold_details = _eval_on_holdout(
                        candidate, search_df, holdout_df,
                        max_features=100, n_folds=1,
                        train_start_ratio=0.5, wf_min_rows=20,
                        computed_feats=(X_search, y_search, meta),
                        use_embargo=False, embargo_days=0, use_nested_wf=False,
                    )

        # 断言：fit() 被调用了
        assert len(fit_calls) > 0, "ProbabilityCalibrator.fit should be called"

        # 断言：fit() 收到的长度等于 validation 集长度
        expected_val_len = len(X_search) - split_point
        for call_info in fit_calls:
            assert call_info["y_true_len"] == expected_val_len, \
                f"fit() should receive y_true of length {expected_val_len} (validation set), got {call_info['y_true_len']}"
            assert call_info["y_prob_len"] == expected_val_len, \
                f"fit() should receive y_prob of length {expected_val_len} (dp_va_raw), got {call_info['y_prob_len']}"

        # 断言：holdout_calibration_comparison 存在且包含正确字段
        assert "holdout_calibration_comparison" in info
        holdout_cal = info["holdout_calibration_comparison"]
        assert holdout_cal.get("calibration_method") == "platt"
        assert holdout_cal.get("use_for_threshold") == True
        # 由于我们按 fold 聚合指标，这里应该有均值指标
        assert "brier_score_raw" in holdout_cal, "Should have brier_score_raw"
        assert "brier_score_calibrated" in holdout_cal, "Should have brier_score_calibrated"
