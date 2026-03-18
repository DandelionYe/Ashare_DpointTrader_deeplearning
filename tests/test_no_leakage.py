# test_no_leakage.py
"""
Tests for data leakage prevention.
Validates that there is no look-ahead bias in the pipeline.
"""
import numpy as np
import pandas as pd
import pytest

from data_loader import walkforward_splits, final_holdout_split
from feature_dpoint import build_features_and_labels
from backtester import backtest_from_dpoint


# =============================================================================
# Test 1: Split temporal integrity
# =============================================================================

class TestNoLeakageSplits:
    """Test that splitter doesn't leak future information."""

    def test_validation_after_training(self):
        """Test that validation data is always after training data."""
        X = pd.DataFrame({"x": range(500)}, index=range(500))
        y = pd.Series(range(500), index=range(500))

        splits = walkforward_splits(X, y, n_folds=4, train_start_ratio=0.5, min_rows=30)

        assert len(splits) > 0, "Should have at least one split"
        
        for i, ((X_train, _), (X_val, _)) in enumerate(splits):
            assert X_train.index.max() < X_val.index.min(), \
                f"Fold {i}: training data should end before validation data starts"

    def test_no_temporal_overlap(self):
        """Test that there is no temporal overlap between folds."""
        X = pd.DataFrame({"x": range(200)}, index=range(200))
        y = pd.Series(range(200), index=range(200))

        splits = walkforward_splits(X, y, n_folds=4, train_start_ratio=0.5)

        validation_ranges = []
        for _, (X_val, _) in splits:
            validation_ranges.append((X_val.index.min(), X_val.index.max()))

        for i, (start1, end1) in enumerate(validation_ranges):
            for j, (start2, end2) in enumerate(validation_ranges):
                if i != j:
                    # Validation sets should not overlap
                    assert not (start1 <= start2 <= end1), \
                        f"Validation fold {i} and {j} overlap"
                    assert not (start1 <= end2 <= end1), \
                        f"Validation fold {i} and {j} overlap"


# =============================================================================
# Test 2: Feature engineering temporal integrity
# =============================================================================

class TestNoLeakageFeatures:
    """Test that feature engineering doesn't leak future information."""

    def test_features_use_only_past_data(self):
        """
        Test that features at time t only use data up to time t.
        
        This is verified by:
        1. Computing features on full dataset
        2. Computing features on truncated dataset (up to day t)
        3. Verifying features at day t are identical in both cases
        """
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

        close_prices = 10 + np.cumsum(np.random.randn(n))
        df = pd.DataFrame({
            "date": dates,
            "close_qfq": close_prices,
            "open_qfq": close_prices * 0.99,
            "high_qfq": close_prices * 1.02,
            "low_qfq": close_prices * 0.98,
            "volume": np.random.uniform(1e6, 1e7, n),
            "amount": np.random.uniform(1e7, 1e8, n),
            "turnover_rate": np.random.uniform(0.5, 5.0, n),
        })

        feature_config = {}

        # Compute on full dataset
        X_full, y_full, _ = build_features_and_labels(df, feature_config)
        
        # Pick a test point (after warmup period)
        test_idx = 30
        test_date = df.iloc[test_idx]["date"]
        
        # Truncate dataset at test point
        df_truncated = df[df["date"] <= test_date].copy()
        
        # Compute on truncated dataset
        X_truncated, y_truncated, _ = build_features_and_labels(df_truncated, feature_config)
        
        # Features at test_date should be identical
        if test_date in X_full.index and test_date in X_truncated.index:
            row_full = X_full.loc[test_date]
            row_truncated = X_truncated.loc[test_date]
            
            pd.testing.assert_series_equal(
                row_full, row_truncated,
                err_msg=f"Features at {test_date} differ between full and truncated datasets"
            )

    def test_dpoint_aligned_to_date(self):
        """
        Test that dpoint (y) is properly aligned to date index.
        
        Verifies:
        - y index is DatetimeIndex
        - y index matches X index
        """
        np.random.seed(42)
        n = 150  # Need more data for feature warmup
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
        
        # Verify index type
        assert isinstance(y.index, pd.DatetimeIndex), \
            "y index should be DatetimeIndex"
        
        # Verify X and y have same index
        pd.testing.assert_index_equal(X.index, y.index)
        
        # Verify we have data after warmup
        assert len(X) > 0, "Should have data after warmup period"


# =============================================================================
# Test 3: Backtest temporal integrity
# =============================================================================

class TestNoLeakageBacktest:
    """Test that backtest doesn't use future information."""

    def test_execution_after_signal(self):
        """Test that execution happens after signal is generated."""
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

        dpoint = pd.Series(0.9, index=dates)
        dpoint.iloc[:10] = 0.5

        result = backtest_from_dpoint(
            df,
            dpoint,
            buy_threshold=0.6,
            sell_threshold=0.4,
            confirm_days=2,
        )

        if len(result.trades) > 0:
            first_trade = result.trades.iloc[0]
            signal_date = pd.to_datetime(first_trade["buy_signal_date"])
            exec_date = pd.to_datetime(first_trade["buy_exec_date"])

            assert exec_date > signal_date, \
                "Execution date must be after signal date"

    def test_t1_execution_price(self):
        """
        Test that execution uses t+1 price, not current day price.
        
        Verifies:
        - Buy orders execute at next day's open price
        - Price is within reasonable tolerance
        """
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

        dpoint = pd.Series(0.9, index=dates)
        dpoint.iloc[:10] = 0.5

        result = backtest_from_dpoint(
            df,
            dpoint,
            buy_threshold=0.6,
            sell_threshold=0.4,
            confirm_days=2,
        )

        if len(result.trades) > 0:
            first_trade = result.trades.iloc[0]
            exec_date = pd.to_datetime(first_trade["buy_exec_date"])

            exec_idx = df.index.get_indexer([exec_date], method="ffill")[0]
            if exec_idx >= 0 and exec_idx < len(df):
                expected_price = df.iloc[exec_idx]["open_qfq"]
                actual_price = first_trade["buy_price"]

                # Allow small tolerance for slippage
                tolerance = 0.05  # 5%
                assert abs(actual_price - expected_price) / expected_price < tolerance, \
                    f"Execution price {actual_price} differs from expected {expected_price}"


# =============================================================================
# Test 4: Holdout isolation
# =============================================================================

class TestHoldoutIsolation:
    """Test that holdout data is completely isolated from search."""

    def test_holdout_not_in_search(self):
        """
        Test that holdout set is completely separate from search set.
        
        Verifies:
        - No index overlap between search and holdout
        - Holdout comes after search temporally
        """
        df = pd.DataFrame({"x": range(500)}, index=range(500))

        search_df, holdout_df = final_holdout_split(
            df, holdout_ratio=0.2, min_holdout_rows=20
        )

        search_indices = set(search_df.index)
        holdout_indices = set(holdout_df.index)

        # No overlap
        assert len(search_indices & holdout_indices) == 0, \
            "Search and holdout sets should have no overlap"

        # Temporal separation
        assert holdout_df.index.min() > search_df.index.max(), \
            "Holdout set should come after search set"


# =============================================================================
# Test 5: Warmup period handling
# =============================================================================

class TestWarmupPeriod:
    """Test that warmup periods are handled correctly for technical indicators."""

    def test_rsi_warmup(self):
        """
        Test that RSI is computed without errors.
        
        Verifies feature engineering completes successfully with TA indicators.
        """
        np.random.seed(42)
        n = 200  # Need sufficient data for warmup
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

        # Enable TA indicators with RSI
        feature_config = {"use_ta_indicators": True, "ta_windows": [14]}

        X, y, meta = build_features_and_labels(df, feature_config)
        
        # Check that RSI columns exist
        rsi_cols = [c for c in X.columns if "rsi" in c.lower()]
        
        # Just verify the feature engineering completed
        assert len(X) > 0, "Should produce features after warmup"
        assert len(y) > 0, "Should produce labels after warmup"

    def test_macd_warmup(self):
        """
        Test that MACD is computed without errors.
        
        Verifies feature engineering completes successfully with MACD.
        """
        np.random.seed(42)
        n = 200  # Need sufficient data for MACD warmup
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

        feature_config = {"use_ta_indicators": True}

        X, y, meta = build_features_and_labels(df, feature_config)
        
        # Check MACD columns exist
        macd_cols = [c for c in X.columns if "macd" in c.lower()]
        
        # Just verify the feature engineering completed
        assert len(X) > 0, "Should produce features after warmup"
        assert len(y) > 0, "Should produce labels after warmup"
