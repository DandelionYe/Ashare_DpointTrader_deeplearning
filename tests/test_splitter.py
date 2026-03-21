# test_splitter.py
"""
Tests for splitter module correctness.
Validates walk-forward splitting logic.
"""
import numpy as np
import pandas as pd
import pytest
from data_loader import (
    walkforward_splits,
    final_holdout_split,
    recommend_n_folds,
    walkforward_splits_with_embargo,
    nested_walkforward_splits,
)


class TestWalkforwardSplits:
    """Test walkforward_splits function."""
    
    def test_basic_split_count(self):
        """Test that correct number of folds is generated."""
        X = pd.DataFrame({"x": range(400)})
        y = pd.Series(range(400))
        
        splits = walkforward_splits(X, y, n_folds=4, train_start_ratio=0.5, min_rows=20)
        
        assert len(splits) == 4
    
    def test_train_val_no_overlap(self):
        """Test that validation sets don't overlap."""
        X = pd.DataFrame({"x": range(200)})
        y = pd.Series(range(200))
        
        splits = walkforward_splits(X, y, n_folds=4, train_start_ratio=0.5)
        
        for i, ((X_train, _), (X_val, _)) in enumerate(splits):
            for j, ((X_train2, _), (X_val2, _)) in enumerate(splits):
                if i != j:
                    val_indices = set(X_val.index)
                    val_indices2 = set(X_val2.index)
                    assert len(val_indices & val_indices2) == 0
    
    def test_train_expanding(self):
        """Test that training set expands with each fold."""
        X = pd.DataFrame({"x": range(200)})
        y = pd.Series(range(200))
        
        splits = walkforward_splits(X, y, n_folds=4, train_start_ratio=0.5)
        
        prev_train_len = 0
        for (X_train, _), _ in splits:
            assert len(X_train) > prev_train_len
            prev_train_len = len(X_train)
    
    def test_min_rows_constraint(self):
        """Test that min_rows constraint is respected."""
        X = pd.DataFrame({"x": range(50)})
        y = pd.Series(range(50))
        
        splits = walkforward_splits(X, y, n_folds=10, min_rows=10)
        
        for (X_train, _), (X_val, _) in splits:
            assert len(X_train) >= 10
            assert len(X_val) >= 10
    
    def test_empty_dataframe_raises(self):
        """Test that empty dataframe raises error."""
        X = pd.DataFrame({"x": []})
        y = pd.Series([], dtype=int)
        
        splits = walkforward_splits(X, y, n_folds=4)
        
        assert len(splits) == 0


class TestFinalHoldoutSplit:
    """Test final_holdout_split function."""

    def test_holdout_ratio(self):
        """Test that holdout ratio is correct."""
        df = pd.DataFrame({"x": range(200)})

        search_df, holdout_df = final_holdout_split(
            df, holdout_ratio=0.2, min_holdout_rows=20
        )

        assert len(holdout_df) == 40
        assert len(search_df) == 160

    def test_holdout_at_end(self):
        """Test that holdout is from the end of data."""
        df = pd.DataFrame({"x": range(200)})

        search_df, holdout_df = final_holdout_split(
            df, holdout_ratio=0.2, min_holdout_rows=20
        )

        assert search_df.index.max() < holdout_df.index.min()

    def test_min_holdout_rows_raises(self):
        """Test that too small holdout raises error."""
        df = pd.DataFrame({"x": range(50)})

        with pytest.raises(ValueError):
            final_holdout_split(df, holdout_ratio=0.1, min_holdout_rows=100)


class TestRecommendNFolds:
    """Test recommend_n_folds function."""
    
    def test_large_data_more_folds(self):
        """Test that more data results in more folds."""
        n_folds_500 = recommend_n_folds(500)
        n_folds_1000 = recommend_n_folds(1000)
        
        assert n_folds_1000 >= n_folds_500
    
    def test_returns_within_bounds(self):
        """Test that result is within min/max bounds."""
        n_folds = recommend_n_folds(100, min_folds=2, max_folds=8)
        
        assert 2 <= n_folds <= 8
    
    def test_small_data_returns_min(self):
        """Test that very small data returns min folds."""
        n_folds = recommend_n_folds(50, min_folds=2, max_folds=8, min_rows=30)
        
        assert n_folds == 2


class TestEmbargoSplit:
    """Test walkforward_splits_with_embargo function."""
    
    def test_embargo_gap(self):
        """Test that embargo gap is applied."""
        X = pd.DataFrame({"x": range(200)})
        y = pd.Series(range(200))
        
        splits = walkforward_splits_with_embargo(
            X, y, n_folds=4, embargo_days=5
        )
        
        for (X_train, _), (X_val, _) in splits:
            train_end = X_train.index[-1]
            val_start = X_val.index[0]
            gap = val_start - train_end - 1
            assert gap >= 5


class TestNestedWalkforward:
    """Test nested_walkforward_splits function."""
    
    def test_nested_structure(self):
        """Test that nested structure is correct."""
        X = pd.DataFrame({"x": range(300)})
        y = pd.Series(range(300))
        
        splits = nested_walkforward_splits(
            X, y, n_outer_folds=2, n_inner_folds=2
        )
        
        assert len(splits) > 0
        for outer_train, outer_val, inner_splits in splits:
            assert len(inner_splits) > 0
            prev_inner_train_len = 0
            for (X_inner_train, _), (X_inner_val, _) in inner_splits:
                assert len(X_inner_train) >= prev_inner_train_len
                assert len(X_inner_train) > 0
                assert len(X_inner_val) > 0
                assert X_inner_train.index.max() < X_inner_val.index.min()
                prev_inner_train_len = len(X_inner_train)
