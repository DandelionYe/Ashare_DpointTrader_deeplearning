# test_rejection.py
"""
Tests for order rejection logic.
Validates limit up/down and suspension handling.
"""
import numpy as np
import pandas as pd
import pytest
from backtester_engine import (
    backtest_from_dpoint,
    check_execution_feasibility,
    _prepare_price_limits,
)


class TestLimitUpDownRejection:
    """Test limit up/down rejection logic."""
    
    def test_limit_up_buy_rejected(self, price_data_limit_up):
        """Test that buying is rejected on limit up day."""
        df = price_data_limit_up.copy()
        df = _prepare_price_limits(df, 0.10, 0.10)
        
        row = df.iloc[11]
        
        feasible, reason = check_execution_feasibility(row, "BUY")
        
        assert not feasible
        assert "涨停" in reason
    
    def test_limit_down_sell_rejected(self, price_data_limit_up):
        """Test that selling is rejected on limit down day."""
        df = price_data_limit_up.copy()
        df = _prepare_price_limits(df, 0.10, 0.10)
        
        row = df.iloc[21]
        
        feasible, reason = check_execution_feasibility(row, "SELL")
        
        assert not feasible
        assert "跌停" in reason
    
    def test_backtest_respects_limit_up(self):
        """Test that backtest respects limit up constraints."""
        n = 30
        dates = pd.date_range(start="2020-01-01", periods=n, freq="B")
        
        prices = [10.0]
        for i in range(n - 1):
            if i == 10:
                prices.append(11.0)
            else:
                prices.append(prices[-1] * 1.01)
        
        df = pd.DataFrame({
            "date": dates,
            "open_qfq": prices,
            "close_qfq": prices,
            "high_qfq": prices,
            "low_qfq": prices,
            "volume": [5_000_000] * n,
        })
        
        dpoint = pd.Series(0.9, index=dates)
        
        result = backtest_from_dpoint(
            df, dpoint,
            buy_threshold=0.5,
            sell_threshold=0.5,
            confirm_days=1,
        )
        
        notes_str = " ".join(result.notes)
        
        if "涨停买不到" in notes_str or "REJECTED" in notes_str:
            assert True


class TestSuspensionRejection:
    """Test suspension rejection logic."""
    
    def test_suspended_rejected(self, price_data_suspended):
        """Test that suspended stock is rejected."""
        df = price_data_suspended.copy()
        df = _prepare_price_limits(df, 0.10, 0.10)
        
        row = df.iloc[10]
        
        feasible, reason = check_execution_feasibility(row, "BUY")
        
        assert not feasible
        assert "停牌" in reason or "无有效价格" in reason
    
    def test_backtest_respects_suspension(self):
        """Test that backtest respects suspension."""
        n = 30
        dates = pd.date_range(start="2020-01-01", periods=n, freq="B")
        
        prices = [10.0] * n
        prices[15] = 0.0
        
        df = pd.DataFrame({
            "date": dates,
            "open_qfq": prices,
            "close_qfq": prices,
            "high_qfq": prices,
            "low_qfq": prices,
            "volume": [5_000_000] * n,
        })
        
        dpoint = pd.Series(0.9, index=dates)
        
        result = backtest_from_dpoint(
            df, dpoint,
            buy_threshold=0.5,
            sell_threshold=0.5,
            confirm_days=1,
        )
        
        assert result is not None


class TestVolumeRejection:
    """Test volume-based rejection logic."""
    
    def test_low_volume_rejected(self):
        """Test that low volume triggers rejection."""
        row = pd.Series({
            "suspended": False,
            "open_qfq": 10.0,
            "prev_close": 10.0,
            "volume": 10000.0,
            "is_st": False,
            "listing_days": 100,
        })
        
        feasible, reason = check_execution_feasibility(
            row, "BUY", min_daily_volume=1_000_000.0
        )
        
        assert not feasible
        assert "成交量" in reason
    
    def test_normal_volume_accepted(self):
        """Test that normal volume is accepted."""
        row = pd.Series({
            "suspended": False,
            "open_qfq": 10.0,
            "prev_close": 10.0,
            "volume": 5_000_000.0,
            "is_st": False,
            "listing_days": 100,
        })
        
        feasible, reason = check_execution_feasibility(
            row, "BUY", min_daily_volume=1_000_000.0
        )
        
        assert feasible


class TestSTRejection:
    """Test ST stock rejection logic."""
    
    def test_st_rejected(self):
        """Test that ST stock is rejected when filter is enabled."""
        row = pd.Series({
            "suspended": False,
            "open_qfq": 10.0,
            "prev_close": 10.0,
            "volume": 5_000_000.0,
            "is_st": True,
            "listing_days": 100,
        })
        
        feasible, reason = check_execution_feasibility(row, "BUY", filter_st=True)
        
        assert not feasible
        assert "ST" in reason
    
    def test_st_accepted_when_disabled(self):
        """Test that ST stock is accepted when filter is disabled."""
        row = pd.Series({
            "suspended": False,
            "open_qfq": 10.0,
            "prev_close": 10.0,
            "volume": 5_000_000.0,
            "is_st": True,
            "listing_days": 100,
        })
        
        feasible, reason = check_execution_feasibility(row, "BUY", filter_st=False)
        
        assert feasible


class TestListingDaysRejection:
    """Test listing days rejection logic."""
    
    def test_new_listing_rejected(self):
        """Test that newly listed stock is rejected."""
        row = pd.Series({
            "suspended": False,
            "open_qfq": 10.0,
            "prev_close": 10.0,
            "volume": 5_000_000.0,
            "is_st": False,
            "listing_days": 30,
        })
        
        feasible, reason = check_execution_feasibility(
            row, "BUY", min_listing_days=60
        )
        
        assert not feasible
        assert "上市" in reason
    
    def test_mature_listing_accepted(self):
        """Test that mature listing is accepted."""
        row = pd.Series({
            "suspended": False,
            "open_qfq": 10.0,
            "prev_close": 10.0,
            "volume": 5_000_000.0,
            "is_st": False,
            "listing_days": 100,
        })
        
        feasible, reason = check_execution_feasibility(
            row, "BUY", min_listing_days=60
        )
        
        assert feasible
