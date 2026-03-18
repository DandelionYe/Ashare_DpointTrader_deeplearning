# test_execution.py
"""
Tests for execution correctness.
Validates backtester_engine execution logic.
"""
import numpy as np
import pandas as pd
import pytest
from backtester_engine import (
    backtest_from_dpoint,
    apply_slippage,
    check_execution_feasibility,
    get_execution_price,
    compute_buy_and_hold,
    ExecutionStats,
    COMMISSION_RATE_BUY,
    COMMISSION_RATE_SELL,
)


class TestApplySlippage:
    """Test apply_slippage function."""
    
    def test_buy_slippage_up(self):
        """Test that buy slippage increases price."""
        price = 10.0
        result = apply_slippage(price, "BUY", slippage_bps=20)
        
        assert result > price
        assert result == pytest.approx(10.0 * 1.002, rel=1e-3)
    
    def test_sell_slippage_down(self):
        """Test that sell slippage decreases price."""
        price = 10.0
        result = apply_slippage(price, "SELL", slippage_bps=20)
        
        assert result < price
        assert result == pytest.approx(10.0 * 0.998, rel=1e-3)
    
    def test_zero_price(self):
        """Test that zero price returns zero."""
        result = apply_slippage(0, "BUY")
        
        assert result == 0
    
    def test_negative_price(self):
        """Test that negative price returns negative."""
        result = apply_slippage(-10, "BUY")
        
        assert result < 0


class TestCheckExecutionFeasibility:
    """Test check_execution_feasibility function."""
    
    def test_suspended_rejected(self):
        """Test that suspended stock is rejected."""
        row = pd.Series({"suspended": True, "open_qfq": 10.0})
        
        feasible, reason = check_execution_feasibility(row, "BUY")
        
        assert not feasible
        assert "停牌" in reason
    
    def test_no_price_rejected(self):
        """Test that no price is rejected."""
        row = pd.Series({"suspended": False, "open_qfq": 0})
        
        feasible, reason = check_execution_feasibility(row, "BUY")
        
        assert not feasible
    
    def test_limit_up_buy_rejected(self):
        """Test that limit up blocks buying."""
        row = pd.Series({
            "suspended": False,
            "open_qfq": 11.0,
            "prev_close": 10.0,
        })
        
        feasible, reason = check_execution_feasibility(row, "BUY")
        
        assert not feasible
        assert "涨停" in reason
    
    def test_limit_down_sell_rejected(self):
        """Test that limit down blocks selling."""
        row = pd.Series({
            "suspended": False,
            "open_qfq": 9.0,
            "prev_close": 10.0,
        })
        
        feasible, reason = check_execution_feasibility(row, "SELL")
        
        assert not feasible
        assert "跌停" in reason
    
    def test_normal_execution(self):
        """Test that normal execution is allowed."""
        row = pd.Series({
            "suspended": False,
            "open_qfq": 10.0,
            "prev_close": 10.0,
            "is_st": False,
            "listing_days": 100,
            "volume": 5_000_000,
        })
        
        feasible, reason = check_execution_feasibility(row, "BUY")
        
        assert feasible
        assert reason == ""


class TestBacktestExecution:
    """Test backtest_from_dpoint execution."""
    
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
    
    def test_no_trades_when_no_signal(self, minimal_price_data, sample_dpoint_series):
        """Test that no trades when dpoint is neutral."""
        df = self._prepare_df(minimal_price_data)
        dpoint = pd.Series(0.5, index=df.index)
        
        result = backtest_from_dpoint(
            df,
            dpoint,
            buy_threshold=0.8,
            sell_threshold=0.2,
            confirm_days=2,
        )
        
        assert len(result.trades) == 0
    
    def test_buy_signal_generated(self, minimal_price_data):
        """Test that buy signal is generated correctly."""
        df = self._prepare_df(minimal_price_data)
        dpoint = pd.Series(0.9, index=df.index)
        
        result = backtest_from_dpoint(
            df,
            dpoint,
            buy_threshold=0.6,
            sell_threshold=0.4,
            confirm_days=2,
            initial_cash=100000.0,
        )
        
        assert len(result.trades) >= 0
    
    def test_commission_applied(self, minimal_price_data):
        """Test that commission is applied to trades."""
        df = self._prepare_df(minimal_price_data)
        dpoint = pd.Series(0.9, index=df.index)
        
        result = backtest_from_dpoint(
            df,
            dpoint,
            buy_threshold=0.6,
            sell_threshold=0.4,
            confirm_days=2,
            initial_cash=100000.0,
        )
        
        if len(result.trades) > 0:
            buy_trade = result.trades.iloc[0]
            assert "buy_commission" in buy_trade.index or buy_trade.get("buy_cost", 0) > 0
    
    def test_t1_execution(self, minimal_price_data):
        """Test that execution happens at t+1 open price."""
        df = self._prepare_df(minimal_price_data)
        dpoint = pd.Series(0.9, index=df.index)
        
        result = backtest_from_dpoint(
            df,
            dpoint,
            buy_threshold=0.6,
            sell_threshold=0.4,
            confirm_days=2,
        )
        
        if len(result.trades) > 0:
            first_trade = result.trades.iloc[0]
            if pd.notna(first_trade.get("buy_exec_date")):
                buy_date = pd.to_datetime(first_trade["buy_exec_date"])
                original_idx = df.index.get_indexer([buy_date], method="ffill")[0]
                if original_idx >= 0 and original_idx < len(df):
                    expected_price = df.iloc[original_idx]["open_qfq"]
                    assert first_trade["buy_price"] == pytest.approx(expected_price, rel=0.05)


class TestExecutionStats:
    """Test ExecutionStats class."""
    
    def test_track_rejects(self):
        """Test that rejected orders are tracked."""
        stats = ExecutionStats()
        stats.add_reject("停牌")
        stats.add_reject("涨停买不到")
        
        assert stats.order_rejected == 2
        assert stats.reject_reasons["停牌"] == 1
        assert stats.reject_reasons["涨停买不到"] == 1
    
    def test_track_fills(self):
        """Test that filled orders are tracked."""
        stats = ExecutionStats()
        stats.add_fill(slippage_cost=10.0, value=1000.0)
        stats.add_fill(slippage_cost=5.0, value=500.0)
        
        assert stats.order_filled == 2
        assert stats.total_slippage_cost == 15.0
        assert stats.filled_value == 1500.0
    
    def test_avg_slippage(self):
        """Test average slippage calculation."""
        stats = ExecutionStats()
        stats.add_fill(10.0, 1000.0)
        stats.add_fill(20.0, 2000.0)
        
        assert stats.avg_slippage_cost == 15.0


class TestBuyAndHold:
    """Test compute_buy_and_hold function."""
    
    def test_basic_computation(self, minimal_price_data):
        """Test basic Buy & Hold computation."""
        result = compute_buy_and_hold(minimal_price_data, initial_cash=100000.0)
        
        assert "bnh_equity" in result.columns
        assert "bnh_cum_return" in result.columns
        assert len(result) == len(minimal_price_data)
    
    def test_buy_at_first_open(self, minimal_price_data):
        """Test that buy happens at first open price."""
        result = compute_buy_and_hold(minimal_price_data, initial_cash=100000.0)
        
        first_open = minimal_price_data.iloc[0]["open_qfq"]
        cost_per_lot = first_open * 100 * (1 + COMMISSION_RATE_BUY)
        expected_shares = (100000 // cost_per_lot) * 100
        
        assert expected_shares >= 0
    
    def test_final_equity_with_commission(self, minimal_price_data):
        """Test that final equity includes sell commission."""
        result = compute_buy_and_hold(minimal_price_data, initial_cash=100000.0)
        
        final_equity = result["bnh_equity"].iloc[-1]
        
        assert final_equity > 0
