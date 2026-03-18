# test_fee_lot.py
"""
Tests for fee and lot size correctness.
Validates A-share trading rules.
"""
import numpy as np
import pandas as pd
import pytest
from backtester import (
    backtest_from_dpoint,
    COMMISSION_RATE_BUY,
    COMMISSION_RATE_SELL,
)
from constants import MIN_CLOSED_TRADES_PER_FOLD, TARGET_CLOSED_TRADES_PER_FOLD


class TestLotSize:
    """Test A-share lot size rules (100 shares minimum)."""
    
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
    
    def test_shares_are_100_multiple(self, minimal_price_data):
        """Test that all trade shares are multiples of 100."""
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
            for _, trade in result.trades.iterrows():
                if pd.notna(trade.get("buy_shares")):
                    assert trade["buy_shares"] % 100 == 0
                if pd.notna(trade.get("sell_shares")):
                    assert trade["sell_shares"] % 100 == 0


class TestCommissionRates:
    """Test commission rate application."""
    
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
    
    def test_buy_commission_rate(self):
        """Test that buy commission rate is correct."""
        assert COMMISSION_RATE_BUY == 0.0003
    
    def test_sell_commission_rate(self):
        """Test that sell commission rate includes stamp duty."""
        assert COMMISSION_RATE_SELL == 0.0013
    
    def test_commission_in_cost(self, minimal_price_data):
        """Test that commission is included in buy cost."""
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
            trade = result.trades.iloc[0]
            buy_price = trade["buy_price"]
            buy_shares = trade["buy_shares"]
            expected_cost = buy_shares * buy_price
            actual_cost = trade["buy_cost"]
            
            commission_rate = actual_cost / expected_cost - 1
            assert abs(commission_rate - COMMISSION_RATE_BUY) < 0.0001
    
    def test_sell_commission_deducted(self, minimal_price_data):
        """Test that sell commission is deducted from proceeds."""
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

        if len(result.trades) > 0 and "status" in result.trades.columns:
            closed_trades = result.trades[result.trades["status"] == "CLOSED"]
            if len(closed_trades) > 0:
                trade = closed_trades.iloc[0]
                sell_price = trade["sell_price"]
                sell_shares = trade["sell_shares"]
                gross_proceeds = sell_shares * sell_price
                actual_proceeds = trade["sell_proceeds"]

                commission_rate = 1 - actual_proceeds / gross_proceeds
                assert abs(commission_rate - COMMISSION_RATE_SELL) < 0.0001
    
    def test_pnl_calculation(self, minimal_price_data):
        """Test that PnL is calculated correctly."""
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

        if len(result.trades) > 0 and "status" in result.trades.columns:
            closed_trades = result.trades[result.trades["status"] == "CLOSED"]
            if len(closed_trades) > 0:
                trade = closed_trades.iloc[0]
                expected_pnl = trade["sell_proceeds"] - trade["buy_cost"]
                actual_pnl = trade["pnl"]

                assert abs(actual_pnl - expected_pnl) < 0.01
    
    def test_return_calculation(self, minimal_price_data):
        """Test that return percentage is calculated correctly."""
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

        if len(result.trades) > 0 and "status" in result.trades.columns:
            closed_trades = result.trades[result.trades["status"] == "CLOSED"]
            if len(closed_trades) > 0:
                trade = closed_trades.iloc[0]
                expected_return = trade["pnl"] / trade["buy_cost"]
                actual_return = trade["return"]

                assert abs(actual_return - expected_return) < 0.0001


class TestConstants:
    """Test constants are properly defined."""
    
    def test_min_closed_trades_positive(self):
        """Test that MIN_CLOSED_TRADES_PER_FOLD is positive."""
        assert MIN_CLOSED_TRADES_PER_FOLD > 0
    
    def test_target_trades_greater_than_min(self):
        """Test that target is greater than or equal to min."""
        assert TARGET_CLOSED_TRADES_PER_FOLD >= MIN_CLOSED_TRADES_PER_FOLD


class TestTradeCalculations:
    """Test trade calculation correctness."""
    
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
    
    def test_pnl_calculation(self, minimal_price_data):
        """Test that PnL is calculated correctly."""
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

        if len(result.trades) > 0 and "status" in result.trades.columns:
            closed_trades = result.trades[result.trades["status"] == "CLOSED"]
            if len(closed_trades) > 0:
                trade = closed_trades.iloc[0]
                expected_pnl = trade["sell_proceeds"] - trade["buy_cost"]
                actual_pnl = trade["pnl"]
                assert abs(actual_pnl - expected_pnl) < 0.01

    def test_return_calculation(self, minimal_price_data):
        """Test that return percentage is calculated correctly."""
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

        if len(result.trades) > 0 and "status" in result.trades.columns:
            closed_trades = result.trades[result.trades["status"] == "CLOSED"]
            if len(closed_trades) > 0:
                trade = closed_trades.iloc[0]
                expected_return = trade["pnl"] / trade["buy_cost"]
                actual_return = trade["return"]
                assert abs(actual_return - expected_return) < 0.0001


class TestCashManagement:
    """Test cash management in backtest."""
    
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
    
    def test_no_negative_cash(self, minimal_price_data):
        """Test that cash never goes negative."""
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
        
        if not result.equity_curve.empty:
            cash_values = result.equity_curve["cash"]
            assert (cash_values >= 0).all()
    
    def test_initial_cash_respected(self, minimal_price_data):
        """Test that initial cash is properly set."""
        initial_cash = 100000.0
        
        df = self._prepare_df(minimal_price_data)
        dpoint = pd.Series(0.5, index=df.index)
        
        result = backtest_from_dpoint(
            df,
            dpoint,
            initial_cash=initial_cash,
        )
        
        if not result.equity_curve.empty:
            first_cash = result.equity_curve["cash"].iloc[0]
            assert first_cash == initial_cash
