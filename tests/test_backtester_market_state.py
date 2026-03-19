# test_backtester_market_state.py
"""
Phase 3: 回归测试 - Backtester 市场状态和流动性过滤

测试目标：
1. _prepare_price_limits 保留真实 is_st/listing_days/suspended 列
2. check_execution_feasibility 默认使用 amount 进行流动性过滤
3. legacy min_daily_volume 参数仍然可用
4. buy-side layered slippage 在 _simulate_execution 中使用估算订单金额
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from backtester import (
    check_execution_feasibility,
    apply_layered_slippage,
    _prepare_price_limits,
    DEFAULT_MIN_DAILY_AMOUNT,
    _simulate_execution,
    _build_signal_frame,
)


class TestPreparePriceLimits:
    """Test _prepare_price_limits preserves external market state columns."""

    def test_prepare_price_limits_preserves_is_st_listing_days_and_suspended(self):
        """构造 df 传入真实 is_st/listing_days/suspended，断言不被覆盖。"""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "open_qfq": [10.0, 10.5, 0, 11.0, 11.5],  # 第 3 行开盘为 0（停牌）
            "close_qfq": [10.2, 10.8, 0, 11.2, 11.8],
            "is_st": [False, True, False, False, False],  # 第 2 行是 ST
            "listing_days": [100, 20, 200, 500, 1000],  # 第 2 行上市仅 20 天
            "suspended": [False, True, False, False, False],  # 第 2 行停牌
        }).set_index("date")

        result = _prepare_price_limits(df, limit_up_pct=0.10, limit_down_pct=0.10)

        # 断言 is_st 保留原始值
        assert result["is_st"].iloc[1] == True, "is_st[1] should be True (preserved)"
        assert result["is_st"].iloc[0] == False, "is_st[0] should be False"

        # 断言 listing_days 保留原始值
        assert result["listing_days"].iloc[1] == 20, "listing_days[1] should be 20 (preserved)"
        assert result["listing_days"].iloc[2] == 200, "listing_days[2] should be 200 (preserved)"

        # 断言 suspended 保留原始值并与计算值 OR
        # 第 2 行原始 suspended=True，应保持 True
        assert result["suspended"].iloc[1] == True, "suspended[1] should be True (preserved)"
        # 第 3 行开盘价=0，计算 suspended=True，应合并为 True
        assert result["suspended"].iloc[2] == True, "suspended[2] should be True (computed from open=0)"


class TestCheckExecutionFeasibility:
    """Test check_execution_feasibility uses amount by default."""

    def test_check_execution_feasibility_uses_amount_by_default(self):
        """构造 amount=500_000, volume=5_000_000，默认参数下应因成交额不足被拒单。"""
        row = pd.Series({
            "open_qfq": 10.0,
            "prev_close": 10.0,
            "volume": 5_000_000,  # 成交量很大
            "amount": 500_000,    # 成交额不足（默认门槛 100 万）
            "is_st": False,
            "listing_days": 100,
            "suspended": False,
        })

        is_feasible, reason = check_execution_feasibility(row, "BUY")

        assert is_feasible == False, "Should reject due to low amount"
        assert "成交额过低" in reason, f"Wrong reject reason: {reason}"

    def test_check_execution_feasibility_amount_sufficient(self):
        """构造 amount=2_000_000，应通过流动性检查。"""
        row = pd.Series({
            "open_qfq": 10.0,
            "prev_close": 10.0,
            "volume": 5_000_000,
            "amount": 2_000_000,  # 成交额足够
            "is_st": False,
            "listing_days": 100,
            "suspended": False,
        })

        is_feasible, reason = check_execution_feasibility(row, "BUY")

        assert is_feasible == True, f"Should pass, got reject reason: {reason}"

    def test_legacy_min_daily_volume_still_works(self):
        """显式传入 min_daily_volume=1_000_000，应使用 volume 检查。"""
        row = pd.Series({
            "open_qfq": 10.0,
            "prev_close": 10.0,
            "volume": 500_000,    # 成交量不足
            "amount": 2_000_000,  # 成交额足够
            "is_st": False,
            "listing_days": 100,
            "suspended": False,
        })

        # 显式传入 min_daily_volume，应使用 volume 检查
        is_feasible, reason = check_execution_feasibility(
            row, "BUY", min_daily_volume=1_000_000
        )

        assert is_feasible == False, "Should reject due to low volume (legacy mode)"
        assert "成交量过低" in reason, f"Wrong reject reason: {reason}"


class TestLayeredSlippage:
    """Test layered slippage in _simulate_execution."""

    def test_buy_layered_slippage_uses_estimated_order_value(self):
        """
        测试 _simulate_execution 的 BUY 路径是否使用估算的订单金额调用 apply_layered_slippage。
        
        这是真正的修复点：之前 BUY 侧传入的 order_value 是 0，
        修复后应该先估算 estimated_buy_shares 和 estimated_order_value。
        """
        # 构造最小 signal_frame
        dates = pd.date_range("2023-01-01", periods=10)
        signal_frame = pd.DataFrame({
            "date": dates,
            "open_qfq": [10.0] * 10,
            "close_qfq": [10.5] * 10,
            "dpoint": [0.7] * 10,  # 高于 buy_threshold，触发买入
            "dp_above_buy": [True] * 10,
            "dp_below_sell": [False] * 10,
            "volume": [1000000] * 10,
            "amount": [10000000] * 10,
            "suspended": [False] * 10,
            "is_st": [False] * 10,
            "listing_days": [100] * 10,
            "prev_close": [10.0] * 10,
        })
        
        # 记录 apply_layered_slippage 的调用参数
        slippage_calls = []
        
        original_apply_layered_slippage = apply_layered_slippage
        
        def mock_apply_layered_slippage(price, action, order_value):
            slippage_calls.append({
                "price": price,
                "action": action,
                "order_value": order_value,
            })
            return original_apply_layered_slippage(price, action, order_value)
        
        with patch('backtester.apply_layered_slippage', side_effect=mock_apply_layered_slippage):
            # 调用 _simulate_execution，开启 use_layered_slippage
            trade_rows, equity_rows, notes, exec_stats = _simulate_execution(
                signal_frame=signal_frame,
                initial_cash=100000.0,
                buy_threshold=0.6,
                sell_threshold=0.4,
                max_hold_days=20,
                take_profit=None,
                stop_loss=None,
                confirm_days=1,
                min_hold_days=1,
                commission_rate_buy=0.0003,
                commission_rate_sell=0.0013,
                slippage_bps=20,
                limit_up_pct=0.10,
                limit_down_pct=0.10,
                filter_st=False,
                min_listing_days=60,
                min_daily_amount=DEFAULT_MIN_DAILY_AMOUNT,
                min_daily_volume=None,
                use_layered_slippage=True,  # 关键：开启分层滑点
            )
        
        # 断言：apply_layered_slippage 被调用
        assert len(slippage_calls) > 0, "apply_layered_slippage should be called"
        
        # 找到 BUY 调用
        buy_calls = [c for c in slippage_calls if c["action"] == "BUY"]
        
        # 断言：BUY 调用存在（有买入信号）
        assert len(buy_calls) > 0, "Should have at least one BUY call"
        
        # 关键断言：BUY 调用的 order_value > 0
        # 修复前这里是 0，修复后应该是 estimated_order_value
        for call in buy_calls:
            assert call["order_value"] > 0, \
                f"BUY order_value should be > 0, got {call['order_value']}. " \
                f"This indicates the bug is not fixed: BUY side should estimate order value before calling apply_layered_slippage."

    def test_sell_layered_slippage_order_value(self):
        """测试 SELL 侧使用 shares * price 计算订单金额。"""
        # 这个测试验证 SELL 侧逻辑保持正确
        price = 10.0
        shares = 10000
        order_value = shares * price  # 10 万
        
        slippage_price = apply_layered_slippage(price, "SELL", order_value)
        
        # 10 万属于中单（10-50 万）：20 bps
        expected_slippage_bps = 20
        actual_slippage_bps = (price - slippage_price) / price * 10000
        
        assert abs(actual_slippage_bps - expected_slippage_bps) < 1, \
            f"SELL slippage should be ~{expected_slippage_bps} bps, got {actual_slippage_bps}"
