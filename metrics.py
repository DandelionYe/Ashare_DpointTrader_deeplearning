# metrics.py
"""
评估指标与折回测统计。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from constants import TARGET_CLOSED_TRADES_PER_FOLD, LAMBDA_TRADE_PENALTY
from backtester_engine import backtest_from_dpoint


def metric_from_fold_ratios(ratios: List[float]) -> float:
    """
    各折净值比率的几何均值。
    比算术均值更惩罚极端亏损折，与复利增长逻辑一致。
    """
    ratios = [r for r in ratios if r > 0]
    if not ratios:
        return 0.0
    return float(np.exp(np.mean(np.log(ratios))))


def trade_penalty(closed_trades_per_fold: List[int]) -> float:
    """
    对偏离 TARGET_CLOSED_TRADES_PER_FOLD 的软性惩罚。
    交易太少（过拟合信号稀疏）或太多（信号噪声大）都会受罚。
    """
    diffs = [abs(int(n) - TARGET_CLOSED_TRADES_PER_FOLD) for n in closed_trades_per_fold]
    return LAMBDA_TRADE_PENALTY * float(np.mean(diffs)) if diffs else float("inf")


def backtest_fold_stats(
    df_full: pd.DataFrame,
    X_val: pd.DataFrame,
    dpoint_val: pd.Series,
    trade_cfg: Dict[str, object],
) -> Dict[str, float]:
    """
    对单个验证折运行回测，返回关键统计量。

    返回字段:
        equity_end  — 验证期末净值
        n_closed    — 已平仓交易数（用于硬约束和惩罚项）
        n_total     — 总交易数（含未平仓）
    """
    start = pd.to_datetime(X_val.index.min())
    end = pd.to_datetime(X_val.index.max())
    df_slice = df_full[(df_full["date"] >= start) & (df_full["date"] <= end)].copy()

    bt = backtest_from_dpoint(
        df=df_slice,
        dpoint=dpoint_val,
        initial_cash=float(trade_cfg["initial_cash"]),
        buy_threshold=float(trade_cfg["buy_threshold"]),
        sell_threshold=float(trade_cfg["sell_threshold"]),
        confirm_days=int(trade_cfg["confirm_days"]),
        min_hold_days=int(trade_cfg["min_hold_days"]),
        max_hold_days=int(trade_cfg.get("max_hold_days", 20)),
        take_profit=trade_cfg.get("take_profit", None),
        stop_loss=trade_cfg.get("stop_loss", None),
    )

    equity_end = (
        float(bt.equity_curve["total_equity"].iloc[-1])
        if not bt.equity_curve.empty
        else float(trade_cfg["initial_cash"])
    )

    if bt.trades is None or bt.trades.empty:
        n_closed, n_total = 0, 0
    else:
        n_total = int(len(bt.trades))
        n_closed = (
            int((bt.trades["status"] == "CLOSED").sum())
            if "status" in bt.trades.columns
            else n_total
        )

    return {
        "equity_end": equity_end,
        "n_closed": float(n_closed),
        "n_total": float(n_total),
    }