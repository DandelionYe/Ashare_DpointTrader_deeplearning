# backtester_engine.py
"""
回测执行引擎（P04 + P1-3 + P1-4 修复版）。

P04 修复：加入 A 股真实交易成本

P1-3 修复：执行价格改用 t+1 日开盘价，消除额外前向偏差
    原版：t 日生成信号，挂单 price = t 日收盘价，t+1 日按此价格成交。
    问题：信号当天收盘价在 t 日盘后已确定，t+1 日实际可成交的最早价格
          是 t+1 日开盘价，用 t 日收盘代替会引入额外的前向偏差。
    修复：执行时改为读取当日（即 t+1 日）的 open_qfq 作为成交价，
          _build_signal_frame 新增 open_qfq 列传入，pending_order 仅记录
          signal_close_price 供日志使用，不再用于定价。

P1-4 修复：持仓天数改用交易日数而非自然日数
    原版：用 (dt_sell - dt_buy).days 计算持仓时长，包含周末和节假日。
    问题：设置 max_hold_days=15 本意是 15 个交易日（约 3 周），
          但自然日计算只有约 3 周，实际约等于 11 个交易日，语义偏差明显。
          对 min_hold_days=1 的 T+1 约束，周五买入、周一卖出自然日差 3 天
          但交易日差仅 1 天，同样需要按交易日计算。
    修复：在 _simulate_execution 开始时预构建 tday_of 字典（日期→交易日序号），
          所有 held_days 相关判断均改为交易日差值。
          max_hold_days、min_hold_days 的单位由自然日变为交易日。
    - 买入：佣金 0.03%（commission_rate_buy，默认 0.0003）
    - 卖出：佣金 0.03% + 印花税 0.10% = 0.13%（commission_rate_sell，默认 0.0013）
    - 两个参数均可在 backtest_from_dpoint 调用时覆盖
    - 印花税说明：2023年8月起已调整为0.05%，此处默认使用较保守的0.10%；
      如需精确模拟可在调用时传入 commission_rate_sell=0.0008

公开 API：
    backtest_from_dpoint(df, dpoint, ...) -> BacktestResult

内部结构（信号/执行分离）：
    _build_signal_frame(df, dpoint, buy_threshold, sell_threshold)
        → 逐日的 dpoint 原始比较结果（纯向量运算，无状态，可独立测试）
    _simulate_execution(df, signal_frame, ...)
        → 含状态的执行模拟（挂单、持仓、净值快照）
    _normalize_open_trade(trade, ...)
        → 统一补全交易记录缺失字段，消除分散的 setdefault 调用

A 股约束：
    - 仅做多，不做空
    - 最小交易单位 100 股
    - T+1 近似：信号在 t 日生成，t+1 日按 t+1 日开盘价（open_qfq）执行
      （P1-3 修复：原用 t 日收盘价，已改为 t+1 日开盘价以消除额外前向偏差）
    - min_hold_days >= 1 强制模拟 T+1 锁定期
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# =========================================================
# P04：A 股交易成本常量
# =========================================================
# 买入佣金（券商收取，通常 0.02%～0.03%，此处取保守值）
COMMISSION_RATE_BUY: float = 0.0003

# 卖出佣金 + 印花税（0.03% + 0.10%；2023年8月后印花税降至0.05%，
# 如需精确模拟请在调用时传入 commission_rate_sell=0.0008）
COMMISSION_RATE_SELL: float = 0.0013


# =========================================================
# 数据类
# =========================================================
@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame    # 含 strategy + benchmark 列（P3-17）
    notes: List[str]
    benchmark_curve: pd.DataFrame # P3-17：Buy & Hold 基准净值曲线（与 equity_curve 同索引）


# =========================================================
# P3-17：Buy & Hold 基准计算
# =========================================================

def compute_buy_and_hold(
    df: pd.DataFrame,
    initial_cash: float = 100_000.0,
    commission_rate_buy: float = COMMISSION_RATE_BUY,
    commission_rate_sell: float = COMMISSION_RATE_SELL,
) -> pd.DataFrame:
    """
    计算同期持有（Buy & Hold）策略的每日净值曲线。

    P3-17 说明：
        策略与 Buy & Hold 的对比是判断是否存在 alpha 的最低标准。
        本函数在第一个可用交易日以开盘价买入，末日以收盘价卖出，
        中间每日净值 = 当日持仓市值 + 剩余现金。

    计算规则：
        - 第一日以 open_qfq 开盘价买入（与策略保持一致）
        - 最后一日持仓市值按 close_qfq 计算（含估算卖出成本）
        - 买入成本含佣金（commission_rate_buy），保持与策略一致
        - 未平仓日的持仓市值按当日 close_qfq 估算（不扣卖出税费）

    Args:
        df: 含 date / open_qfq / close_qfq 列的日频行情 DataFrame
        initial_cash: 初始资金（元），应与策略保持一致
        commission_rate_buy:  买入佣金率
        commission_rate_sell: 卖出佣金 + 印花税合计率（仅用于估算末日实收）

    Returns:
        DataFrame，列：
            date            — 交易日
            bnh_equity      — Buy & Hold 每日总净值（元）
            bnh_cum_return  — 累计收益率（相对初始资金）
    """
    df = df.copy()
    
    if "date" in df.columns and df.index.name == "date":
        df = df.reset_index(drop=True)
    elif "date" not in df.columns and df.index.name == "date":
        df = df.reset_index()
        
    if "date" not in df.columns:
        raise KeyError("compute_buy_and_hold: 找不到 'date' 列。")
    
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    rows = []
    shares = 0
    cash = float(initial_cash)

    for i, row in df.iterrows():
        dt = row["date"]
        close_t = float(row["close_qfq"])

        if i == 0:
            # 第一日开盘买入（与策略执行节奏保持一致）
            buy_price = float(row["open_qfq"])
            if buy_price > 0:
                cost_per_lot = buy_price * 100 * (1.0 + commission_rate_buy)
                max_lot = int(cash // cost_per_lot)
                shares = max_lot * 100
                if shares > 0:
                    cash -= shares * buy_price * (1.0 + commission_rate_buy)

        # 每日净值：持仓按收盘价估算（末日估算含卖出成本）
        if i == len(df) - 1 and shares > 0:
            equity = cash + shares * close_t * (1.0 - commission_rate_sell)
        else:
            equity = cash + shares * close_t

        rows.append({
            "date": dt,
            "bnh_equity": round(equity, 4),
            "bnh_cum_return": round(equity / initial_cash - 1.0, 6),
        })

    bnh = pd.DataFrame(rows)
    return bnh


# =========================================================
# 私有工具函数
# =========================================================
def _calc_buy_shares(cash: float, price: float, commission_rate_buy: float) -> int:
    """
    按 A 股 100 股最小单位计算可买入股数。
    price <= 0 时返回 0。
    P04：将买入佣金计入每手实际成本，避免因佣金导致现金略微透支。
    """
    if price <= 0:
        return 0
    cost_per_lot = price * 100 * (1.0 + commission_rate_buy)
    max_lot = int(cash // cost_per_lot)
    return max_lot * 100


def _normalize_open_trade(
    trade: Dict[str, object],
    buy_threshold: float,
    sell_threshold: float,
    confirm_days: int,
    min_hold_days: int,
) -> Dict[str, object]:
    """
    统一补全交易记录所有可能缺失的字段，避免 DataFrame 列不对齐。
    对 CLOSED 和 OPEN 两种状态均适用，缺失字段填 NaN / NaT。
    """
    # 卖出侧（未平仓时为空）
    trade.setdefault("sell_signal_date", pd.NaT)
    trade.setdefault("sell_exec_date", pd.NaT)
    trade.setdefault("sell_price", np.nan)
    trade.setdefault("sell_shares", np.nan)
    trade.setdefault("sell_proceeds", np.nan)          # 扣除卖出成本后的实收金额
    trade.setdefault("sell_commission", np.nan)        # P04：卖出成本（佣金+印花税）
    trade.setdefault("cash_after_sell", np.nan)

    # 平仓指标（未平仓时不可用）
    trade.setdefault("pnl", np.nan)
    trade.setdefault("return", np.nan)
    trade.setdefault("success", np.nan)

    # 信号诊断字段
    trade.setdefault("buy_dpoint_signal_day", np.nan)
    trade.setdefault("sell_dpoint_signal_day", np.nan)
    trade.setdefault("buy_above_cnt_at_signal", np.nan)
    trade.setdefault("sell_below_cnt_at_signal", np.nan)

    # P04：买入成本字段（未平仓时也应存在）
    trade.setdefault("buy_commission", np.nan)         # 买入佣金

    # 策略参数快照（方便事后对账）
    trade.setdefault("buy_threshold", float(buy_threshold))
    trade.setdefault("sell_threshold", float(sell_threshold))
    trade.setdefault("confirm_days", int(confirm_days))
    trade.setdefault("min_hold_days", int(min_hold_days))

    return trade


# =========================================================
# 第一层：信号帧构建（无状态，可独立测试）
# =========================================================
def _build_signal_frame(
    df: pd.DataFrame,
    dpoint: pd.Series,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.DataFrame:
    """
    对齐 dpoint 与行情数据，逐日计算原始阈值比较结果。

    此函数不含任何持仓状态或计数器，仅做向量化的比较运算，
    可以独立于执行模拟进行单元测试。

    返回 DataFrame，列：
        date          — 交易日
        open_qfq      — 后复权开盘价（P1-3：用于 t+1 日执行成交价）
        close_qfq     — 后复权收盘价
        dpoint        — 当日 Dpoint 值（NaN 表示无信号）
        dp_above_buy  — dpoint > buy_threshold（用于累计 above_cnt）
        dp_below_sell — dpoint < sell_threshold（用于累计 below_cnt）
    """
    open_ = df["open_qfq"].astype(float)   # P1-3：新增开盘价
    close = df["close_qfq"].astype(float)
    dpoint_aligned = dpoint.reindex(df.index)

    signal_frame = pd.DataFrame({
        "date": df.index,
        "open_qfq": open_,                  # P1-3：新增
        "close_qfq": close,
        "dpoint": dpoint_aligned,
        "dp_above_buy": dpoint_aligned > buy_threshold,
        "dp_below_sell": dpoint_aligned < sell_threshold,
    })

    # NaN 的 dpoint 不触发任何方向
    signal_frame.loc[dpoint_aligned.isna(), ["dp_above_buy", "dp_below_sell"]] = False

    return signal_frame.reset_index(drop=True)


# =========================================================
# 第二层：执行模拟（有状态，按日循环）
# =========================================================
def _simulate_execution(
    signal_frame: pd.DataFrame,
    initial_cash: float,
    buy_threshold: float,
    sell_threshold: float,
    max_hold_days: int,
    take_profit: Optional[float],
    stop_loss: Optional[float],
    confirm_days: int,
    min_hold_days: int,
    commission_rate_buy: float,    # P04 新增
    commission_rate_sell: float,   # P04 新增
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[str]]:
    """
    有状态的逐日执行模拟。读取 _build_signal_frame 的输出，
    维护持仓状态机（挂单 → 成交 → 净值快照）。

    P04 修复：买入和卖出均计入真实交易成本：
        - 买入实付 = 股数 × 价格 × (1 + commission_rate_buy)
        - 卖出实收 = 股数 × 价格 × (1 - commission_rate_sell)
        - pnl = 卖出实收 - 买入实付（净盈亏，含所有成本）

    P1-3 修复：执行价改用 t+1 日开盘价：
        pending_order 的 "price" 字段保留 t 日收盘价供日志记录，
        实际执行时改为读取当日 open_qfq（即信号次日的开盘价）。

    P1-4 修复：所有持仓天数判断改用交易日数：
        在循环开始前预构建 tday_of 字典（日期→交易日序号），
        min_hold_days / max_hold_days 的单位均为交易日。

    返回 (trade_rows, equity_rows, notes)，由 backtest_from_dpoint 组装为 BacktestResult。
    """
    notes: List[str] = []
    trade_rows: List[Dict[str, object]] = []
    equity_rows: List[Dict[str, object]] = []

    dates = list(signal_frame.index)

    # P1-4：预构建交易日序号映射，用于交易日数计算
    # key: pd.Timestamp, value: 从 0 开始的交易日序号
    tday_of: Dict[pd.Timestamp, int] = {
        pd.Timestamp(signal_frame.iloc[idx]["date"]): idx
        for idx in range(len(signal_frame))
    }

    cash: float = float(initial_cash)
    shares: int = 0
    position_entry_date: Optional[pd.Timestamp] = None
    pending_order: Optional[Dict[str, object]] = None
    open_trade: Optional[Dict[str, object]] = None

    above_cnt: int = 0
    below_cnt: int = 0

    for i in range(len(dates)):
        row = signal_frame.iloc[i]
        dt: pd.Timestamp = row["date"]
        price_open_t: float = float(row["open_qfq"])   # P1-3：当日开盘价，用于挂单执行
        price_close_t: float = float(row["close_qfq"])  # 收盘价：用于信号生成和净值快照
        dp: float = float(row["dpoint"]) if pd.notna(row["dpoint"]) else float("nan")
        dp_above: bool = bool(row["dp_above_buy"])
        dp_below: bool = bool(row["dp_below_sell"])

        exec_action_today = "NONE"
        exec_price_used = np.nan

        # -----------------------------------------------------------
        # 阶段一：执行前一日挂单
        # -----------------------------------------------------------
        if pending_order is not None and pending_order.get("exec_date") == dt:
            action = str(pending_order["action"])
            # P1-3：实际成交价改用 t+1 日开盘价（当日 open_qfq）
            # pending_order["price"] 存储的是 t 日收盘价，仅作日志参考
            exec_price = price_open_t
            exec_price_used = exec_price
            signal_date = pd.to_datetime(pending_order["signal_date"])

            if action == "BUY":
                if shares == 0:
                    # P04：买入股数计算时纳入佣金，避免现金轻微透支
                    buy_shares = _calc_buy_shares(cash, exec_price, commission_rate_buy)
                    if buy_shares > 0:
                        # P04：实付成本 = 股数 × 价格 × (1 + 佣金率)
                        buy_commission = buy_shares * exec_price * commission_rate_buy
                        cost = buy_shares * exec_price + buy_commission
                        cash -= cost
                        shares += buy_shares
                        position_entry_date = dt
                        exec_action_today = "BUY_EXEC"
                        open_trade = {
                            "buy_signal_date": signal_date,
                            "buy_exec_date": dt,
                            "buy_price": exec_price,                   # P1-3：t+1 开盘成交价
                            "buy_signal_close": float(pending_order.get("price", np.nan)),  # P1-3：t 收盘参考价
                            "buy_shares": buy_shares,
                            "buy_cost": cost,                      # P04：含佣金的实付总额
                            "buy_commission": buy_commission,      # P04：佣金明细
                            "cash_after_buy": cash,
                            "buy_dpoint_signal_day": float(pending_order.get("signal_dpoint", np.nan)),
                            "buy_threshold": float(buy_threshold),
                            "sell_threshold": float(sell_threshold),
                            "confirm_days": int(confirm_days),
                            "min_hold_days": int(min_hold_days),
                            "buy_above_cnt_at_signal": int(pending_order.get("above_cnt_at_signal", 0)),
                        }
                    else:
                        notes.append(f"{dt.date()}: BUY skipped (insufficient cash for 100 shares).")
                else:
                    notes.append(f"{dt.date()}: BUY pending but already in position; skipped.")

            elif action == "SELL":
                if shares > 0:
                    # P1-4：持仓时长改用交易日数（原版为自然日数）
                    if position_entry_date is not None and position_entry_date in tday_of:
                        held_tdays = tday_of[dt] - tday_of[position_entry_date]
                    else:
                        held_tdays = 999_999
                    if held_tdays >= min_hold_days:
                        # P04：卖出实收 = 股数 × 价格 × (1 - 佣金率 - 印花税率)
                        sell_commission = shares * exec_price * commission_rate_sell
                        proceeds = shares * exec_price - sell_commission
                        sell_shares = shares
                        cash += proceeds
                        shares = 0
                        position_entry_date = None
                        exec_action_today = "SELL_EXEC"

                        if open_trade is None:
                            open_trade = {}
                        open_trade.update({
                            "sell_signal_date": signal_date,
                            "sell_exec_date": dt,
                            "sell_price": exec_price,
                            "sell_shares": sell_shares,
                            "sell_proceeds": proceeds,             # P04：扣除成本后的实收
                            "sell_commission": sell_commission,    # P04：卖出成本明细
                            "cash_after_sell": cash,
                            "sell_dpoint_signal_day": float(pending_order.get("signal_dpoint", np.nan)),
                            "sell_below_cnt_at_signal": int(pending_order.get("below_cnt_at_signal", 0)),
                        })

                        # P04：pnl = 卖出实收 - 买入实付（净盈亏，含全部成本）
                        buy_cost = float(open_trade.get("buy_cost", 0.0))
                        pnl = proceeds - buy_cost
                        open_trade["pnl"] = pnl
                        open_trade["return"] = pnl / buy_cost if buy_cost > 0 else np.nan
                        open_trade["success"] = bool(pnl > 0)
                        open_trade["status"] = "CLOSED"

                        open_trade = _normalize_open_trade(
                            open_trade, buy_threshold, sell_threshold,
                            confirm_days, min_hold_days,
                        )
                        trade_rows.append(open_trade)
                        open_trade = None
                    else:
                        notes.append(
                            f"{dt.date()}: SELL blocked by min_hold_days "
                            f"(held {held_tdays} tdays < {min_hold_days})."
                        )
                else:
                    notes.append(f"{dt.date()}: SELL pending but no shares; skipped.")

            pending_order = None

        # -----------------------------------------------------------
        # 阶段二：更新计数器
        # -----------------------------------------------------------
        above_cnt = (above_cnt + 1) if dp_above else 0
        below_cnt = (below_cnt + 1) if dp_below else 0

        buy_condition_met = bool(
            (shares == 0) and (above_cnt >= confirm_days) and (pending_order is None)
        )

        # -----------------------------------------------------------
        # 阶段三：检查强制平仓条件
        # -----------------------------------------------------------
        force_sell = False
        force_reason = ""

        if shares > 0 and position_entry_date is not None and i < len(dates) - 1:
            # P1-4：max_hold_days 判断改用交易日数（+1 表示下一交易日执行时的持仓交易日数）
            held_tdays_next = (i + 1) - tday_of.get(position_entry_date, i + 1)

            if held_tdays_next >= max_hold_days:
                force_sell = True
                force_reason = (
                    f"max_hold_days reached ({held_tdays_next}>={max_hold_days} tdays) -> FORCE_SELL"
                )

            if open_trade is not None:
                buy_price = float(open_trade.get("buy_price", np.nan))
                if buy_price > 0:
                    pnl_ratio = (price_close_t / buy_price) - 1.0
                    if take_profit is not None and pnl_ratio >= float(take_profit):
                        force_sell = True
                        force_reason = (
                            f"take_profit reached ({pnl_ratio:.2%}>={take_profit:.2%}) -> FORCE_SELL"
                        )
                    if stop_loss is not None and pnl_ratio <= -float(stop_loss):
                        force_sell = True
                        force_reason = (
                            f"stop_loss reached ({pnl_ratio:.2%}<={-stop_loss:.2%}) -> FORCE_SELL"
                        )

        # -----------------------------------------------------------
        # 阶段四：生成今日信号，挂单至 t+1
        # -----------------------------------------------------------
        signal_today = "NONE"
        order_scheduled_for = pd.NaT
        reason = ""

        sell_condition_met = False
        if shares > 0 and (below_cnt >= confirm_days or force_sell) and (pending_order is None):
            if position_entry_date is None:
                sell_condition_met = True
            elif i < len(dates) - 1:
                # P1-4：改用交易日差（+1 表示下一交易日执行时的持仓交易日数）
                held_tdays_next = (i + 1) - tday_of.get(position_entry_date, i + 1)
                sell_condition_met = (held_tdays_next >= min_hold_days)

        if force_sell and shares > 0 and pending_order is None:
            sell_condition_met = True

        if i < len(dates) - 1 and pending_order is None and not np.isnan(dp):
            next_dt = signal_frame.iloc[i + 1]["date"]

            if buy_condition_met:
                signal_today = "BUY_SIGNAL"
                order_scheduled_for = next_dt
                reason = f"dpoint连续{confirm_days}天>{buy_threshold} 且空仓 -> BUY_SIGNAL"
                pending_order = {
                    "action": "BUY",
                    "signal_date": dt,
                    "exec_date": next_dt,
                    "price": price_close_t,   # P1-3：t 日收盘参考价，仅供日志记录，不用于定价
                    "signal_dpoint": dp,
                    "above_cnt_at_signal": int(above_cnt),
                }
                above_cnt = 0
                below_cnt = 0

            elif sell_condition_met:
                signal_today = "SELL_SIGNAL"
                order_scheduled_for = next_dt
                reason = (
                    force_reason if force_sell
                    else f"dpoint连续{confirm_days}天<{sell_threshold} "
                         f"且满足最短持有{min_hold_days}天 -> SELL_SIGNAL"
                )
                pending_order = {
                    "action": "SELL",
                    "signal_date": dt,
                    "exec_date": next_dt,
                    "price": price_close_t,   # P1-3：t 日收盘参考价，仅供日志记录，不用于定价
                    "signal_dpoint": dp,
                    "below_cnt_at_signal": int(below_cnt),
                }
                above_cnt = 0
                below_cnt = 0

        # -----------------------------------------------------------
        # 阶段五：净值快照（P04：市值计算不受成本影响，成本已体现在 cash 中）
        # -----------------------------------------------------------
        market_value = shares * price_close_t
        equity_rows.append({
            "date": dt,
            "close_qfq": price_close_t,
            "cash": cash,
            "shares": shares,
            "market_value": market_value,
            "total_equity": cash + market_value,
            "dpoint": dp if not np.isnan(dp) else np.nan,
            "above_cnt": int(above_cnt),
            "below_cnt": int(below_cnt),
            "buy_condition_met": bool(buy_condition_met),
            "sell_condition_met": bool(sell_condition_met),
            "signal_today": signal_today,
            "order_scheduled_for": order_scheduled_for,
            "exec_action_today": exec_action_today,
            "exec_price_used": exec_price_used,
            "reason": reason,
        })

    # -----------------------------------------------------------
    # 期末：处理未平仓持仓
    # -----------------------------------------------------------
    if open_trade is not None:
        last_row = signal_frame.iloc[-1]
        last_close = float(last_row["close_qfq"])
        buy_cost = float(open_trade.get("buy_cost", 0.0))
        buy_shares_held = float(open_trade.get("buy_shares", 0.0))
        mkt_value = buy_shares_held * last_close
        # P04：未实现盈亏也应扣除假设卖出时的成本，给出保守估计
        estimated_sell_commission = buy_shares_held * last_close * commission_rate_sell
        unreal_pnl = (mkt_value - estimated_sell_commission) - buy_cost if buy_cost > 0 else np.nan

        open_trade["status"] = "OPEN"
        open_trade["unrealized_pnl"] = unreal_pnl
        open_trade["unrealized_return"] = (unreal_pnl / buy_cost) if buy_cost > 0 else np.nan
        open_trade["estimated_sell_commission"] = estimated_sell_commission  # P04：估算卖出成本

        open_trade = _normalize_open_trade(
            open_trade, buy_threshold, sell_threshold,
            confirm_days, min_hold_days,
        )
        trade_rows.append(open_trade)

    return trade_rows, equity_rows, notes


# =========================================================
# 公开 API（向后兼容：新增参数均有默认值，原调用方无需修改）
# =========================================================
def backtest_from_dpoint(
    df: pd.DataFrame,
    dpoint: pd.Series,
    initial_cash: float = 100_000.0,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    max_hold_days: int = 20,
    take_profit: Optional[float] = None,
    stop_loss: Optional[float] = None,
    confirm_days: int = 2,
    min_hold_days: int = 1,
    # P04 新增：交易成本参数，默认值符合 A 股主流水平
    commission_rate_buy: float = COMMISSION_RATE_BUY,
    commission_rate_sell: float = COMMISSION_RATE_SELL,
    mode_note: str = (
        "Execution: signal at t (close), execute at t+1 open. "
        "Hold days counted in trading days."
    ),
) -> BacktestResult:
    """
    将 Dpoint 序列转化为 A 股回测结果。

    P04 新增参数：
        commission_rate_buy  — 买入佣金率（默认 0.03%）
        commission_rate_sell — 卖出佣金 + 印花税合计率（默认 0.13%）
            注：2023年8月后印花税调整为0.05%，可传入 0.0008 进行精确模拟

    P1-3 修复：执行价格改为 t+1 日开盘价（原为 t 日收盘价）。
    P1-4 修复：max_hold_days / min_hold_days 单位改为交易日（原为自然日）。

    参数说明：
        df             — 含 date / open_qfq / close_qfq 列的日频行情 DataFrame
        dpoint         — P(close_{t+1} > close_t | X_t)，index 为日期
        initial_cash   — 初始资金（元）
        buy_threshold  — Dpoint 连续高于此值 confirm_days 天触发买入信号
        sell_threshold — Dpoint 连续低于此值 confirm_days 天触发卖出信号
        max_hold_days  — 最大持仓交易日数（P1-4：已改为交易日）
        take_profit    — 止盈比例（如 0.12 表示 12%），None 表示不启用
        stop_loss      — 止损比例（如 0.08 表示 8%），None 表示不启用
        confirm_days   — 连续满足条件天数，用于平滑信号
        min_hold_days  — 最短持仓交易日数（P1-4：已改为交易日，T+1 约束设为 1）
    """
    # --- 数据预处理 ---
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date", drop=False)

    dpoint = dpoint.copy()
    dpoint.index = pd.to_datetime(dpoint.index)

    # --- 第一步：构建信号帧（无状态）---
    signal_frame = _build_signal_frame(df, dpoint, buy_threshold, sell_threshold)

    # --- 第二步：执行模拟（有状态）---
    trade_rows, equity_rows, exec_notes = _simulate_execution(
        signal_frame=signal_frame,
        initial_cash=initial_cash,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_hold_days=max_hold_days,
        take_profit=take_profit,
        stop_loss=stop_loss,
        confirm_days=confirm_days,
        min_hold_days=min_hold_days,
        commission_rate_buy=commission_rate_buy,   # P04
        commission_rate_sell=commission_rate_sell, # P04
    )

    # --- 第三步：组装结果 ---
    # P04：在 notes 中记录实际使用的成本参数，便于核查
    cost_note = (
        f"Transaction costs: buy={commission_rate_buy:.4%}, "
        f"sell={commission_rate_sell:.4%} "
        f"(commission + stamp duty)"
    )
    notes = [mode_note, cost_note] + exec_notes
    trades = pd.DataFrame(trade_rows)
    equity_curve = pd.DataFrame(equity_rows)

    if not equity_curve.empty:
        equity_curve = equity_curve.sort_values("date").reset_index(drop=True)
        equity_curve["cum_max_equity"] = equity_curve["total_equity"].cummax()
        equity_curve["drawdown"] = (
            equity_curve["total_equity"] / equity_curve["cum_max_equity"] - 1.0
        )

    # P3-17：计算 Buy & Hold 基准，并将其净值列合并进 equity_curve 便于 Excel 对齐
    benchmark_curve = compute_buy_and_hold(
        df, initial_cash=initial_cash,
        commission_rate_buy=commission_rate_buy,
        commission_rate_sell=commission_rate_sell,
    )
    if not equity_curve.empty and not benchmark_curve.empty:
        equity_curve = equity_curve.merge(
            benchmark_curve[["date", "bnh_equity", "bnh_cum_return"]],
            on="date", how="left",
        )
        # 在 notes 中追加 alpha 快速摘要
        strat_final = float(equity_curve["total_equity"].iloc[-1])
        bnh_final   = float(equity_curve["bnh_equity"].iloc[-1]) if "bnh_equity" in equity_curve.columns else initial_cash
        alpha_pct   = (strat_final - bnh_final) / initial_cash * 100.0
        notes.append(
            f"Benchmark (Buy&Hold) final equity: {bnh_final:.2f}  |  "
            f"Strategy final equity: {strat_final:.2f}  |  "
            f"Alpha vs B&H: {alpha_pct:+.2f}% (vs initial_cash)"
        )

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        notes=notes,
        benchmark_curve=benchmark_curve,
    )
