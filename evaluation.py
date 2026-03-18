# evaluation.py
"""
评估模块 - 合并版

本文件由以下三个文件合并而成：
1. backtester_engine.py - 回测执行引擎（P04 + P1-3 + P1-4 + P0/P1/P2 增强版）
2. metrics.py - 评估指标与折回测统计（P0 + P1 + P2）
3. regime.py - 市场状态检测与分层评估模块（P0 + P1 + P2）

合并目的：
    - 统一管理所有评估相关代码
    - 减少模块间循环依赖
    - 简化导入路径

主要功能：
    - 回测引擎：backtest_from_dpoint, BacktestResult, ExecutionStats
    - 风险指标：calculate_risk_metrics, format_metrics_summary
    - 市场状态检测：RegimeDetector, RegimeAwareBacktester, RegimeEnsemble

版本：merged
日期：2026-03-17
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# 全局常量
# =========================================================
# P04：A 股交易成本常量
COMMISSION_RATE_BUY: float = 0.0003      # 买入佣金（券商收取，通常 0.02%～0.03%）
COMMISSION_RATE_SELL: float = 0.0013     # 卖出佣金 + 印花税（0.03% + 0.10%）

# P0/P1: 执行层常量
DEFAULT_SLIPPAGE_BPS: int = 20           # 固定滑点 20 bps = 0.2%
DEFAULT_LIMIT_UP_PCT: float = 0.10       # 涨停幅度（A 股默认 10%）
DEFAULT_LIMIT_DOWN_PCT: float = 0.10     # 跌停幅度
ST_LIMIT_PCT: float = 0.05               # ST 股涨停幅度
DEFAULT_MIN_LISTING_DAYS: int = 60       # 最小上市天数要求
DEFAULT_MIN_DAILY_VOLUME: float = 1_000_000.0  # 最小成交量要求（100 万成交额）
DEFAULT_FILTER_ST: bool = True           # 过滤 ST 股

# Regime 常量
REGIME_TREND = "trend"
REGIME_NON_TREND = "non_trend"
REGIME_HIGH_VOL = "high_vol"
REGIME_LOW_VOL = "low_vol"
REGIME_MEDIUM_VOL = "medium_vol"


# =========================================================
# 数据类
# =========================================================

@dataclass
class ExecutionStats:
    """P1: 执行统计"""
    order_submitted: int = 0
    order_filled: int = 0
    order_rejected: int = 0
    reject_reasons: Dict[str, int] = field(default_factory=dict)
    total_slippage_cost: float = 0.0
    filled_value: float = 0.0

    def add_reject(self, reason: str):
        self.order_rejected += 1
        self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1

    def add_fill(self, slippage_cost: float, value: float):
        self.order_filled += 1
        self.total_slippage_cost += slippage_cost
        self.filled_value += value

    @property
    def avg_slippage_cost(self) -> float:
        return self.total_slippage_cost / self.order_filled if self.order_filled > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "order_submitted": self.order_submitted,
            "order_filled": self.order_filled,
            "order_rejected": self.order_rejected,
            "reject_reasons": self.reject_reasons,
            "total_slippage_cost": self.total_slippage_cost,
            "avg_slippage_cost": self.avg_slippage_cost,
        }


@dataclass
class BacktestResult:
    """回测结果数据类"""
    trades: pd.DataFrame
    equity_curve: pd.DataFrame    # 含 strategy + benchmark 列
    notes: List[str]
    benchmark_curve: pd.DataFrame
    execution_stats: Optional[ExecutionStats] = None  # P1: 执行统计


@dataclass
class PartialFillResult:
    """P2: 部分成交结果"""
    filled_shares: int
    remaining_shares: int
    exec_price: float
    status: str  # "full" | "partial" | "rejected"


# =========================================================
# Part 1: 回测引擎 (backtester_engine.py)
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


def apply_slippage(
    price: float,
    action: str,
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
) -> float:
    """
    P0: 应用固定滑点模型。

    参数：
        price: 基准价格（开盘价）
        action: "BUY" 或 "SELL"
        slippage_bps: 滑点基数（bps），默认 20 = 0.2%

    返回：
        滑点后的成交价格
    """
    if price <= 0:
        return price

    slippage = price * slippage_bps / 10000.0
    if action == "BUY":
        # 买入时滑点向上（高价买）
        return price + slippage
    else:  # SELL
        # 卖出时滑点向下（低价卖）
        return price - slippage


def check_execution_feasibility(
    row: pd.Series,
    action: str,
    limit_up_pct: float = DEFAULT_LIMIT_UP_PCT,
    limit_down_pct: float = DEFAULT_LIMIT_DOWN_PCT,
    filter_st: bool = DEFAULT_FILTER_ST,
    min_listing_days: int = DEFAULT_MIN_LISTING_DAYS,
    min_daily_volume: float = DEFAULT_MIN_DAILY_VOLUME,
) -> tuple[bool, str]:
    """
    P0: 检查订单是否可执行。

    检查项：
    1. 涨跌停：涨停不能买，跌停不能卖
    2. 停牌：无有效价格
    3. ST 股过滤（可选）
    4. 上市天数不足过滤（可选）
    5. 成交量过低过滤（可选）

    参数：
        row: 包含 open_qfq, close_qfq, limit_up, limit_down, suspended 等字段的行
        action: "BUY" 或 "SELL"
        limit_up_pct: 涨停幅度
        limit_down_pct: 跌停幅度
        filter_st: 是否过滤 ST 股
        min_listing_days: 最小上市天数
        min_daily_volume: 最小日成交额

    返回：
        (is_feasible, reject_reason)
    """
    # 1. 检查停牌
    if row.get("suspended", False):
        return False, "停牌"

    # 2. 检查有效价格
    price = row.get("open_qfq", 0)
    if price <= 0 or pd.isna(price):
        return False, "无有效价格"

    # 3. 检查涨跌停（使用前一日收盘价判断）
    prev_close = row.get("prev_close", price)
    if pd.isna(prev_close) or prev_close <= 0:
        prev_close = price

    limit_up_price = prev_close * (1 + limit_up_pct)
    limit_down_price = prev_close * (1 - limit_down_pct)

    if action == "BUY":
        # 涨停不能买
        if price >= limit_up_price:
            return False, "涨停买不到"
    else:  # SELL
        # 跌停不能卖
        if price <= limit_down_price:
            return False, "跌停卖不掉"

    # 4. 检查 ST 股
    if filter_st and row.get("is_st", False):
        return False, "ST 股过滤"

    # 5. 检查上市天数
    listing_days = row.get("listing_days", 999999)
    if listing_days < min_listing_days:
        return False, "上市天数不足"

    # 6. 检查成交量（使用 amount 成交额，单位：元）
    # P2 修复：原代码使用 volume（股数），但 min_daily_volume 参数名暗示成交额（元）
    # 修复为使用 amount 字段，更符合流动性过滤的实际需求
    daily_amount = row.get("amount", 0)
    if daily_amount < min_daily_volume:
        return False, "成交量过低"

    return True, ""


def get_execution_price(
    row: pd.Series,
    action: str,
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
    use_open: bool = True,
) -> float:
    """
    P0: 获取执行价格。

    P1-3 修复：默认使用开盘价（t+1 日开盘）避免前向偏差
    P0: 加入滑点

    参数：
        row: 包含 open_qfq, close_qfq 等字段
        action: "BUY" 或 "SELL"
        slippage_bps: 滑点基数
        use_open: 是否使用开盘价（默认 True）

    返回：
        滑点后的执行价格
    """
    if use_open:
        base_price = float(row.get("open_qfq", 0))
    else:
        base_price = float(row.get("close_qfq", 0))

    if base_price <= 0:
        base_price = float(row.get("close_qfq", 0))

    return apply_slippage(base_price, action, slippage_bps)


def apply_layered_slippage(
    price: float,
    action: str,
    order_value: float,
) -> float:
    """
    P2: 分层滑点模型。

    滑点随订单规模增加：
        - 小单 (< 10 万): 10 bps
        - 中单 (10-50 万): 20 bps
        - 大单 (> 50 万): 30 bps

    参数：
        price: 基准价格
        action: "BUY" 或 "SELL"
        order_value: 订单金额（元）

    返回：
        滑点后的成交价格
    """
    if price <= 0 or order_value <= 0:
        return price

    # 分层滑点
    if order_value < 100_000:
        slippage_bps = 10
    elif order_value < 500_000:
        slippage_bps = 20
    else:
        slippage_bps = 30

    return apply_slippage(price, action, slippage_bps)


def simulate_limit_execution(
    row: pd.Series,
    action: str,
    shares: int,
    limit_up_pct: float = DEFAULT_LIMIT_UP_PCT,
    limit_down_pct: float = DEFAULT_LIMIT_DOWN_PCT,
) -> tuple[float, float, str]:
    """
    P2: 更细的涨跌停成交近似模型。

    当发生涨跌停时：
    - 如果是涨停且想买：无法买入（全天封板）
    - 如果是跌停且想卖：无法卖出（全天封板）
    - 如果不是涨跌停但接近涨跌停价：按实际价格成交

    返回：
        (exec_price, filled_shares, status)
        - status: "filled" | "partial" | "rejected"
    """
    open_price = float(row.get("open_qfq", 0))
    prev_close = float(row.get("prev_close", open_price))
    close_price = float(row.get("close_qfq", open_price))

    if open_price <= 0:
        return 0, 0, "rejected"

    limit_up = prev_close * (1 + limit_up_pct)
    limit_down = prev_close * (1 - limit_down_pct)

    if action == "BUY":
        # 涨停检查
        if open_price >= limit_up:
            # 全天涨停，无法买入
            return 0, 0, "rejected"
        elif close_price >= limit_up * 0.98:  # 收盘接近涨停
            # 按涨停价成交
            return limit_up * 0.99, shares, "filled"
        else:
            return open_price, shares, "filled"
    else:  # SELL
        # 跌停检查
        if open_price <= limit_down:
            # 全天跌停，无法卖出
            return 0, 0, "rejected"
        elif close_price <= limit_down * 1.02:  # 收盘接近跌停
            # 按跌停价成交
            return limit_down * 1.01, shares, "filled"
        else:
            return open_price, shares, "filled"


def simulate_partial_fill(
    row: pd.Series,
    action: str,
    requested_shares: int,
    order_value: float,
    max_position_pct: float = 0.3,
    daily_volume: float = 10_000_000.0,
) -> PartialFillResult:
    """
    P2: 部分成交模拟。

    考虑因素：
    - 单日成交量限制（默认最多占成交量的 30%）
    - 持仓比例限制（默认单只股票最多 30% 仓位）

    参数：
        row: 当日行情数据
        action: "BUY" 或 "SELL"
        requested_shares: 请求成交股数
        order_value: 订单金额
        max_position_pct: 最大持仓比例
        daily_volume: 当日成交额

    返回：
        PartialFillResult
    """
    if requested_shares <= 0:
        return PartialFillResult(0, 0, 0, "rejected")

    price = float(row.get("open_qfq", 0))
    if price <= 0:
        return PartialFillResult(0, requested_shares, 0, "rejected")

    # 成交量约束：最多成交 30% 的日成交量
    max_volume_share = daily_volume * 0.3 / price
    volume_limited_shares = int(min(max_volume_share, requested_shares))

    # 取两者较小值
    filled_shares = min(volume_limited_shares, requested_shares)
    remaining = requested_shares - filled_shares

    if filled_shares == 0:
        return PartialFillResult(0, requested_shares, 0, "rejected")
    elif remaining > 0:
        return PartialFillResult(filled_shares, remaining, price, "partial")
    else:
        return PartialFillResult(filled_shares, 0, price, "full")


def calculate_position_size(
    cash: float,
    price: float,
    target_position_pct: float = 0.3,
    max_position_pct: float = 0.5,
    commission_rate: float = COMMISSION_RATE_BUY,
) -> int:
    """
    P2: 计算建仓股数。

    参数：
        cash: 可用资金
        price: 买入价格
        target_position_pct: 目标持仓比例（默认 30%）
        max_position_pct: 最大持仓比例（默认 50%）
        commission_rate: 佣金率

    返回：
        可买入股数（100 股整数倍）
    """
    if price <= 0 or cash <= 0:
        return 0

    # 目标买入金额
    target_value = cash * target_position_pct

    # 考虑佣金后的实际可用金额
    available_cash = cash * max_position_pct
    cost_per_share = price * (1 + commission_rate)
    max_shares = int(available_cash // cost_per_share)

    # 取 100 股整数倍
    return (max_shares // 100) * 100


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
    trade.setdefault("sell_commission", np.nan)        # P04：卖出成本（佣金 + 印花税）
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
        volume        — 成交量（用于流动性过滤）
        amount        — 成交额（用于流动性过滤）
        suspended     — 停牌标记
        is_st         — ST 股标记
        listing_days  — 上市天数
        prev_close    — 前一日收盘价（用于涨跌停判断）
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
        # P2 修复：保留必要字段供 check_execution_feasibility 使用
        "volume": df["volume"].astype(float),
        "amount": df["amount"].astype(float),
        "suspended": df.get("suspended", False),
        "is_st": df.get("is_st", False),
        "listing_days": df.get("listing_days", 999999),
        "prev_close": df.get("prev_close", close),
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
    commission_rate_buy: float,
    commission_rate_sell: float,
    # P0/P1 新增参数
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
    limit_up_pct: float = DEFAULT_LIMIT_UP_PCT,
    limit_down_pct: float = DEFAULT_LIMIT_DOWN_PCT,
    filter_st: bool = DEFAULT_FILTER_ST,
    min_listing_days: int = DEFAULT_MIN_LISTING_DAYS,
    min_daily_volume: float = DEFAULT_MIN_DAILY_VOLUME,
    use_layered_slippage: bool = False,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[str], ExecutionStats]:
    """
    有状态的逐日执行模拟。

    P04 修复：买入和卖出均计入真实交易成本
    P1-3 修复：执行价改用 t+1 日开盘价
    P1-4 修复：持仓天数改用交易日数
    P0 增强：统一 execution layer（滑点、涨跌停、停牌检查）
    P1 增强：执行统计

    返回 (trade_rows, equity_rows, notes, execution_stats)
    """
    notes: List[str] = []
    trade_rows: List[Dict[str, object]] = []
    equity_rows: List[Dict[str, object]] = []
    exec_stats = ExecutionStats()  # P1: 执行统计

    dates = list(signal_frame.index)

    # P1-4：预构建交易日序号映射
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
        price_open_t: float = float(row["open_qfq"])
        price_close_t: float = float(row["close_qfq"])
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
            signal_date = pd.to_datetime(pending_order["signal_date"])

            # P0: 检查订单可行性（涨跌停、停牌、ST 等）
            is_feasible, reject_reason = check_execution_feasibility(
                row, action,
                limit_up_pct=limit_up_pct,
                limit_down_pct=limit_down_pct,
                filter_st=filter_st,
                min_listing_days=min_listing_days,
                min_daily_volume=min_daily_volume,
            )

            exec_stats.order_submitted += 1

            if not is_feasible:
                # P0: 订单被拒绝
                exec_stats.add_reject(reject_reason)
                notes.append(f"{dt.date()}: {action} REJECTED - {reject_reason}")
                pending_order = None
                # 继续执行后续逻辑（不成交）
            else:
                # P0: 执行订单 - 获取滑点后的价格
                if use_layered_slippage:
                    # P2: 分层滑点
                    order_value = shares * price_open_t if action == "SELL" else 0
                    exec_price = apply_layered_slippage(price_open_t, action, order_value)
                else:
                    # P0: 固定滑点
                    exec_price = get_execution_price(row, action, slippage_bps)

                # P0: 记录滑点成本
                slippage_cost = abs(exec_price - price_open_t) * (shares if action == "SELL" else 0)
                exec_price_used = exec_price

                if action == "BUY":
                    if shares == 0:
                        # P04：买入股数计算时纳入佣金
                        buy_shares = _calc_buy_shares(cash, exec_price, commission_rate_buy)
                        if buy_shares > 0:
                            # P04：实付成本 = 股数 × 价格 × (1 + 佣金率)
                            buy_commission = buy_shares * exec_price * commission_rate_buy
                            cost = buy_shares * exec_price + buy_commission
                            cash -= cost
                            shares += buy_shares
                            position_entry_date = dt
                            exec_action_today = "BUY_EXEC"
                            # P1: 记录滑点成本
                            order_value = buy_shares * exec_price
                            slippage_cost = abs(exec_price - price_open_t) * buy_shares
                            exec_stats.add_fill(slippage_cost, order_value)
                            open_trade = {
                                "buy_signal_date": signal_date,
                                "buy_exec_date": dt,
                                "buy_price": exec_price,
                                "buy_price_before_slippage": price_open_t,
                                "buy_slippage_bps": (exec_price - price_open_t) / price_open_t * 10000 if price_open_t > 0 else 0,
                                "buy_signal_close": float(pending_order.get("price", np.nan)),
                                "buy_shares": buy_shares,
                                "buy_cost": cost,
                                "buy_commission": buy_commission,
                                "cash_after_buy": cash,
                                "buy_dpoint_signal_day": float(pending_order.get("signal_dpoint", np.nan)),
                                "buy_threshold": float(buy_threshold),
                                "sell_threshold": float(sell_threshold),
                                "confirm_days": int(confirm_days),
                                "min_hold_days": int(min_hold_days),
                                "buy_above_cnt_at_signal": int(pending_order.get("above_cnt_at_signal", 0)),
                            }
                        else:
                            exec_stats.add_reject("资金不足")
                            notes.append(f"{dt.date()}: BUY skipped (insufficient cash for 100 shares).")
                    else:
                        notes.append(f"{dt.date()}: BUY pending but already in position; skipped.")

                elif action == "SELL":
                    if shares > 0:
                        # P1-4：持仓时长改用交易日数
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
                            # P1: 记录滑点成本
                            order_value = sell_shares * exec_price
                            slippage_cost = abs(exec_price - price_open_t) * sell_shares
                            exec_stats.add_fill(slippage_cost, order_value)

                            if open_trade is None:
                                open_trade = {}
                            open_trade.update({
                                "sell_signal_date": signal_date,
                                "sell_exec_date": dt,
                                "sell_price": exec_price,
                                "sell_price_before_slippage": price_open_t,
                                "sell_slippage_bps": (price_open_t - exec_price) / price_open_t * 10000 if price_open_t > 0 else 0,
                                "sell_shares": sell_shares,
                                "sell_proceeds": proceeds,
                                "sell_commission": sell_commission,
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
        force_trigger = None  # P2 修复：初始化 force_trigger

        if shares > 0 and position_entry_date is not None and i < len(dates) - 1:
            # P1-4：max_hold_days 判断改用交易日数（+1 表示下一交易日执行时的持仓交易日数）
            held_tdays_next = (i + 1) - tday_of.get(position_entry_date, i + 1)

            if held_tdays_next >= max_hold_days:
                force_sell = True
                force_reason = (
                    f"max_hold_days reached ({held_tdays_next}>={max_hold_days} tdays) -> FORCE_SELL"
                )
                force_trigger = "max_hold_days"  # P2 修复：添加 force_trigger

            if open_trade is not None:
                buy_price = float(open_trade.get("buy_price", np.nan))
                if buy_price > 0:
                    pnl_ratio = (price_close_t / buy_price) - 1.0
                    if take_profit is not None and pnl_ratio >= float(take_profit):
                        force_sell = True
                        force_reason = (
                            f"take_profit reached ({pnl_ratio:.2%}>={take_profit:.2%}) -> FORCE_SELL"
                        )
                        force_trigger = "take_profit"
                    if stop_loss is not None and pnl_ratio <= -float(stop_loss):
                        force_sell = True
                        force_reason = (
                            f"stop_loss reached ({pnl_ratio:.2%}<={-stop_loss:.2%}) -> FORCE_SELL"
                        )
                        force_trigger = "stop_loss"

        # -----------------------------------------------------------
        # P0: 处理止盈止损的执行逻辑（按可执行价格成交）
        # -----------------------------------------------------------
        force_exec_price = np.nan
        if force_sell and shares > 0 and pending_order is None:
            # 检查次日是否可执行
            if i < len(dates) - 1:
                next_row = signal_frame.iloc[i + 1]
                next_dt = next_row["date"]

                # P0: 检查可执行性
                is_feasible, reject_reason = check_execution_feasibility(
                    next_row, "SELL",
                    limit_up_pct=limit_up_pct,
                    limit_down_pct=limit_down_pct,
                    filter_st=filter_st,
                    min_listing_days=min_listing_days,
                    min_daily_volume=min_daily_volume,
                )

                if is_feasible:
                    # P0: 获取滑点后的执行价格
                    next_open = float(next_row["open_qfq"])
                    if use_layered_slippage:
                        order_value = shares * next_open
                        force_exec_price = apply_layered_slippage(next_open, "SELL", order_value)
                    else:
                        force_exec_price = get_execution_price(next_row, "SELL", slippage_bps)

                    # 记录挂单信息
                    pending_order = {
                        "action": "SELL",
                        "action_reason": force_trigger,  # 记录触发原因
                        "signal_date": dt,
                        "exec_date": next_dt,
                        "price": price_close_t,
                        "exec_price_planned": force_exec_price,
                        "signal_dpoint": dp,
                        "below_cnt_at_signal": int(below_cnt),
                        "pnl_ratio_at_signal": pnl_ratio,
                    }
                    notes.append(f"{next_dt.date()}: {force_reason} -> SELL order submitted at {force_exec_price:.2f}")
                    force_sell = False  # 重置，避免重复处理
                else:
                    # P0: 止盈止损被拒绝
                    exec_stats.order_submitted += 1
                    exec_stats.add_reject(reason=f"stop_loss/take_profit_{reject_reason}")
                    notes.append(f"{next_dt.date()}: {force_reason} REJECTED - {reject_reason}")
                    force_sell = False  # 重置

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
                reason = f"dpoint 连续{confirm_days}天>{buy_threshold} 且空仓 -> BUY_SIGNAL"
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
                    else f"dpoint 连续{confirm_days}天<{sell_threshold} "
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

    return trade_rows, equity_rows, notes, exec_stats


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
    # P0/P1 新增：执行层参数
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
    limit_up_pct: float = DEFAULT_LIMIT_UP_PCT,
    limit_down_pct: float = DEFAULT_LIMIT_DOWN_PCT,
    filter_st: bool = DEFAULT_FILTER_ST,
    min_listing_days: int = DEFAULT_MIN_LISTING_DAYS,
    min_daily_volume: float = DEFAULT_MIN_DAILY_VOLUME,
    use_layered_slippage: bool = False,  # P2: 分层滑点
    mode_note: str = (
        "Execution: signal at t (close), execute at t+1 open. "
        "Hold days counted in trading days. "
        "P0: includes slippage, limit-up/down, suspension checks."
    ),
) -> BacktestResult:
    """
    将 Dpoint 序列转化为 A 股回测结果。

    P04 新增参数：
        commission_rate_buy  — 买入佣金率（默认 0.03%）
        commission_rate_sell — 卖出佣金 + 印花税合计率（默认 0.13%）

    P0 新增参数（Execution Layer）：
        slippage_bps       — 滑点（默认 20 bps = 0.2%）
        limit_up_pct       — 涨停幅度（默认 10%）
        limit_down_pct     — 跌停幅度（默认 10%）
        filter_st          — 是否过滤 ST 股（默认 True）
        min_listing_days  — 最小上市天数（默认 60）
        min_daily_volume  — 最小日成交额（默认 100 万）

    P1 新增参数：
        use_layered_slippage — 是否使用分层滑点（默认 False）

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

    # P0: 预处理涨跌停和停牌标记
    df = _prepare_price_limits(df, limit_up_pct, limit_down_pct)

    dpoint = dpoint.copy()
    dpoint.index = pd.to_datetime(dpoint.index)

    # --- 第一步：构建信号帧（无状态）---
    signal_frame = _build_signal_frame(df, dpoint, buy_threshold, sell_threshold)

    # --- 第二步：执行模拟（有状态）---
    trade_rows, equity_rows, exec_notes, exec_stats = _simulate_execution(
        signal_frame=signal_frame,
        initial_cash=initial_cash,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_hold_days=max_hold_days,
        take_profit=take_profit,
        stop_loss=stop_loss,
        confirm_days=confirm_days,
        min_hold_days=min_hold_days,
        commission_rate_buy=commission_rate_buy,
        commission_rate_sell=commission_rate_sell,
        # P0/P1 参数
        slippage_bps=slippage_bps,
        limit_up_pct=limit_up_pct,
        limit_down_pct=limit_down_pct,
        filter_st=filter_st,
        min_listing_days=min_listing_days,
        min_daily_volume=min_daily_volume,
        use_layered_slippage=use_layered_slippage,
    )

    # --- 第三步：组装结果 ---
    # P04：在 notes 中记录实际使用的成本参数，便于核查
    # P0: 记录执行层参数
    cost_note = (
        f"Transaction costs: buy={commission_rate_buy:.4%}, "
        f"sell={commission_rate_sell:.4%} "
        f"(commission + stamp duty)"
    )
    exec_note = (
        f"Execution: slippage={slippage_bps}bps, "
        f"limit_up={limit_up_pct:.0%}, limit_down={limit_down_pct:.0%}, "
        f"filter_ST={filter_st}, min_listing_days={min_listing_days}"
    )
    notes = [mode_note, cost_note, exec_note] + exec_notes

    # P1: 添加执行统计到 notes
    if exec_stats:
        notes.append(
            f"Execution stats: submitted={exec_stats.order_submitted}, "
            f"filled={exec_stats.order_filled}, "
            f"rejected={exec_stats.order_rejected}, "
            f"avg_slippage={exec_stats.avg_slippage_cost:.4f}"
        )

    trades = pd.DataFrame(trade_rows)
    equity_curve = pd.DataFrame(equity_rows)

    if not equity_curve.empty:
        equity_curve = equity_curve.sort_values("date").reset_index(drop=True)
        equity_curve["cum_max_equity"] = equity_curve["total_equity"].cummax()
        equity_curve["drawdown"] = (
            equity_curve["total_equity"] / equity_curve["cum_max_equity"] - 1.0
        )

    # P3-17：计算 Buy & Hold 基准
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
            f"Benchmark (Buy&Hold) final equity：{bnh_final:.2f}  |  "
            f"Strategy final equity: {strat_final:.2f}  |  "
            f"Alpha vs B&H: {alpha_pct:+.2f}% (vs initial_cash)"
        )

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        notes=notes,
        benchmark_curve=benchmark_curve,
        execution_stats=exec_stats,  # P1: 返回执行统计
    )


def _prepare_price_limits(
    df: pd.DataFrame,
    limit_up_pct: float,
    limit_down_pct: float,
) -> pd.DataFrame:
    """
    P0: 预处理涨跌停和停牌标记。
    """
    df = df.copy()

    # 前一日收盘价
    df["prev_close"] = df["close_qfq"].shift(1)

    # 涨跌停价格
    df["limit_up_price"] = df["prev_close"] * (1 + limit_up_pct)
    df["limit_down_price"] = df["prev_close"] * (1 - limit_down_pct)

    # 标记是否涨停/跌停（当日开盘触及涨跌停）
    df["at_limit_up"] = df["open_qfq"] >= df["limit_up_price"]
    df["at_limit_down"] = df["open_qfq"] <= df["limit_down_price"]

    # 停牌标记（开盘价为 0 或 NaN）
    df["suspended"] = (df["open_qfq"] <= 0) | df["open_qfq"].isna()

    # ST 标记（需要从外部数据源传入，这里默认 False）
    df["is_st"] = False

    # 上市天数（从第一行开始计数）
    df["listing_days"] = range(1, len(df) + 1)

    return df


# =========================================================
# Part 2: 评估指标 (metrics.py)
# =========================================================

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
    # 使用本地默认值，避免依赖 constants 模块
    TARGET_CLOSED_TRADES_PER_FOLD = 10
    LAMBDA_TRADE_PENALTY = 0.01
    
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


# =========================================================
# P0: 统一 metrics 层 - 核心风险指标
# =========================================================

def calculate_risk_metrics(
    equity_curve: pd.DataFrame,
    trades: Optional[pd.DataFrame],
    initial_cash: float,
    annual_trading_days: int = 252,
    benchmark_curve: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    P0: 统一的完整风险指标计算。

    计算以下核心指标：
    - total_return: 总收益率
    - annual_return: 年化收益率
    - annual_vol: 年化波动率
    - max_drawdown: 最大回撤
    - sharpe: 夏普比率
    - sortino: 索提诺比率

    P1 扩展指标：
    - max_drawdown_duration: 最大回撤持续天数
    - calmar: 卡玛比率
    - profit_factor: 盈利因子
    - expectancy: 期望收益
    - avg_holding_days: 平均持仓天数
    - turnover: 换手率
    - win_rate: 胜率
    - payoff_ratio: 盈亏比
    - avg_win: 平均盈利
    - avg_loss: 平均亏损

    P1 Benchmark 对照：
    - bnh_return: 买入持有收益率
    - excess_return: 超额收益
    - alpha: Alpha
    - beta: Beta

    P2 扩展：
    - rolling_sharpe: 滚动夏普
    - rolling_max_dd: 滚动最大回撤
    - tail_risk: 尾部风险
    - downside_deviation: 下行偏差
    - monthly_returns: 月度收益
    - yearly_returns: 年度收益

    参数：
        equity_curve: 包含 total_equity 列的 DataFrame
        trades: 交易记录 DataFrame
        initial_cash: 初始资金
        annual_trading_days: 年化交易日数，默认 252
        benchmark_curve: 可选的基准净值曲线

    返回：
        包含所有指标的字典
    """
    metrics = {}

    if equity_curve.empty:
        return _empty_metrics(initial_cash)

    # 基本数据
    equity = equity_curve["total_equity"].values
    n_days = len(equity)

    # P0: 核心指标
    total_return = (equity[-1] - initial_cash) / initial_cash
    metrics["total_return"] = float(total_return)
    metrics["total_return_pct"] = float(total_return * 100)

    # 年化收益率
    years = n_days / annual_trading_days
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    metrics["annual_return"] = float(annual_return)
    metrics["annual_return_pct"] = float(annual_return * 100)

    # 日收益率
    daily_returns = np.diff(equity) / equity[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    # 年化波动率
    annual_vol = np.std(daily_returns) * np.sqrt(annual_trading_days) if len(daily_returns) > 0 else 0
    metrics["annual_vol"] = float(annual_vol)
    metrics["annual_vol_pct"] = float(annual_vol * 100)

    # 最大回撤和回撤持续天数
    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
    metrics["max_drawdown"] = float(max_dd)
    metrics["max_drawdown_pct"] = float(max_dd * 100)

    # P1: 最大回撤持续天数
    in_drawdown = drawdown < -0.001  # 阈值 0.1%
    dd_durations = []
    current_dd = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_dd += 1
        else:
            if current_dd > 0:
                dd_durations.append(current_dd)
            current_dd = 0
    if current_dd > 0:
        dd_durations.append(current_dd)
    metrics["max_drawdown_duration"] = int(max(dd_durations)) if dd_durations else 0

    # P0: 夏普比率 (假设无风险利率为 0)
    risk_free_rate = 0.0
    excess_returns = daily_returns - risk_free_rate / annual_trading_days
    sharpe = (np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(annual_trading_days)) if np.std(excess_returns) > 0 else 0
    metrics["sharpe"] = float(sharpe)

    # P0: 索提诺比率 (只考虑下行波动)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(annual_trading_days) if len(downside_returns) > 0 else 0
    sortino = (np.mean(excess_returns) / downside_std * np.sqrt(annual_trading_days)) if downside_std > 0 else 0
    metrics["sortino"] = float(sortino)

    # P1: 卡玛比率
    metrics["calmar"] = float(annual_return / abs(max_dd)) if max_dd != 0 else 0

    # P2: 下行偏差
    metrics["downside_deviation"] = float(downside_std)

    # P2: 尾部风险 (VaR 95% 和 CVaR 95%)
    var_95 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
    cvar_95 = np.mean(daily_returns[daily_returns <= var_95]) if len(daily_returns[daily_returns <= var_95]) > 0 else var_95
    metrics["var_95"] = float(var_95)
    metrics["cvar_95"] = float(cvar_95)
    metrics["tail_risk"] = float(abs(cvar_95))

    # P0: 交易统计
    if trades is not None and not trades.empty:
        closed_trades = trades[trades.get("status", "CLOSED") == "CLOSED"] if "status" in trades.columns else trades

        n_trades = len(closed_trades)
        metrics["trade_count"] = int(n_trades)

        if n_trades > 0:
            # P1: 胜率
            if "pnl" in closed_trades.columns:
                wins = closed_trades[closed_trades["pnl"] > 0]
                losses = closed_trades[closed_trades["pnl"] < 0]
                win_count = len(wins)
                loss_count = len(losses)
                metrics["win_rate"] = float(win_count / n_trades) if n_trades > 0 else 0
                metrics["win_count"] = int(win_count)
                metrics["loss_count"] = int(loss_count)

                # P1: 平均盈利/亏损
                avg_win = float(wins["pnl"].mean()) if win_count > 0 else 0
                avg_loss = float(losses["pnl"].mean()) if loss_count > 0 else 0
                metrics["avg_win"] = float(avg_win)
                metrics["avg_loss"] = float(avg_loss)

                # P1: 盈亏比
                metrics["payoff_ratio"] = float(abs(avg_win / avg_loss)) if avg_loss != 0 else 0

                # P1: 盈利因子
                gross_profit = float(wins["pnl"].sum()) if win_count > 0 else 0
                gross_loss = float(abs(losses["pnl"].sum())) if loss_count > 0 else 0
                metrics["profit_factor"] = float(gross_profit / gross_loss) if gross_loss > 0 else 0

                # P1: 期望收益
                total_pnl = float(closed_trades["pnl"].sum())
                metrics["expectancy"] = float(total_pnl / n_trades) if n_trades > 0 else 0
            else:
                metrics["win_rate"] = 0.0
                metrics["avg_win"] = 0.0
                metrics["avg_loss"] = 0.0
                metrics["payoff_ratio"] = 0.0
                metrics["profit_factor"] = 0.0
                metrics["expectancy"] = 0.0

            # P1: 平均持仓天数
            if "holding_days" in closed_trades.columns:
                metrics["avg_holding_days"] = float(closed_trades["holding_days"].mean())
            else:
                metrics["avg_holding_days"] = 0.0
        else:
            metrics["win_rate"] = 0.0
            metrics["avg_win"] = 0.0
            metrics["avg_loss"] = 0.0
            metrics["payoff_ratio"] = 0.0
            metrics["profit_factor"] = 0.0
            metrics["expectancy"] = 0.0
            metrics["avg_holding_days"] = 0.0
    else:
        metrics["trade_count"] = 0

    # P1: 换手率 (基于交易次数和资金规模估算)
    if trades is not None and not trades.empty and "value" in trades.columns:
        total_volume = trades["value"].sum() if "value" in trades.columns else 0
        avg_equity = np.mean(equity)
        metrics["turnover"] = float(total_volume / (avg_equity * years)) if years > 0 and avg_equity > 0 else 0
    else:
        metrics["turnover"] = 0.0

    # P1: Benchmark 对照
    if benchmark_curve is not None and not benchmark_curve.empty:
        bnh_equity = benchmark_curve["bnh_equity"].values
        bnh_return = (bnh_equity[-1] - initial_cash) / initial_cash
        metrics["bnh_return"] = float(bnh_return)
        metrics["bnh_return_pct"] = float(bnh_return * 100)

        # 超额收益
        metrics["excess_return"] = float(total_return - bnh_return)
        metrics["excess_return_pct"] = float((total_return - bnh_return) * 100)

        # Alpha 和 Beta
        if len(daily_returns) > 1 and "bnh_returns" in benchmark_curve.columns:
            bnh_daily = benchmark_curve["bnh_returns"].values[1:]
            bnh_daily = bnh_daily[np.isfinite(bnh_daily)]
            min_len = min(len(daily_returns), len(bnh_daily))
            if min_len > 1:
                cov = np.cov(daily_returns[:min_len], bnh_daily[:min_len])[0, 1]
                var_bnh = np.var(bnh_daily[:min_len])
                beta = cov / var_bnh if var_bnh > 0 else 1.0
                alpha = annual_return - beta * bnh_return
                metrics["beta"] = float(beta)
                metrics["alpha"] = float(alpha)
            else:
                metrics["beta"] = 1.0
                metrics["alpha"] = 0.0
        else:
            metrics["beta"] = 1.0
            metrics["alpha"] = 0.0
    else:
        metrics["bnh_return"] = 0.0
        metrics["excess_return"] = 0.0
        metrics["alpha"] = 0.0
        metrics["beta"] = 1.0

    # P2: 月度收益
    if "date" in equity_curve.columns or equity_curve.index.dtype != 'int64':
        dates = pd.to_datetime(equity_curve.index if "date" not in equity_curve.columns else equity_curve["date"])
        monthly = pd.DataFrame({"equity": equity}, index=dates).resample("M").last()
        monthly_returns = monthly["equity"].pct_change().dropna()
        metrics["monthly_returns"] = monthly_returns.values.tolist() if len(monthly_returns) > 0 else []
        metrics["monthly_win_rate"] = float((monthly_returns > 0).mean()) if len(monthly_returns) > 0 else 0.0
    else:
        metrics["monthly_returns"] = []
        metrics["monthly_win_rate"] = 0.0

    # P2: 年度收益
    if "date" in equity_curve.columns or equity_curve.index.dtype != 'int64':
        dates = pd.to_datetime(equity_curve.index if "date" not in equity_curve.columns else equity_curve["date"])
        yearly = pd.DataFrame({"equity": equity}, index=dates).resample("Y").last()
        yearly_returns = yearly["equity"].pct_change().dropna()
        metrics["yearly_returns"] = yearly_returns.values.tolist() if len(yearly_returns) > 0 else []
        metrics["yearly_win_rate"] = float((yearly_returns > 0).mean()) if len(yearly_returns) > 0 else 0.0
    else:
        metrics["yearly_returns"] = []
        metrics["yearly_win_rate"] = 0.0

    # P2: 滚动夏普 (60 天窗口)
    if len(daily_returns) >= 60:
        rolling_mean = pd.Series(daily_returns).rolling(60).mean()
        rolling_std = pd.Series(daily_returns).rolling(60).std()
        rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(252)).dropna()
        metrics["rolling_sharpe_mean"] = float(rolling_sharpe.mean()) if len(rolling_sharpe) > 0 else 0.0
        metrics["rolling_sharpe_min"] = float(rolling_sharpe.min()) if len(rolling_sharpe) > 0 else 0.0
        metrics["rolling_sharpe_max"] = float(rolling_sharpe.max()) if len(rolling_sharpe) > 0 else 0.0
    else:
        metrics["rolling_sharpe_mean"] = 0.0
        metrics["rolling_sharpe_min"] = 0.0
        metrics["rolling_sharpe_max"] = 0.0

    # P2: 滚动最大回撤
    if len(daily_returns) >= 60:
        rolling_dd = []
        for i in range(60, len(equity)):
            window_equity = equity[i-60:i]
            window_cummax = np.maximum.accumulate(window_equity)
            window_dd = (window_equity - window_cummax) / window_cummax
            rolling_dd.append(np.min(window_dd))
        metrics["rolling_max_dd_mean"] = float(np.mean(rolling_dd)) if rolling_dd else 0.0
        metrics["rolling_max_dd_max"] = float(np.min(rolling_dd)) if rolling_dd else 0.0
    else:
        metrics["rolling_max_dd_mean"] = 0.0
        metrics["rolling_max_dd_max"] = 0.0

    # 附加信息
    metrics["n_days"] = int(n_days)
    metrics["years"] = float(years)
    metrics["initial_cash"] = float(initial_cash)
    metrics["final_equity"] = float(equity[-1])

    return metrics


def _empty_metrics(initial_cash: float) -> Dict[str, float]:
    """返回空的指标字典"""
    return {
        "total_return": 0.0,
        "total_return_pct": 0.0,
        "annual_return": 0.0,
        "annual_return_pct": 0.0,
        "annual_vol": 0.0,
        "annual_vol_pct": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_pct": 0.0,
        "max_drawdown_duration": 0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "downside_deviation": 0.0,
        "var_95": 0.0,
        "cvar_95": 0.0,
        "tail_risk": 0.0,
        "trade_count": 0,
        "win_rate": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "payoff_ratio": 0.0,
        "profit_factor": 0.0,
        "expectancy": 0.0,
        "avg_holding_days": 0.0,
        "turnover": 0.0,
        "bnh_return": 0.0,
        "bnh_return_pct": 0.0,
        "excess_return": 0.0,
        "excess_return_pct": 0.0,
        "alpha": 0.0,
        "beta": 1.0,
        "monthly_returns": [],
        "monthly_win_rate": 0.0,
        "yearly_returns": [],
        "yearly_win_rate": 0.0,
        "rolling_sharpe_mean": 0.0,
        "rolling_sharpe_min": 0.0,
        "rolling_sharpe_max": 0.0,
        "rolling_max_dd_mean": 0.0,
        "rolling_max_dd_max": 0.0,
        "n_days": 0,
        "years": 0.0,
        "initial_cash": float(initial_cash),
        "final_equity": float(initial_cash),
    }


def format_metrics_summary(metrics: Dict[str, float]) -> str:
    """
    P0: 格式化风险指标为可读字符串。
    """
    lines = [
        f"Total Return     : {metrics.get('total_return_pct', 0):+.2f}%",
        f"Annual Return    : {metrics.get('annual_return_pct', 0):+.2f}%",
        f"Annual Vol       : {metrics.get('annual_vol_pct', 0):.2f}%",
        f"Sharpe Ratio     : {metrics.get('sharpe', 0):.3f}",
        f"Sortino Ratio    : {metrics.get('sortino', 0):.3f}",
        f"Max Drawdown     : {metrics.get('max_drawdown_pct', 0):.2f}%",
        f"Max DD Duration  : {metrics.get('max_drawdown_duration', 0)} days",
        f"Calmar Ratio     : {metrics.get('calmar', 0):.3f}",
    ]

    if metrics.get("trade_count", 0) > 0:
        lines.extend([
            f"",
            f"Trade Count      : {metrics.get('trade_count', 0)}",
            f"Win Rate         : {metrics.get('win_rate', 0)*100:.1f}%",
            f"Avg Win          : {metrics.get('avg_win', 0):.2f}",
            f"Avg Loss         : {metrics.get('avg_loss', 0):.2f}",
            f"Payoff Ratio     : {metrics.get('payoff_ratio', 0):.3f}",
            f"Profit Factor    : {metrics.get('profit_factor', 0):.3f}",
            f"Expectancy       : {metrics.get('expectancy', 0):.2f}",
            f"Avg Holding Days : {metrics.get('avg_holding_days', 0):.1f}",
        ])

    if metrics.get("bnh_return", 0) != 0:
        lines.extend([
            f"",
            f"B&H Return       : {metrics.get('bnh_return_pct', 0):+.2f}%",
            f"Excess Return    : {metrics.get('excess_return_pct', 0):+.2f}%",
            f"Alpha            : {metrics.get('alpha', 0):+.4f}",
            f"Beta             : {metrics.get('beta', 1):.3f}",
        ])

    return "\n".join(lines)


# =========================================================
# P2: Regime-based 风险分析
# =========================================================

def calculate_regime_metrics(
    equity_curve: pd.DataFrame,
    trades: Optional[pd.DataFrame],
    initial_cash: float,
    regime_column: str = "volatility_regime",
) -> Dict[str, Any]:
    """
    P2: 按市场状态（regime）拆分风险指标。

    简单 regime 定义：
        - high_vol: 年化波动率 > 20%
        - low_vol: 年化波动率 < 10%
        - medium_vol: 其他

    返回：
        按 regime 分类的指标字典
    """
    if equity_curve.empty:
        return {}

    equity = equity_curve["total_equity"].values
    n = len(equity)

    # 计算滚动波动率 (60 天窗口)
    daily_returns = np.diff(equity) / equity[:-1]
    rolling_vol = pd.Series(daily_returns).rolling(60).std() * np.sqrt(252) * 100

    # 标记 regime
    regimes = []
    for vol in rolling_vol:
        if pd.isna(vol):
            regimes.append("unknown")
        elif vol > 20:
            regimes.append("high_vol")
        elif vol < 10:
            regimes.append("low_vol")
        else:
            regimes.append("medium_vol")

    # 补齐前面的 regime
    regimes = ["unknown"] * 59 + regimes

    # 计算各 regime 下的指标
    regime_metrics = {}
    for regime_name in ["high_vol", "medium_vol", "low_vol"]:
        regime_indices = [i for i, r in enumerate(regimes) if r == regime_name]
        # 确保索引在 equity 有效范围内
        regime_indices = [i for i in regime_indices if i < n]
        if len(regime_indices) < 20:  # 至少 20 天
            continue

        # daily_returns 长度比 equity/regimes 少 1，需要调整索引
        # 只取 daily_returns 有效范围内的索引
        valid_return_indices = [i for i in regime_indices if i < len(daily_returns)]
        if len(valid_return_indices) < 5:
            continue

        regime_returns = daily_returns[valid_return_indices]
        regime_equity = equity[regime_indices]

        if len(regime_returns) < 5:
            continue

        total_ret = (regime_equity[-1] - regime_equity[0]) / regime_equity[0] if regime_equity[0] > 0 else 0
        annual_ret = (1 + total_ret) ** (252 / len(regime_returns)) - 1 if len(regime_returns) > 0 else 0
        vol = np.std(regime_returns) * np.sqrt(252)

        # 最大回撤
        cummax = np.maximum.accumulate(regime_equity)
        drawdown = (regime_equity - cummax) / cummax
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0

        regime_metrics[regime_name] = {
            "n_days": len(regime_indices),
            "total_return": float(total_ret),
            "annual_return": float(annual_ret),
            "annual_vol": float(vol),
            "max_drawdown": float(max_dd),
            "sharpe": float(annual_ret / vol) if vol > 0 else 0,
            "win_rate": float((regime_returns > 0).mean()),
        }

    return regime_metrics


def calculate_trade_distribution(
    trades: Optional[pd.DataFrame],
    equity_curve: pd.DataFrame,
) -> Dict[str, Any]:
    """
    P2: 交易分布统计分析。

    返回：
        - pnl_distribution: 盈利/亏损分布
        - holding_days_distribution: 持仓天数分布
        - monthly_trade_count: 月度交易次数
        - yearly_trade_count: 年度交易次数
    """
    dist = {}

    if trades is None or trades.empty:
        return dist

    # PnL 分布
    if "pnl" in trades.columns:
        wins = trades[trades["pnl"] > 0]["pnl"]
        losses = trades[trades["pnl"] < 0]["pnl"]

        dist["pnl_distribution"] = {
            "win_count": int(len(wins)),
            "loss_count": int(len(losses)),
            "win_mean": float(wins.mean()) if len(wins) > 0 else 0,
            "win_median": float(wins.median()) if len(wins) > 0 else 0,
            "loss_mean": float(losses.mean()) if len(losses) > 0 else 0,
            "loss_median": float(losses.median()) if len(losses) > 0 else 0,
            "largest_win": float(wins.max()) if len(wins) > 0 else 0,
            "largest_loss": float(losses.min()) if len(losses) > 0 else 0,
        }

    # 持仓天数分布
    if "holding_days" in trades.columns:
        dist["holding_days_distribution"] = {
            "mean": float(trades["holding_days"].mean()),
            "median": float(trades["holding_days"].median()),
            "min": int(trades["holding_days"].min()),
            "max": int(trades["holding_days"].max()),
            "std": float(trades["holding_days"].std()),
        }

    # 月度/年度交易次数
    if "date" in trades.columns or "buy_date" in trades.columns:
        date_col = "date" if "date" in trades.columns else "buy_date"
        trades_df = trades.copy()
        trades_df["date"] = pd.to_datetime(trades_df[date_col])

        monthly = trades_df.groupby(trades_df["date"].dt.to_period("M")).size()
        dist["monthly_trade_count"] = {
            "mean": float(monthly.mean()),
            "max": int(monthly.max()),
            "min": int(monthly.min()),
            "std": float(monthly.std()),
        }

        yearly = trades_df.groupby(trades_df["date"].dt.year).size()
        dist["yearly_trade_count"] = {
            "mean": float(yearly.mean()),
            "max": int(yearly.max()),
            "min": int(yearly.min()),
            "std": float(yearly.std()),
        }

    return dist


# =========================================================
# Part 3: 市场状态检测 (regime.py)
# =========================================================

class RegimeDetector:
    """
    市场状态检测器。

    支持多种 regime 定义：
    - trend / non-trend: 基于移动平均线斜率
    - high_vol / low_vol / medium_vol: 基于波动率
    - combined: trend + volatility 组合
    """

    def __init__(
        self,
        ma_short: int = 5,
        ma_long: int = 20,
        vol_window: int = 20,
        vol_high_threshold: float = 0.20,
        vol_low_threshold: float = 0.10,
    ):
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.vol_window = vol_window
        self.vol_high_threshold = vol_high_threshold
        self.vol_low_threshold = vol_low_threshold

    def compute_ma_slope(self, close: pd.Series, window: int) -> pd.Series:
        """计算移动平均线的斜率。"""
        ma = close.rolling(window).mean()
        slope = ma.pct_change(window)
        return slope

    def compute_volatility(self, close: pd.Series) -> pd.Series:
        """计算历史波动率（日收益率标准差）。"""
        returns = close.pct_change()
        vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        return vol

    def detect_trend(self, close: pd.Series) -> pd.Series:
        """
        检测趋势状态。

        trend: 短期均线上穿长期均线，或短期均线斜率为正
        non_trend: 其他情况
        """
        ma_short = close.rolling(self.ma_short).mean()
        ma_long = close.rolling(self.ma_long).mean()

        trend = (ma_short > ma_long).astype(int)

        trend = trend.replace({1: REGIME_TREND, 0: REGIME_NON_TREND})

        return trend

    def detect_volatility(self, close: pd.Series) -> pd.Series:
        """
        检测波动率状态。

        high_vol: 年化波动率 > vol_high_threshold (20%)
        low_vol: 年化波动率 < vol_low_threshold (10%)
        medium_vol: 其他
        """
        vol = self.compute_volatility(close)

        vol_regime = pd.Series(index=close.index, data="medium_vol", dtype=object)
        vol_regime[vol > self.vol_high_threshold] = REGIME_HIGH_VOL
        vol_regime[vol < self.vol_low_threshold] = REGIME_LOW_VOL

        return vol_regime

    def detect_combined(
        self,
        close: pd.Series,
    ) -> pd.DataFrame:
        """
        检测组合 regime（trend + volatility）。

        Returns:
            DataFrame with columns: trend, volatility, combined
        """
        trend_regime = self.detect_trend(close)
        vol_regime = self.detect_volatility(close)

        combined = []
        for t, v in zip(trend_regime, vol_regime):
            combined.append(f"{t}_{v}")

        result = pd.DataFrame({
            "trend": trend_regime,
            "volatility": vol_regime,
            "combined": combined,
        }, index=close.index)

        return result

    def fit_predict(self, df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """
        对数据集进行 regime 检测。

        Args:
            df: 包含价格数据的 DataFrame
            price_col: 价格列名

        Returns:
            包含 regime 标签的 DataFrame
        """
        close = df[price_col]

        regimes = self.detect_combined(close)

        regimes["ma_slope_short"] = self.compute_ma_slope(close, self.ma_short)
        regimes["ma_slope_long"] = self.compute_ma_slope(close, self.ma_long)
        regimes["volatility"] = self.compute_volatility(close)

        return regimes


class RegimeAwareBacktester:
    """
    支持 regime 分层的回测器。

    可以在不同 regime 下使用不同参数进行回测。
    """

    def __init__(
        self,
        regime_detector: RegimeDetector,
        regime_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.detector = regime_detector
        self.regime_params = regime_params or {}

    def get_regime_params(self, regime: str) -> Dict[str, Any]:
        """获取特定 regime 下的参数。"""
        return self.regime_params.get(regime, {})

    def backtest_by_regime(
        self,
        df: pd.DataFrame,
        dpoint: pd.Series,
        base_trade_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        按 regime 分层回测。

        Args:
            df: 价格数据
            dpoint: D-point 信号
            base_trade_cfg: 基础交易配置

        Returns:
            按 regime 分层的回测结果
        """
        regimes = self.detector.fit_predict(df)

        results = {}

        for regime_name in regimes["combined"].unique():
            if pd.isna(regime_name):
                continue

            mask = regimes["combined"] == regime_name

            if mask.sum() < 10:
                continue

            regime_dpoint = dpoint[mask]
            regime_df = df[mask]

            regime_cfg = self.get_regime_params(regime_name)
            trade_cfg = {**base_trade_cfg, **regime_cfg}

            bt = backtest_from_dpoint(
                df=regime_df,
                dpoint=regime_dpoint,
                **trade_cfg,
            )

            results[regime_name] = {
                "n_samples": int(mask.sum()),
                "n_trades": len(bt.trades) if bt.trades is not None else 0,
                "equity_curve": bt.equity_curve,
                "trades": bt.trades,
                "config": trade_cfg,
            }

        return results


def compute_regime_metrics(
    equity_curve: pd.DataFrame,
    trades: Optional[pd.DataFrame],
    initial_cash: float,
    regime_labels: Optional[pd.Series] = None,
    regime_type: str = "combined",
) -> Dict[str, Dict[str, float]]:
    """
    计算各 regime 下的性能指标。

    Args:
        equity_curve: 净值曲线
        trades: 交易记录
        initial_cash: 初始资金
        regime_labels: regime 标签序列
        regime_type: regime 类型 ("trend", "volatility", "combined")

    Returns:
        各 regime 下的指标字典
    """
    if regime_labels is None or equity_curve.empty:
        return {}

    regime_labels = regime_labels.fillna("unknown")
    equity = equity_curve["total_equity"].values

    results = {}

    for regime in regime_labels.unique():
        mask = regime_labels == regime

        if mask.sum() < 20:
            continue

        regime_equity = equity[mask.values]

        if len(regime_equity) < 10:
            continue

        total_return = (regime_equity[-1] - initial_cash) / initial_cash

        daily_returns = np.diff(regime_equity) / regime_equity[:-1]
        daily_returns = daily_returns[np.isfinite(daily_returns)]

        if len(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0

            cummax = np.maximum.accumulate(regime_equity)
            drawdown = (regime_equity - cummax) / cummax
            max_dd = np.min(drawdown)
        else:
            sharpe = 0
            max_dd = 0

        n_trades = 0
        if trades is not None and not trades.empty:
            if "date" in trades.columns:
                trade_mask = trades["date"].isin(regime_labels[mask].index)
                n_trades = trade_mask.sum()

        results[regime] = {
            "n_days": int(mask.sum()),
            "total_return": float(total_return),
            "total_return_pct": float(total_return * 100),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "max_drawdown_pct": float(max_dd * 100),
            "trade_count": int(n_trades),
        }

    return results


def create_regime_visualization(
    df: pd.DataFrame,
    regimes: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    创建 regime 可视化数据。

    Returns:
        包含价格和 regime 颜色的 DataFrame
    """
    result = pd.DataFrame(index=df.index)
    result["price"] = df[price_col]
    result["ma_5"] = df[price_col].rolling(5).mean()
    result["ma_20"] = df[price_col].rolling(20).mean()

    if "volatility" in regimes.columns:
        result["volatility"] = regimes["volatility"]

    if "trend" in regimes.columns:
        result["trend"] = regimes["trend"]

    if "combined" in regimes.columns:
        result["regime"] = regimes["combined"]

    regime_colors = {
        f"{REGIME_TREND}_{REGIME_LOW_VOL}": "#2ecc71",
        f"{REGIME_TREND}_{REGIME_MEDIUM_VOL}": "#27ae60",
        f"{REGIME_TREND}_{REGIME_HIGH_VOL}": "#f1c40f",
        f"{REGIME_NON_TREND}_{REGIME_LOW_VOL}": "#3498db",
        f"{REGIME_NON_TREND}_{REGIME_MEDIUM_VOL}": "#2980b9",
        f"{REGIME_NON_TREND}_{REGIME_HIGH_VOL}": "#e74c3c",
    }

    result["regime_color"] = regimes["combined"].map(regime_colors).fillna("#95a5a6")

    return result


class RegimeEnsemble:
    """
    P2: Regime-aware ensemble。

    支持根据当前 regime 选择不同模型或调整权重。
    """

    def __init__(
        self,
        models: Dict[str, Any],
        regime_detector: RegimeDetector,
        weights: Optional[Dict[str, List[float]]] = None,
    ):
        """
        Args:
            models: 模型字典，key 为 regime 名称
            regime_detector: regime 检测器
            weights: 各 regime 下的权重配置
        """
        self.models = models
        self.detector = regime_detector
        self.weights = weights or {}

    def predict(
        self,
        X: pd.DataFrame,
        df: pd.DataFrame,
        mode: str = "hard",
    ) -> np.ndarray:
        """
        预测函数。

        Args:
            X: 特征数据
            df: 价格数据（用于检测 regime）
            mode: "hard" (硬切换) 或 "soft" (软权重)

        Returns:
            预测概率
        """
        regimes = self.detector.fit_predict(df)
        current_regime = regimes["combined"].iloc[-1] if len(regimes) > 0 else "non_trend_medium_vol"

        if mode == "hard":
            if current_regime in self.models:
                return self.models[current_regime].predict_proba(X)[:, 1]
            else:
                base_model = self.models.get("default")
                if base_model:
                    return base_model.predict_proba(X)[:, 1]
                else:
                    return np.zeros(len(X))

        elif mode == "soft":
            predictions = []
            weights = []

            for regime, model in self.models.items():
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X)[:, 1]
                    predictions.append(pred)
                    w = self.weights.get(regime, [1.0])[0]
                    weights.append(w)

            if predictions:
                weights = np.array(weights)
                weights = weights / weights.sum()

                ensemble_pred = np.zeros(len(X))
                for pred, w in zip(predictions, weights):
                    ensemble_pred += pred * w

                return ensemble_pred

        return np.zeros(len(X))


def compute_regime_transition_matrix(
    regimes: pd.Series,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    计算 regime 转移概率矩阵。

    Args:
        regimes: regime 序列
        normalize: 是否归一化为概率

    Returns:
        转移矩阵 DataFrame
    """
    regimes = regimes.fillna("unknown")

    transition_counts = pd.crosstab(regimes[:-1].values, regimes[1:].values)

    if normalize:
        transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)
        transition_probs = transition_probs.fillna(0)
        return transition_probs

    return transition_counts


def get_regime_stationary_distribution(
    transition_matrix: pd.DataFrame,
    n_iter: int = 100,
) -> pd.Series:
    """
    计算 regime 的稳态分布。

    Args:
        transition_matrix: 转移概率矩阵
        n_iter: 迭代次数

    Returns:
        各 regime 的稳态概率
    """
    n = len(transition_matrix)
    pi = np.ones(n) / n

    P = transition_matrix.values

    for _ in range(n_iter):
        pi = pi @ P

    return pd.Series(pi, index=transition_matrix.index)


# =========================================================
# 公开 API 导出列表
# =========================================================
__all__ = [
    # 常量
    "COMMISSION_RATE_BUY",
    "COMMISSION_RATE_SELL",
    "DEFAULT_SLIPPAGE_BPS",
    "DEFAULT_LIMIT_UP_PCT",
    "DEFAULT_LIMIT_DOWN_PCT",
    "ST_LIMIT_PCT",
    "DEFAULT_MIN_LISTING_DAYS",
    "DEFAULT_MIN_DAILY_VOLUME",
    "DEFAULT_FILTER_ST",
    "REGIME_TREND",
    "REGIME_NON_TREND",
    "REGIME_HIGH_VOL",
    "REGIME_LOW_VOL",
    "REGIME_MEDIUM_VOL",
    
    # 数据类
    "ExecutionStats",
    "BacktestResult",
    "PartialFillResult",
    
    # 回测引擎核心函数
    "backtest_from_dpoint",
    "compute_buy_and_hold",
    "apply_slippage",
    "apply_layered_slippage",
    "check_execution_feasibility",
    "get_execution_price",
    "simulate_limit_execution",
    "simulate_partial_fill",
    "calculate_position_size",
    
    # 评估指标
    "calculate_risk_metrics",
    "format_metrics_summary",
    "metric_from_fold_ratios",
    "trade_penalty",
    "backtest_fold_stats",
    "calculate_regime_metrics",
    "calculate_trade_distribution",
    
    # 市场状态检测
    "RegimeDetector",
    "RegimeAwareBacktester",
    "RegimeEnsemble",
    "compute_regime_metrics",
    "create_regime_visualization",
    "compute_regime_transition_matrix",
    "get_regime_stationary_distribution",
]
