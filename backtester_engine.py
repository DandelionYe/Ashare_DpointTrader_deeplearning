# backtester_engine.py
"""
回测执行引擎（P04 + P1-3 + P1-4 + P0/P1/P2 增强版）。

P04 修复：加入 A 股真实交易成本

P1-3 修复：执行价格改用 t+1 日开盘价，消除额外前向偏差
P1-4 修复：持仓天数改用交易日数而非自然日数

P0 增强：统一 execution layer
    - 固定滑点模型
    - 涨跌停/停牌不可成交逻辑
    - 订单拒绝原因记录

P1 增强：
    - 成交量约束
    - 开盘跳空处理
    - ST/上市天数过滤
    - execution stats 输出

P2 增强：
    - 分层滑点模型
    - 部分成交逻辑

公开 API：
    backtest_from_dpoint(df, dpoint, ...) -> BacktestResult
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# =========================================================
# P04：A 股交易成本常量
# =========================================================
# 买入佣金（券商收取，通常 0.02%～0.03%，此处取保守值）
COMMISSION_RATE_BUY: float = 0.0003

# 卖出佣金 + 印花税（0.03% + 0.10%；2023年8月后印花税降至0.05%）
COMMISSION_RATE_SELL: float = 0.0013


# =========================================================
# P0/P1: 执行层常量
# =========================================================
# 固定滑点（按成交价百分比）
DEFAULT_SLIPPAGE_BPS: int = 20  # 20 bps = 0.2%

# 涨跌停幅度（A 股默认 10%，ST 为 5%）
DEFAULT_LIMIT_UP_PCT: float = 0.10
DEFAULT_LIMIT_DOWN_PCT: float = 0.10
ST_LIMIT_PCT: float = 0.05

# 最小上市天数要求（默认 60 个交易日）
DEFAULT_MIN_LISTING_DAYS: int = 60

# 最小成交量要求（默认 100 万成交额）
DEFAULT_MIN_DAILY_VOLUME: float = 1_000_000.0

# 过滤 ST 股
DEFAULT_FILTER_ST: bool = True


# =========================================================
# P0: Execution Stats 数据类
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


# =========================================================
# P0: 数据类
# =========================================================
@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame    # 含 strategy + benchmark 列
    notes: List[str]
    benchmark_curve: pd.DataFrame
    execution_stats: Optional[ExecutionStats] = None  # P1: 执行统计


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
# P0: 统一 Execution Layer
# =========================================================

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
        return False, "ST股过滤"

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

    P1-3 修复：默认使用开盘价（t+1日开盘）避免前向偏差
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


# =========================================================
# P2: 分层滑点模型
# =========================================================

def apply_layered_slippage(
    price: float,
    action: str,
    order_value: float,
) -> float:
    """
    P2: 分层滑点模型。

    滑点随订单规模增加：
        - 小单 (< 10万): 10 bps
        - 中单 (10-50万): 20 bps
        - 大单 (> 50万): 30 bps

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


# =========================================================
# P2: 更细的涨跌停成交近似
# =========================================================

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


# =========================================================
# P2: 部分成交逻辑
# =========================================================

@dataclass
class PartialFillResult:
    """P2: 部分成交结果"""
    filled_shares: int
    remaining_shares: int
    exec_price: float
    status: str  # "full" | "partial" | "rejected"


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


# =========================================================
# P2: 组合资金分配
# =========================================================

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
        "volume": df["volume"].astype(float) if "volume" in df.columns else pd.Series(0.0, index=df.index),
        "amount": df["amount"].astype(float) if "amount" in df.columns else pd.Series(0.0, index=df.index),
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

            # P0: 检查订单可行性（涨跌停、停牌、ST等）
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
            f"Benchmark (Buy&Hold) final equity: {bnh_final:.2f}  |  "
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

    # 停牌标记（开盘价为0或NaN）
    df["suspended"] = (df["open_qfq"] <= 0) | df["open_qfq"].isna()

    # ST 标记（需要从外部数据源传入，这里默认 False）
    df["is_st"] = False

    # 上市天数（从第一行开始计数）
    df["listing_days"] = range(1, len(df) + 1)

    return df
