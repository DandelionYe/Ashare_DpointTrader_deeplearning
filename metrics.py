# metrics.py
"""
评估指标与折回测统计。

P0: 统一 metrics 层，所有回测结果统一使用同一套指标计算
P1: 增加完整风险指标
P2: 细分风险分析
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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

    # 计算滚动波动率 (60天窗口)
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