# regime.py
"""
市场状态检测与分层评估模块。

P0:
    - 引入 regime detector
    - 最简单两维：trend/non-trend, high-vol/low-vol
    - 分层评估：return, sharpe, max drawdown, trade count

P1:
    - 支持不同 regime 下使用不同阈值/止盈止损/持仓规则
    - 支持以指数或市场宽度指标作为 regime 输入
    - 增加 regime 序列可视化

P2:
    - 支持按 regime 训练不同模型
    - 支持 regime probability / soft regime
    - 支持 regime-aware ensemble
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REGIME_TREND = "trend"
REGIME_NON_TREND = "non_trend"
REGIME_HIGH_VOL = "high_vol"
REGIME_LOW_VOL = "low_vol"
REGIME_MEDIUM_VOL = "medium_vol"


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
            
            from backtester_engine import backtest_from_dpoint
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
    
    from metrics import calculate_risk_metrics
    
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
