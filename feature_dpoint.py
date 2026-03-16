# feature_dpoint.py
"""
特征工程（P3-19 扩展版）。

P3-19 修复：新增经典技术指标特征族 use_ta_indicators
    原版特征全部基于 OHLCV 统计量（动量、波动率、成交量比率等），
    缺席 RSI / MACD / 布林带宽 / OBV 等被广泛使用的技术指标，
    导致搜索空间存在明显盲区。

    新增 4 类指标，均不引入前向偏差（只使用 t 日及以前的数据）：

    ① RSI（相对强弱指数）
        对每个 window 计算 RSI，衡量超买超卖程度。
        rsi_{w} = 100 - 100 / (1 + avg_gain_{w} / avg_loss_{w})
        归一化到 [0, 1]。

    ② MACD（移动平均收敛/发散）
        固定使用 (fast=12, slow=26, signal=9) 参数组合（最通行参数）。
        输出：macd_line（MACD 线）、macd_hist（柱状图，即 MACD 线 - 信号线）。
        两者均做 rolling z-score 归一化，消除量纲。

    ③ 布林带宽（Bollinger Band Width）
        对每个 window 计算：bband_width_{w} = (upper - lower) / mid
        = 2 × std_{w} / sma_{w}，衡量波动率的相对水平。

    ④ OBV（能量潮 / On-Balance Volume）
        OBV_t = OBV_{t-1} + volume_t × sign(close_t - close_{t-1})
        做 rolling z-score 归一化消除量级漂移。

    配置参数：
        use_ta_indicators: bool  — 是否启用（默认 False，向后兼容）
        ta_windows: List[int]    — RSI 和布林带的计算窗口，默认 [6, 14, 20]

    搜索空间：
        search_engine.py 中新增 use_ta_indicators 到 _sample_explore /
        _sample_exploit 的特征配置采样空间。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureMeta:
    feature_names: List[str]
    params: Dict[str, object]
    dpoint_explainer: str


def _safe_log1p(x: pd.Series) -> pd.Series:
    """对序列做 log1p 变换，先 clip 负值为 0，避免对数域报错。"""
    return np.log1p(np.clip(x.astype(float), 0.0, None))


def _rolling_mad(x: pd.Series, window: int) -> pd.Series:
    """滚动中位数绝对偏差（MAD），比标准差更鲁棒的波动率代理。"""
    med = x.rolling(window, min_periods=window).median()
    mad = (x - med).abs().rolling(window, min_periods=window).median()
    return mad


def _rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    """滚动 Z-score 标准化；标准差为 0 时返回 NaN 避免除零。"""
    mu = x.rolling(window, min_periods=window).mean()
    sd = x.rolling(window, min_periods=window).std()
    return (x - mu) / sd.replace(0, np.nan)


# =========================================================
# P3-19：技术指标计算函数
# =========================================================

def _calc_rsi(close: pd.Series, window: int) -> pd.Series:
    """
    计算 RSI（相对强弱指数），归一化到 [0, 1]。

    使用 Wilder 平滑（EMA 变体，alpha=1/window），与 TradingView 等工具一致。
    前 window 个交易日用简单平均初始化，之后滚动更新。

    无前向偏差：t 日 RSI 只使用 close[0..t]。
    """
    delta = close.diff(1)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder 平滑：com = window - 1 等价于 alpha = 1/window
    avg_gain = gain.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window, adjust=False).mean()

    #标准 RSI 公式：RS 越大 → RSI 越高 → 越超买）
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)   # 标准 RSI 公式
    return (rsi / 100.0)                 # 归一化到 [0, 1]


def _calc_macd(close: pd.Series,
               fast: int = 12, slow: int = 26, signal: int = 9
               ) -> Tuple[pd.Series, pd.Series]:
    """
    计算 MACD 线和 MACD 柱状图（histogram）。

    macd_line = EMA(fast) - EMA(slow)
    signal_line = EMA(macd_line, signal)
    macd_hist = macd_line - signal_line

    返回值均做全局 rolling z-score 归一化（消除量级漂移），
    window=slow+signal 确保有足够历史数据时才输出有效值。

    无前向偏差：EMA 只使用 t 日及以前的收盘价。
    """
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd_line - signal_line

    # rolling z-score：用 slow+signal 窗口归一化，避免量纲随价格漂移
    norm_window = slow + signal
    macd_line_z = _rolling_zscore(macd_line, norm_window)
    macd_hist_z = _rolling_zscore(macd_hist, norm_window)

    return macd_line_z, macd_hist_z


def _calc_bband_width(close: pd.Series, window: int, n_std: float = 2.0) -> pd.Series:
    """
    计算布林带宽（Bollinger Band Width）。

    bband_width = (upper - lower) / mid = 2 × n_std × std(w) / sma(w)

    衡量波动率的相对水平，宽带意味着高波动，窄带意味着低波动（挤压）。
    归一化：除以中轨（sma），使其无量纲，便于跨标的、跨价格水平比较。

    无前向偏差：t 日宽度只使用 close[t-window+1..t]。
    """
    sma = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    bwidth = 2.0 * n_std * std / sma.replace(0, np.nan)
    return bwidth


def _calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    计算 OBV（On-Balance Volume，能量潮），并做 rolling z-score 归一化。

    OBV_t = OBV_{t-1} + volume_t × sign(close_t - close_{t-1})
    涨时加量，跌时减量，平盘加零。

    OBV 本身随时间单调增减，无归一化的 OBV 值域跨度极大（难以与其他特征并排）。
    做 rolling z-score 后：OBV 的短期相对变化被保留，绝对量级被消除。

    无前向偏差：t 日 OBV 只累加 close[0..t] 和 volume[0..t]。
    """
    direction = np.sign(close.diff(1))
    direction.iloc[0] = 0.0
    obv_raw = (direction * volume).cumsum()

    # rolling z-score：默认 20 周期，消除量级漂移
    obv_z = _rolling_zscore(obv_raw, window=20)
    return obv_z


# =========================================================
# 主特征构建函数
# =========================================================

def build_features_and_labels(
    df: pd.DataFrame,
    config: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.Series, FeatureMeta]:
    """
    Dpoint_t = P(close_{t+1} > close_t | X_t)
    All features are computed using info <= t (no leakage).

    P3-19 新增配置参数：
        use_ta_indicators: bool     — 是否启用技术指标族（默认 False）
        ta_windows: List[int]       — RSI 和布林带宽的计算窗口（默认 [6, 14, 20]）
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    windows: List[int] = list(config.get("windows", [3, 5, 10, 20]))

    # 原始特征族开关
    use_momentum:       bool = bool(config.get("use_momentum",       True))
    use_volatility:     bool = bool(config.get("use_volatility",     True))
    use_volume:         bool = bool(config.get("use_volume",         True))
    use_candle:         bool = bool(config.get("use_candle",         True))
    use_turnover:       bool = bool(config.get("use_turnover",       True))
    # P3-19：新增技术指标族开关（向后兼容，默认 False）
    use_ta_indicators:  bool = bool(config.get("use_ta_indicators",  False))

    vol_metric:     str = str(config.get("vol_metric",     "std")).lower()
    liq_transform:  str = str(config.get("liq_transform",  "ratio")).lower()
    # P3-19：RSI 和布林带宽的计算窗口，独立于 windows 参数
    ta_windows: List[int] = list(config.get("ta_windows", [6, 14, 20]))

    close    = df["close_qfq"].astype(float)
    open_    = df["open_qfq"].astype(float)
    high     = df["high_qfq"].astype(float)
    low      = df["low_qfq"].astype(float)
    volume   = df["volume"].astype(float)
    amount   = df["amount"].astype(float)
    turnover = df["turnover_rate"].astype(float)

    feats: Dict[str, pd.Series] = {}

    # base return
    ret1 = close.pct_change(1)
    feats["ret_1"] = ret1

    if use_momentum:
        for k in windows:
            feats[f"ret_{k}"] = close.pct_change(k)
            ma = close.rolling(k, min_periods=k).mean()
            feats[f"ma_{k}_ratio"] = close / ma - 1.0

    if use_volatility:
        feats["hl_range"] = (high - low) / close.replace(0, np.nan)

        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1
        ).max(axis=1)
        feats["true_range_norm"] = tr / close.replace(0, np.nan)

        for k in windows:
            if vol_metric == "mad":
                feats[f"vol_mad_{k}"] = _rolling_mad(ret1, k)
            else:
                feats[f"vol_std_{k}"] = ret1.rolling(k, min_periods=k).std()

    if use_volume:
        feats["log_volume"] = _safe_log1p(volume)
        feats["log_amount"] = _safe_log1p(amount)
        for k in windows:
            if liq_transform == "zscore":
                feats[f"volume_z_{k}"]  = _rolling_zscore(volume, k)
                feats[f"amount_z_{k}"]  = _rolling_zscore(amount, k)
            else:
                vma = volume.rolling(k, min_periods=k).mean()
                ama = amount.rolling(k, min_periods=k).mean()
                feats[f"volume_ma_{k}_ratio"] = volume / vma.replace(0, np.nan)
                feats[f"amount_ma_{k}_ratio"] = amount / ama.replace(0, np.nan)

    if use_turnover:
        feats["turnover"] = turnover
        for k in windows:
            if liq_transform == "zscore":
                feats[f"turnover_z_{k}"] = _rolling_zscore(turnover, k)
            else:
                feats[f"turnover_ma_{k}"]  = turnover.rolling(k, min_periods=k).mean()
                feats[f"turnover_std_{k}"] = turnover.rolling(k, min_periods=k).std()

    if use_candle:
        feats["body"]         = (close - open_) / open_.replace(0, np.nan)
        feats["upper_shadow"] = (high - np.maximum(open_, close)) / close.replace(0, np.nan)
        feats["lower_shadow"] = (np.minimum(open_, close) - low)  / close.replace(0, np.nan)

    # P3-19：技术指标族
    if use_ta_indicators:
        # ① RSI：对每个 ta_window 计算，归一化到 [0, 1]
        for w in ta_windows:
            feats[f"rsi_{w}"] = _calc_rsi(close, window=w)

        # ② MACD：固定参数 (12, 26, 9)，输出归一化后的 macd_line 和 macd_hist
        macd_line_z, macd_hist_z = _calc_macd(close, fast=12, slow=26, signal=9)
        feats["macd_line_z"] = macd_line_z
        feats["macd_hist_z"] = macd_hist_z

        # ③ 布林带宽：对每个 ta_window 计算（2σ 标准布林带）
        for w in ta_windows:
            feats[f"bband_width_{w}"] = _calc_bband_width(close, window=w)

        # ④ OBV（rolling z-score 归一化）
        feats["obv_z"] = _calc_obv(close, volume)

    X = pd.DataFrame(feats)

    # label: next day up or not
    y_diff = close.shift(-1) - close
    y = (y_diff > 0).astype(int)

    # 过滤条件：X 所有特征非空 & y_diff 非空（排除最后一行）
    valid = X.notna().all(axis=1) & y_diff.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    X.index = df.loc[X.index, "date"].values
    y.index = X.index

    meta = FeatureMeta(
        feature_names=list(X.columns),
        params={
            "windows":           windows,
            "use_momentum":      use_momentum,
            "use_volatility":    use_volatility,
            "use_volume":        use_volume,
            "use_candle":        use_candle,
            "use_turnover":      use_turnover,
            "vol_metric":        vol_metric,
            "liq_transform":     liq_transform,
            # P3-19 新增
            "use_ta_indicators": use_ta_indicators,
            "ta_windows":        ta_windows,
        },
        dpoint_explainer=(
            "Dpoint_t = P(close_{t+1} > close_t | X_t). "
            "X_t is built from OHLCV/amount/turnover data up to t only (no future leakage). "
            "P3-19: optional TA indicators (RSI, MACD, BB-width, OBV) available via use_ta_indicators=True."
        ),
    )
    return X, y, meta
