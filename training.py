# training.py
"""
训练模块合并版 - 整合校准、解释器、持久化、搜索引擎和训练优化器。

本模块整合了以下五个原文件的功能：
1. calibration.py - 概率校准模块
2. explainer.py - 特征和模型解释模块
3. persistence.py - 最优配置的持久化 I/O
4. search_engine.py - 随机搜索主引擎
5. trainer_optimizer.py - 训练器公开 API

主要功能：
- 概率校准：支持 Platt Scaling 和 Isotonic Regression
- 特征解释：支持树模型重要性、Permutation Importance、SHAP
- 持久化：管理 best_so_far.json 和 best_pool.json
- 随机搜索：多轮随机搜索，支持探索/开采平衡
- 训练优化：最终模型训练和 Dpoint 预测

使用示例：
    from training import random_search_train, train_final_model_and_dpoint, TrainResult

    # 执行随机搜索
    result = random_search_train(df_clean, runs=50, seed=42)

    # 训练最终模型
    dpoint, artifacts = train_final_model_and_dpoint(df_clean, result.best_config)

依赖：
    - constants.py (必须)
    - feature_dpoint.py
    - model_builder.py
    - dl_model_builder.py
    - splitter.py
    - metrics.py

版本历史：
    P0: 基础功能整合，修复多进程缓存、Top-K 池、XGBoost CUDA 检测等问题
    P1: 添加概率校准、特征使用跟踪、特征重要性
    P2: 添加参数敏感性分析、滚动校准监控
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# =========================================================
# 外部依赖导入
# =========================================================
from constants import (
    MIN_CLOSED_TRADES_PER_FOLD,
    TARGET_CLOSED_TRADES_PER_FOLD,
    LAMBDA_TRADE_PENALTY,
    BEST_SO_FAR_FILENAME,
    BEST_POOL_FILENAME,
)
from feature_dpoint import build_features_and_labels, FeatureMeta
from models import (
    make_model, predict_dpoint, _try_import_xgboost,
    MLP, LSTM, GRU, CNN1D, Transformer,
    _get_device, train_pytorch_model, predict_pytorch_model,
)
from data import walkforward_splits, final_holdout_split, walkforward_splits_with_embargo, nested_walkforward_splits
from evaluation import metric_from_fold_ratios, trade_penalty, backtest_fold_stats

# SHAP 和 Permutation Importance 可选导入
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance
    PERMUTATION_AVAILABLE = True
except ImportError:
    PERMUTATION_AVAILABLE = False

# =========================================================
# calibration.py - 概率校准模块
# =========================================================
"""
概率校准模块。

P0:
    - 支持 none / platt / isotonic 三种校准方法
    - 校准只在 validation set 上拟合
    - 推理时支持 raw prob → calibrated prob
    - 输出 Brier score 和 calibration curve

P1:
    - 不同模型可使用不同校准方法
    - 阈值策略支持基于 calibrated probability 运行

P2:
    - 加 ECE/MCE 等更细指标
    - 加 rolling calibration drift 检查
"""

CALIBRATION_METHODS = ["none", "platt", "isotonic"]


class ProbabilityCalibrator:
    """
    概率校准器。

    支持三种校准方法:
        - none: 不校准
        - platt: Platt Scaling (使用 logistic regression)
        - isotonic: Isotonic Regression

    校准只在 validation set 上拟合，推理时将 raw probability 转换为 calibrated probability。
    """

    def __init__(self, method: str = "none"):
        """
        初始化校准器。

        Args:
            method: 校准方法，可选 "none", "platt", "isotonic"
        """
        if method not in CALIBRATION_METHODS:
            raise ValueError(f"Unknown calibration method: {method}. Must be one of {CALIBRATION_METHODS}")
        self.method = method
        self.calibrator: Optional[Any] = None
        self._is_fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "ProbabilityCalibrator":
        """
        在 validation set 上拟合校准器。

        Args:
            y_true: 真实标签 (0 或 1)
            y_prob: 模型输出的原始概率

        Returns:
            self
        """
        if self.method == "none":
            self._is_fitted = True
            return self

        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()

        if self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            self.calibrator.fit(y_prob.reshape(-1, 1), y_true)

        elif self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(y_prob, y_true)

        self._is_fitted = True
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """
        将原始概率转换为校准后的概率。

        Args:
            y_prob: 原始概率

        Returns:
            校准后的概率
        """
        y_prob = np.asarray(y_prob).flatten()

        if self.method == "none":
            return y_prob

        if not self._is_fitted:
            raise RuntimeError("Calibrator must be fitted before transform")

        if self.method == "platt":
            calibrated = self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]

        elif self.method == "isotonic":
            calibrated = self.calibrator.transform(y_prob)

        return np.clip(calibrated, 0.0, 1.0)

    def fit_transform(self, y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        """
        拟合并转换。

        Args:
            y_true: 真实标签
            y_prob: 原始概率

        Returns:
            校准后的概率
        """
        return self.fit(y_true, y_prob).transform(y_prob)

    def is_fitted(self) -> bool:
        """检查校准器是否已拟合。"""
        return self._is_fitted

    def get_params(self) -> Dict[str, Any]:
        """获取校准器参数。"""
        return {
            "method": self.method,
            "is_fitted": self._is_fitted,
        }


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    计算 Brier Score。

    Brier Score = (1/N) * Σ(predicted_prob - actual_outcome)^2

    值越小越好，范围 [0, 1]。

    Args:
        y_true: 真实标签 (0 或 1)
        y_prob: 预测概率

    Returns:
        Brier Score
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()
    from sklearn.metrics import brier_score_loss
    return float(brier_score_loss(y_true, y_prob))


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    计算校准曲线数据。

    Args:
        y_true: 真实标签
        y_prob: 预测概率
        n_bins: 分箱数量

    Returns:
        包含以下键的字典:
            - bin_centers: 各箱的中心概率
            - bin_true_fractions: 各箱中正例的实际比例
            - bin_counts: 各箱的样本数量
            - sample_count: 总样本数
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins)

    bin_centers = []
    bin_true_fractions = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        count = int(mask.sum())

        if count > 0:
            bin_center = (bins[i] + bins[i + 1]) / 2
            true_fraction = float(y_true[mask].mean())

            bin_centers.append(bin_center)
            bin_true_fractions.append(true_fraction)
            bin_counts.append(count)

    return {
        "bin_centers": bin_centers,
        "bin_true_fractions": bin_true_fractions,
        "bin_counts": bin_counts,
        "sample_count": len(y_true),
    }


def compute_ece_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    计算 Expected Calibration Error (ECE) 和 Maximum Calibration Error (MCE)。

    ECE = Σ (|B_m| / N) * |acc(B_m) - conf(B_m)|
    MCE = max_m |acc(B_m) - conf(B_m)|

    Args:
        y_true: 真实标签
        y_prob: 预测概率
        n_bins: 分箱数量

    Returns:
        包含 ECE 和 MCE 的字典
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins)

    ece = 0.0
    mce = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        count = int(mask.sum())

        if count > 0:
            bin_accuracy = float(y_true[mask].mean())
            bin_confidence = float(y_prob[mask].mean())
            bin_error = abs(bin_accuracy - bin_confidence)

            ece += (count / total_samples) * bin_error
            mce = max(mce, bin_error)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "n_bins": n_bins,
    }


def compute_all_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    计算所有校准指标。

    Args:
        y_true: 真实标签
        y_prob: 预测概率
        n_bins: 分箱数量

    Returns:
        包含所有校准指标的字典:
            - brier_score: Brier Score
            - ece: Expected Calibration Error
            - mce: Maximum Calibration Error
            - calibration_curve: 校准曲线数据
    """
    brier = compute_brier_score(y_true, y_prob)
    ece_mce = compute_ece_mce(y_true, y_prob, n_bins)
    curve = compute_calibration_curve(y_true, y_prob, n_bins)

    return {
        "brier_score": brier,
        "ece": ece_mce["ece"],
        "mce": ece_mce["mce"],
        "n_bins": n_bins,
        "calibration_curve": curve,
    }


class RollingCalibrationMonitor:
    """
    滚动校准漂移监控器。

    用于检测模型在校准上的漂移，为后续滚动再训练提供依据。
    """

    def __init__(
        self,
        window_size: int = 500,
        n_bins: int = 10,
        drift_threshold: float = 0.05,
    ):
        """
        初始化滚动校准监控器。

        Args:
            window_size: 滚动窗口大小
            n_bins: ECE 计算的分箱数
            drift_threshold: 漂移阈值，超过此值认为校准失效
        """
        self.window_size = window_size
        self.n_bins = n_bins
        self.drift_threshold = drift_threshold

        self._history: List[Dict[str, Any]] = []
        self._baseline_ece: Optional[float] = None

    def update(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, Any]:
        """
        更新监控器状态。

        Args:
            y_true: 真实标签
            y_prob: 预测概率

        Returns:
            包含以下键的字典:
                - ece: 当前窗口 ECE
                - mce: 当前窗口 MCE
                - brier: 当前窗口 Brier Score
                - drift: 相对基线的漂移量
                - is_drifted: 是否检测到校准漂移
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        if len(y_true) < self.window_size:
            return {
                "ece": None,
                "mce": None,
                "brier": None,
                "drift": None,
                "is_drifted": False,
                "sample_size": len(y_true),
            }

        window_true = y_true[-self.window_size:]
        window_prob = y_prob[-self.window_size:]

        metrics = compute_all_calibration_metrics(window_true, window_prob, self.n_bins)

        record = {
            "ece": metrics["ece"],
            "mce": metrics["mce"],
            "brier": metrics["brier_score"],
        }
        self._history.append(record)

        if self._baseline_ece is None:
            self._baseline_ece = metrics["ece"]

        drift = abs(metrics["ece"] - self._baseline_ece) if self._baseline_ece is not None else 0.0
        is_drifted = drift > self.drift_threshold

        return {
            "ece": metrics["ece"],
            "mce": metrics["mce"],
            "brier": metrics["brier_score"],
            "drift": drift,
            "is_drifted": is_drifted,
            "sample_size": self.window_size,
        }

    def reset_baseline(self):
        """重置基线，使用当前 ECE 作为新的基线。"""
        if self._history:
            self._baseline_ece = self._history[-1]["ece"]

    def get_history(self) -> List[Dict[str, Any]]:
        """获取历史记录。"""
        return self._history.copy()

    def get_status(self) -> Dict[str, Any]:
        """获取当前监控状态。"""
        if not self._history:
            return {
                "baseline_ece": None,
                "current_ece": None,
                "drift": None,
                "is_drifted": False,
            }

        current_ece = self._history[-1]["ece"]
        drift = abs(current_ece - self._baseline_ece) if self._baseline_ece is not None else 0.0

        return {
            "baseline_ece": self._baseline_ece,
            "current_ece": current_ece,
            "drift": drift,
            "is_drifted": drift > self.drift_threshold,
            "n_records": len(self._history),
        }


def create_calibrator_from_config(config: Dict[str, Any]) -> ProbabilityCalibrator:
    """
    从配置创建校准器。

    Args:
        config: 包含 "calibration_method" 键的配置字典

    Returns:
        ProbabilityCalibrator 实例
    """
    method = config.get("calibration_method", "none")
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Invalid calibration method: {method}")
    return ProbabilityCalibrator(method=method)


# =========================================================
# explainer.py - 特征解释和模型解释模块
# =========================================================
"""
特征解释和模型解释模块。

P0:
    - 记录 feature usage frequency
    - 计算 global importance（树模型 importance 和 permutation importance）
    - 支持特征排名和特征组排名

P1:
    - 统一 explainer 接口
    - 对树模型增加 SHAP summary
    - 输出 feature ranking, feature group ranking, window length usage stats
    - 支持按 fold 汇总 importance 稳定性

P2:
    - 增加 local explanation
    - 增加不同 regime 下的重要特征对比
    - 增加不同 ticker 下的重要特征对比
    - 增加特征删减实验
"""

FEATURE_GROUPS = {
    "momentum": ["rsi", "macd", "mom", "return", "atr"],
    "volatility": ["vol", "std", "bb_width", "atr_ratio"],
    "volume": ["vol", "amount", "turnover", "vol_ma"],
    "candle": ["open", "high", "low", "close", "body", "shadow"],
    "turnover": ["turnover", "pe", "pb"],
    "ta": ["rsi", "macd", "bb", "atr"],
}


class FeatureImportanceExplainer:
    """
    统一的特征重要性解释器。

    支持:
    - 树模型的原生 feature_importances_
    - Permutation Importance
    - SHAP values (如果可用)
    """

    def __init__(
        self,
        model: Any,
        model_type: str,
        feature_names: List[str],
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.model_type = model_type
        self.feature_names = list(feature_names)
        self.X_train = X_train
        self.y_train = y_train

        self._importance: Optional[np.ndarray] = None
        self._shap_values: Optional[np.ndarray] = None
        self._permutation_importance: Optional[np.ndarray] = None

    def get_tree_importance(self) -> Optional[np.ndarray]:
        """获取树模型的原生 feature importance。"""
        if hasattr(self.model, "feature_importances_"):
            self._importance = self.model.feature_importances_
            return self._importance

        if hasattr(self.model, "named_steps"):
            for step_name, step in self.model.named_steps.items():
                if hasattr(step, "feature_importances_"):
                    self._importance = step.feature_importances_
                    return self._importance

        return None

    def compute_permutation_importance(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> Optional[np.ndarray]:
        """计算 Permutation Importance。"""
        if not PERMUTATION_AVAILABLE:
            return None

        try:
            if hasattr(self.model, "predict"):
                result = permutation_importance(
                    self.model, X_val.values, y_val,
                    n_repeats=n_repeats, random_state=random_state, n_jobs=-1
                )
                self._permutation_importance = result.importances_mean
                return self._permutation_importance
        except Exception:
            pass

        return None

    def compute_shap_values(
        self,
        X: pd.DataFrame,
        background_size: int = 100,
    ) -> Optional[np.ndarray]:
        """计算 SHAP values。"""
        if not SHAP_AVAILABLE:
            return None

        try:
            if self.model_type in ["xgb", "xgboost"]:
                import xgboost as xgb
                if isinstance(self.model, xgb.XGBClassifier):
                    explainer = shap.TreeExplainer(self.model)
                    self._shap_values = explainer.shap_values(X.values)
                    return self._shap_values

            elif self.model_type in ["logreg", "sgd"]:
                from sklearn.linear_model import LogisticRegression
                if isinstance(self.model, LogisticRegression):
                    explainer = shap.LinearExplainer(self.model, X.values)
                    self._shap_values = explainer.shap_values(X.values)
                    return self._shap_values

            elif hasattr(self.model, "predict_proba"):
                if self.X_train is not None and len(self.X_train) > background_size:
                    background = shap.sample(self.X_train.values, background_size)
                else:
                    background = X.values[:min(background_size, len(X))]

                if self.model_type in ["mlp", "lstm", "gru", "cnn", "transformer"]:
                    pass
                else:
                    explainer = shap.KernelExplainer(self.model.predict_proba, background)
                    self._shap_values = explainer.shap_values(X.values)
                    return self._shap_values
        except Exception:
            pass

        return None

    def get_global_importance(
        self,
        method: str = "auto",
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        获取全局特征重要性。

        Args:
            method: "tree", "permutation", "shap", "auto"
            X_val: 验证集特征
            y_val: 验证集标签

        Returns:
            包含 importance 排名的字典
        """
        importance = None
        method_used = method

        if method == "auto":
            if self.model_type == "xgb":
                importance = self.get_tree_importance()
                method_used = "tree" if importance is not None else "permutation"
            else:
                method_used = "permutation"

        if method == "tree" or (method == "auto" and importance is None):
            importance = self.get_tree_importance()
            method_used = "tree"

        if importance is None and X_val is not None and y_val is not None:
            importance = self.compute_permutation_importance(X_val, y_val)
            method_used = "permutation"

        if importance is None and X_val is not None:
            importance = self.compute_shap_values(X_val)
            method_used = "shap"

        if importance is None:
            return {"error": "No importance method available"}

        sorted_idx = np.argsort(importance)[::-1]

        return {
            "method": method_used,
            "importance": importance.tolist(),
            "feature_names": self.feature_names,
            "ranking": [
                {
                    "rank": i + 1,
                    "feature": self.feature_names[sorted_idx[i]],
                    "importance": float(importance[sorted_idx[i]]),
                }
                for i in range(len(sorted_idx))
            ],
        }

    def get_shap_summary(
        self,
        X: pd.DataFrame,
        plot_type: str = "bar",
    ) -> Optional[Dict[str, Any]]:
        """获取 SHAP summary。"""
        if not SHAP_AVAILABLE:
            return None

        shap_values = self.compute_shap_values(X)
        if shap_values is None:
            return None

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]

        return {
            "shap_values": shap_values.tolist() if shap_values.shape[0] <= 100 else None,
            "mean_abs_shap": mean_abs_shap.tolist(),
            "feature_names": self.feature_names,
            "ranking": [
                {
                    "rank": i + 1,
                    "feature": self.feature_names[sorted_idx[i]],
                    "mean_abs_shap": float(mean_abs_shap[sorted_idx[i]]),
                }
                for i in range(min(20, len(sorted_idx)))
            ],
        }


class FeatureUsageTracker:
    """跟踪搜索过程中的特征使用频率。"""

    def __init__(self):
        self._usage_counts: Dict[str, int] = {}
        self._total_candidates: int = 0

    def record_candidate(self, feature_config: Dict[str, Any]):
        """记录一个候选的特征使用情况。"""
        self._total_candidates += 1

        if feature_config.get("use_momentum"):
            self._increment("momentum")
        if feature_config.get("use_volatility"):
            self._increment("volatility")
        if feature_config.get("use_volume"):
            self._increment("volume")
        if feature_config.get("use_candle"):
            self._increment("candle")
        if feature_config.get("use_turnover"):
            self._increment("turnover")
        if feature_config.get("use_ta_indicators"):
            self._increment("ta_indicators")

        windows = feature_config.get("windows", [])
        for w in windows:
            self._increment(f"window_{w}")

        vol_metric = feature_config.get("vol_metric")
        if vol_metric:
            self._increment(f"vol_metric_{vol_metric}")

        liq_transform = feature_config.get("liq_transform")
        if liq_transform:
            self._increment(f"liq_transform_{liq_transform}")

    def _increment(self, key: str):
        self._usage_counts[key] = self._usage_counts.get(key, 0) + 1

    def get_usage_stats(self) -> Dict[str, Any]:
        """获取特征使用统计。"""
        if self._total_candidates == 0:
            return {"total_candidates": 0}

        group_usage = {}
        for key, count in self._usage_counts.items():
            group_usage[key] = {
                "count": count,
                "frequency": count / self._total_candidates,
            }

        return {
            "total_candidates": self._total_candidates,
            "group_usage": group_usage,
        }

    def get_feature_group_stats(self) -> Dict[str, float]:
        """获取特征组使用频率。"""
        if self._total_candidates == 0:
            return {}

        return {
            "momentum": self._usage_counts.get("momentum", 0) / self._total_candidates,
            "volatility": self._usage_counts.get("volatility", 0) / self._total_candidates,
            "volume": self._usage_counts.get("volume", 0) / self._total_candidates,
            "candle": self._usage_counts.get("candle", 0) / self._total_candidates,
            "turnover": self._usage_counts.get("turnover", 0) / self._total_candidates,
            "ta_indicators": self._usage_counts.get("ta_indicators", 0) / self._total_candidates,
        }

    def get_window_stats(self) -> Dict[str, float]:
        """获取窗口长度使用统计。"""
        if self._total_candidates == 0:
            return {}

        window_counts = {}
        for key, count in self._usage_counts.items():
            if key.startswith("window_"):
                window_counts[key.replace("window_", "")] = count

        return {
            w: c / self._total_candidates
            for w, c in window_counts.items()
        }


class LocalExplainer:
    """局部特征解释器，用于解释单个预测。"""

    def __init__(
        self,
        model: Any,
        model_type: str,
        feature_names: List[str],
    ):
        self.model = model
        self.model_type = model_type
        self.feature_names = list(feature_names)

    def explain_instance(
        self,
        X_instance: np.ndarray,
        method: str = "shap",
    ) -> Dict[str, Any]:
        """解释单个样本的预测。"""
        if method == "shap" and SHAP_AVAILABLE:
            return self._explain_with_shap(X_instance)
        else:
            return self._explain_with_lime(X_instance)

    def _explain_with_shap(self, X_instance: np.ndarray) -> Dict[str, Any]:
        """使用 SHAP 解释单个样本。"""
        try:
            if self.model_type == "xgb":
                import xgboost as xgb
                if isinstance(self.model, xgb.XGBClassifier):
                    explainer = shap.TreeExplainer(self.model)
                    shap_values = explainer.shap_values(X_instance.reshape(1, -1))

                    sorted_idx = np.argsort(np.abs(shap_values[0]))[::-1]

                    return {
                        "method": "shap",
                        "shap_values": shap_values[0].tolist(),
                        "explanation": [
                            {
                                "feature": self.feature_names[sorted_idx[i]],
                                "shap_value": float(shap_values[0][sorted_idx[i]]),
                                "abs_value": float(np.abs(shap_values[0][sorted_idx[i]])),
                            }
                            for i in range(min(10, len(sorted_idx)))
                        ],
                    }
        except Exception:
            pass

        return {"error": "SHAP explanation failed"}

    def _explain_with_lime(self, X_instance: np.ndarray) -> Dict[str, Any]:
        """使用 LIME 风格解释（基于特征值差异）。"""
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X_instance.reshape(1, -1))[0]

                if hasattr(self.model, "coef_"):
                    coefs = self.model.coef_.flatten()
                    feature_contribution = coefs * X_instance
                    sorted_idx = np.argsort(np.abs(feature_contribution))[::-1]

                    return {
                        "method": "coefficient",
                        "probabilities": proba.tolist(),
                        "explanation": [
                            {
                                "feature": self.feature_names[sorted_idx[i]],
                                "contribution": float(feature_contribution[sorted_idx[i]]),
                            }
                            for i in range(min(10, len(sorted_idx)))
                        ],
                    }
        except Exception:
            pass

        return {"error": "Local explanation failed"}


def compute_feature_group_ranking(
    importance: np.ndarray,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """计算特征组的重要性排名。"""
    group_importance: Dict[str, float] = {}

    for group_name, keywords in FEATURE_GROUPS.items():
        group_importance[group_name] = 0.0
        for i, fname in enumerate(feature_names):
            fname_lower = fname.lower()
            if any(kw in fname_lower for kw in keywords):
                group_importance[group_name] += importance[i]

    sorted_groups = sorted(
        group_importance.items(), key=lambda x: x[1], reverse=True
    )

    return [
        {"rank": i + 1, "group": g, "importance": float(imp)}
        for i, (g, imp) in enumerate(sorted_groups)
    ]


def compute_feature_deletion_experiment(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    metric: str = "accuracy",
    cv: int = 5,
) -> Dict[str, Any]:
    """
    特征删减实验：逐步删除重要特征，检查性能变化。

    用于验证哪些特征是真正必要的。
    """
    from sklearn.model_selection import cross_val_score

    baseline_score = np.mean(cross_val_score(model, X.values, y, cv=cv, scoring=metric))

    feature_importance = np.zeros(len(feature_names))
    if hasattr(model, "feature_importances_"):
        feature_importance = model.feature_importances_
    elif hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_importances_"):
                feature_importance = step.feature_importances_
                break

    sorted_idx = np.argsort(feature_importance)[::-1]

    results = []
    for n_keep in [len(feature_names), len(feature_names) // 2, len(feature_names) // 4, 3]:
        if n_keep >= len(feature_names):
            keep_idx = list(range(len(feature_names)))
            score = baseline_score
        else:
            keep_idx = sorted_idx[:n_keep].tolist()
            X_subset = X.iloc[:, keep_idx]
            try:
                score = np.mean(cross_val_score(model, X_subset.values, y, cv=cv, scoring=metric))
            except Exception:
                score = 0.0

        results.append({
            "n_features": n_keep,
            "score": float(score),
            "score_drop": float(baseline_score - score),
            "features_kept": [feature_names[i] for i in keep_idx[:10]],
        })

    return {
        "baseline_score": float(baseline_score),
        "feature_deletion_results": results,
    }


def compute_regime_feature_importance(
    explainer: FeatureImportanceExplainer,
    X: pd.DataFrame,
    regime_labels: np.ndarray,
) -> Dict[str, Any]:
    """
    计算不同 regime 下的特征重要性对比。

    Args:
        explainer: 特征解释器
        X: 特征数据
        regime_labels: regime 标签 (如 "high_vol", "low_vol", "medium_vol")

    Returns:
        各 regime 下的特征重要性
    """
    regimes = np.unique(regime_labels)
    regime_importance = {}

    for regime in regimes:
        mask = regime_labels == regime
        X_regime = X.iloc[mask]

        if len(X_regime) < 10:
            continue

        importance_dict = explainer.get_global_importance(
            method="tree", X_val=X_regime, y_val=None
        )

        regime_importance[str(regime)] = {
            "n_samples": int(mask.sum()),
            "top_features": importance_dict.get("ranking", [])[:10],
        }

    return regime_importance


# =========================================================
# persistence.py - 最优配置的持久化 I/O
# =========================================================
"""
最优配置的持久化 I/O。
管理 best_so_far.json（全局最优）和 best_pool.json（Top-K 池）。
"""


def config_hash(cfg: Dict[str, object]) -> str:
    """对配置字典做 SHA-256，用于去重和缓存 key。"""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def best_so_far_path(output_dir: str) -> str:
    return os.path.join(output_dir, BEST_SO_FAR_FILENAME)


def best_pool_path(output_dir: str) -> str:
    return os.path.join(output_dir, BEST_POOL_FILENAME)


def load_best_so_far(output_dir: str) -> Optional[Dict[str, object]]:
    path = best_so_far_path(output_dir)
    if not output_dir or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        return blob.get("best_config")
    except Exception:
        return None


def load_best_so_far_metric(output_dir: str) -> Optional[float]:
    path = best_so_far_path(output_dir)
    if not output_dir or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        m = blob.get("best_metric", None)
        return float(m) if m is not None else None
    except Exception:
        return None


def save_best_so_far(
    output_dir: str,
    best_config: Dict[str, object],
    best_metric: float,
) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    blob = {
        "saved_at": pd.Timestamp.now().isoformat(),
        "best_metric": float(best_metric),
        "best_config": best_config,
    }
    with open(best_so_far_path(output_dir), "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)


def load_best_pool(output_dir: str) -> List[Dict[str, object]]:
    path = best_pool_path(output_dir)
    if not output_dir or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        items = blob.get("items", [])
        return items if isinstance(items, list) else []
    except Exception:
        return []


def save_best_pool(output_dir: str, items: List[Dict[str, object]]) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    blob = {
        "saved_at": pd.Timestamp.now().isoformat(),
        "items": items,
    }
    with open(best_pool_path(output_dir), "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)


def update_best_pool(
    output_dir: str,
    candidate_config: Dict[str, object],
    candidate_metric: float,
    top_k: int,
) -> None:
    """将候选配置加入 Top-K 池，按 metric 降序维护，去重。"""
    items = load_best_pool(output_dir)
    cand_hash = config_hash(candidate_config)
    items = [it for it in items if it.get("hash") != cand_hash]
    items.append({"metric": float(candidate_metric), "hash": cand_hash, "config": candidate_config})
    items = sorted(items, key=lambda x: float(x.get("metric", -np.inf)), reverse=True)[: int(top_k)]
    save_best_pool(output_dir, items)


# =========================================================
# search_engine.py - 随机搜索主引擎
# =========================================================
"""
随机搜索主引擎（P0 缺陷修复版）。

P03 修复：XGBoost CUDA 检测语法错误
    - 原版 `from xgb import callback` 模块名错误，且 xgb 变量作用域不对
    - 修复后：直接使用 _try_import_xgboost() 返回的模块对象，无多余 import

P01 修复：feat_cache 在多进程下完全失效
    - 原版：feat_cache 是闭包变量，loky 后端 fork 子进程后每个 worker 拿到的是
      主进程内存的独立副本，写入无效，缓存命中率永远为零
    - 修复后：在主进程按 feature_config hash 预计算所有唯一特征，
      将计算结果作为显式参数传给 _eval_candidate，完全绕开跨进程共享问题

P02 修复：update_best_pool 死代码，Top-K 池从未被写入或利用
    - 修复后：每轮评估结束后将有效结果写入 Top-K 池；
      exploit 阶段以 30% 概率从池中按 metric 加权采样基础配置，
      增加搜索多样性，避免 exploit 退化为单点微扰

P05 修复：exploit 候选全部在搜索开始前生成，无在线反馈
    - 原版：先生成全部候选，再并行评估，中途无论发现多好的配置都不会
      影响后续候选的生成，exploit_ratio 形同虚设
    - 修复后：将 runs 均分为 n_rounds 轮，每轮并行评估后立即更新 incumbent
      和诊断状态，下一轮 exploit 候选基于最新 incumbent 生成
"""


# =========================================================
# P03 修复：_detect_cuda 语法与作用域错误修正
# =========================================================
def _detect_cuda() -> bool:
    """
    轻量级 CUDA 检测，不实际完整训练模型。

    P03 修复点：
        原版 `from xgb import callback` 模块名写错（应为 xgboost），
        且后续 `xgb.XGBClassifier` 引用的是内层作用域中未定义的 xgb，
        导致每次都走 except 分支，XGBoost GPU 加速永远被静默禁用。
    """
    # 第一步：优先检测 PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass

    # 第二步：检测 XGBoost GPU 支持
    # 使用 _try_import_xgboost() 返回的模块对象，不再单独 import
    xgb = _try_import_xgboost()
    if xgb is None:
        return False
    try:
        _X = np.random.rand(10, 4).astype("float32")
        _y = np.array([0, 1] * 5)
        clf = xgb.XGBClassifier(
            n_estimators=1,
            max_depth=1,
            device="cuda",
            tree_method="hist",
            verbosity=0,
        )
        clf.fit(_X, _y)
        return True
    except Exception:
        return False


#  None 占位，首次真正需要时才检测
_CUDA_AVAILABLE: bool | None = None


def _get_cuda_available() -> bool:
    """懒加载单例：首次调用时执行检测，结果缓存，后续调用直接返回。"""
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is None:
        _CUDA_AVAILABLE = _detect_cuda()
    return _CUDA_AVAILABLE


# =========================================================
# 搜索空间定义
# =========================================================
@dataclass
class SearchSpaces:
    # 模型类型共享的离散维度列表
    hidden_dims: List[int]           # [32, 64, 128]
    learning_rates: List[float]      # [0.0005, 0.001, 0.003, 0.005]
    batch_sizes: List[int]           # [32, 64, 128, 256]
    seq_lens: List[int]              # [10, 20]
    num_layers_pool: List[int]       # [1, 2]
    # LogReg / SGD 组合少，保留预展开
    logreg_choices: List[Dict[str, Any]]
    sgd_choices: List[Dict[str, Any]]
    # XGBoost 参数少，保留预展开
    xgb_param_pool: List[Dict[str, Any]]
    # 特征和交易参数
    window_pool: List[List[int]]
    xgb_available: bool
    vol_metric_pool: List[str]
    liq_transform_pool: List[str]
    buy_pool: List[float]
    sell_pool: List[float]
    confirm_pool: List[int]
    min_hold_pool: List[int]
    max_hold_pool: List[int]
    take_profit_pool: List[Optional[float]]
    stop_loss_pool: List[Optional[float]]
    ta_window_pool: List[List[int]]    # P3-19：RSI / 布林带宽的窗口候选列表
    calibration_pool: List[str]          # P1: 校准方法候选列表
    # 运行时传入
    input_dim: int                     # 从调用方传入，用于运行时构建 config
    cuda_available: bool


def _build_search_spaces(seed: int, input_dim: int) -> SearchSpaces:
    xgb_available = _try_import_xgboost() is not None
    cuda_available = _get_cuda_available()

    # LogReg / SGD 组合少，保持预展开
    C_pool = list(np.logspace(-2, 2, 13))
    logreg_choices = [
        {"penalty": p, "solver": s, "C": C, "class_weight": cw}
        for C in C_pool
        for cw in [None, "balanced"]
        for p, s in [("l2", "lbfgs"), ("l1", "liblinear")]
    ]
    sgd_choices = [
        {"alpha": alpha, "penalty": "l2", "class_weight": "balanced"}
        for alpha in np.logspace(-5, -2, 10)
    ]

    # XGBoost 参数少，保持预展开
    device_args = {"device": "cuda", "tree_method": "hist"} if cuda_available else {"n_jobs": 4}
    xgb_param_pool = [
        dict(n_estimators=200, max_depth=d, learning_rate=0.05,
             objective="binary:logistic", eval_metric="logloss", **device_args)
        for d in [2, 3, 4]
    ] if xgb_available else []

    return SearchSpaces(
        hidden_dims=[32, 64, 128],
        learning_rates=[0.0005, 0.001, 0.003, 0.005],
        batch_sizes=[32, 64, 128, 256],
        seq_lens=[10, 20],
        num_layers_pool=[1, 2],
        logreg_choices=logreg_choices,
        sgd_choices=sgd_choices,
        xgb_param_pool=xgb_param_pool,
        window_pool=[[2, 3, 5, 8, 13, 21], [5, 10, 20, 60], [3, 7, 14, 28]],
        xgb_available=xgb_available,
        vol_metric_pool=["std", "mad"],
        liq_transform_pool=["ratio", "zscore"],
        buy_pool=[0.52, 0.55, 0.58, 0.62],
        sell_pool=[0.38, 0.42, 0.45, 0.48],
        confirm_pool=[1, 2],
        min_hold_pool=[1, 2],
        max_hold_pool=[15, 30, 60],
        take_profit_pool=[None, 0.10, 0.15],
        stop_loss_pool=[None, 0.05, 0.08],
        ta_window_pool=[[6, 14], [14, 20], [6, 14, 20]],
        calibration_pool=["none"] * 7 + ["platt", "platt", "isotonic"],
        input_dim=input_dim,
        cuda_available=cuda_available,
    )


# =========================================================
# 工具函数
# =========================================================
def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def _diagnose_from_incumbent(inc_info: Dict[str, Any]) -> Dict[str, float]:
    diag = {"trade_too_few": 0.0, "trade_too_many": 0.0}
    avg_trades = float(inc_info.get("avg_closed_trades", float("nan")))
    target = float(TARGET_CLOSED_TRADES_PER_FOLD)
    if not np.isnan(avg_trades):
        if avg_trades < target:
            diag["trade_too_few"] = float(np.clip((target - avg_trades) / target, 0.0, 1.0))
        elif avg_trades > target:
            diag["trade_too_many"] = float(np.clip((avg_trades - target) / target, 0.0, 1.0))
    return diag


_EARLY_STOP_RATIO: float = 0.85


# =========================================================
# 候选评估（_eval_candidate 签名不变，但 computed_feats 现在由主进程传入）
# =========================================================
def _eval_candidate(
    candidate: Dict[str, Any],
    df_clean: pd.DataFrame,
    max_features: int,
    n_folds: int,
    train_start_ratio: float,
    wf_min_rows: int,
    computed_feats: Optional[Tuple[pd.DataFrame, pd.Series, FeatureMeta]],
) -> Tuple[float, float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    评估单个候选配置。

    P01 修复说明：
        computed_feats 现在始终由主进程预计算后传入，不再在 worker 内部
        维护 feat_cache 闭包（多进程下闭包缓存完全无效）。
        当 computed_feats 为 None 时仍支持惰性计算（兼容单独调用）。
    """
    fold_details: List[Dict[str, Any]] = []

    if computed_feats is None:
        feat_cfg = candidate["feature_config"]
        computed_feats = build_features_and_labels(df_clean, feat_cfg)
        if computed_feats is None:
            return (-np.inf, 100000.0, {"skip": "feat_fail"}, [])

    X, y, meta = computed_feats
    trade_cfg = candidate["trade_config"]
    initial_cash = float(trade_cfg["initial_cash"])
    early_stop_floor = initial_cash * _EARLY_STOP_RATIO
    cand_seed = int(candidate.get("candidate_seed", 42))

    if len(X.columns) > max_features:
        return (-np.inf, initial_cash, {"skip": "too_many_feats", "n_features": len(X.columns)}, [])

    splits = walkforward_splits(X, y, n_folds=n_folds, train_start_ratio=train_start_ratio, min_rows=wf_min_rows)
    if not splits:
        return (-np.inf, initial_cash, {"skip": "no_splits"}, [])

    ratios, equities, closed_trades = [], [], []
    device = _get_device()
    fold_idx = 0

    calibration_method = str(candidate.get("calibration_config", {}).get("method", "none"))
    use_calibration = calibration_method != "none"
    use_calibrated_threshold = candidate.get("calibration_config", {}).get("use_for_threshold", False)

    all_y_true: List[float] = []
    all_y_prob_raw: List[float] = []
    all_y_prob_calibrated: List[float] = []

    fold_calibration_metrics: List[Dict[str, Any]] = []

    # 用实际特征维度动态覆盖 input_dim，避免 PyTorch 线性层维度不匹配）
    for (X_tr, y_tr), (X_va, y_va) in splits:
        model_type = str(candidate["model_config"]["model_type"])
        if model_type in ["mlp", "lstm", "gru", "cnn", "transformer"]:
            actual_cfg = {**candidate["model_config"], "input_dim": X_tr.shape[1]}
            trained_model = train_pytorch_model(X_tr, y_tr, actual_cfg, device,
                                                X_val=X_va, y_val=y_va)
            dp_val_raw = predict_pytorch_model(
                trained_model, X_va, device,
                seq_len=int(actual_cfg.get("seq_len", 20))
            )
        else:
            model = make_model(candidate, seed=cand_seed)
            model.fit(
                X_tr.values if isinstance(model, Pipeline) else X_tr,
                y_tr.values if isinstance(model, Pipeline) else y_tr,
            )
            dp_val_raw = predict_dpoint(model, X_va)

        dp_val = dp_val_raw.copy()

        if use_calibration and len(y_va) >= 20:
            try:
                calibrator = ProbabilityCalibrator(method=calibration_method)
                calibrator.fit(y_va.values, dp_val_raw.values)
                dp_val = pd.Series(
                    calibrator.transform(dp_val_raw.values),
                    index=dp_val_raw.index,
                    name=dp_val_raw.name
                )

                cal_metrics = compute_all_calibration_metrics(
                    y_va.values, dp_val.values, n_bins=10
                )
                fold_calibration_metrics.append(cal_metrics)

                all_y_true.extend(y_va.values.tolist())
                all_y_prob_raw.extend(dp_val_raw.values.tolist())
                all_y_prob_calibrated.extend(dp_val.values.tolist())
            except Exception:
                pass

        fold_stats = backtest_fold_stats(df_clean, X_va, dp_val, trade_cfg)
        equity_end = float(fold_stats["equity_end"])
        n_closed = int(fold_stats["n_closed"])

        if n_closed < MIN_CLOSED_TRADES_PER_FOLD:
            return (-np.inf, initial_cash, {"skip": "too_few_trades"}, [])
        if equity_end < early_stop_floor:
            return (-np.inf, initial_cash, {"skip": "early_stop"}, [])

        equities.append(equity_end)
        ratios.append(equity_end / initial_cash)
        closed_trades.append(n_closed)

        fold_details.append({
            "fold_idx": fold_idx,
            "equity_end": equity_end,
            "ratio": equity_end / initial_cash,
            "n_closed_trades": n_closed,
        })
        fold_idx += 1

    calibration_summary: Dict[str, Any] = {}
    if use_calibration and all_y_true:
        try:
            overall_cal_metrics = compute_all_calibration_metrics(
                np.array(all_y_true), np.array(all_y_prob_calibrated), n_bins=10
            )
            raw_cal_metrics = compute_all_calibration_metrics(
                np.array(all_y_true), np.array(all_y_prob_raw), n_bins=10
            )
            calibration_summary = {
                "calibration_method": calibration_method,
                "brier_score_raw": raw_cal_metrics["brier_score"],
                "brier_score_calibrated": overall_cal_metrics["brier_score"],
                "ece_raw": raw_cal_metrics["ece"],
                "ece_calibrated": overall_cal_metrics["ece"],
                "mce_raw": raw_cal_metrics["mce"],
                "mce_calibrated": overall_cal_metrics["mce"],
            }
        except Exception:
            calibration_summary = {"calibration_method": calibration_method}

    geom = metric_from_fold_ratios(ratios)
    min_r = float(np.min(ratios))
    metric_raw = 0.8 * geom + 0.2 * min_r
    penalty = trade_penalty(closed_trades)

    worst_fold_penalty = 0.0
    if ratios:
        worst_ratio = min(ratios)
        if worst_ratio < 0.8:
            worst_fold_penalty = 0.1 * (0.8 - worst_ratio)

    variance_penalty = 0.0
    if len(ratios) > 1:
        ratio_std = float(np.std(ratios))
        if ratio_std > 0.15:
            variance_penalty = 0.05 * (ratio_std - 0.15)

    few_trades_penalty = 0.0
    avg_trades = float(np.mean(closed_trades))
    if avg_trades < TARGET_CLOSED_TRADES_PER_FOLD * 0.7:
        few_trades_penalty = 0.08

    extra_penalty = worst_fold_penalty + variance_penalty + few_trades_penalty

    return (
        float(metric_raw - penalty - extra_penalty),
        float(np.mean(equities)),
        {
            "n_features": len(X.columns),
            "geom_mean_ratio": geom,
            "min_fold_ratio": min_r,
            "metric_raw": metric_raw,
            "penalty": penalty,
            "extra_penalty": extra_penalty,
            "worst_fold_penalty": worst_fold_penalty,
            "variance_penalty": variance_penalty,
            "few_trades_penalty": few_trades_penalty,
            "avg_closed_trades": float(np.mean(closed_trades)),
            "fold_details": fold_details,
            "calibration_summary": calibration_summary,
        },
        fold_details,
    )


# =========================================================
# Holdout 评估函数
# =========================================================
def _eval_on_holdout(
    candidate: Dict[str, Any],
    search_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    max_features: int,
    n_folds: int,
    train_start_ratio: float,
    wf_min_rows: int,
    computed_feats: Optional[Tuple[pd.DataFrame, pd.Series, FeatureMeta]],
) -> Tuple[float, float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    在 holdout 数据上评估候选配置。
    使用搜索阶段相同的特征构建，但在 holdout 数据上进行回测。
    """
    if computed_feats is None:
        feat_cfg = candidate["feature_config"]
        computed_feats = build_features_and_labels(search_df, feat_cfg)
        if computed_feats is None:
            return (-np.inf, 100000.0, {"skip": "feat_fail"}, [])

    X_search, y_search, meta = computed_feats

    holdout_computed = build_features_and_labels(holdout_df, candidate["feature_config"])
    if holdout_computed is None:
        return (-np.inf, 100000.0, {"skip": "holdout_feat_fail"}, [])

    X_holdout, y_holdout, _ = holdout_computed

    if len(X_search.columns) > max_features:
        return (-np.inf, float(candidate["trade_config"]["initial_cash"]), {"skip": "too_many_feats", "n_features": len(X_search.columns)}, [])

    splits = walkforward_splits(X_search, y_search, n_folds=n_folds, train_start_ratio=train_start_ratio, min_rows=wf_min_rows)
    if not splits:
        return (-np.inf, float(candidate["trade_config"]["initial_cash"]), {"skip": "no_splits"}, [])

    trade_cfg = candidate["trade_config"]
    initial_cash = float(trade_cfg["initial_cash"])
    cand_seed = int(candidate.get("candidate_seed", 42))
    device = _get_device()

    calibration_method = str(candidate.get("calibration_config", {}).get("method", "none"))
    use_calibration = calibration_method != "none"
    use_calibrated_threshold = candidate.get("calibration_config", {}).get("use_for_threshold", False)

    all_equities = []
    all_ratios = []
    all_trades = []
    fold_details = []

    holdout_calibration_comparison: Dict[str, Any] = {}
    all_y_true: List[float] = []
    all_y_prob_raw: List[float] = []
    all_y_prob_calibrated: List[float] = []

    for fold_idx, ((X_tr, y_tr), (X_va, y_va)) in enumerate(splits):
        model_type = str(candidate["model_config"]["model_type"])
        if model_type in ["mlp", "lstm", "gru", "cnn", "transformer"]:
            actual_cfg = {**candidate["model_config"], "input_dim": X_tr.shape[1]}
            trained_model = train_pytorch_model(X_tr, y_tr, actual_cfg, device,
                                                X_val=X_va, y_val=y_va)
            dp_val_raw = predict_pytorch_model(
                trained_model, X_holdout, device,
                seq_len=int(actual_cfg.get("seq_len", 20))
            )
        else:
            model = make_model(candidate, seed=cand_seed)
            model.fit(
                X_tr.values if isinstance(model, Pipeline) else X_tr,
                y_tr.values if isinstance(model, Pipeline) else y_tr,
            )
            dp_val_raw = predict_dpoint(model, X_holdout)

        dp_val = dp_val_raw.copy()

        if use_calibration and len(y_va) >= 20:
            try:
                calibrator = ProbabilityCalibrator(method=calibration_method)
                calibrator.fit(y_va.values, dp_val_raw.values)
                dp_val = pd.Series(
                    calibrator.transform(dp_val_raw.values),
                    index=dp_val_raw.index,
                    name=dp_val_raw.name
                )

                all_y_true.extend(y_holdout.values.tolist())
                all_y_prob_raw.extend(dp_val_raw.values.tolist())
                all_y_prob_calibrated.extend(dp_val.values.tolist())
            except Exception:
                pass

        fold_stats = backtest_fold_stats(holdout_df, X_holdout, dp_val, trade_cfg)
        equity_end = float(fold_stats["equity_end"])
        n_closed = int(fold_stats["n_closed"])

        if n_closed < MIN_CLOSED_TRADES_PER_FOLD:
            return (-np.inf, initial_cash, {"skip": "too_few_trades"}, [])

        all_equities.append(equity_end)
        all_ratios.append(equity_end / initial_cash)
        all_trades.append(n_closed)

        fold_details.append({
            "fold_idx": fold_idx,
            "equity_end": equity_end,
            "ratio": equity_end / initial_cash,
            "n_closed_trades": n_closed,
        })

    if use_calibration and all_y_true:
        try:
            raw_metrics = compute_all_calibration_metrics(
                np.array(all_y_true), np.array(all_y_prob_raw), n_bins=10
            )
            calibrated_metrics = compute_all_calibration_metrics(
                np.array(all_y_true), np.array(all_y_prob_calibrated), n_bins=10
            )
            holdout_calibration_comparison = {
                "calibration_method": calibration_method,
                "use_for_threshold": use_calibrated_threshold,
                "brier_score_raw": raw_metrics["brier_score"],
                "brier_score_calibrated": calibrated_metrics["brier_score"],
                "ece_raw": raw_metrics["ece"],
                "ece_calibrated": calibrated_metrics["ece"],
                "mce_raw": raw_metrics["mce"],
                "mce_calibrated": calibrated_metrics["mce"],
            }
        except Exception:
            holdout_calibration_comparison = {"calibration_method": calibration_method}

    geom = metric_from_fold_ratios(all_ratios)
    min_r = float(np.min(all_ratios))
    metric_raw = 0.8 * geom + 0.2 * min_r
    penalty = trade_penalty(all_trades)

    return (
        float(metric_raw - penalty),
        float(np.mean(all_equities)),
        {
            "n_features": len(X_search.columns),
            "geom_mean_ratio": geom,
            "min_fold_ratio": min_r,
            "metric_raw": metric_raw,
            "penalty": penalty,
            "avg_closed_trades": float(np.mean(all_trades)),
            "holdout_calibration_comparison": holdout_calibration_comparison,
        },
        fold_details,
    )


# =========================================================
# 多种子稳定性评估
# =========================================================
def _multi_seed_evaluation(
    candidate: Dict[str, Any],
    df_clean: pd.DataFrame,
    max_features: int,
    n_folds: int,
    train_start_ratio: float,
    wf_min_rows: int,
    n_seeds: int = 3,
) -> Dict[str, Any]:
    """
    使用多个随机种子评估候选配置的稳定性。
    """
    feat_cfg = candidate["feature_config"]
    computed_feats = build_features_and_labels(df_clean, feat_cfg)
    if computed_feats is None:
        return {
            "stability_metric": -np.inf,
            "mean_metric": -np.inf,
            "std_metric": 0.0,
            "seeds_valid": 0,
            "seed_details": [],
        }

    seed_metrics = []
    seed_details = []

    for seed in range(n_seeds):
        seed_candidate = {**candidate, "candidate_seed": seed}
        metric, equity, info, _ = _eval_candidate(
            seed_candidate,
            df_clean,
            max_features,
            n_folds,
            train_start_ratio,
            wf_min_rows,
            computed_feats,
        )

        seed_details.append({
            "seed": seed,
            "metric": metric,
            "equity": equity,
            "avg_trades": info.get("avg_closed_trades", 0),
        })

        if metric > -np.inf:
            seed_metrics.append(metric)

    if not seed_metrics:
        return {
            "stability_metric": -np.inf,
            "mean_metric": -np.inf,
            "std_metric": 0.0,
            "seeds_valid": 0,
            "seed_details": seed_details,
        }

    mean_metric = float(np.mean(seed_metrics))
    std_metric = float(np.std(seed_metrics))

    stability_penalty = 0.0
    if std_metric > 0.1:
        stability_penalty = 0.1 * std_metric

    stability_metric = mean_metric - stability_penalty

    return {
        "stability_metric": stability_metric,
        "mean_metric": mean_metric,
        "std_metric": std_metric,
        "seeds_valid": len(seed_metrics),
        "seed_details": seed_details,
    }


# =========================================================
# P2: 参数敏感性分析
# =========================================================
def _parameter_sensitivity_analysis(
    candidate: Dict[str, Any],
    df_search: pd.DataFrame,
    n_folds: int,
    train_start_ratio: float,
    wf_min_rows: int,
    n_perturbations: int = 5,
    perturbation_scale: float = 0.1,
) -> Dict[str, Any]:
    """
    P2: 参数敏感性分析。

    检查最优解是否过于"尖锐"：
        - 对关键参数进行微扰，观察性能变化
        - 如果微扰导致性能大幅下降，说明解不稳定/过拟合
        - 返回敏感性指标供决策参考

    参数扰动范围：
        - buy_threshold: ±perturbation_scale
        - sell_threshold: ±perturbation_scale
        - model C / alpha / learning_rate: ±perturbation_scale * 100%

    返回：
        包含各参数扰动结果的敏感性报告
    """
    sensitivity_results = []
    base_metric, base_equity, base_info, _ = _eval_candidate(
        candidate, df_search, max_features=100, n_folds=n_folds,
        train_start_ratio=train_start_ratio, wf_min_rows=wf_min_rows, computed_feats=None
    )

    if base_metric == -np.inf:
        return {"error": "base_eval_failed", "base_metric": -np.inf}

    tc = candidate["trade_config"]
    mc = candidate["model_config"]

    # 1. 阈值敏感性
    buy_thresh = float(tc.get("buy_threshold", 0.55))
    sell_thresh = float(tc.get("sell_threshold", 0.45))

    for direction in [-1, 1]:
        perturbed_tc = {
            **tc,
            "buy_threshold": buy_thresh + direction * perturbation_scale,
            "sell_threshold": sell_thresh - direction * perturbation_scale,  # 保持 buy > sell
        }
        perturbed_cand = {**candidate, "trade_config": perturbed_tc}

        m, eq, info, _ = _eval_candidate(
            perturbed_cand, df_search, max_features=100, n_folds=n_folds,
            train_start_ratio=train_start_ratio, wf_min_rows=wf_min_rows, computed_feats=None
        )

        if m > -np.inf:
            sensitivity_results.append({
                "param": f"threshold_{'up' if direction > 0 else 'down'}",
                "metric": m,
                "delta": m - base_metric,
                "delta_pct": (m - base_metric) / abs(base_metric) if base_metric != 0 else 0,
            })

    # 2. 模型超参敏感性
    model_type = str(mc.get("model_type", "logreg"))

    if model_type in ["logreg", "sgd"]:
        # C 或 alpha 扰动
        param_name = "C" if model_type == "logreg" else "alpha"
        base_val = float(mc.get(param_name, 0.01))

        for direction in [-1, 1]:
            perturbed_val = base_val * (1 + direction * perturbation_scale * 5)
            perturbed_mc = {**mc, param_name: perturbed_val}
            perturbed_cand = {**candidate, "model_config": perturbed_mc}

            m, eq, info, _ = _eval_candidate(
                perturbed_cand, df_search, max_features=100, n_folds=n_folds,
                train_start_ratio=train_start_ratio, wf_min_rows=wf_min_rows, computed_feats=None
            )

            if m > -np.inf:
                sensitivity_results.append({
                    "param": f"{model_type}_{param_name}_{'up' if direction > 0 else 'down'}",
                    "metric": m,
                    "delta": m - base_metric,
                    "delta_pct": (m - base_metric) / abs(base_metric) if base_metric != 0 else 0,
                })

    elif model_type in ["mlp", "lstm", "gru", "cnn", "transformer"]:
        # learning_rate 扰动
        base_lr = float(mc.get("learning_rate", 0.001))

        for direction in [-1, 1]:
            perturbed_lr = base_lr * (1 + direction * perturbation_scale * 3)
            perturbed_mc = {**mc, "learning_rate": perturbed_lr}
            perturbed_cand = {**candidate, "model_config": perturbed_mc}

            m, eq, info, _ = _eval_candidate(
                perturbed_cand, df_search, max_features=100, n_folds=n_folds,
                train_start_ratio=train_start_ratio, wf_min_rows=wf_min_rows, computed_feats=None
            )

            if m > -np.inf:
                sensitivity_results.append({
                    "param": f"{model_type}_lr_{'up' if direction > 0 else 'down'}",
                    "metric": m,
                    "delta": m - base_metric,
                    "delta_pct": (m - base_metric) / abs(base_metric) if base_metric != 0 else 0,
                })

    # 计算敏感性指标
    valid_deltas = [r["delta"] for r in sensitivity_results if r["delta"] != 0]
    if valid_deltas:
        avg_delta = float(np.mean(valid_deltas))
        max_delta = float(np.max(valid_deltas))
        sensitivity_score = abs(avg_delta)  # 越高越敏感

        # 判断是否过于尖锐
        is_sharp = sensitivity_score > 0.05 or max_delta > 0.1
    else:
        avg_delta = 0.0
        max_delta = 0.0
        sensitivity_score = 0.0
        is_sharp = False

    return {
        "base_metric": base_metric,
        "n_perturbations": len(sensitivity_results),
        "avg_delta": avg_delta,
        "max_delta": max_delta,
        "sensitivity_score": sensitivity_score,
        "is_sharp": is_sharp,
        "perturbation_details": sensitivity_results,
    }


# =========================================================
# 候选采样
# =========================================================
def _sample_explore(
    rng: np.random.Generator,
    spaces: SearchSpaces,
    trade_params: Dict[str, Any],
) -> Dict[str, Any]:
    """随机采样一个全新的候选配置（探索模式）。"""
    cand_seed = int(rng.integers(0, 1_000_000))
    feat_cfg = {
        "windows": spaces.window_pool[int(rng.integers(0, len(spaces.window_pool)))],
        "use_momentum":      bool(rng.integers(0, 2)),
        "use_volatility":    bool(rng.integers(0, 2)),
        "use_volume":        bool(rng.integers(0, 2)),
        "use_candle":        bool(rng.integers(0, 2)),
        "use_turnover":      bool(rng.integers(0, 2)),
        "vol_metric":        rng.choice(spaces.vol_metric_pool),
        "liq_transform":     rng.choice(spaces.liq_transform_pool),
        # P3-19：以 30% 概率启用技术指标族，避免过度增加特征数量
        "use_ta_indicators": bool(rng.random() < 0.3),
        "ta_windows":        list(spaces.ta_window_pool[int(rng.integers(0, len(spaces.ta_window_pool)))]),
    }
    # 至少保留一个基础特征族（技术指标族不计入此约束）
    if not any([feat_cfg["use_momentum"], feat_cfg["use_volatility"],
                feat_cfg["use_volume"], feat_cfg["use_candle"], feat_cfg["use_turnover"]]):
        feat_cfg["use_momentum"] = True

    dl_models = ["mlp", "lstm", "gru", "cnn", "transformer"]
    mt = rng.choice(["logreg", "sgd"] + dl_models + (["xgb"] if spaces.xgb_available else []))

    if mt == "xgb":
        model_cfg = {"model_type": "xgb", "params": {**dict(rng.choice(spaces.xgb_param_pool)), "random_state": cand_seed}}
    elif mt == "mlp":
        model_cfg = dict(
            model_type="mlp",
            input_dim=spaces.input_dim,
            hidden_dim=int(rng.choice(spaces.hidden_dims)),
            learning_rate=float(rng.choice(spaces.learning_rates)),
            batch_size=int(rng.choice(spaces.batch_sizes)),
            epochs=30,
            dropout_rate=0.4,
        )
    elif mt == "lstm":
        model_cfg = dict(
            model_type="lstm",
            input_dim=spaces.input_dim,
            hidden_dim=int(rng.choice(spaces.hidden_dims)),
            learning_rate=float(rng.choice(spaces.learning_rates)),
            batch_size=int(rng.choice(spaces.batch_sizes)),
            num_layers=int(rng.choice(spaces.num_layers_pool)),
            seq_len=int(rng.choice(spaces.seq_lens)),
            bidirectional=bool(rng.integers(0, 2)),
            epochs=30,
            dropout_rate=0.3,
        )
    elif mt == "gru":
        model_cfg = dict(
            model_type="gru",
            input_dim=spaces.input_dim,
            hidden_dim=int(rng.choice(spaces.hidden_dims)),
            learning_rate=float(rng.choice(spaces.learning_rates)),
            batch_size=int(rng.choice(spaces.batch_sizes)),
            num_layers=int(rng.choice(spaces.num_layers_pool)),
            seq_len=int(rng.choice(spaces.seq_lens)),
            bidirectional=False,
            epochs=30,
            dropout_rate=0.3,
        )
    elif mt == "cnn":
        model_cfg = dict(
            model_type="cnn",
            input_dim=spaces.input_dim,
            num_filters=int(rng.choice(spaces.hidden_dims)),
            learning_rate=float(rng.choice(spaces.learning_rates)),
            batch_size=int(rng.choice(spaces.batch_sizes)),
            seq_len=int(rng.choice(spaces.seq_lens)),
            kernel_sizes=[2, 3, 5],
            epochs=30,
            dropout_rate=0.3,
        )
    elif mt == "transformer":
        d_model = int(rng.choice(spaces.hidden_dims))
        # nhead 必须能整除 d_model
        nhead_choices = [h for h in [2, 4, 8] if d_model % h == 0]
        model_cfg = dict(
            model_type="transformer",
            input_dim=spaces.input_dim,
            d_model=d_model,
            nhead=int(rng.choice(nhead_choices)) if nhead_choices else 2,
            num_layers=int(rng.choice(spaces.num_layers_pool)),
            dim_feedforward=d_model * 4,
            learning_rate=float(rng.choice(spaces.learning_rates)),
            batch_size=int(rng.choice(spaces.batch_sizes)),
            seq_len=int(rng.choice(spaces.seq_lens)),
            epochs=30,
            dropout_rate=0.1,
        )
    else:
        model_cfg = {"model_type": mt, **rng.choice(spaces.logreg_choices if mt == "logreg" else spaces.sgd_choices)}

    buy = float(rng.choice(spaces.buy_pool))
    sell = float(rng.choice(spaces.sell_pool))
    if sell >= buy:
        sell = buy - 0.10

    return {
        "candidate_seed": cand_seed,
        "feature_config": feat_cfg,
        "model_config": model_cfg,
        "calibration_config": {
            "method": rng.choice(spaces.calibration_pool),
            "use_for_threshold": bool(rng.integers(0, 2)),
        },
        "trade_config": {
            "initial_cash": float(trade_params["initial_cash"]),
            "buy_threshold": buy,
            "sell_threshold": sell,
            "confirm_days": int(rng.choice(spaces.confirm_pool)),
            "min_hold_days": 1,
            "max_hold_days": int(rng.choice(spaces.max_hold_pool)),
            "take_profit": rng.choice(spaces.take_profit_pool),
            "stop_loss": rng.choice(spaces.stop_loss_pool),
        },
    }


def _sample_exploit(
    incumbent: Dict[str, Any],
    diag: Dict[str, float],
    rng: np.random.Generator,
    spaces: SearchSpaces,
    trade_params: Dict[str, Any],
) -> Dict[str, Any]:
    """在 incumbent 基础上做小扰动（开采模式）。"""
    cand_seed = int(rng.integers(0, 1_000_000))
    c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in incumbent.items()}
    c["candidate_seed"] = cand_seed
    tf, tm = diag.get("trade_too_few", 0.0), diag.get("trade_too_many", 0.0)
    buy = float(c["trade_config"]["buy_threshold"])
    sell = float(c["trade_config"]["sell_threshold"])

    if tf > 0.1:
        buy -= rng.uniform(0, 0.05 * tf)
        sell += rng.uniform(0, 0.05 * tf)
    elif tm > 0.1:
        buy += rng.uniform(0, 0.05 * tm)
        sell -= rng.uniform(0, 0.05 * tm)
    else:
        buy += rng.uniform(-0.02, 0.02)
        sell += rng.uniform(-0.02, 0.02)

    #在阈值扰动基础上，独立概率扰动特征配置和模型超参
    c["trade_config"]["buy_threshold"] = float(np.clip(buy, 0.50, 0.75))
    c["trade_config"]["sell_threshold"] = float(np.clip(sell, 0.25, 0.60))

    # ── 特征配置扰动（30% 概率）──────────────────────────────────────
    # 随机翻转一个特征族开关，或随机替换 windows 组合
    # 保留"至少一个基础特征族开启"的约束
    if rng.random() < 0.3:
        feat_toggles = ["use_momentum", "use_volatility", "use_volume",
                        "use_candle", "use_turnover"]
        key = feat_toggles[int(rng.integers(0, len(feat_toggles)))]
        c["feature_config"] = dict(c["feature_config"])   # 浅拷贝，避免污染 incumbent
        c["feature_config"][key] = not bool(c["feature_config"].get(key, True))
        # 维持约束：至少一个基础特征族开启
        if not any(c["feature_config"].get(k, True) for k in feat_toggles):
            c["feature_config"][key] = True   # 翻转回来

    if rng.random() < 0.2:
        c["feature_config"] = dict(c["feature_config"])
        c["feature_config"]["windows"] = list(
            spaces.window_pool[int(rng.integers(0, len(spaces.window_pool)))]
        )

    # ── 模型超参扰动（25% 概率）──────────────────────────────────────
    # 针对不同 model_type 扰动其最关键的一个连续超参
    # 扰动范围保守（±相对 20%），避免跳出有效区间
    if rng.random() < 0.25:
        mc = dict(c["model_config"])   # 浅拷贝
        mt = str(mc.get("model_type", ""))
        if mt == "logreg":
            current_C = float(mc.get("C", 0.01))
            delta = float(rng.uniform(0.8, 1.25))
            mc["C"] = float(np.clip(current_C * delta, 1e-4, 1e3))
        elif mt == "sgd":
            current_alpha = float(mc.get("alpha", 0.001))
            delta = float(rng.uniform(0.8, 1.25))
            mc["alpha"] = float(np.clip(current_alpha * delta, 1e-6, 0.1))
        elif mt in ["mlp", "lstm", "gru", "cnn", "transformer"]:
            current_lr = float(mc.get("learning_rate", 0.001))
            delta = float(rng.uniform(0.7, 1.4))
            mc["learning_rate"] = float(np.clip(current_lr * delta, 1e-5, 0.05))
        elif mt == "xgb":
            mc["params"] = dict(mc.get("params", {}))
            current_lr = float(mc["params"].get("learning_rate", 0.05))
            delta = float(rng.uniform(0.7, 1.4))
            mc["params"]["learning_rate"] = float(np.clip(current_lr * delta, 0.005, 0.3))
            mc["params"]["random_state"] = cand_seed
        c["model_config"] = mc
    elif c["model_config"]["model_type"] == "xgb":
        # 即使不扰动超参，XGBoost 也需更新 random_state 保证多样性
        c["model_config"] = dict(c["model_config"])
        c["model_config"]["params"] = dict(c["model_config"]["params"])
        c["model_config"]["params"]["random_state"] = cand_seed

    # ── P1: 校准配置扰动（15% 概率）──────────────────────────────────────
    if "calibration_config" not in c:
        c["calibration_config"] = {"method": "none", "use_for_threshold": False}

    if rng.random() < 0.15:
        c["calibration_config"] = dict(c.get("calibration_config", {}))
        c["calibration_config"]["method"] = rng.choice(spaces.calibration_pool)

    if rng.random() < 0.15:
        if "calibration_config" not in c:
            c["calibration_config"] = {"method": "none", "use_for_threshold": False}
        else:
            c["calibration_config"] = dict(c["calibration_config"])
        c["calibration_config"]["use_for_threshold"] = bool(rng.integers(0, 2))

    return c


def _sample_exploit_from_pool(
    pool_items: List[Dict[str, Any]],
    diag: Dict[str, float],
    rng: np.random.Generator,
    spaces: SearchSpaces,
    trade_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    P02：从 Top-K 池中按 metric 加权随机选取一个配置作为变异基础，
    再调用 _sample_exploit 做小扰动。

    相比只从单一 incumbent 变异，此方式保留了历史上多个优质配置的
    多样性，减少搜索陷入局部最优的风险。
    """
    # 按 metric 计算采样权重，metric 较高的配置被选中概率更大
    metrics = np.array([max(float(it.get("metric", 0.0)), 0.0) for it in pool_items])
    total = metrics.sum()
    if total <= 0:
        idx = int(rng.integers(0, len(pool_items)))
    else:
        probs = metrics / total
        idx = int(rng.choice(len(pool_items), p=probs))

    base_cfg = pool_items[idx]["config"]
    return _sample_exploit(base_cfg, diag, rng, spaces, trade_params)


# =========================================================
# TrainResult 数据类（接口与原版完全一致）
# =========================================================
@dataclass
class TrainResult:
    best_config: Dict[str, Any]
    best_val_metric: float
    best_val_final_equity_proxy: float
    search_log: pd.DataFrame
    feature_meta: Dict[str, Any]
    training_notes: List[str]
    global_best_updated: bool
    global_best_metric_prev: float
    global_best_metric_new: float
    candidate_best_metric: float
    epsilon: float
    not_updated_reason: str
    best_so_far_path: str
    best_pool_path: str
    holdout_metric: float = -np.inf
    holdout_equity: float = 0.0
    holdout_fold_details: List[Dict[str, Any]] = field(default_factory=list)
    search_data_rows: int = 0
    holdout_data_rows: int = 0
    stability_report: Dict[str, Any] = field(default_factory=dict)
    holdout_calibration_comparison: Dict[str, Any] = field(default_factory=dict)
    feature_usage_stats: Dict[str, Any] = field(default_factory=dict)
    best_model_importance: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# 主搜索函数
# =========================================================
def random_search_train(
    df_clean: pd.DataFrame,
    runs: int = 50,
    seed: int = 42,
    base_best_config: Optional[Dict[str, Any]] = None,
    trade_params: Optional[Dict[str, Any]] = None,
    max_features: int = 80,
    output_dir: str = "./output",
    epsilon: float = 0.01,
    exploit_ratio: float = 0.7,
    top_k: int = 10,
    n_folds: int = 4,
    train_start_ratio: float = 0.5,
    wf_min_rows: int = 40,
    n_jobs: int = -1,
    n_rounds: int = 4,
    pool_exploit_prob: float = 0.3,
    use_holdout: bool = True,
    holdout_ratio: float = 0.15,
    min_holdout_rows: int = 60,
    cross_ticker_paths: Optional[List[str]] = None,
    # P2: 新增参数
    use_embargo: bool = False,
    embargo_days: int = 5,
    use_nested_wf: bool = False,
    use_sensitivity_analysis: bool = True,
) -> TrainResult:
    """
    随机搜索训练主函数。

    P01 修复：
        主进程按 feature_config hash 预计算所有唯一特征，
        以显式参数传给 _eval_candidate，彻底解决多进程缓存失效问题。

    P02 修复：
        - 每轮评估后将有效结果写入 Top-K 持久化池（update_best_pool）
        - exploit 阶段以 pool_exploit_prob 概率从池中加权采样基础配置，
          提升搜索多样性

    P05 修复：
        将 runs 均分为 n_rounds 轮，每轮并行评估后立即更新 incumbent
        和诊断状态，下一轮 exploit 候选反映最新搜索成果。

    参数说明（新增部分）：
        n_rounds         — 分轮数，默认 4；runs 会被均分，每轮后更新 incumbent
        pool_exploit_prob — exploit 时从 Top-K 池采样的概率，默认 0.3
        wf_min_rows       — Walk-Forward 每折最小行数，默认 40（原 60），降低以支持较少数据量的场景
    """
    rng = np.random.default_rng(seed)
    tp = trade_params or {"initial_cash": 100000.0}
    device = _get_device()

    # --- Holdout split ---
    _search_df = df_clean
    _holdout_df = None
    holdout_metric = -np.inf
    holdout_equity = 0.0
    holdout_fold_details = []
    training_notes: List[str] = []  # P2: 提前初始化，供敏感性分析使用

    if use_holdout and len(df_clean) >= min_holdout_rows + wf_min_rows * n_folds:
        split_result = final_holdout_split(
            df_clean,
            holdout_ratio=holdout_ratio,
            min_holdout_rows=min_holdout_rows
        )
        if split_result is not None:
            _search_df, _holdout_df = split_result
            logger.info("SEARCH Holdout split: search=%d, holdout=%d", len(_search_df), len(_holdout_df))

    search_data_rows = len(_search_df)
    holdout_data_rows = len(_holdout_df) if _holdout_df is not None else 0

    # --- 初始化搜索空间 ---
    _, _, init_meta = build_features_and_labels(_search_df, {
        "windows": [3, 5, 10, 20], "use_momentum": True, "use_volatility": True,
        "use_volume": True, "use_candle": True, "use_turnover": True,
        "vol_metric": "std", "liq_transform": "ratio",
    })
    spaces = _build_search_spaces(seed, len(init_meta.feature_names))

    # --- 加载初始 incumbent ---
    best_cfg = load_best_so_far(output_dir) or base_best_config or _sample_explore(rng, spaces, tp)

    # P02：加载已有 Top-K 池（跨 run 保留历史优质配置）
    pool_items = load_best_pool(output_dir)

    # --- 评估初始 incumbent（P1-5：如有持久化 metric 则直接复用，跳过重复评估）---
    # （统一处理所有 DL 模型类型）
    X_inc, y_inc, meta_inc = build_features_and_labels(_search_df, best_cfg["feature_config"])
    if str(best_cfg["model_config"]["model_type"]) in ["mlp", "lstm", "gru", "cnn", "transformer"]:
        best_cfg["model_config"]["input_dim"] = X_inc.shape[1]

    # P1-5：load_best_so_far_metric 读取上次持久化的 best_metric，
    # 如果本次加载的配置来自 best_so_far.json（同一文件），则其 metric 已保存，
    # 无需再执行一次完整的 walk-forward 评估。
    # 仅当 metric 无法读取（首次运行 / base_best_config 来自外部 / 新采样）时才重新评估。
    saved_metric = load_best_so_far_metric(output_dir)
    if saved_metric is not None and load_best_so_far(output_dir) is not None:
        # 配置来自 best_so_far.json，metric 已有持久化值，直接复用
        best_m = float(saved_metric)
        best_eq = float(tp.get("initial_cash", 100000.0))  # equity 无持久化，用 initial_cash 占位
        # 构造一个最小化的 info_inc 供 _diagnose_from_incumbent 使用
        info_inc: Dict[str, Any] = {"avg_closed_trades": float("nan")}
        training_notes_extra = [
            f"P1-5: incumbent metric 从持久化文件复用（{saved_metric:.6f}），跳过重复 walk-forward 评估。"
        ]
    else:
        # 首次运行或配置来自外部，必须重新评估
        best_m_raw, best_eq_raw, info_inc, _ = _eval_candidate(
            best_cfg, _search_df, max_features, n_folds,
            train_start_ratio, wf_min_rows, (X_inc, y_inc, meta_inc)
        )
        best_m = float(best_m_raw)
        best_eq = float(best_eq_raw)
        training_notes_extra = [
            "P1-5: incumbent 为新配置，已执行完整 walk-forward 评估。"
        ]

    initial_m = best_m   # 保存初始值，用于最终 global_best_metric_prev

    # --- P0: 特征使用频率跟踪器 ---
    feature_usage_tracker = FeatureUsageTracker()

    # --- P05：分轮搜索主循环 ---
    all_search_rows: List[Dict[str, Any]] = []
    all_candidates: List[Dict[str, Any]] = []
    cand_best_m: float = -np.inf
    cand_best_cfg: Optional[Dict[str, Any]] = None
    cand_best_eq: float = best_eq
    final_feat_map: Dict[str, Tuple] = {}   # 用于最终 feature_meta 获取

    runs_per_round = max(1, runs // n_rounds)

    for round_idx in range(n_rounds):
        # 计算本轮实际 runs 数（最后一轮补齐余数）
        if round_idx < n_rounds - 1:
            actual_runs = runs_per_round
        else:
            actual_runs = max(1, runs - round_idx * runs_per_round)

        logger.info("SEARCH Round %d/%d, evaluating %d candidates", round_idx + 1, n_rounds, actual_runs)

        # P05：基于当前最新 incumbent 生成本轮候选
        diag = _diagnose_from_incumbent(info_inc)
        round_c: List[Dict[str, Any]] = []
        # P2-1：与 round_c 并行记录每个候选的生成模式，用于 search_log 诊断
        round_c_modes: List[str] = []          # "explore" | "exploit" | "pool_exploit"
        incumbent_m_at_gen: float = best_m     # 本轮生成时的 incumbent metric 快照

        for _ in range(actual_runs):
            roll = rng.random()
            if roll < exploit_ratio:
                # P02：exploit 时以 pool_exploit_prob 概率从 Top-K 池采样
                if pool_items and rng.random() < pool_exploit_prob:
                    c = _sample_exploit_from_pool(pool_items, diag, rng, spaces, tp)
                    mode = "pool_exploit"
                else:
                    c = _sample_exploit(best_cfg, diag, rng, spaces, tp)
                    mode = "exploit"
            else:
                c = _sample_explore(rng, spaces, tp)
                mode = "explore"
            round_c.append(c)
            round_c_modes.append(mode)

            # P0: 记录候选的特征使用情况
            feature_usage_tracker.record_candidate(c["feature_config"])

        # P01：主进程统一预计算本轮所有唯一 feature_config
        # 相同 feature_config 的候选只计算一次，多进程下通过参数传递，不依赖共享内存
        feat_map: Dict[str, Tuple] = {}
        for c in round_c:
            fhash = config_hash(c["feature_config"])
            if fhash not in feat_map:
                feat_map[fhash] = build_features_and_labels(_search_df, c["feature_config"])

        # P01：将预计算特征作为显式参数传入，完全绕开多进程缓存问题
        round_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_eval_candidate)(
                c,
                _search_df,
                max_features,
                n_folds,
                train_start_ratio,
                wf_min_rows,
                feat_map[config_hash(c["feature_config"])],   # P01：主进程预计算结果
            )
            for c in round_c
        )

        # --- 收集本轮结果，实时更新 incumbent ---
        for i, (m, eq, info, fold_details) in enumerate(round_results):
            all_candidates.append(round_c[i])
            cand = round_c[i]
            mc = cand["model_config"]
            tc = cand["trade_config"]

            # P2-1：提取关键模型超参（对不同 model_type 按需取值，缺失时留空字符串）
            # 这些字段是调试搜索行为的核心：知道是什么模型、什么配置拿到了什么分数
            _mt = str(mc.get("model_type", ""))
            _hidden_dim   = mc.get("hidden_dim", mc.get("d_model", ""))   # MLP/LSTM/GRU/Transformer
            _seq_len      = mc.get("seq_len", "")                         # 序列模型
            _num_layers   = mc.get("num_layers", "")                      # LSTM/GRU/Transformer
            _n_estimators = (mc.get("params") or {}).get("n_estimators", "")  # XGBoost
            _C            = mc.get("C", "")                               # LogReg/SGD
            _lr           = mc.get("learning_rate", "")                   # DL 模型

            all_search_rows.append({
                # ── 进度 ──────────────────────────────────────────────────
                "round":              round_idx + 1,
                "iter":               len(all_search_rows) + 1,
                # ── 生成模式（P2-1 新增）──────────────────────────────────
                # exploit_mode 是理解搜索行为最直接的字段：
                #   "explore"      — 完全随机采样，用于发现新区域
                #   "exploit"      — 在 incumbent 周围小扰动，用于精细化
                #   "pool_exploit" — 从 Top-K 池中选基础配置再扰动，保留历史多样性
                "exploit_mode":       round_c_modes[i],
                # ── 候选状态 ─────────────────────────────────────────────
                "status":             info.get("skip", "ok"),
                # ── 核心指标 ─────────────────────────────────────────────
                "val_metric_final":   m,
                "val_equity_proxy":   eq,
                "geom_mean_ratio":    info.get("geom_mean_ratio", ""),
                "min_fold_ratio":     info.get("min_fold_ratio", ""),
                "metric_raw":         info.get("metric_raw", ""),
                "penalty":            info.get("penalty", ""),
                "avg_closed_trades":  info.get("avg_closed_trades", ""),
                # ── 参照基线（P2-1 新增）──────────────────────────────────
                # 记录生成此候选时的 incumbent metric，方便事后计算每次改进幅度
                "incumbent_m_at_gen": incumbent_m_at_gen,
                "delta_vs_incumbent": (m - incumbent_m_at_gen) if m > -np.inf else "",
                # ── 特征配置 ─────────────────────────────────────────────
                "n_features":         info.get("n_features", ""),
                # ── 模型标识 ─────────────────────────────────────────────
                "model_type":         _mt,
                # ── 关键模型超参（P2-1 新增）──────────────────────────────
                # 不同 model_type 只有部分字段有值，其余留空，Reporter 导出时自然对齐
                "hidden_dim":         _hidden_dim,
                "seq_len":            _seq_len,
                "num_layers":         _num_layers,
                "n_estimators":       _n_estimators,
                "C":                  _C,
                "learning_rate":      _lr,
                # ── 信号阈值（P2-1 新增）──────────────────────────────────
                # 买卖阈值直接影响交易频率和 penalty，是诊断 avg_closed_trades
                # 偏少/偏多问题时最需要对照的字段
                "buy_threshold":      tc.get("buy_threshold", ""),
                "sell_threshold":     tc.get("sell_threshold", ""),
                "confirm_days":       tc.get("confirm_days", ""),
                "max_hold_days":      tc.get("max_hold_days", ""),
            })

            # P02：有效结果（未被跳过）写入 Top-K 持久化池
            if m > -np.inf and "skip" not in info:
                update_best_pool(output_dir, round_c[i], m, top_k)

            # 更新全局候选最优
            if m > cand_best_m:
                cand_best_m = m
                cand_best_cfg = round_c[i]
                cand_best_eq = eq

            # P05：轮内实时更新 incumbent，使后续 exploit 候选受益
            if m > best_m:
                best_m = m
                best_cfg = round_c[i]
                best_eq = eq
                info_inc = info   # 更新诊断状态，影响下一轮 exploit 的阈值扰动方向

        # P02：每轮结束后重新加载 pool，确保下一轮 exploit 用到最新的 Top-K 池
        pool_items = load_best_pool(output_dir)

        # 保留最后一轮的 feat_map，用于获取 final_meta
        final_feat_map = feat_map

    # --- 更新全局 best_so_far ---
    updated = False
    reason = ""
    logger.info("SEARCH Search complete: initial_m=%.6f, cand_best_m=%.6f, epsilon=%.6f", initial_m, cand_best_m, epsilon)
    if cand_best_cfg is not None and cand_best_m > initial_m + epsilon:
        updated = True
        save_best_so_far(output_dir, best_cfg, best_m)
        logger.info("SEARCH Best updated: new best_m=%.6f", best_m)
    else:
        reason = "not_exceed_epsilon" if cand_best_cfg is not None else "no_valid_cand"
        logger.info("SEARCH Best NOT updated: reason=%s", reason)

    # --- 获取最终 feature_meta ---
    final_fhash = config_hash(best_cfg["feature_config"])
    if final_fhash in final_feat_map:
        final_meta = final_feat_map[final_fhash][2]
    else:
        _, _, final_meta = build_features_and_labels(_search_df, best_cfg["feature_config"])

    # --- Holdout evaluation ---
    if _holdout_df is not None:
        logger.info("SEARCH Evaluating best config on holdout set")
        X_best, y_best, meta_best = build_features_and_labels(_search_df, best_cfg["feature_config"])
        if str(best_cfg["model_config"]["model_type"]) in ["mlp", "lstm", "gru", "cnn", "transformer"]:
            best_cfg_holdout = {**best_cfg, "model_config": {**best_cfg["model_config"], "input_dim": X_best.shape[1]}}
        else:
            best_cfg_holdout = best_cfg

        holdout_m, holdout_eq, holdout_info, holdout_fold_details = _eval_on_holdout(
            best_cfg_holdout,
            _search_df,
            _holdout_df,
            max_features,
            n_folds,
            train_start_ratio,
            wf_min_rows,
            (X_best, y_best, meta_best),
        )
        holdout_metric = holdout_m
        holdout_equity = holdout_eq
        holdout_calibration_comparison = holdout_info.get("holdout_calibration_comparison", {})
        logger.info("SEARCH Holdout metric: %.6f, equity: %.2f", holdout_metric, holdout_equity)
        if holdout_calibration_comparison:
            logger.info("SEARCH Holdout calibration: method=%s", holdout_calibration_comparison.get('calibration_method', 'none'))
    else:
        holdout_fold_details = []
        holdout_calibration_comparison = {}

    # --- Multi-seed stability evaluation ---
    stability_report = {}
    if cand_best_cfg is not None and cand_best_m > -np.inf:
        logger.info("SEARCH Running multi-seed stability evaluation")
        stability_report = _multi_seed_evaluation(
            cand_best_cfg,
            _search_df,
            max_features,
            n_folds,
            train_start_ratio,
            wf_min_rows,
            n_seeds=3,
        )
        logger.info("SEARCH Stability: mean_metric=%.6f, std=%.6f", stability_report.get('mean_metric', -np.inf), stability_report.get('std_metric', 0.0))

    # --- Cross-ticker evaluation ---
    cross_ticker_results = []
    if cross_ticker_paths and cand_best_cfg is not None:
        logger.info("SEARCH Running cross-ticker evaluation")
        for ticker_path in cross_ticker_paths:
            if os.path.exists(ticker_path):
                try:
                    ticker_df = pd.read_excel(ticker_path)
                    ticker_name = os.path.basename(ticker_path)
                    logger.info("SEARCH Evaluating on %s", ticker_name)

                    X_ticker, y_ticker, meta_ticker = build_features_and_labels(ticker_df, cand_best_cfg["feature_config"])
                    if X_ticker is not None:
                        mt, eq, info, _ = _eval_candidate(
                            cand_best_cfg,
                            ticker_df,
                            max_features,
                            n_folds,
                            train_start_ratio,
                            wf_min_rows,
                            (X_ticker, y_ticker, meta_ticker),
                        )
                        cross_ticker_results.append({
                            "ticker": ticker_name,
                            "metric": mt,
                            "equity": eq,
                            "avg_trades": info.get("avg_closed_trades", 0),
                        })
                        logger.info("SEARCH %s: metric=%.6f, equity=%.2f", ticker_name, mt, eq)
                except Exception as e:
                    logger.warning("SEARCH Failed to evaluate on %s: %s", ticker_path, e)
        stability_report["cross_ticker_results"] = cross_ticker_results

    # P2: 参数敏感性分析
    if use_sensitivity_analysis and best_cfg is not None:
        logger.info("P2: Running parameter sensitivity analysis on best config")
        sensitivity_report = _parameter_sensitivity_analysis(
            best_cfg, _search_df,
            n_folds=n_folds,
            train_start_ratio=train_start_ratio,
            wf_min_rows=wf_min_rows,
            n_perturbations=5,
            perturbation_scale=0.1,
        )
        stability_report["sensitivity_analysis"] = sensitivity_report
        if sensitivity_report.get("is_sharp"):
            training_notes.append(
                f"P2: ⚠️ WARNING - Best config is SENSITIVE (score={sensitivity_report.get('sensitivity_score', 0):.4f}). "
                f"Consider choosing a more robust configuration."
            )
        else:
            training_notes.append(
                f"P2: Parameter sensitivity OK (score={sensitivity_report.get('sensitivity_score', 0):.4f})"
            )

    # P2: 扩展 training_notes，而不是重新赋值
    training_notes.extend([
        f"Device: {device}",
        f"n_rounds: {n_rounds}，每轮 ~{runs_per_round} 个候选",
        f"总候选数：{len(all_candidates)}",
        f"CUDA 可用：{_get_cuda_available()}",
        f"P01: 特征预计算已在主进程完成，多进程缓存问题已修复",
        f"P02: Top-K 池已接入，pool_exploit_prob={pool_exploit_prob}",
        f"P03: XGBoost CUDA 检测语法已修复",
        f"P05: 分轮搜索已启用，每轮后更新 incumbent",
    ] + training_notes_extra)

    if use_holdout:
        training_notes.append(f"P0: Final holdout enabled - ratio={holdout_ratio}")
    if use_embargo:
        training_notes.append(f"P2: Embargo enabled - days={embargo_days}")
    if use_sensitivity_analysis:
        training_notes.append("P2: Parameter sensitivity analysis enabled")

    # P0: 获取特征使用统计
    feature_usage_stats = feature_usage_tracker.get_usage_stats()
    feature_group_stats = feature_usage_tracker.get_feature_group_stats()
    window_stats = feature_usage_tracker.get_window_stats()

    training_notes.append(f"P0: Feature usage tracked - {feature_usage_stats.get('total_candidates', 0)} candidates")

    # P0-P1: 计算最佳模型的全局重要性
    best_model_importance: Dict[str, Any] = {}
    if best_cfg is not None:
        try:
            X_best, y_best, meta_best = build_features_and_labels(_search_df, best_cfg["feature_config"])
            model_type = str(best_cfg["model_config"]["model_type"])

            if model_type in ["logreg", "sgd", "xgb"]:
                model = make_model(best_cfg, seed=42)
                model.fit(X_best.values, y_best.values)

                explainer = FeatureImportanceExplainer(
                    model=model,
                    model_type=model_type,
                    feature_names=list(X_best.columns),
                    X_train=X_best,
                    y_train=y_best.values,
                )

                importance_dict = explainer.get_global_importance(method="auto", X_val=X_best, y_val=y_best.values)
                best_model_importance = importance_dict

                if importance_dict.get("ranking"):
                    top_features = importance_dict["ranking"][:10]
                    training_notes.append(
                        f"P0: Top 10 features: {', '.join([f['feature'] for f in top_features])}"
                    )

                if model_type == "xgb":
                    group_ranking = compute_feature_group_ranking(
                        np.array(importance_dict.get("importance", [])),
                        importance_dict.get("feature_names", [])
                    )
                    best_model_importance["feature_group_ranking"] = group_ranking

        except Exception as e:
            logger.warning("SEARCH Failed to compute feature importance: %s", e)

    return TrainResult(
        best_config=best_cfg,
        best_val_metric=best_m,
        best_val_final_equity_proxy=best_eq,
        search_log=pd.DataFrame(all_search_rows),
        feature_meta=final_meta.__dict__,
        training_notes=training_notes,
        global_best_updated=updated,
        global_best_metric_prev=initial_m,
        global_best_metric_new=best_m,
        candidate_best_metric=cand_best_m,
        epsilon=epsilon,
        not_updated_reason=reason,
        best_so_far_path=best_so_far_path(output_dir),
        best_pool_path=best_pool_path(output_dir),
        holdout_metric=holdout_metric,
        holdout_equity=holdout_equity,
        holdout_fold_details=holdout_fold_details,
        search_data_rows=search_data_rows,
        holdout_data_rows=holdout_data_rows,
        stability_report=stability_report,
        holdout_calibration_comparison=holdout_calibration_comparison,
        feature_usage_stats=feature_usage_stats,
        best_model_importance=best_model_importance,
    )


# =========================================================
# trainer_optimizer.py - 训练器公开 API
# =========================================================
"""
训练器公开 API 入口（Medium-01 重构后的精简版）。

本文件职责：
  1. train_final_model_and_dpoint — 全样本最终模型拟合（保留在此）
  2. 从 search_engine 重新导出 random_search_train 和 TrainResult，
     保证 main_cli.py 的 import 语句无需任何改动。

子模块依赖关系：
    constants.py
        ↑
    persistence.py  model_builder.py  splitter.py  metrics.py
        ↑                ↑                ↑             ↑
                     search_engine.py（随机搜索主循环）
                         ↑
                     trainer_optimizer.py（本文件，公开 API）
                         ↑
                     main_cli.py
"""


def train_final_model_and_dpoint(
    df_clean: pd.DataFrame,
    best_config: Dict[str, Any],
    seed: int = 42,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    在全部有标签数据上拟合最终模型，输出 Dpoint 序列和模型参数工件。
    支持 sklearn 兼容模型和 PyTorch MLP 模型。

    ⚠️  WARNING — IN-SAMPLE LOOK-AHEAD BIAS:
    The model is trained on the full dataset and then used to predict dpoint on that same dataset.
    The resulting equity curve in the final report is therefore an IN-SAMPLE fit, NOT a true
    out-of-sample backtest. It WILL overstate real trading performance.

    ⚠️  警告 — 全样本前向偏差：
    此函数用全部历史数据训练模型，再对同段数据预测 dpoint，最终报告中的 equity curve
    是【样本内拟合展示】，不代表任何真实可操作的交易表现，数值必然偏乐观。
    真实样本外表现请参考 walk-forward 各折验证期的 out-of-sample 指标（见 SearchLog sheet）。
    """
    feat_cfg = best_config["feature_config"]
    model_cfg = best_config["model_config"]

    X, y, meta = build_features_and_labels(df_clean, feat_cfg)
    model_params: Optional[Dict[str, Any]] = None

    model_type = str(model_cfg["model_type"])
    device = _get_device()

    # 深度学习模型（MLP/LSTM/GRU/CNN/Transformer）
    if model_type in ["mlp", "lstm", "gru", "cnn", "transformer"]:
        input_dim = X.shape[1]
        model_cfg_with_input_dim = {**model_cfg, "input_dim": input_dim}
        trained_model = train_pytorch_model(X, y, model_cfg_with_input_dim, device)
        seq_len = int(model_cfg.get("seq_len", 20)) if model_type != "mlp" else 1
        dpoint = predict_pytorch_model(trained_model, X, device, seq_len=seq_len)
        model_params = model_cfg_with_input_dim

    else:  # sklearn compatible models (LogReg, SGD, XGBoost)
        model = make_model(best_config, seed=seed)
        if isinstance(model, Pipeline):
            model.fit(X.values, y.values)
            proba = model.predict_proba(X.values)[:, 1]
            try:
                from sklearn.pipeline import Pipeline as PipelineType
                scaler = model.named_steps["scaler"]
                clf = model.named_steps["clf"]
                model_params = {
                    "feature_names": meta.feature_names,
                    "mean": np.asarray(scaler.mean_, dtype=float).tolist(),
                    "scale": np.asarray(scaler.scale_, dtype=float).tolist(),
                    "coef": np.asarray(clf.coef_[0], dtype=float).tolist(),
                    "intercept": float(clf.intercept_[0]),
                }
            except Exception:
                model_params = None
        else:
            model.fit(X, y)
            proba = model.predict_proba(X)[:, 1]
        dpoint = pd.Series(proba, index=X.index, name="dpoint")

    artifacts = {
        "feature_meta": {
            "feature_names": meta.feature_names,
            "feature_params": meta.params,
            "dpoint_explainer": meta.dpoint_explainer,
        },
        "model": {
            "type": model_type,
            "params": model_cfg,
        },
    }
    if model_params is not None:
        artifacts["model_params"] = model_params

    return dpoint, artifacts


# =========================================================
# 公开 API 导出
# =========================================================
__all__ = [
    # 校准模块
    "CALIBRATION_METHODS",
    "ProbabilityCalibrator",
    "RollingCalibrationMonitor",
    "compute_brier_score",
    "compute_calibration_curve",
    "compute_ece_mce",
    "compute_all_calibration_metrics",
    "create_calibrator_from_config",
    # 解释器模块
    "FEATURE_GROUPS",
    "FeatureImportanceExplainer",
    "FeatureUsageTracker",
    "LocalExplainer",
    "compute_feature_group_ranking",
    "compute_feature_deletion_experiment",
    "compute_regime_feature_importance",
    # 持久化模块
    "config_hash",
    "best_so_far_path",
    "best_pool_path",
    "load_best_so_far",
    "load_best_so_far_metric",
    "save_best_so_far",
    "load_best_pool",
    "save_best_pool",
    "update_best_pool",
    # 搜索引擎模块
    "SearchSpaces",
    "TrainResult",
    "random_search_train",
    # 训练优化器模块
    "train_final_model_and_dpoint",
    # 常量导出（兼容性）
    "MIN_CLOSED_TRADES_PER_FOLD",
    "TARGET_CLOSED_TRADES_PER_FOLD",
    "LAMBDA_TRADE_PENALTY",
]
