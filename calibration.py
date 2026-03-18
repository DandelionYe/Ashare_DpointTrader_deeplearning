# calibration.py
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
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


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
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            self.calibrator.fit(y_prob.reshape(-1, 1), y_true)
        
        elif self.method == "isotonic":
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
