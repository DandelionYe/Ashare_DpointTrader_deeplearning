# explainer.py
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
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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
