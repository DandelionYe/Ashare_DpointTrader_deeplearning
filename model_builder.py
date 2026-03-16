# model_builder.py
"""
模型构建与预测。
支持 LogisticRegression、SGDClassifier（均含 StandardScaler Pipeline）、XGBoost 和 PyTorch MLP。
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
from dl_model_builder import MLP, _get_device, train_pytorch_model, predict_pytorch_model

# 模型训练超参数常量（可通过配置覆盖）
LOGREG_MAX_ITER: int = 8000
SGD_MAX_ITER: int = 3000
SGD_TOL: float = 1e-3


def _try_import_xgboost() -> Any:
    try:
        import xgboost as xgb  # type: ignore
        return xgb
    except Exception:
        return None


def make_model(candidate: Dict[str, Any], seed: int) -> Any:
    """
    根据 candidate["model_config"] 构建未拟合的 sklearn 兼容模型（含 predict_proba）或 PyTorch 模型。

    支持的 model_type:
        logreg  — LogisticRegression + StandardScaler Pipeline
        sgd     — SGDClassifier(log_loss) + StandardScaler Pipeline
        xgb     — XGBClassifier（需安装 xgboost）
        mlp     — PyTorch MLP（多层感知机）
    """
    model_type = str(candidate["model_config"]["model_type"])
    model_config = candidate["model_config"]

    if model_type == "logreg":
        C = float(model_config["C"])
        penalty = str(model_config["penalty"])
        solver = str(model_config["solver"])
        class_weight = model_config.get("class_weight", None)
        l1_ratio = model_config.get("l1_ratio", None)
        max_iter = int(model_config.get("max_iter", LOGREG_MAX_ITER))
        clf = LogisticRegression(
            C=C, penalty=penalty, solver=solver, l1_ratio=l1_ratio,
            class_weight=class_weight, max_iter=max_iter, random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_type == "sgd":
        alpha = float(model_config["alpha"])
        penalty = str(model_config["penalty"])
        l1_ratio = float(model_config.get("l1_ratio", 0.15))
        class_weight = model_config.get("class_weight", None)
        max_iter = int(model_config.get("max_iter", SGD_MAX_ITER))
        tol = float(model_config.get("tol", SGD_TOL))
        clf = SGDClassifier(
            loss="log_loss", alpha=alpha, penalty=penalty,
            l1_ratio=l1_ratio if penalty == "elasticnet" else None,
            class_weight=class_weight, max_iter=max_iter, tol=tol, random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_type == "xgb":
        xgb = _try_import_xgboost()
        if xgb is None:
            raise RuntimeError("xgboost_not_installed")
        # 移除 verbose，改为 verbosity
        xgb_params = dict(model_config["params"])
        xgb_params.pop("verbose", None) # 确保移除老旧的 verbose 参数
        if "eval_metric" not in xgb_params: # 确保 eval_metric 存在
            xgb_params["eval_metric"] = "logloss"
        clf = xgb.XGBClassifier(**xgb_params)
        return clf

    if model_type == "mlp":
        # PyTorch 模型不需要在 make_model 阶段训练，只返回模型结构
        input_dim = int(model_config["input_dim"])
        hidden_dim = int(model_config["hidden_dim"])
        dropout_rate = float(model_config.get("dropout_rate", 0.5))
        return MLP(input_dim, hidden_dim, dropout_rate=dropout_rate)

    raise ValueError(f"Unknown model_type: {model_type}")


def predict_dpoint(model: Any, X: pd.DataFrame) -> pd.Series:
    """
    调用 predict_proba（或 PyTorch 模型前向传播），返回 class=1 的概率作为 Dpoint Series。
    """
    if isinstance(model, MLP):
        device = _get_device()
        return predict_pytorch_model(model, X, device)

    if hasattr(model, "predict_proba"):
        if isinstance(model, Pipeline):
            proba = model.predict_proba(X.values)[:, 1]
        else:
            proba = model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=X.index, name="dpoint")

    raise ValueError("model has no predict_proba or is not a supported PyTorch MLP")