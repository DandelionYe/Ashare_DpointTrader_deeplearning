# trainer_optimizer.py
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
from __future__ import annotations

from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from feature_dpoint import build_features_and_labels
from model_builder import make_model
from dl_model_builder import MLP, _get_device, train_pytorch_model, predict_pytorch_model

# 从 search_engine 重新导出，保持 main_cli.py 的 import 不变
from search_engine import random_search_train, TrainResult  # noqa: F401

# 从 constants 重新导出，保持 reporter.py 旧版 import 的兼容性
# （reporter.py 已在 High-02 中改为直接从 constants import，此处仅作保险层）
from constants import (  # noqa: F401
    MIN_CLOSED_TRADES_PER_FOLD,
    TARGET_CLOSED_TRADES_PER_FOLD,
    LAMBDA_TRADE_PENALTY,
)

__all__ = [
    "random_search_train",
    "TrainResult",
    "train_final_model_and_dpoint",
    "MIN_CLOSED_TRADES_PER_FOLD",
    "TARGET_CLOSED_TRADES_PER_FOLD",
    "LAMBDA_TRADE_PENALTY",
]


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