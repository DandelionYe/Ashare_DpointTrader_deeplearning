# search_engine.py
"""
随机搜索主引擎（P0 缺陷修复版）。

P03 修复：XGBoost CUDA 检测语法错误
    - 原版 `from xgb import callback` 模块名错误，且 xgb 变量作用域不对
    - 修复后：直接使用 _try_import_xgboost() 返回的模块对象，无多余 import

P01 修复：feat_cache 在多进程下完全失效
    - 原版：feat_cache 是闭包变量，loky 后端 fork 子进程后每个 worker 拿到的是
      主进程内存的独立副本，写入无效，缓存命中率永远为零
    - 修复后：在主进程中按 feature_config hash 预计算所有唯一特征，
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
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline

from constants import (
    MIN_CLOSED_TRADES_PER_FOLD,
    TARGET_CLOSED_TRADES_PER_FOLD,
    LAMBDA_TRADE_PENALTY,
)
from feature_dpoint import build_features_and_labels, FeatureMeta
from model_builder import make_model, predict_dpoint, _try_import_xgboost
from dl_model_builder import MLP, _get_device, train_pytorch_model, predict_pytorch_model
from splitter import walkforward_splits
from metrics import metric_from_fold_ratios, trade_penalty, backtest_fold_stats
from persistence import (
    config_hash,
    best_so_far_path,
    best_pool_path,
    load_best_so_far,
    load_best_so_far_metric,   # P02：新增导入
    save_best_so_far,
    update_best_pool,          # P02：现在真正被调用
    load_best_pool,            # P02：新增导入
)


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
    window_pool: List[List[int]]
    logreg_choices: List[Dict[str, Any]]
    sgd_choices: List[Dict[str, Any]]
    xgb_param_pool: List[Dict[str, Any]]
    mlp_param_pool: List[Dict[str, Any]]
    lstm_param_pool: List[Dict[str, Any]]
    gru_param_pool: List[Dict[str, Any]]
    cnn_param_pool: List[Dict[str, Any]]
    transformer_param_pool: List[Dict[str, Any]]
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


def _build_search_spaces(seed: int, input_dim: int) -> SearchSpaces:
    xgb_available = _try_import_xgboost() is not None
    C_pool = list(np.logspace(-2, 2, 13))
    logreg_choices = []
    for C in C_pool:
        for cw in [None, "balanced"]:
            logreg_choices.append({"penalty": "l2", "solver": "lbfgs", "C": C, "class_weight": cw})
            logreg_choices.append({"penalty": "l1", "solver": "liblinear", "C": C, "class_weight": cw})

    sgd_choices = []
    for alpha in list(np.logspace(-5, -2, 10)):
        sgd_choices.append({"alpha": alpha, "penalty": "l2", "class_weight": "balanced"})

    xgb_param_pool = []
    if xgb_available:
        device_args = {"device": "cuda", "tree_method": "hist"} if _get_cuda_available() else {"n_jobs": 4}
        for depth in [2, 3, 4]:
            xgb_param_pool.append(dict(
                n_estimators=200, max_depth=depth, learning_rate=0.05,
                objective="binary:logistic", eval_metric="logloss", **device_args
            ))

    mlp_param_pool = []
    for hd in [32, 64, 128]:
        for lr in [0.001, 0.005]:
            for bs in [64, 128, 256]:
                mlp_param_pool.append(dict(
                    input_dim=input_dim, hidden_dim=hd, learning_rate=lr,
                    epochs=30, batch_size=bs, dropout_rate=0.4, model_type="mlp"
                ))

    lstm_param_pool = []
    for hd in [32, 64, 128]:
        for lr in [0.001, 0.003]:
            for bs in [32, 64, 128]:
                for layers in [1, 2]:
                    for bidir in [False, True]:
                        lstm_param_pool.append(dict(
                            input_dim=input_dim, model_type="lstm", hidden_dim=hd,
                            learning_rate=lr, epochs=30, batch_size=bs, dropout_rate=0.3,
                            seq_len=20, num_layers=layers, bidirectional=bidir
                        ))

    gru_param_pool = []
    for hd in [32, 64, 128]:
        for lr in [0.001, 0.003]:
            for bs in [32, 64, 128]:
                for layers in [1, 2]:
                    gru_param_pool.append(dict(
                        input_dim=input_dim, model_type="gru", hidden_dim=hd,
                        learning_rate=lr, epochs=30, batch_size=bs, dropout_rate=0.3,
                        seq_len=20, num_layers=layers, bidirectional=False
                    ))

    cnn_param_pool = []
    for nf in [32, 64, 128]:
        for lr in [0.001, 0.003]:
            for bs in [32, 64, 128]:
                cnn_param_pool.append(dict(
                    input_dim=input_dim, model_type="cnn", num_filters=nf,
                    learning_rate=lr, epochs=30, batch_size=bs, dropout_rate=0.3,
                    seq_len=20, kernel_sizes=[2, 3, 5]
                ))

    transformer_param_pool = []
    for d_model in [32, 64, 128]:
        for nhead in [2, 4]:
            for lr in [0.0005, 0.001]:
                for bs in [32, 64, 128]:
                    transformer_param_pool.append(dict(
                        input_dim=input_dim, model_type="transformer",
                        d_model=d_model, nhead=nhead, learning_rate=lr,
                        epochs=30, batch_size=bs, dropout_rate=0.1, seq_len=20,
                        num_layers=2, dim_feedforward=128
                    ))

    return SearchSpaces(
        window_pool=[[2, 3, 5, 8, 13, 21], [5, 10, 20, 60], [3, 7, 14, 28]],
        logreg_choices=logreg_choices,
        sgd_choices=sgd_choices,
        xgb_param_pool=xgb_param_pool,
        mlp_param_pool=mlp_param_pool,
        lstm_param_pool=lstm_param_pool,
        gru_param_pool=gru_param_pool,
        cnn_param_pool=cnn_param_pool,
        transformer_param_pool=transformer_param_pool,
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
        # P3-19：RSI / 布林带宽的 ta_windows 候选（短期、中期、长期三档）
        ta_window_pool=[[6, 14], [14, 20], [6, 14, 20]],
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
) -> Tuple[float, float, Dict[str, Any]]:
    """
    评估单个候选配置。

    P01 修复说明：
        computed_feats 现在始终由主进程预计算后传入，不再在 worker 内部
        维护 feat_cache 闭包（多进程下闭包缓存完全无效）。
        当 computed_feats 为 None 时仍支持惰性计算（兼容单独调用）。
    """
    if computed_feats is None:
        feat_cfg = candidate["feature_config"]
        computed_feats = build_features_and_labels(df_clean, feat_cfg)
        if computed_feats is None:
            return (-np.inf, 100000.0, {"skip": "feat_fail"})

    X, y, meta = computed_feats
    trade_cfg = candidate["trade_config"]
    initial_cash = float(trade_cfg["initial_cash"])
    early_stop_floor = initial_cash * _EARLY_STOP_RATIO
    cand_seed = int(candidate.get("candidate_seed", 42))

    if len(X.columns) > max_features:
        return (-np.inf, initial_cash, {"skip": "too_many_feats", "n_features": len(X.columns)})

    splits = walkforward_splits(X, y, n_folds=n_folds, train_start_ratio=train_start_ratio, min_rows=wf_min_rows)
    if not splits:
        return (-np.inf, initial_cash, {"skip": "no_splits"})

    ratios, equities, closed_trades = [], [], []
    device = _get_device()

    # 用实际特征维度动态覆盖 input_dim，避免 PyTorch 线性层维度不匹配）
    for (X_tr, y_tr), (X_va, y_va) in splits:
        model_type = str(candidate["model_config"]["model_type"])
        if model_type in ["mlp", "lstm", "gru", "cnn", "transformer"]:
            actual_cfg = {**candidate["model_config"], "input_dim": X_tr.shape[1]}
            trained_model = train_pytorch_model(X_tr, y_tr, actual_cfg, device)
            dp_val = predict_pytorch_model(
                trained_model, X_va, device,
                seq_len=int(actual_cfg.get("seq_len", 20))
            )
        else:
            model = make_model(candidate, seed=cand_seed)
            model.fit(
                X_tr.values if isinstance(model, Pipeline) else X_tr,
                y_tr.values if isinstance(model, Pipeline) else y_tr,
            )
            dp_val = predict_dpoint(model, X_va)

        fold_stats = backtest_fold_stats(df_clean, X_va, dp_val, trade_cfg)
        eq_end = float(fold_stats["equity_end"])

        if int(fold_stats["n_closed"]) < MIN_CLOSED_TRADES_PER_FOLD:
            return (-np.inf, initial_cash, {"skip": "too_few_trades"})
        if eq_end < early_stop_floor:
            return (-np.inf, initial_cash, {"skip": "early_stop"})

        equities.append(eq_end)
        ratios.append(eq_end / initial_cash)
        closed_trades.append(int(fold_stats["n_closed"]))

    geom = metric_from_fold_ratios(ratios)
    min_r = float(np.min(ratios))
    metric_raw = 0.8 * geom + 0.2 * min_r
    penalty = trade_penalty(closed_trades)

    return (
        float(metric_raw - penalty),
        float(np.mean(equities)),
        {
            "n_features": len(X.columns),
            "geom_mean_ratio": geom,
            "min_fold_ratio": min_r,
            "metric_raw": metric_raw,
            "penalty": penalty,
            "avg_closed_trades": float(np.mean(closed_trades)),
        },
    )


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
        model_cfg = dict(rng.choice(spaces.mlp_param_pool))
    elif mt == "lstm":
        model_cfg = dict(rng.choice(spaces.lstm_param_pool))
    elif mt == "gru":
        model_cfg = dict(rng.choice(spaces.gru_param_pool))
    elif mt == "cnn":
        model_cfg = dict(rng.choice(spaces.cnn_param_pool))
    elif mt == "transformer":
        model_cfg = dict(rng.choice(spaces.transformer_param_pool))
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
    wf_min_rows: int = 60,
    n_jobs: int = -1,
    n_rounds: int = 4,             # P05 新增：分轮数，每轮后更新 incumbent
    pool_exploit_prob: float = 0.3, # P02 新增：从 Top-K 池采样的概率
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
    """
    rng = np.random.default_rng(seed)
    tp = trade_params or {"initial_cash": 100000.0}
    device = _get_device()

    # --- 初始化搜索空间 ---
    _, _, init_meta = build_features_and_labels(df_clean, {
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
    X_inc, y_inc, meta_inc = build_features_and_labels(df_clean, best_cfg["feature_config"])
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
        best_m_raw, best_eq_raw, info_inc = _eval_candidate(
            best_cfg, df_clean, max_features, n_folds,
            train_start_ratio, wf_min_rows, (X_inc, y_inc, meta_inc)
        )
        best_m = float(best_m_raw)
        best_eq = float(best_eq_raw)
        training_notes_extra = [
            "P1-5: incumbent 为新配置，已执行完整 walk-forward 评估。"
        ]

    initial_m = best_m   # 保存初始值，用于最终 global_best_metric_prev

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
        
        print(f"[SEARCH] Round {round_idx + 1}/{n_rounds}, evaluating {actual_runs} candidates...")

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

        # P01：主进程统一预计算本轮所有唯一 feature_config
        # 相同 feature_config 的候选只计算一次，多进程下通过参数传递，不依赖共享内存
        feat_map: Dict[str, Tuple] = {}
        for c in round_c:
            fhash = config_hash(c["feature_config"])
            if fhash not in feat_map:
                feat_map[fhash] = build_features_and_labels(df_clean, c["feature_config"])

        # P01：将预计算特征作为显式参数传入，完全绕开多进程缓存问题
        round_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_eval_candidate)(
                c,
                df_clean,
                max_features,
                n_folds,
                train_start_ratio,
                wf_min_rows,
                feat_map[config_hash(c["feature_config"])],   # P01：主进程预计算结果
            )
            for c in round_c
        )

        # --- 收集本轮结果，实时更新 incumbent ---
        for i, (m, eq, info) in enumerate(round_results):
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
    print(f"[SEARCH] Search complete. initial_m={initial_m:.6f}, cand_best_m={cand_best_m:.6f}, epsilon={epsilon}")
    if cand_best_cfg is not None and cand_best_m > initial_m + epsilon:
        updated = True
        save_best_so_far(output_dir, best_cfg, best_m)
        print(f"[SEARCH] Best updated! new best_m={best_m:.6f}")
    else:
        reason = "not_exceed_epsilon" if cand_best_cfg is not None else "no_valid_cand"
        print(f"[SEARCH] Best NOT updated. reason={reason}")

    # --- 获取最终 feature_meta ---
    final_fhash = config_hash(best_cfg["feature_config"])
    if final_fhash in final_feat_map:
        final_meta = final_feat_map[final_fhash][2]
    else:
        _, _, final_meta = build_features_and_labels(df_clean, best_cfg["feature_config"])

    training_notes = [
        f"Device: {device}",
        f"n_rounds: {n_rounds}，每轮 ~{runs_per_round} 个候选",
        f"总候选数: {len(all_candidates)}",
        f"CUDA 可用: {_get_cuda_available()}",
        f"P01: 特征预计算已在主进程完成，多进程缓存问题已修复",
        f"P02: Top-K 池已接入，pool_exploit_prob={pool_exploit_prob}",
        f"P03: XGBoost CUDA 检测语法已修复",
        f"P05: 分轮搜索已启用，每轮后更新 incumbent",
    ] + training_notes_extra

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
    )
