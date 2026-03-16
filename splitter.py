# splitter.py
"""
Walk-forward 时序切分。
验证集不重叠，训练集累积扩展（expanding window）。

P3-20 新增：recommend_n_folds — 根据数据量自适应推算合理折数
    原版 n_folds=4 固定，数据较少时每折验证期可能只有 60 个交易日，
    统计置信度不足（4 折 × ~15 次交易/折 = 60 次，样本偏少）。
    数据较多时 4 折又会浪费大量可用的验证机会。

    推算逻辑：
        val_rows_per_fold = n_total × (1 - train_start_ratio) / n_folds
        expected_trades   = val_rows_per_fold × assumed_trade_freq
        最优折数 ≈ 使 expected_trades ≈ target_trades_per_fold 的最大 n_folds，
        同时满足 min_rows 约束和 [min_folds, max_folds] 范围。
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def walkforward_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 4,
    train_start_ratio: float = 0.5,
    min_rows: int = 80,
) -> List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]]:
    """
    生成 walk-forward 时序切分。

    参数说明：
        n_folds          : 验证折数，默认 4。
        train_start_ratio: 第一折训练集占全部数据的比例，默认 0.5。
        min_rows         : 训练集或验证集的最小行数，不足时跳过该折并打印警告。

    切分示意（n_folds=4, train_start_ratio=0.5）：
        折1: train=[0%~50%]   val=[50%~62.5%]
        折2: train=[0%~62%]   val=[62.5%~75%]
        折3: train=[0%~75%]   val=[75%~87.5%]
        折4: train=[0%~87%]   val=[87.5%~100%]
        （共 n_folds = 4 个验证折，首段 0~50% 数据仅用作初始训练集）

    注意：验证集不重叠，训练集累积扩展。
    """
    n = len(X)
    cuts = [
        train_start_ratio + (1.0 - train_start_ratio) * i / n_folds
        for i in range(n_folds + 1)
    ]

    splits = []
    for k in range(len(cuts) - 1):
        train_end = int(n * cuts[k])
        val_end = int(n * cuts[k + 1])
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]

        if len(X_train) < min_rows or len(X_val) < min_rows:
            print(
                f"[WARN] walkforward_splits: fold {k + 1} skipped "
                f"(train={len(X_train)}, val={len(X_val)}, min_rows={min_rows}). "
                f"Consider reducing n_folds or min_rows."
            )
            continue
        splits.append(((X_train, y_train), (X_val, y_val)))

    if not splits:
        print(
            f"[WARN] walkforward_splits: ALL {n_folds} folds skipped. "
            f"Total rows={n}, train_start_ratio={train_start_ratio}, min_rows={min_rows}."
        )
    return splits


def recommend_n_folds(
    n_samples: int,
    train_start_ratio: float = 0.5,
    target_trades_per_fold: int = 4,
    assumed_trade_freq: float = 1.0 / 15.0,
    min_rows: int = 80,
    min_folds: int = 2,
    max_folds: int = 8,
) -> int:
    """
    P3-20：根据数据量自适应推算合理的 walk-forward 折数。

    推算原则：
        在满足以下三个约束的前提下，选取尽可能大的折数：
            ① 每折验证期行数 ≥ min_rows（确保回测有足够交易日）
            ② 每折期望交易次数 ≈ target_trades_per_fold（与 penalty 对齐）
            ③ 折数在 [min_folds, max_folds] 范围内

    参数说明：
        n_samples             — 总样本数（特征工程后的有效行数）
        train_start_ratio     — 初始训练集比例，应与 walkforward_splits 保持一致
        target_trades_per_fold — 每折目标交易次数（与 constants.TARGET_CLOSED_TRADES_PER_FOLD 对齐）
        assumed_trade_freq    — 假设的每个交易日触发一次交易的概率（保守估计）
                                默认 1/15 ≈ 平均每 15 个交易日一次，可根据标的特性调整
        min_rows              — 每折验证期的最小行数（与 walkforward_splits.min_rows 一致）
        min_folds             — 最少折数下界
        max_folds             — 最多折数上界

    返回值：
        推荐的折数（int），已经过 [min_folds, max_folds] 的 clip。

    示例（基于 A 股 5 年数据约 1200 个交易日）：
        >>> recommend_n_folds(n_samples=1200)
        6

    示例（数据较少，约 2 年 480 个交易日）：
        >>> recommend_n_folds(n_samples=480)
        3
    """
    val_pool = n_samples * (1.0 - train_start_ratio)  # 总验证池大小（行数）

    best_n = min_folds
    for n in range(max_folds, min_folds - 1, -1):
        val_rows_per_fold = val_pool / n
        # 约束①：验证折行数 ≥ min_rows
        if val_rows_per_fold < min_rows:
            continue
        # 约束②：期望交易次数 ≥ target_trades_per_fold
        expected_trades = val_rows_per_fold * assumed_trade_freq
        if expected_trades < target_trades_per_fold:
            continue
        # 满足所有约束，选取当前 n（从大到小遍历，第一个满足的即最大可行折数）
        best_n = n
        break

    return max(min_folds, min(max_folds, best_n))
