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


def final_holdout_split(
    df: pd.DataFrame,
    holdout_ratio: float = 0.15,
    min_holdout_rows: int = 60,
) -> Tuple[
    pd.DataFrame,  # search_df
    pd.DataFrame,  # holdout_df
]:
    """
    P0: Final holdout split - 从数据末尾切出 holdout 集，确保搜索流程完全不接触。

    三阶段验证流程：
        1. Search OOS: walk-forward splits 在 search 数据上评估
        2. Selection OOS: top-K 候选在 search 数据上重新验证
        3. Final Holdout OOS: 最优配置在 holdout 集上做最终评估

    参数说明：
        holdout_ratio    : holdout 集占总数据的比例，默认 15%
        min_holdout_rows : holdout 集最小行数，不足时抛出异常

    返回：
        (search_df, holdout_df) — 原始 DataFrame 切分结果
        （调用处需自行调用 build_features_and_labels 生成 X, y）
    """
    n = len(df)
    holdout_size = int(n * holdout_ratio)

    if holdout_size < min_holdout_rows:
        raise ValueError(
            f"holdout_size={holdout_size} < min_holdout_rows={min_holdout_rows}. "
            f"Increase holdout_ratio or use more data."
        )

    split_idx = n - holdout_size
    search_df = df.iloc[:split_idx].copy()
    holdout_df = df.iloc[split_idx:].copy()

    print(
        f"[INFO] P0 Final Holdout Split: search={len(search_df)} rows, "
        f"holdout={len(holdout_df)} rows ({holdout_ratio*100:.1f}%)"
    )

    return search_df, holdout_df


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


# =========================================================
# P2: Nested Walk-Forward（防止模型选择时的前向偏差）
# =========================================================
def nested_walkforward_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_outer_folds: int = 3,
    n_inner_folds: int = 2,
    train_start_ratio: float = 0.5,
    min_rows: int = 60,
    embargo_days: int = 5,
) -> List[Tuple[
    Tuple[pd.DataFrame, pd.Series],  # outer_train
    Tuple[pd.DataFrame, pd.Series],  # outer_val
    List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]],  # inner_splits
]]:
    """
    P2: 嵌套 Walk-Forward 切分。

    与普通 walk-forward 的区别：
        - 外层：标准的 walk-forward split（训练集扩展到当前点，验证集下一段）
        - 内层：在外层训练集上再做一次 walk-forward，用于模型选择/超参调优
        - 这样可以避免"用验证集选择模型，再用同一验证集评估模型"的前向偏差

    另外增加了 embargo_days：
        - 在训练集和验证集之间留出 gap，防止特征工程中的滚动窗口
          （如 moving average）导致的信息泄露

    参数说明：
        n_outer_folds    : 外层折数，默认 3
        n_inner_folds    : 内层折数，默认 2（在外层训练集上）
        train_start_ratio: 外层初始训练集比例
        min_rows         : 最小行数约束
        embargo_days     : 训练集和验证集之间的 embargo 天数

    返回：
        List[(
            (outer_train_X, outer_train_y),
            (outer_val_X, outer_val_y),
            [inner_splits...]
        )]
    """
    n = len(X)
    cuts = [
        train_start_ratio + (1.0 - train_start_ratio) * i / n_outer_folds
        for i in range(n_outer_folds + 1)
    ]

    splits = []
    for k in range(len(cuts) - 1):
        outer_train_end = int(n * cuts[k])
        outer_val_end = int(n * cuts[k + 1])

        # 应用 embargo：验证集向后推移
        outer_val_start = outer_train_end + embargo_days
        if outer_val_start >= outer_val_end:
            print(f"[WARN] nested_walkforward: fold {k+1} skipped due to embargo_days={embargo_days}")
            continue

        X_outer_train = X.iloc[:outer_train_end]
        y_outer_train = y.iloc[:outer_train_end]
        X_outer_val = X.iloc[outer_val_start:outer_val_end]
        y_outer_val = y.iloc[outer_val_start:outer_val_end]

        if len(X_outer_train) < min_rows or len(X_outer_val) < min_rows:
            print(f"[WARN] nested_walkforward: fold {k+1} skipped (train={len(X_outer_train)}, val={len(X_outer_val)})")
            continue

        # 内层 walk-forward：在外层训练集上做
        inner_cuts = [
            1.0 * i / n_inner_folds
            for i in range(n_inner_folds + 1)
        ]
        inner_splits = []
        for j in range(len(inner_cuts) - 1):
            inner_train_end = int(len(X_outer_train) * inner_cuts[j])
            inner_val_end = int(len(X_outer_train) * inner_cuts[j + 1])

            if inner_train_end < min_rows or (inner_val_end - inner_train_end) < min_rows:
                continue

            X_inner_train = X_outer_train.iloc[:inner_train_end]
            y_inner_train = y_outer_train.iloc[:inner_train_end]
            X_inner_val = X_outer_train.iloc[inner_train_end:inner_val_end]
            y_inner_val = y_outer_train.iloc[inner_train_end:inner_val_end]

            inner_splits.append(((X_inner_train, y_inner_train), (X_inner_val, y_inner_val)))

        if not inner_splits:
            print(f"[WARN] nested_walkforward: fold {k+1} skipped (no valid inner splits)")
            continue

        splits.append((
            (X_outer_train, y_outer_train),
            (X_outer_val, y_outer_val),
            inner_splits
        ))

    if not splits:
        print(f"[WARN] nested_walkforward: ALL {n_outer_folds} folds skipped.")
    return splits


# =========================================================
# P2: Embargo Gap Walk-Forward（带 embargo 的标准 walk-forward）
# =========================================================
def walkforward_splits_with_embargo(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 4,
    train_start_ratio: float = 0.5,
    min_rows: int = 60,
    embargo_days: int = 5,
) -> List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]]:
    """
    P2: 带 embargo 的 Walk-Forward 切分。

    在标准 walk-forward 基础上增加 embargo_days：
        - 训练集和验证集之间留出 gap
        - 防止滚动窗口特征（如 MA、RSI 等）导致的信息泄露

    参数说明：
        embargo_days : 训练集和验证集之间的 embargo 天数，默认 5 天

    切分示意（n_folds=4, train_start_ratio=0.5, embargo_days=5）：
        折1: train=[0%~50%]   val=[50%+5天~62.5%]
        折2: train=[0%~62%]   val=[62%+5天~75%]
        ...
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

        # 应用 embargo
        val_start = train_end + embargo_days
        if val_start >= val_end:
            print(f"[WARN] walkforward_splits_with_embargo: fold {k+1} skipped (embargo={embargo_days})")
            continue

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[val_start:val_end], y.iloc[val_start:val_end]

        if len(X_train) < min_rows or len(X_val) < min_rows:
            print(f"[WARN] walkforward_splits_with_embargo: fold {k+1} skipped (train={len(X_train)}, val={len(X_val)})")
            continue

        splits.append(((X_train, y_train), (X_val, y_val)))

    if not splits:
        print(f"[WARN] walkforward_splits_with_embargo: ALL {n_folds} folds skipped.")
    return splits
