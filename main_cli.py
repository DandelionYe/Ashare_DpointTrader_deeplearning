# main_cli.py
"""
P1-6 修复：n_jobs 与 CUDA 冲突（见 _resolve_n_jobs）

P2-4 修复：conda 激活逻辑无限循环风险
    原版仅依赖 CONDA_DEFAULT_ENV 环境变量判断是否需要激活，
    在部分 IDE / 调度系统中该变量即使激活后也未被设置，导致脚本无限重启自身。
    此外 Windows 分支使用 shell=True 拼接含路径的命令字符串，存在路径注入风险。

    修复方案：
        1. 在重新启动子进程前，将 _ASHARE_RELAUNCHED=1 写入子进程环境变量。
        2. 脚本启动时优先检查 _ASHARE_RELAUNCHED，若已设置则跳过所有激活逻辑，
           彻底打破递归。
        3. Windows 分支改为 list 参数 + shell=False，消除路径注入风险。

P2-5 修复：trade_params 双来源逻辑混乱
    原版 trade_params 字典中既有 initial_cash，又混入 buy_threshold /
    sell_threshold / confirm_days / min_hold_days，
    但这些阈值字段在搜索中不起实质作用（搜索空间自己采样），
    而最终 backtest 又从 best_config["trade_config"] 取参数，
    造成"传了但没用"的迷惑性。

    修复方案：
        1. 将变量重命名为 search_initial_params，仅保留 initial_cash。
        2. 在参数旁加注释说明其唯一用途：为随机采样的候选提供初始资金基准。
        3. backtest_from_dpoint 调用保持从 best_config["trade_config"] 取全部参数，
           来源单一，无歧义。

P2（本次修复）：
    ① docstring 注释一致性
        backtester_engine.py 的 A 股约束说明已在 P1-3 中改为 t+1 开盘价执行，
        main_cli.py 内无直接过时注释，但相关调用处均已与实际行为一致。
    ② 硬编码本地路径（跨平台可用性）
        移除 Windows 绝对路径 fallback，DEFAULT_DATA_PATH 改为 None。
        main() 内新增空值检查，路径缺失时打印清晰提示后退出，
        而非让底层文件 IO 抛出难以理解的异常。
        路径提供方式（优先级从高到低）：
            1. 环境变量：export ASHARE_DATA_PATH=/path/to/data.xlsx
            2. 命令行参数：python main_cli.py --data_path /path/to/data.xlsx
            3. 两者均未设置时，程序启动后打印提示并退出
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
from typing import Dict, Optional, List, Any

# =========================================================
# P2-4：conda 激活块 — 防无限递归版
# =========================================================
# 核心机制：子进程启动时注入 _ASHARE_RELAUNCHED=1 环境变量；
# 脚本一进入就检查该变量，若已设置则完全跳过激活逻辑，打破递归。
_already_relaunched: bool = os.environ.get("_ASHARE_RELAUNCHED") == "1"

# Skip conda check if SKIP_CONDA env var is set
_skip_conda: bool = os.environ.get("SKIP_CONDA") == "1"

if not _already_relaunched and not _skip_conda:
    _target_env = "ashare_dpoint"
    _in_correct_env = os.environ.get("CONDA_DEFAULT_ENV") == _target_env

    if not _in_correct_env:
        # 检查 conda 是否可用
        import shutil
        if shutil.which("conda") is None:
            print("[WARNING] conda not found in PATH, skipping conda activation. Set SKIP_CONDA=1 to suppress this warning.")
        else:
            # 构造子进程环境：继承当前环境，加入防递归标记
            _child_env = {**os.environ, "_ASHARE_RELAUNCHED": "1"}

            if os.name == "nt":  # Windows
                # P2-4：改用 list 参数，避免 shell=True 的路径注入风险
                _cmd = [
                    "conda", "run",
                    "--no-capture-output",
                    "-n", _target_env,
                    sys.executable,   # 使用绝对路径的 Python 解释器
                ] + sys.argv
                print(f"[INFO] P2-4: 以 conda 环境 '{_target_env}' 重新启动...")
                subprocess.run(_cmd, env=_child_env, check=True)
            else:  # Linux / macOS
                _cmd = [
                    "conda", "run",
                    "--no-capture-output",
                    "-n", _target_env,
                    sys.executable,
                ] + sys.argv
                print(f"[INFO] P2-4: 以 conda 环境 '{_target_env}' 重新启动...")
                subprocess.run(_cmd, env=_child_env, check=True)

            sys.exit(0)
# =========================================================
# End P2-4 conda 激活块
# =========================================================


import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from data_loader import load_stock_excel
from trainer_optimizer import random_search_train, train_final_model_and_dpoint
from backtester_engine import backtest_from_dpoint, compute_buy_and_hold  # P3-17
from reporter import save_run_outputs, find_latest_run
from splitter import recommend_n_folds  # P3-20


# ====== 数据路径配置 ======
# P2 修复：移除硬编码 Windows 绝对路径，改为 None。
# 路径提供方式（优先级从高到低）：
#   1. 环境变量：export ASHARE_DATA_PATH=/path/to/data.xlsx（跨平台推荐）
#   2. 命令行参数：python main_cli.py --data_path /path/to/data.xlsx
#   3. 两者均未设置时，main() 会打印明确提示并以非零状态退出
ENV_DATA_PATH = os.environ.get("ASHARE_DATA_PATH")
DEFAULT_DATA_PATH: Optional[str] = ENV_DATA_PATH or None
# ==========================


def _get_latest_run_id(output_dir: str) -> int:
    latest = find_latest_run(output_dir)
    if latest is None:
        return 0
    run_id, _, _ = latest
    return int(run_id)


def _load_previous_best(output_dir: str) -> Optional[Dict[str, Any]]:
    latest = find_latest_run(output_dir)
    if latest is None:
        return None
    _, cfg_path, _ = latest
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        return blob.get("best_config")
    except Exception:
        return None


def _evaluate_config_on_ticker(
    data_path: str,
    best_config: Dict[str, Any],
    seed: int,
    initial_cash: float,
) -> Dict[str, Any]:
    """
    P3-18：在单个外部标的上评估已训练好的最优配置。

    评估方式：参数迁移（hyperparameter transfer），而非权重迁移。
        - 使用与主标的相同的 feature_config / model_config / trade_config
        - 在目标标的数据上重新从头训练模型（feature_config 和 model_config 不变）
        - 对全样本进行预测和回测，输出结果
        - 不做任何超参搜索，纯粹测试配置的跨标的泛化性

    通俗解释：
        这是验证"这套特征+模型架构是否对其他股票也适用"，而不是验证模型权重
        能否直接迁移（后者通常效果更差，因为不同股票的特征分布差异很大）。

    Args:
        data_path:   外部标的 Excel 路径
        best_config: 主标的搜索得到的最优配置
        seed:        随机种子
        initial_cash: 初始资金

    Returns:
        包含 ticker_path / final_equity / bnh_equity / alpha_pct / n_trades /
        error 等字段的字典；失败时 error 字段非 None。
    """
    result: Dict[str, Any] = {
        "ticker_path": data_path,
        "final_equity": None,
        "bnh_equity": None,
        "alpha_pct": None,
        "n_trades": None,
        "error": None,
    }

    try:
        df_other, _ = load_stock_excel(data_path)
        if len(df_other) < 100:
            result["error"] = f"数据行数不足（{len(df_other)} < 100），跳过"
            return result

        # 使用相同的 best_config 训练并预测
        dpoint_other, _ = train_final_model_and_dpoint(df_other, best_config, seed=seed)

        tc = best_config["trade_config"]
        bt = backtest_from_dpoint(
            df=df_other,
            dpoint=dpoint_other,
            initial_cash=initial_cash,
            buy_threshold=float(tc["buy_threshold"]),
            sell_threshold=float(tc["sell_threshold"]),
            confirm_days=int(tc["confirm_days"]),
            min_hold_days=int(tc["min_hold_days"]),
            max_hold_days=int(tc.get("max_hold_days", 20)),
            take_profit=tc.get("take_profit", None),
            stop_loss=tc.get("stop_loss", None),
        )

        final_equity = (
            float(bt.equity_curve["total_equity"].iloc[-1])
            if not bt.equity_curve.empty else initial_cash
        )
        bnh_equity = (
            float(bt.equity_curve["bnh_equity"].iloc[-1])
            if not bt.equity_curve.empty and "bnh_equity" in bt.equity_curve.columns
            else initial_cash
        )
        n_trades = len(bt.trades) if bt.trades is not None else 0

        result["final_equity"] = round(final_equity, 2)
        result["bnh_equity"]   = round(bnh_equity, 2)
        result["alpha_pct"]    = round((final_equity - bnh_equity) / initial_cash * 100, 2)
        result["n_trades"]     = n_trades

    except Exception as exc:
        result["error"] = str(exc)

    return result


def _resolve_n_folds(n_folds_arg: int, df_clean: "pd.DataFrame") -> int:
    """
    P3-20：根据 --n_folds 参数和数据量自动决定实际折数。
    """
    if n_folds_arg == -1:
        # 修复：乘以 0.88 收缩系数，补偿特征工程滚动窗口丢弃的头部 NaN 行
        # 实测数据：1210 原始行 → 1090 有效行，丢弃约 10%
        # min_rows 从 80 降至 60，给边界情况留出余量（60行≈3个月，统计上仍可接受）
        effective_samples = int(len(df_clean) * 0.88)
        n = recommend_n_folds(
            n_samples=effective_samples,
            target_trades_per_fold=4,
            assumed_trade_freq=1.0 / 15.0,
            min_rows=60,              # 原来是 80，改为 60
            min_folds=2,
            max_folds=8,
        )
        print(
            f"[INFO] P3-20: n_folds 自动推算 = {n} "
            f"（原始样本={len(df_clean)}，收缩后={effective_samples}，"
            f"assumed_trade_freq=1/15，target=4 trades/fold）"
        )
    else:
        n = max(2, n_folds_arg)
        if n != n_folds_arg:
            print(f"[WARN] P3-20: --n_folds={n_folds_arg} < 2，已自动提升为 {n}。")
        else:
            print(f"[INFO] P3-20: 使用用户指定 n_folds={n}。")
    return n


def _resolve_n_jobs(n_jobs_arg: int) -> int:
    """
    P1-6：根据 CUDA 可用性自动决定并行度。

    规则：
        --n_jobs -1（默认，自动）：
            - CUDA 可用 → 强制 1（joblib fork 与 CUDA 不兼容）
            - CUDA 不可用 → 4（保守默认，sklearn 模型可安全并行）
        --n_jobs N（N != -1，用户显式指定）：
            - 直接使用 N，但若 CUDA 可用且 N > 1，打印警告提示潜在风险
    """
    # 轻量检测 CUDA，不导入全部 torch（避免在 conda 激活检查前触发 CUDA 初始化）
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        pass

    if n_jobs_arg == -1:
        # 自动模式
        if cuda_available:
            resolved = 1
            print(
                "[INFO] P1-6: CUDA 可用，n_jobs 自动设置为 1，"
                "避免 joblib fork 进程与 CUDA 上下文冲突。"
            )
        else:
            resolved = 4
            print("[INFO] P1-6: 未检测到 CUDA，n_jobs 自动设置为 4（CPU 并行）。")
    else:
        resolved = n_jobs_arg
        if cuda_available and n_jobs_arg > 1:
            print(
                f"[WARN] P1-6: CUDA 可用但 --n_jobs={n_jobs_arg} > 1。"
                f"若搜索空间包含 DL 模型（MLP/LSTM/GRU/CNN/Transformer），"
                f"joblib fork 可能导致 CUDA 上下文冲突。"
                f"建议使用 --n_jobs 1 或让系统自动决定（--n_jobs -1）。"
            )

    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="A-share single-stock ML Dpoint trader (2.0).")
    parser.add_argument("--mode", choices=["first", "continue"], default="first", help="Run mode.")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to Excel data file.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for runs.")
    parser.add_argument("--runs", type=int, default=100, help="Random search iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--initial_cash", type=float, default=100000.0, help="Initial cash.")
    # P1-6：新增 --n_jobs 参数，-1 表示自动检测
    parser.add_argument(
        "--n_jobs", type=int, default=-1,
        help=(
            "并行进程数。-1（默认）= 自动检测：CUDA 可用时强制 1，否则 4。"
            "显式传入正整数可覆盖自动值（CUDA 环境下建议保持 1）。"
        ),
    )
    # P3-20：新增 --n_folds 参数，-1 表示根据数据量自动推算
    parser.add_argument(
        "--n_folds", type=int, default=-1,
        help=(
            "Walk-forward 折数。-1（默认）= 根据数据量自动推算（基于 recommend_n_folds）。"
            "显式传入正整数（如 4）可固定折数，最小值为 2。"
        ),
    )
    # P3-18：新增 --eval_tickers 参数，逗号分隔多个 Excel 路径
    parser.add_argument(
        "--eval_tickers", type=str, default="",
        help=(
            "P3-18 泛化性评估：逗号分隔的额外标的 Excel 文件路径列表，"
            "将使用主标的搜索出的最优配置（参数迁移）在每个标的上重新训练并回测。"
            "示例：--eval_tickers 600036_5Y.xlsx,601318_5Y.xlsx"
        ),
    )
    args = parser.parse_args()

    # P2 修复：data_path 为 None 时（环境变量和命令行均未指定），提前给出明确提示。
    # 避免让底层 load_stock_excel 抛出难以理解的 FileNotFoundError 或 TypeError。
    if not args.data_path:
        print(
            "[ERROR] 未指定数据文件路径。请通过以下任意方式提供：\n"
            "  命令行参数: --data_path /path/to/data.xlsx\n"
            "  环境变量:   export ASHARE_DATA_PATH=/path/to/data.xlsx  （Linux/macOS）\n"
            "              set ASHARE_DATA_PATH=C:\\path\\to\\data.xlsx  （Windows）"
        )
        sys.exit(1)

    if not os.path.exists(args.data_path):
        print(f"[ERROR] Data path not found: {args.data_path}")
        print("Please set ASHARE_DATA_PATH environment variable or check the path.")
        sys.exit(1)

    df_clean, data_report = load_stock_excel(args.data_path)
    print(f"[INFO] Loaded clean data rows: {len(df_clean)}")
    if len(df_clean) == 0:
        raise ValueError("DataLoader produced 0 rows after cleaning. Check Excel columns and date parsing.")

    base_best_config = None
    if args.mode == "continue":
        base_best_config = _load_previous_best(args.output_dir)
        if base_best_config is None:
            print("[WARN] Continue mode but no previous config found. Falling back to first mode behavior.")
        else:
            print("[INFO] Loaded previous best_config as incumbent.")

    # P2-5 修复：trade_params 仅传 initial_cash
    # ─────────────────────────────────────────────────────────────────────
    # 原版把 buy_threshold / sell_threshold / confirm_days / min_hold_days
    # 也塞进 trade_params 传给 random_search_train，但搜索引擎从不读取这些字段——
    # 每个候选的阈值都由 _sample_explore / _sample_exploit 在采样空间里自主生成。
    # 唯一真正被消费的字段是 initial_cash（用于构造 trade_config["initial_cash"]）。
    #
    # 最终 backtest 也不从这里取阈值，而是从 best_config["trade_config"] 取，
    # 所以那四个字段"传了但没用"，造成"参数到底来自哪里"的混乱。
    #
    # 修复：只保留 initial_cash；其余阈值参数均由搜索引擎在采样空间内自由决定，
    #       最终参数以 best_config["trade_config"] 为唯一权威来源。
    # ─────────────────────────────────────────────────────────────────────
    search_initial_params: Dict[str, Any] = {
        "initial_cash": float(args.initial_cash),   # 唯一被搜索引擎使用的字段
    }

    latest_run_id = _get_latest_run_id(args.output_dir) if args.mode == "continue" else 0
    seed_effective = int(args.seed) + int(latest_run_id)

    # P1-6：根据 CUDA 可用性决定实际并行度
    n_jobs_effective = _resolve_n_jobs(args.n_jobs)
    print(f"[INFO] P1-6: 实际使用 n_jobs={n_jobs_effective}")

    # P3-20：根据数据量自适应推算折数
    n_folds_effective = _resolve_n_folds(args.n_folds, df_clean)

    train_res = random_search_train(
        df_clean=df_clean,
        runs=int(args.runs),
        seed=int(seed_effective),
        base_best_config=base_best_config,
        output_dir=str(args.output_dir),
        epsilon=0.01,
        exploit_ratio=0.7,
        top_k=10,
        trade_params=search_initial_params,   # P2-5：仅传 initial_cash，阈值由搜索自主决定
        max_features=80,
        n_jobs=n_jobs_effective,              # P1-6：使用自动决定的安全并行度
        n_folds=n_folds_effective,            # P3-20：自适应折数
    )

    best_config = train_res.best_config
    print(f"[INFO] Best validation metric (geom mean ratio): {train_res.best_val_metric:.6f}")
    print(f"[INFO] Best validation equity proxy (mean): {train_res.best_val_final_equity_proxy:.2f}")

    dpoint, artifacts = train_final_model_and_dpoint(df_clean, best_config, seed=int(args.seed))

    tc = best_config["trade_config"]
    bt = backtest_from_dpoint(
        df=df_clean,
        dpoint=dpoint,
        initial_cash=float(tc["initial_cash"]),
        buy_threshold=float(tc["buy_threshold"]),
        sell_threshold=float(tc["sell_threshold"]),
        confirm_days=int(tc["confirm_days"]),
        min_hold_days=int(tc["min_hold_days"]),
        max_hold_days=int(tc.get("max_hold_days", 20)),
        take_profit=tc.get("take_profit", None),
        stop_loss=tc.get("stop_loss", None),
        # P04 成本参数使用默认值（A 股标准：买入 0.03%，卖出 0.13%）
    )

    final_equity = (
        float(bt.equity_curve["total_equity"].iloc[-1])
        if not bt.equity_curve.empty
        else float(args.initial_cash)
    )
    print(f"[INFO] Full-sample final equity: {final_equity:.2f}")
    print(f"[INFO] Trades executed: {len(bt.trades)}")

    log_notes: List[str] = []
    log_notes.append("=== DataLoader Report ===")
    log_notes.append(f"Data path: {args.data_path}")
    log_notes.append(f"Sheet used: {data_report.sheet_used}")
    log_notes.append(f"Rows raw: {data_report.rows_raw}")
    log_notes.append(f"Rows after dropna core: {data_report.rows_after_dropna_core}")
    log_notes.append(f"Rows after filters: {data_report.rows_after_filters}")
    log_notes.append(f"Duplicate dates: {data_report.duplicate_dates}")
    log_notes.append(f"Bad OHLC rows dropped: {data_report.bad_ohlc_rows}")
    log_notes.extend(data_report.notes)

    log_notes.append("")
    log_notes.append("=== Training Summary / Improvement Confirmation ===")
    log_notes.append(f"Mode: {args.mode}")
    log_notes.append(f"Runs (search iterations): {args.runs}")
    log_notes.append(f"Base seed (CLI): {args.seed}")
    log_notes.append(f"Effective seed (base + latest_run_id): {seed_effective} (latest_run_id={latest_run_id})")
    log_notes.append(f"n_jobs (CLI arg): {args.n_jobs} → n_jobs (effective): {n_jobs_effective}")  # P1-6
    log_notes.append(f"search_initial_cash: {search_initial_params['initial_cash']:.2f}")  # P2-5
    log_notes.append(f"Best validation metric (geom-mean ratio): {train_res.best_val_metric:.6f}")
    log_notes.append(f"Best validation equity proxy (mean): {train_res.best_val_final_equity_proxy:.2f}")
    log_notes.append(f"Global best metric prev: {train_res.global_best_metric_prev:.6f}")
    log_notes.append(f"Candidate best metric this run: {train_res.candidate_best_metric:.6f}")
    log_notes.append(f"Global best metric new: {train_res.global_best_metric_new:.6f}")
    log_notes.append(f"Epsilon (min improvement): {train_res.epsilon:.6f}")
    log_notes.append(f"Global best updated: {train_res.global_best_updated}")
    log_notes.append(f"Not-updated reason: {train_res.not_updated_reason}")
    log_notes.append(f"Best-so-far file: {train_res.best_so_far_path}")
    log_notes.append(f"Best pool file: {train_res.best_pool_path}")
    log_notes.extend(train_res.training_notes)

    log_notes.append("")
    log_notes.append("=== Backtest Notes ===")
    log_notes.append(
        "⚠️  WARNING: This equity curve is an IN-SAMPLE result (model trained & predicted on same data). "
        "It overstates real performance. See SearchLog for out-of-sample walk-forward metrics."
    )
    log_notes.append(
        "⚠️  警告：此净值曲线为全样本内拟合结果，模型在训练集上预测，存在前向偏差，数值偏乐观。"
        "真实样本外表现请查看 SearchLog sheet 中的 walk-forward 验证指标。"
    )
    log_notes.append(f"Full-sample final equity: {final_equity:.2f}")
    log_notes.append(f"Trades executed: {len(bt.trades)}")
    log_notes.extend(bt.notes)

    # P3-17：Buy & Hold 对比摘要（bt.notes 中已包含 alpha 行，此处汇总到 log）
    if not bt.equity_curve.empty and "bnh_equity" in bt.equity_curve.columns:
        bnh_final = float(bt.equity_curve["bnh_equity"].iloc[-1])
        strat_cum  = (final_equity - float(args.initial_cash)) / float(args.initial_cash) * 100.0
        bnh_cum    = (bnh_final    - float(args.initial_cash)) / float(args.initial_cash) * 100.0
        log_notes.append("")
        log_notes.append("=== P3-17 Buy & Hold 基准对比（In-sample，仅供参考）===")
        log_notes.append(f"Strategy cumulative return : {strat_cum:+.2f}%")
        log_notes.append(f"Buy & Hold cumulative return: {bnh_cum:+.2f}%")
        log_notes.append(f"Alpha (strategy - B&H)      : {strat_cum - bnh_cum:+.2f}%")
        log_notes.append("⚠️  注意：上述数字为全样本内拟合结果，非真实样本外 alpha。")

    # P3-18：多标的泛化性评估
    eval_paths = [p.strip() for p in str(args.eval_tickers).split(",") if p.strip()]
    if eval_paths:
        log_notes.append("")
        log_notes.append("=== P3-18 多标的泛化性评估（参数迁移，超参不变，数据独立）===")
        log_notes.append(
            "评估方式：使用主标的最优配置（feature_config + model_config + trade_config），"
            "在每个目标标的上从头训练模型，全样本回测（In-sample 结果，仅供参考）。"
        )
        for ep in eval_paths:
            if not os.path.exists(ep):
                log_notes.append(f"  [{ep}] 文件不存在，跳过")
                continue
            print(f"[INFO] P3-18: 评估标的 {ep} ...")
            eval_r = _evaluate_config_on_ticker(
                data_path=ep,
                best_config=best_config,
                seed=int(args.seed),
                initial_cash=float(args.initial_cash),
            )
            if eval_r["error"]:
                log_notes.append(f"  [{ep}] 评估失败: {eval_r['error']}")
            else:
                log_notes.append(
                    f"  [{ep}] final_equity={eval_r['final_equity']:.2f}  "
                    f"bnh_equity={eval_r['bnh_equity']:.2f}  "
                    f"alpha={eval_r['alpha_pct']:+.2f}%  "
                    f"n_trades={eval_r['n_trades']}"
                )

    excel_path, config_path, run_id = save_run_outputs(
        output_dir=str(args.output_dir),
        df_clean=df_clean,
        log_notes=log_notes,
        trades=bt.trades,
        equity_curve=bt.equity_curve,
        config=best_config,
        feature_meta=artifacts["feature_meta"],
        search_log=train_res.search_log,
        model_params=artifacts.get("model_params"),
    )

    print(f"[DONE] Saved run {run_id:03d}")
    print(f"  - Excel : {os.path.abspath(excel_path)}")
    print(f"  - Config: {os.path.abspath(config_path)}")


if __name__ == "__main__":
    main()
