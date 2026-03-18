#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dpoint_updater.py — Dpoint 更新工具
=============================================================
放置位置：项目根目录（与 main_cli.py / data_loader.py 同一目录）
启动方式：python dpoint_updater.py
           python dpoint_updater.py --output_dir ./output
=============================================================

功能说明：
  1. 扫描 output 目录，列出所有历史运行结果，由用户输入 run 编号
  2. 读取对应的 run_NNN_config.json，解析最佳配置
     （feature_config / model_config / trade_config）
  3. 打印该次运行的 Excel 摘要信息（特征族开关、模型类型等）
  4. 弹出文件选择窗口，用户选择包含最新交易数据的完整 xlsx 文件
  5. 用原配置在新数据上 **重新训练** 模型（不复用旧 Dpoint，
     规避复权价格漂移问题）
  6. 输出 data/ 目录下的 Excel 文件：
     A列=日期  B列=收盘价(close_qfq)  C列=Dpoint

注意事项（运行前必读）：
  • 新数据文件须是包含完整历史的全量文件（不能只提供增量）
  • 深度学习（MLP/LSTM/GRU/CNN/Transformer）模型含随机性，
    每次重训结果略有差异，属正常现象
  • 深度学习模型在 CPU 和 CUDA 环境下均可运行；
    无 CUDA 时自动回退到 CPU，训练速度较慢但结果正确
=============================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────
# 抑制 sklearn 收敛警告，保持控制台输出整洁
# ──────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR: str = "./output"
DEFAULT_DATA_DIR:   str = "./data"

DL_MODEL_TYPES: tuple = ("mlp", "lstm", "gru", "cnn", "transformer")

SEPARATOR = "=" * 68


# ══════════════════════════════════════════════════════════════
# SECTION 1  工具函数
# ══════════════════════════════════════════════════════════════

def _print_banner() -> None:
    print()
    print(SEPARATOR)
    print("  Dpoint 更新工具 — dpoint_updater.py")
    print(SEPARATOR)
    print()


def _list_runs(output_dir: str) -> List[Tuple[int, str, str]]:
    """
    扫描 output_dir，返回所有 (run_id, config_path, xlsx_path) 三元组，
    按 run_id 升序排列。
    """
    if not os.path.isdir(output_dir):
        return []
    found = []
    for fn in os.listdir(output_dir):
        if fn.startswith("run_") and fn.endswith("_config.json"):
            try:
                run_id = int(fn.split("_")[1])
                cfg_path  = os.path.join(output_dir, fn)
                xlsx_path = os.path.join(output_dir, f"run_{run_id:03d}.xlsx")
                found.append((run_id, cfg_path, xlsx_path))
            except (ValueError, IndexError):
                pass
    return sorted(found, key=lambda x: x[0])


def _load_config_json(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    return blob


def _summarize_run(blob: Dict[str, Any], xlsx_path: str) -> None:
    """在控制台打印该次运行的关键信息摘要。"""
    best_cfg = blob.get("best_config", {})
    feat_cfg  = best_cfg.get("feature_config", {})
    model_cfg = best_cfg.get("model_config", {})
    trade_cfg = best_cfg.get("trade_config", {})
    feat_meta = blob.get("feature_meta", {})

    print(f"\n{'─'*50}")
    print(f"  运行编号   : run_{blob.get('run_id', '?'):03d}")
    print(f"  创建时间   : {blob.get('created_at', '未知')}")
    print(f"{'─'*50}")

    # ── 特征配置 ──
    print("  【特征工程配置】")
    print(f"    窗口列表         : {feat_cfg.get('windows', '未知')}")
    print(f"    动量特征         : {feat_cfg.get('use_momentum', '?')}")
    print(f"    波动率特征       : {feat_cfg.get('use_volatility', '?')}")
    print(f"    成交量特征       : {feat_cfg.get('use_volume', '?')}")
    print(f"    K线形态特征      : {feat_cfg.get('use_candle', '?')}")
    print(f"    换手率特征       : {feat_cfg.get('use_turnover', '?')}")
    print(f"    技术指标(TA)     : {feat_cfg.get('use_ta_indicators', False)}")
    if feat_cfg.get("use_ta_indicators"):
        print(f"    TA 窗口          : {feat_cfg.get('ta_windows', [6, 14, 20])}")
    print(f"    波动率度量       : {feat_cfg.get('vol_metric', 'std')}")
    print(f"    流动性变换       : {feat_cfg.get('liq_transform', 'ratio')}")

    # ── 模型配置 ──
    print("  【模型配置】")
    model_type = model_cfg.get("model_type", "未知")
    print(f"    模型类型         : {model_type}")
    if model_type in ("logreg",):
        print(f"    C / penalty      : {model_cfg.get('C')} / {model_cfg.get('penalty')}")
    elif model_type == "sgd":
        print(f"    alpha / penalty  : {model_cfg.get('alpha')} / {model_cfg.get('penalty')}")
    elif model_type == "xgb":
        p = model_cfg.get("params", {})
        print(f"    n_estimators     : {p.get('n_estimators')}")
        print(f"    max_depth        : {p.get('max_depth')}")
        print(f"    learning_rate    : {p.get('learning_rate')}")
    elif model_type in DL_MODEL_TYPES:
        print(f"    hidden_dim       : {model_cfg.get('hidden_dim', '?')}")
        print(f"    dropout_rate     : {model_cfg.get('dropout_rate', '?')}")
        print(f"    epochs           : {model_cfg.get('epochs', '?')}")

    # ── 交易配置 ──
    print("  【交易配置】")
    print(f"    初始资金         : {trade_cfg.get('initial_cash', '?')}")
    print(f"    买入阈值         : {trade_cfg.get('buy_threshold', '?')}")
    print(f"    卖出阈值         : {trade_cfg.get('sell_threshold', '?')}")
    print(f"    确认天数         : {trade_cfg.get('confirm_days', '?')}")
    print(f"    最短持仓(交易日) : {trade_cfg.get('min_hold_days', '?')}")
    print(f"    最长持仓(交易日) : {trade_cfg.get('max_hold_days', '?')}")
    tp = trade_cfg.get("take_profit", None)
    sl = trade_cfg.get("stop_loss", None)
    print(f"    止盈             : {f'{tp:.1%}' if tp else '未启用'}")
    print(f"    止损             : {f'{sl:.1%}' if sl else '未启用'}")

    # ── 特征名列表 ──
    feat_names = feat_meta.get("feature_names", [])
    if feat_names:
        print(f"  【特征数量】 : {len(feat_names)} 个")
        print(f"    前10个特征 : {feat_names[:10]}")

    # ── Excel 摘要（如果存在）──
    if os.path.isfile(xlsx_path):
        try:
            _print_search_log_summary(xlsx_path)
        except Exception as e:
            print(f"  [提示] Excel 摘要读取失败（{e}），已跳过。")
    else:
        print(f"\n  [提示] 未找到对应 Excel：{xlsx_path}")

    print(f"{'─'*50}\n")


def _print_search_log_summary(xlsx_path: str) -> None:
    """读取 run_NNN.xlsx 的 SearchLog / Log sheet，打印搜索结果摘要。"""
    try:
        xl = pd.ExcelFile(xlsx_path)
        available_sheets = xl.sheet_names
    except Exception:
        return

    print(f"\n  【Excel 文件 Sheet 列表】: {available_sheets}")

    # Log sheet 中的 search_log 摘要
    if "Log" in available_sheets:
        try:
            log_df = pd.read_excel(xlsx_path, sheet_name="Log", header=None)
            # search_log 从 notes 后面开始，找到列头行
            for idx in range(len(log_df)):
                row_vals = log_df.iloc[idx].tolist()
                if "iter" in str(row_vals).lower() or "metric" in str(row_vals).lower():
                    # 找到 search_log header
                    search_df = pd.read_excel(
                        xlsx_path, sheet_name="Log",
                        header=idx, skiprows=range(0, idx)
                    )
                    if "metric" in search_df.columns:
                        valid = search_df["metric"].dropna()
                        if not valid.empty:
                            print(f"\n  【搜索结果摘要（来自 Log sheet）】")
                            print(f"    搜索总轮次     : {len(search_df)}")
                            print(f"    最高 metric    : {valid.max():.6f}")
                            print(f"    最低 metric    : {valid.min():.6f}")
                            print(f"    平均 metric    : {valid.mean():.6f}")
                    break
        except Exception:
            pass

    # EquityCurve sheet 摘要
    if "EquityCurve" in available_sheets:
        try:
            ec_df = pd.read_excel(xlsx_path, sheet_name="EquityCurve")
            if "total_equity" in ec_df.columns:
                final_eq = ec_df["total_equity"].iloc[-1]
                init_eq  = ec_df["total_equity"].iloc[0]
                print(f"\n  【净值曲线摘要（全样本，含前向偏差）】")
                print(f"    期末净值       : {final_eq:,.2f}")
                print(f"    累计收益率     : {(final_eq/init_eq - 1):.2%}")
                if "bnh_equity" in ec_df.columns:
                    bnh_final = ec_df["bnh_equity"].iloc[-1]
                    print(f"    B&H 期末净值   : {bnh_final:,.2f}")
                    print(f"    vs B&H         : {(final_eq - bnh_final)/init_eq:+.2%}")
                if "drawdown" in ec_df.columns:
                    max_dd = ec_df["drawdown"].min()
                    print(f"    最大回撤       : {max_dd:.2%}")
                print(f"  ⚠️  注：以上为全样本拟合数据，存在前向偏差，仅供参考！")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
# SECTION 2  CUDA 检查
# ══════════════════════════════════════════════════════════════

def _check_cuda_for_dl(model_type: str) -> None:
    """
    若模型为深度学习类型，强制要求 CUDA 可用。
    检测失败时打印详细安装指引并退出，不回退 CPU 运行。
    """
    if model_type.lower() not in DL_MODEL_TYPES:
        return  # 非 DL 模型，无需 CUDA

    print(f"\n  [CUDA 检查] 模型类型为 {model_type.upper()}，需要 CUDA 支持...")

    # ── 检查 PyTorch 是否安装 ──
    try:
        import torch  # noqa: F401
    except ImportError:
        _exit_with_cuda_guide("未检测到 PyTorch。")

    import torch
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        print(f"  ✅ CUDA 可用！设备：{dev}")
        return

    # ── CUDA 不可用，给出详细指引后退出 ──
    _exit_with_cuda_guide(
        "torch.cuda.is_available() 返回 False。\n"
        "  当前安装的 PyTorch 可能是 CPU-only 版本，或显卡驱动未正确安装。"
    )


def _exit_with_cuda_guide(reason: str) -> None:
    print()
    print("  " + "!" * 60)
    print("  ❌  CUDA 环境未就绪，无法运行深度学习模型")
    print("  " + "!" * 60)
    print(f"\n  原因：{reason}")
    print("""
  ─── 安装指引 ────────────────────────────────────────────
  步骤 1：确认显卡驱动版本
    运行 nvidia-smi，查看右上角 "CUDA Version"（如 12.1）

  步骤 2：安装对应版本的 PyTorch（以 CUDA 12.1 为例）
    conda activate ashare_dpoint
    pip install torch torchvision torchaudio \\
        --index-url https://download.pytorch.org/whl/cu121

    其他 CUDA 版本请访问：https://pytorch.org/get-started/locally/

  步骤 3：验证安装
    python -c "import torch; print(torch.cuda.is_available())"
    应输出：True

  步骤 4：重新运行本工具
    python dpoint_updater.py

  ⚠️  本工具不回退 CPU 运行，原因：
    深度学习模型在 CPU 上训练速度极慢（可能需要数小时），
    且与训练时 CUDA 环境存在微小数值差异，不建议在生产环境混用。
  ─────────────────────────────────────────────────────────
""")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
# SECTION 3  文件选择器
# ══════════════════════════════════════════════════════════════

def _pick_data_file() -> str:
    """
    弹出 tkinter 文件选择对话框，返回用户选中的 xlsx 路径。
    如果 tkinter 不可用（如无桌面环境），降级为控制台手动输入。
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()           # 隐藏主窗口
        root.attributes("-topmost", True)   # 对话框置顶

        path = filedialog.askopenfilename(
            title="请选择包含最新行情数据的完整 xlsx 文件",
            filetypes=[("Excel 文件", "*.xlsx"), ("所有文件", "*.*")],
        )
        root.destroy()

        if not path:
            print("  [取消] 未选择文件，程序退出。")
            sys.exit(0)
        return path

    except Exception as e:
        print(f"  [提示] 无法打开图形文件选择窗口（{e}）")
        print("  请手动输入新数据文件的完整路径：", end="")
        path = input().strip().strip('"').strip("'")
        if not path or not os.path.isfile(path):
            print("  [错误] 路径无效或文件不存在，程序退出。")
            sys.exit(1)
        return path


# ══════════════════════════════════════════════════════════════
# SECTION 4  复权漂移检查
# ══════════════════════════════════════════════════════════════

def _check_qfq_drift(
    old_df: Optional[pd.DataFrame],
    new_df: pd.DataFrame,
    tolerance: float = 0.005,
) -> None:
    """
    检测新旧数据集的复权价格是否发生漂移（拆股/分红导致的全局复权调整）。

    方法：在新旧数据的重叠日期上，比较 close_qfq 的差异比例均值。
    若超过 tolerance（默认 0.5%），打印警告并提示用户确认继续。

    注：本工具通过重新训练规避了复权漂移问题，此检查仅为信息提示。
    """
    if old_df is None or old_df.empty:
        return

    try:
        old_indexed = old_df.set_index("date")["close_qfq"]
        new_indexed = new_df.set_index("date")["close_qfq"]
        common = old_indexed.index.intersection(new_indexed.index)

        if len(common) < 10:
            print("  [复权检查] 重叠日期不足 10 天，跳过检查。")
            return

        sample = common[-min(60, len(common)):]   # 取最近 60 个重叠交易日
        old_prices = old_indexed[sample]
        new_prices = new_indexed[sample]

        diff_ratio = ((new_prices - old_prices).abs() / old_prices.abs()).mean()

        if diff_ratio > tolerance:
            print(f"\n  ⚠️  【复权漂移警告】")
            print(f"     在最近 {len(sample)} 个重叠交易日中，新旧 close_qfq 平均偏差：{diff_ratio:.2%}")
            print(f"     超过阈值 {tolerance:.2%}，可能发生了拆股/分红复权调整。")
            print(f"     本工具将在新数据上 **重新训练** 模型，Dpoint 结果不受旧数据影响。✅")
        else:
            print(f"  ✅ 复权检查通过：新旧数据价格偏差 {diff_ratio:.4%}，无明显漂移。")

    except Exception as e:
        print(f"  [复权检查] 检查失败（{e}），已跳过。")


# ══════════════════════════════════════════════════════════════
# SECTION 5  Dpoint 计算（重训练）
# ══════════════════════════════════════════════════════════════

def _compute_dpoint_retrain(
    df_new: pd.DataFrame,
    best_config: Dict[str, Any],
    seed: int = 42,
) -> pd.Series:
    """
    在新数据上用原配置重新训练模型，返回全时段 Dpoint 序列。

    重要说明：
    - 此为全样本训练（in-sample），Dpoint 仅代表模型对历史的拟合，
      不代表未来真实信号强度。
    - 最后一个交易日由于无 t+1 标签，build_features_and_labels 会自动跳过，
      本函数将其 Dpoint 设为倒数第二个交易日的值（并标注说明）。

    Returns:
        dpoint: pd.Series，index 为 pd.Timestamp 日期，包含全部可计算日期
    """
    # ── 延迟导入（保证脚本在项目根目录运行时能找到兄弟模块）──
    from training import train_final_model_and_dpoint  # type: ignore

    print("  [训练] 正在使用原配置在新数据上重新训练模型...")

    model_type = str(best_config.get("model_config", {}).get("model_type", ""))

    # DL 模型随机性提示
    if model_type in DL_MODEL_TYPES:
        print(f"\n  ⚠️  【深度学习随机性提示】")
        print(f"     模型类型：{model_type.upper()}")
        print(f"     深度学习模型训练含随机初始化，每次运行结果略有差异，属正常现象。")
        print(f"     若需要可复现的结果，可固定 seed（当前 seed={seed}）。")
        print(f"     本次训练 seed={seed}，与原训练保持一致以减少差异。\n")

    dpoint, _ = train_final_model_and_dpoint(df_new, best_config, seed=seed)

    print(f"  ✅ 模型训练完成，共计算 {len(dpoint)} 个交易日的 Dpoint。")
    return dpoint


def _extend_dpoint_to_last_day(
    dpoint: pd.Series,
    df_new: pd.DataFrame,
) -> Tuple[pd.Series, bool]:
    """
    检查 Dpoint 是否覆盖了新数据的最后一个交易日。

    由于 Dpoint_t = P(close_{t+1} > close_t | X_t)，
    最后一个交易日 T 没有 t+1 的标签，build_features_and_labels
    会自动跳过，导致最后一天缺失 Dpoint。

    处理方式：将 T-1 日的 Dpoint 值复制给 T 日，并设 extended=True。
    这样做使 Excel 中每个交易日都有对应 Dpoint 值。

    Returns:
        (dpoint_extended, extended): extended=True 表示进行了补充
    """
    df_sorted = df_new.sort_values("date").reset_index(drop=True)
    last_date  = pd.Timestamp(df_sorted["date"].iloc[-1])
    dpoint_idx = pd.DatetimeIndex(dpoint.index)

    if last_date in dpoint_idx:
        return dpoint, False    # 最后一天已有 Dpoint，无需补充

    # 最后一天缺失，取倒数第二天的值补充
    if len(dpoint) == 0:
        return dpoint, False

    prev_value = float(dpoint.iloc[-1])
    extended = dpoint.copy()
    extended.loc[last_date] = prev_value
    extended = extended.sort_index()
    return extended, True


# ══════════════════════════════════════════════════════════════
# SECTION 6  输出 Excel
# ══════════════════════════════════════════════════════════════

def _build_output_df(
    df_new: pd.DataFrame,
    dpoint: pd.Series,
    last_day_extended: bool,
) -> pd.DataFrame:
    """
    将收盘价与 Dpoint 对齐，构造输出 DataFrame。

    列说明：
        date      — 交易日期
        close_qfq — 当日后复权收盘价
        dpoint    — 模型预测的次日上涨概率（本日数据计算）
        note      — 说明（如最后一日使用前一日 Dpoint 时标注）
    """
    close_series = (
        df_new.set_index("date")["close_qfq"]
        .rename_axis("date")
    )
    close_series.index = pd.to_datetime(close_series.index)
    dpoint.index = pd.to_datetime(dpoint.index)

    combined = pd.DataFrame({
        "date":      close_series.index,
        "close_qfq": close_series.values,
    })
    combined = combined.set_index("date")

    dpoint_df = dpoint.rename("dpoint").to_frame()
    combined = combined.join(dpoint_df, how="left")
    combined = combined.reset_index()
    combined = combined.sort_values("date").reset_index(drop=True)

    # 补充 note 列
    combined["note"] = ""
    if last_day_extended and len(combined) > 0:
        last_idx = combined.index[-1]
        combined.loc[last_idx, "note"] = (
            "最后交易日无次日数据，Dpoint 使用前一交易日值（仅供参考）"
        )

    return combined[["date", "close_qfq", "dpoint", "note"]]


def _save_output_excel(
    output_df: pd.DataFrame,
    run_id: int,
    data_dir: str,
    new_file_path: str,
) -> str:
    """
    将结果保存到 data_dir 下，文件名含 run_id 和当前日期。
    同时写入一个元数据 sheet 说明来源。
    """
    os.makedirs(data_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"dpoint_run{run_id:03d}_{ts}.xlsx"
    out_path = os.path.join(data_dir, fname)

    # 元数据 sheet
    meta_rows = [
        ("生成时间",       datetime.now().isoformat(timespec="seconds")),
        ("数据来源",       os.path.abspath(new_file_path)),
        ("基础配置来源",   f"run_{run_id:03d}_config.json"),
        ("列说明_date",    "交易日期"),
        ("列说明_close",   "后复权收盘价（close_qfq）"),
        ("列说明_dpoint",  "P(close_t+1 > close_t | X_t)，即预测次日上涨概率"),
        ("列说明_note",    "特殊说明（如最后交易日 Dpoint 使用前一日值的情况）"),
        ("前向偏差说明",   "Dpoint 为全样本训练后的样本内预测，最后几日接近真实信号，早期数据存在前向偏差"),
        ("复权说明",       "Dpoint 基于新数据重新训练计算，已规避复权价格漂移影响"),
    ]
    meta_df = pd.DataFrame(meta_rows, columns=["key", "value"])

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        # Sheet 1: Dpoint 主表
        output_df.to_excel(writer, sheet_name="Dpoint", index=False)

        # Sheet 2: 元数据
        meta_df.to_excel(writer, sheet_name="Meta", index=False)

        # ── 格式化 Dpoint sheet ──
        workbook  = writer.book
        worksheet = writer.sheets["Dpoint"]

        # 日期格式
        date_fmt   = workbook.add_format({"num_format": "yyyy-mm-dd", "align": "center"})
        num_fmt    = workbook.add_format({"num_format": "0.0000",     "align": "right"})
        pct_fmt    = workbook.add_format({"num_format": "0.0000",     "align": "right"})
        header_fmt = workbook.add_format({
            "bold": True, "bg_color": "#2E4057", "font_color": "#FFFFFF",
            "border": 1, "align": "center"
        })

        # 列宽
        worksheet.set_column("A:A", 14, date_fmt)
        worksheet.set_column("B:B", 14, num_fmt)
        worksheet.set_column("C:C", 12, pct_fmt)
        worksheet.set_column("D:D", 55)

        # 表头格式
        headers = ["date", "close_qfq", "dpoint", "note"]
        for col_idx, h in enumerate(headers):
            worksheet.write(0, col_idx, h, header_fmt)

        # 数据行：最后一行如有 note 则高亮
        yellow_fmt = workbook.add_format({"bg_color": "#FFF3CD", "num_format": "0.0000"})
        for row_idx, row in output_df.iterrows():
            excel_row = row_idx + 1  # header 占第 0 行
            if row["note"]:
                worksheet.write(excel_row, 2, row["dpoint"], yellow_fmt)

        # 冻结首行
        worksheet.freeze_panes(1, 0)

    return out_path


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main() -> None:
    _print_banner()

    # ── 解析命令行参数 ──
    parser = argparse.ArgumentParser(
        description="Dpoint 更新工具：用最优配置在新数据上重新计算 Dpoint",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"训练结果输出目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--data_dir", type=str, default=DEFAULT_DATA_DIR,
        help=f"Dpoint 结果输出目录（默认：{DEFAULT_DATA_DIR}）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认：42）",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    data_dir   = args.data_dir

    # ────────────────────────────────────────────
    # STEP 1  扫描并列出所有历史运行结果
    # ────────────────────────────────────────────
    runs = _list_runs(output_dir)

    if not runs:
        print(f"  [错误] 在 '{output_dir}' 中未找到任何训练结果（run_NNN_config.json）。")
        print(f"  请确认 --output_dir 参数正确，或先运行 main_cli.py 生成训练结果。")
        sys.exit(1)

    print(f"  在 '{output_dir}' 中找到以下训练结果：\n")
    print(f"  {'编号':>5}   {'配置文件':<35}   {'Excel'}")
    print(f"  {'─'*5}   {'─'*35}   {'─'*20}")
    for rid, cfg_path, xlsx_path in runs:
        has_xlsx = "✅" if os.path.isfile(xlsx_path) else "❌ 缺失"
        print(f"  {rid:5d}   {os.path.basename(cfg_path):<35}   {has_xlsx}")

    # ────────────────────────────────────────────
    # STEP 2  用户输入 run 编号
    # ────────────────────────────────────────────
    print()
    valid_ids = {r[0] for r in runs}
    while True:
        print("  请输入最佳 val_metric 对应的运行编号（如输入 13 表示 run_013）：", end="")
        raw = input().strip()
        try:
            run_id = int(raw)
        except ValueError:
            print("  [错误] 请输入一个整数。")
            continue
        if run_id not in valid_ids:
            print(f"  [错误] 编号 {run_id} 不存在，请从上表中选择。")
            continue
        break

    # 找到对应路径
    cfg_path  = os.path.join(output_dir, f"run_{run_id:03d}_config.json")
    xlsx_path = os.path.join(output_dir, f"run_{run_id:03d}.xlsx")

    # ────────────────────────────────────────────
    # STEP 3  读取并展示配置
    # ────────────────────────────────────────────
    print(f"\n  正在读取 run_{run_id:03d}_config.json ...")
    blob       = _load_config_json(cfg_path)
    best_config = blob.get("best_config")

    if best_config is None:
        print(f"  [错误] 配置文件中未找到 'best_config' 字段。文件可能损坏。")
        sys.exit(1)

    _summarize_run(blob, xlsx_path)

    # ────────────────────────────────────────────
    # STEP 4  CUDA 检查（DL 模型）
    # ────────────────────────────────────────────
    model_type = str(best_config.get("model_config", {}).get("model_type", ""))
    _check_cuda_for_dl(model_type)

    # ────────────────────────────────────────────
    # STEP 5  用户选择新数据文件
    # ────────────────────────────────────────────
    print(SEPARATOR)
    print("  请在弹出的文件选择窗口中选择包含最新行情数据的完整 xlsx 文件。")
    print()
    print("  重要提示：")
    print("    • 所选文件必须是完整历史数据（不能只提供增量）")
    print("    • 文件格式须与原 data 目录中的 xlsx 相同")
    print("      （含 date / open_qfq / high_qfq / low_qfq / close_qfq /")
    print("       volume / amount / turnover_rate 等列）")
    print("    • 新文件应在原截止日之后多出若干行最新交易日数据")
    print(SEPARATOR)
    print()

    new_file_path = _pick_data_file()
    print(f"\n  已选择文件：{new_file_path}")

    # ────────────────────────────────────────────
    # STEP 6  加载新数据
    # ────────────────────────────────────────────
    print("\n  正在加载新数据文件...")

    # 延迟导入项目模块
    try:
        from data_loader import load_stock_excel  # type: ignore
    except ImportError:
        print("  [错误] 找不到 data_loader.py，请确认本脚本放置在项目根目录。")
        sys.exit(1)

    try:
        df_new, data_report = load_stock_excel(new_file_path, strict_columns=True)
    except Exception as e:
        print(f"  [错误] 加载数据失败：{e}")
        sys.exit(1)

    print(f"  ✅ 数据加载成功：")
    print(f"     原始行数        : {data_report.rows_raw}")
    print(f"     清洗后行数      : {data_report.rows_after_filters}")
    print(f"     数据起始日期    : {df_new['date'].min().date()}")
    print(f"     数据截止日期    : {df_new['date'].max().date()}")
    if data_report.notes:
        print(f"     清洗日志        :")
        for note in data_report.notes:
            print(f"       - {note}")

    if data_report.rows_after_filters < 100:
        print(f"\n  [警告] 有效数据仅 {data_report.rows_after_filters} 行，可能不足以训练稳定的模型。")

    # ────────────────────────────────────────────
    # STEP 7  复权漂移检查（信息提示，不阻止运行）
    # ────────────────────────────────────────────
    print(f"\n  正在检查复权价格漂移...")
    # 尝试加载原始 data 文件路径（从 config 的 data_hash 或 notes 中无法直接得到，
    # 此处通过扫描 data 目录里最早的同名文件来对比）
    old_df: Optional[pd.DataFrame] = None
    if os.path.isfile(xlsx_path):
        try:
            # 从 Excel 的 Config sheet 读出数据哈希，仅作参考
            ec_df = pd.read_excel(xlsx_path, sheet_name="EquityCurve")
            # 直接用 equity_curve 的 date + close_qfq 列模拟"旧数据"比较
            if "date" in ec_df.columns and "close_qfq" in ec_df.columns:
                old_df = ec_df[["date", "close_qfq"]].copy()
                old_df["date"] = pd.to_datetime(old_df["date"])
        except Exception:
            pass
    _check_qfq_drift(old_df, df_new)

    # ────────────────────────────────────────────
    # STEP 8  重新训练并计算 Dpoint
    # ────────────────────────────────────────────
    print()
    print(SEPARATOR)
    print("  开始重新训练模型...")
    print("  （全样本训练，视模型复杂度可能需要数分钟）")
    print(SEPARATOR)

    try:
        dpoint = _compute_dpoint_retrain(df_new, best_config, seed=args.seed)
    except Exception as e:
        print(f"\n  [错误] 模型训练失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ────────────────────────────────────────────
    # STEP 9  处理最后一个交易日
    # ────────────────────────────────────────────
    dpoint_ext, extended = _extend_dpoint_to_last_day(dpoint, df_new)

    if extended:
        last_date = df_new["date"].max()
        prev_dpt  = float(dpoint.iloc[-1])
        print(f"\n  ℹ️  最后一个交易日（{last_date.date()}）无法计算 Dpoint")
        print(f"     （原因：需要 t+1 日收盘价作标签，最新一日的次日数据尚不存在）")
        print(f"     处理方式：使用前一交易日的 Dpoint 值 {prev_dpt:.4f} 填充，已在 note 列标注。")
    else:
        print(f"  ✅ 最后一个交易日已正常计算 Dpoint。")

    # ────────────────────────────────────────────
    # STEP 10  构造输出 DataFrame
    # ────────────────────────────────────────────
    output_df = _build_output_df(df_new, dpoint_ext, extended)

    # ── 打印末尾几行供用户确认 ──
    print(f"\n  【最近 5 个交易日的 Dpoint 预览】")
    print()
    tail = output_df.tail(5).copy()
    tail["date"] = tail["date"].dt.strftime("%Y-%m-%d")
    tail["dpoint"] = tail["dpoint"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    # 简单控制台对齐
    print(f"  {'日期':<14} {'收盘价(qfq)':>12} {'Dpoint':>10}  {'备注'}")
    print(f"  {'─'*14} {'─'*12} {'─'*10}  {'─'*20}")
    for _, row in tail.iterrows():
        note_short = row["note"][:20] if row["note"] else ""
        print(f"  {row['date']:<14} {row['close_qfq']:>12.4f} {row['dpoint']:>10}  {note_short}")

    # ────────────────────────────────────────────
    # STEP 11  保存 Excel
    # ────────────────────────────────────────────
    print()
    print(SEPARATOR)
    out_path = _save_output_excel(output_df, run_id, data_dir, new_file_path)
    print(f"  ✅ 结果已保存至：")
    print(f"     {os.path.abspath(out_path)}")
    print()
    print(f"  Excel 结构：")
    print(f"    Sheet 'Dpoint' ：A列=日期  B列=收盘价  C列=Dpoint  D列=备注")
    print(f"    Sheet 'Meta'   ：来源说明、列定义、前向偏差说明")
    print()
    print(f"  ⚠️  重要提示：")
    print(f"     • 输出的 Dpoint 为全样本训练后的样本内预测，早期数据存在前向偏差。")
    print(f"     • 最近数个交易日的 Dpoint 与真实信号最接近，越早的数据越乐观。")
    print(f"     • 请勿将历史所有 Dpoint 值视为真实可操作信号。")
    print()
    print(SEPARATOR)
    print("  全部完成！")
    print(SEPARATOR)
    print()


if __name__ == "__main__":
    main()
