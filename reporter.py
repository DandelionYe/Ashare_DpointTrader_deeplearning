from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from constants import (
    MIN_CLOSED_TRADES_PER_FOLD,
    TARGET_CLOSED_TRADES_PER_FOLD,
    LAMBDA_TRADE_PENALTY,
)


def escape_excel_formulas(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Prevent Excel from treating strings as formulas (which can trigger repair prompts),
    by prefixing strings starting with = + - @ with a single quote.

    Args:
        df: DataFrame to process
        inplace: If True, modify in place; if False, return a copy

    Returns:
        Processed DataFrame
    """
    if not inplace:
        df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda v: ("'" + v) if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v
            )
    return df


def _hash_dataframe(df: pd.DataFrame) -> str:
    """
    对 DataFrame 内容做 SHA-256 哈希，用于检测数据变化。
    使用 pandas 内置哈希（比 to_csv 快约 10x），结果为 16 进制字符串。
    """
    raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(raw).hexdigest()


def _next_run_id(output_dir: str) -> int:
    """
    扫描 output_dir 中已有的 run_XXX_config.json，返回下一个可用的 run_id（从 1 开始）。
    若目录为空则返回 1。
    """
    os.makedirs(output_dir, exist_ok=True)
    existing = []
    for fn in os.listdir(output_dir):
        if fn.startswith("run_") and fn.endswith("_config.json"):
            try:
                n = int(fn.split("_")[1])
                existing.append(n)
            except Exception:
                pass
    return (max(existing) + 1) if existing else 1


def find_latest_run(output_dir: str) -> Optional[Tuple[int, str, str]]:
    if not os.path.isdir(output_dir):
        return None

    candidates = []
    for fn in os.listdir(output_dir):
        if fn.startswith("run_") and fn.endswith("_config.json"):
            try:
                run_id = int(fn.split("_")[1])
                cfg_path = os.path.join(output_dir, fn)
                xlsx_path = os.path.join(output_dir, f"run_{run_id:03d}.xlsx")
                candidates.append((run_id, cfg_path, xlsx_path))
            except Exception:
                continue

    if not candidates:
        return None

    return sorted(candidates, key=lambda x: x[0])[-1]


def save_run_outputs(
    output_dir: str,
    df_clean: pd.DataFrame,
    log_notes: List[str],
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    config: Dict[str, object],
    feature_meta: Dict[str, object],
    search_log: pd.DataFrame,
    model_params: Optional[Dict[str, object]] = None,
) -> Tuple[str, str, int]:
    run_id = _next_run_id(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    excel_path = os.path.join(output_dir, f"run_{run_id:03d}.xlsx")
    config_path = os.path.join(output_dir, f"run_{run_id:03d}_config.json")

    df_hash = _hash_dataframe(df_clean)

    # ---------- build config rows FIRST ----------
    config_blob = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_hash": df_hash,
        "best_config": config,
        "feature_meta": feature_meta,
        "notes": {
            "execution_assumption": "Signal uses day t data; order executes on t+1 at t+1 open price (open_qfq). P1-3: changed from t close to t+1 open to remove forward bias.",
            "a_share_constraints": "Long-only, buy before sell, no short, min 100 shares, full-in/out, T+1 approximated via min_hold_days>=1.",
        },
    }

    config_rows = []
    config_rows.append(("run_id", run_id))
    config_rows.append(("created_at", config_blob["created_at"]))
    config_rows.append(("data_hash", df_hash))
    config_rows.append(("split_mode", config.get("split_mode", "")))

    for k, v in config.get("feature_config", {}).items():
        config_rows.append((f"feature.{k}", str(v)))
    for k, v in config.get("model_config", {}).items():
        config_rows.append((f"model.{k}", str(v)))
    for k, v in config.get("trade_config", {}).items():
        config_rows.append((f"trade.{k}", str(v)))

    config_rows.append(("constraint.min_closed_trades_per_fold", MIN_CLOSED_TRADES_PER_FOLD))
    config_rows.append(("penalty.target_closed_trades_per_fold", TARGET_CLOSED_TRADES_PER_FOLD))
    config_rows.append(("penalty.lambda_trade_penalty", LAMBDA_TRADE_PENALTY))

    config_rows.append(("dpoint_definition", feature_meta.get("dpoint_explainer", "")))
    config_df = pd.DataFrame(config_rows, columns=["key", "value"])

    notes_df = pd.DataFrame({"notes": log_notes})

    # ---------- escape Excel formulas BEFORE writing Excel ----------
    # 使用 inplace=True 减少 DataFrame 复制，优化内存效率
    escape_excel_formulas(trades, inplace=True)
    escape_excel_formulas(equity_curve, inplace=True)
    escape_excel_formulas(config_df, inplace=True)
    escape_excel_formulas(notes_df, inplace=True)
    escape_excel_formulas(search_log, inplace=True)

    model_params_effective = model_params
    if model_params_effective is None and isinstance(feature_meta, dict):
        model_params_effective = feature_meta.get("model_params")

    model_params_df: Optional[pd.DataFrame] = None
    if isinstance(model_params_effective, dict):
        feature_names = list(model_params_effective.get("feature_names", []))
        coef = list(model_params_effective.get("coef", []))
        scaler_mean = model_params_effective.get("mean", model_params_effective.get("scaler_mean", []))
        scaler_scale = model_params_effective.get("scale", model_params_effective.get("scaler_scale", []))
        scaler_mean = list(scaler_mean) if isinstance(scaler_mean, (list, tuple)) else []
        scaler_scale = list(scaler_scale) if isinstance(scaler_scale, (list, tuple)) else []

        n = max(len(feature_names), len(coef), len(scaler_mean), len(scaler_scale))
        rows = []
        for i in range(n):
            rows.append(
                {
                    "feature_name": feature_names[i] if i < len(feature_names) else "",
                    "coef": coef[i] if i < len(coef) else "",
                    "scaler_mean": scaler_mean[i] if i < len(scaler_mean) else "",
                    "scaler_scale": scaler_scale[i] if i < len(scaler_scale) else "",
                }
            )

        intercept = model_params_effective.get("intercept")
        if intercept is not None:
            rows.append(
                {
                    "feature_name": "__intercept__",
                    "coef": intercept,
                    "scaler_mean": "",
                    "scaler_scale": "",
                }
            )

        if rows:
            model_params_df = pd.DataFrame(
                rows,
                columns=["feature_name", "coef", "scaler_mean", "scaler_scale"],
            )
            escape_excel_formulas(model_params_df, inplace=True)

    # ---------- write config json (ONLY json) ----------
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_blob, f, ensure_ascii=False, indent=2)

    # ---------- write Excel ----------
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        trades.to_excel(writer, sheet_name="Trades", index=False)
        equity_curve.to_excel(writer, sheet_name="EquityCurve", index=False)
        config_df.to_excel(writer, sheet_name="Config", index=False)
        notes_df.to_excel(writer, sheet_name="Log", index=False, startrow=0)

        startrow = len(notes_df) + 3
        search_log.to_excel(writer, sheet_name="Log", index=False, startrow=startrow)

        if model_params_df is not None:
            model_params_df.to_excel(writer, sheet_name="ModelParams", index=False)

    return excel_path, config_path, run_id