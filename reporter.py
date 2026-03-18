from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from constants import (
    MIN_CLOSED_TRADES_PER_FOLD,
    TARGET_CLOSED_TRADES_PER_FOLD,
    LAMBDA_TRADE_PENALTY,
)
from metrics import calculate_risk_metrics, format_metrics_summary, calculate_regime_metrics, calculate_trade_distribution
from regime import RegimeDetector, compute_regime_metrics, create_regime_visualization
from html_reporter import save_html_report, generate_leaderboard_html, save_leaderboard_html


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
    扫描 output_dir 中已有的 run_XXX_config.json 或 exp_XXX 目录，
    返回下一个可用的 run_id（从 1 开始）。
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
        if fn.startswith("exp_") and os.path.isdir(os.path.join(output_dir, fn)):
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
    feature_usage_stats: Optional[Dict[str, Any]] = None,
    best_model_importance: Optional[Dict[str, Any]] = None,
    use_regime_analysis: bool = False,
    regime_config: Optional[Dict[str, Any]] = None,
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
    for k, v in config.get("calibration_config", {}).items():
        config_rows.append((f"calibration.{k}", str(v)))

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

    # ---------- P0: 计算统一风险指标 ----------
    # 获取初始资金
    initial_cash = float(config.get("trade_config", {}).get("initial_cash", 100000.0))

    # 准备 benchmark 数据
    benchmark_curve = None
    if "bnh_equity" in equity_curve.columns:
        benchmark_curve = pd.DataFrame({"bnh_equity": equity_curve["bnh_equity"]})

    # 计算风险指标
    risk_metrics = calculate_risk_metrics(
        equity_curve=equity_curve,
        trades=trades,
        initial_cash=initial_cash,
        benchmark_curve=benchmark_curve,
    )

    # P0: 计算日收益率并添加到 equity_curve
    equity_with_returns = equity_curve.copy()
    if "total_equity" in equity_with_returns.columns:
        equity_with_returns["daily_return"] = equity_with_returns["total_equity"].pct_change()
        # 补齐第一行
        equity_with_returns.loc[equity_with_returns.index[0], "daily_return"] = 0.0

    # P2: 计算 regime 指标
    regime_metrics = calculate_regime_metrics(equity_curve, trades, initial_cash)

    # P2: 计算交易分布
    trade_dist = calculate_trade_distribution(trades, equity_curve)

    # 格式化风险指标为 DataFrame
    risk_metrics_rows = []
    for key, value in risk_metrics.items():
        if isinstance(value, list):
            continue  # 跳过列表类型
        risk_metrics_rows.append({"metric": key, "value": value})
    risk_metrics_df = pd.DataFrame(risk_metrics_rows)

    # P2: Regime 指标 DataFrame
    regime_rows = []
    for regime, reg_metrics in regime_metrics.items():
        for k, v in reg_metrics.items():
            regime_rows.append({"regime": regime, "metric": k, "value": v})
    regime_df = pd.DataFrame(regime_rows) if regime_rows else None

    # P0: 使用新的 RegimeDetector 进行更详细的分层分析
    regime_analysis_df = None
    regime_visualization_df = None
    if use_regime_analysis and df_clean is not None and not df_clean.empty:
        try:
            detector = RegimeDetector(
                ma_short=regime_config.get("ma_short", 5) if regime_config else 5,
                ma_long=regime_config.get("ma_long", 20) if regime_config else 20,
                vol_window=regime_config.get("vol_window", 20) if regime_config else 20,
                vol_high_threshold=regime_config.get("vol_high_threshold", 0.20) if regime_config else 0.20,
                vol_low_threshold=regime_config.get("vol_low_threshold", 0.10) if regime_config else 0.10,
            )
            
            if "close" in df_clean.columns:
                regimes = detector.fit_predict(df_clean)
                
                regime_labels = regimes["combined"]
                
                detailed_regime_metrics = compute_regime_metrics(
                    equity_curve, trades, initial_cash, regime_labels, "combined"
                )
                
                regime_analysis_rows = []
                for regime_name, metrics in detailed_regime_metrics.items():
                    regime_analysis_rows.append({
                        "regime": regime_name,
                        "n_days": metrics.get("n_days", 0),
                        "total_return_pct": metrics.get("total_return_pct", 0),
                        "sharpe": metrics.get("sharpe", 0),
                        "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
                        "trade_count": metrics.get("trade_count", 0),
                    })
                
                if regime_analysis_rows:
                    regime_analysis_df = pd.DataFrame(regime_analysis_rows)
                
                vis_df = create_regime_visualization(df_clean, regimes)
                if "regime" in vis_df.columns:
                    regime_visualization_df = vis_df[["price", "ma_5", "ma_20", "volatility", "regime", "regime_color"]].reset_index()
        except Exception as e:
            print(f"[WARN] Regime analysis failed: {e}")

    # P2: 交易分布 DataFrame
    dist_rows = []
    for cat, cat_metrics in trade_dist.items():
        for k, v in cat_metrics.items():
            dist_rows.append({"category": cat, "metric": k, "value": v})
    trade_dist_df = pd.DataFrame(dist_rows) if dist_rows else None

    # P1: 校准指标 DataFrame - 从 config 中提取校准信息
    calibration_config = config.get("calibration_config", {})
    calibration_rows = []
    if calibration_config:
        for k, v in calibration_config.items():
            calibration_rows.append({"metric": f"calibration.{k}", "value": str(v)})
    
    # 从 holdout_calibration_comparison 中提取校准对比信息（如果在 feature_meta 中）
    if isinstance(feature_meta, dict):
        holdout_cal = feature_meta.get("holdout_calibration_comparison", {})
        if holdout_cal:
            calibration_rows.append({"metric": "holdout.comparison.available", "value": "true"})
            for k, v in holdout_cal.items():
                if isinstance(v, (int, float)):
                    calibration_rows.append({"metric": f"holdout.{k}", "value": float(v)})
                else:
                    calibration_rows.append({"metric": f"holdout.{k}", "value": str(v)})
    
    calibration_df = pd.DataFrame(calibration_rows) if calibration_rows else None

    # 格式化摘要
    metrics_summary = format_metrics_summary(risk_metrics)

    # ---------- write Excel ----------
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        trades.to_excel(writer, sheet_name="Trades", index=False)
        equity_with_returns.to_excel(writer, sheet_name="EquityCurve", index=False)  # P0: 包含 daily_return
        config_df.to_excel(writer, sheet_name="Config", index=False)
        notes_df.to_excel(writer, sheet_name="Log", index=False, startrow=0)

        startrow = len(notes_df) + 3
        search_log.to_excel(writer, sheet_name="Log", index=False, startrow=startrow)

        if model_params_df is not None:
            model_params_df.to_excel(writer, sheet_name="ModelParams", index=False)

        # P0: 新增 RiskMetrics sheet
        risk_metrics_df.to_excel(writer, sheet_name="RiskMetrics", index=False)

        # P2: Regime 分析 sheet
        if regime_df is not None and not regime_df.empty:
            regime_df.to_excel(writer, sheet_name="RegimeAnalysis", index=False)

        # P0: 详细 Regime 分层评估 sheet
        if regime_analysis_df is not None and not regime_analysis_df.empty:
            regime_analysis_df.to_excel(writer, sheet_name="RegimeStratified", index=False)

        # P0: Regime 可视化数据 sheet
        if regime_visualization_df is not None and not regime_visualization_df.empty:
            regime_visualization_df.to_excel(writer, sheet_name="RegimeVisualization", index=False)

        # P2: 交易分布 sheet
        if trade_dist_df is not None and not trade_dist_df.empty:
            trade_dist_df.to_excel(writer, sheet_name="TradeDistribution", index=False)

        # P1: 校准指标 sheet
        if calibration_df is not None and not calibration_df.empty:
            calibration_df.to_excel(writer, sheet_name="CalibrationMetrics", index=False)

        # P0: 特征使用频率 sheet
        if feature_usage_stats:
            fus = feature_usage_stats
            fus_rows = []
            fus_rows.append({"stat": "total_candidates", "value": fus.get("total_candidates", 0)})
            group_usage = fus.get("group_usage", {})
            for key, data in group_usage.items():
                fus_rows.append({"stat": key, "value": f"{data.get('frequency', 0)*100:.2f}%", "count": data.get("count", 0)})
            fus_df = pd.DataFrame(fus_rows)
            fus_df.to_excel(writer, sheet_name="FeatureUsage", index=False)

        # P0-P1: 最佳模型特征重要性 sheet
        if best_model_importance:
            bmi = best_model_importance
            bmi_rows = []
            bmi_rows.append({"type": "method", "value": bmi.get("method", "")})
            
            ranking = bmi.get("ranking", [])
            for item in ranking:
                bmi_rows.append({
                    "type": "feature",
                    "rank": item.get("rank", ""),
                    "name": item.get("feature", ""),
                    "importance": item.get("importance", ""),
                })
            
            group_ranking = bmi.get("feature_group_ranking", [])
            for item in group_ranking:
                bmi_rows.append({
                    "type": "group",
                    "rank": item.get("rank", ""),
                    "name": item.get("group", ""),
                    "importance": item.get("importance", ""),
                })
            
            if bmi_rows:
                bmi_df = pd.DataFrame(bmi_rows)
                bmi_df.to_excel(writer, sheet_name="FeatureImportance", index=False)

    html_path = None
    if use_regime_analysis or True:
        try:
            initial_cash = float(config.get("trade_config", {}).get("initial_cash", 100000.0))
            
            holdout_metric = None
            holdout_equity = None
            if isinstance(feature_meta, dict):
                holdout_metric = feature_meta.get("holdout_metric")
                holdout_equity = feature_meta.get("holdout_equity")
            
            calibration_data = None
            if isinstance(feature_meta, dict):
                calibration_data = feature_meta.get("holdout_calibration_comparison")
            
            html_path = save_html_report(
                output_dir=output_dir,
                run_id=run_id,
                config=config,
                metrics=risk_metrics,
                equity_curve=equity_curve,
                trades=trades,
                initial_cash=initial_cash,
                holdout_metric=holdout_metric,
                holdout_equity=holdout_equity,
                calibration_data=calibration_data,
                feature_importance=best_model_importance,
                feature_usage=feature_usage_stats,
                monthly_returns=risk_metrics.get("monthly_returns"),
                yearly_returns=risk_metrics.get("yearly_returns"),
                benchmark_return=risk_metrics.get("bnh_return"),
                created_at=config_blob.get("created_at"),
                notes=log_notes if isinstance(log_notes, list) else None,
            )
            print(f"[REPORT] HTML report saved: {html_path}")
        except Exception as e:
            print(f"[WARN] Failed to generate HTML report: {e}")

    return excel_path, config_path, run_id


def generate_multi_run_report(output_dir: str) -> str:
    """
    P2: 生成多 run 对比报告和索引页。
    
    Returns:
        生成的 leaderboard.html 路径
    """
    runs = []
    
    for fn in os.listdir(output_dir):
        if fn.startswith("run_") and fn.endswith("_config.json"):
            try:
                config_path = os.path.join(output_dir, fn)
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_blob = json.load(f)
                
                run_id = config_blob.get("run_id")
                run_dir = output_dir
                
                excel_path = os.path.join(output_dir, f"run_{run_id:03d}.xlsx")
                if not os.path.exists(excel_path):
                    continue
                
                from metrics import calculate_risk_metrics
                
                try:
                    equity_df = pd.read_excel(excel_path, sheet_name="EquityCurve")
                    trades_df = pd.read_excel(excel_path, sheet_name="Trades")
                    
                    initial_cash = config_blob.get("best_config", {}).get("trade_config", {}).get("initial_cash", 100000.0)
                    
                    risk_metrics = calculate_risk_metrics(
                        equity_curve=equity_df,
                        trades=trades_df,
                        initial_cash=initial_cash,
                    )
                    
                    runs.append({
                        "run_id": run_id,
                        "created_at": config_blob.get("created_at", ""),
                        "total_return_pct": risk_metrics.get("total_return_pct", 0),
                        "sharpe": risk_metrics.get("sharpe", 0),
                        "max_drawdown_pct": risk_metrics.get("max_drawdown_pct", 0),
                        "trade_count": risk_metrics.get("trade_count", 0),
                        "win_rate": risk_metrics.get("win_rate", 0),
                        "annual_return_pct": risk_metrics.get("annual_return_pct", 0),
                    })
                except Exception as e:
                    print(f"[WARN] Failed to load run {run_id}: {e}")
            except Exception as e:
                print(f"[WARN] Failed to load config {fn}: {e}")
    
    if not runs:
        print("[INFO] No runs found for multi-run report")
        return ""
    
    runs_sorted = sorted(runs, key=lambda x: x.get('total_return_pct', 0), reverse=True)
    
    leaderboard_path = save_leaderboard_html(output_dir, runs_sorted)
    print(f"[REPORT] Leaderboard saved: {leaderboard_path}")
    
    return leaderboard_path