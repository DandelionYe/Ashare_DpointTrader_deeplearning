# compare_runs.py
"""
P2: 结果比对工具

支持两次 run 的配置与指标差异对比
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Any, Optional

from tabulate import tabulate


def load_experiment_data(exp_dir: str) -> Optional[Dict[str, Any]]:
    """加载实验数据"""
    manifest_path = os.path.join(exp_dir, "manifest.json")
    config_path = os.path.join(exp_dir, "config.json")
    
    data = {"experiment_dir": exp_dir}
    
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data["manifest"] = json.load(f)
    
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data["config"] = json.load(f)
    
    return data if "manifest" in data else None


def find_all_experiments(output_dir: str) -> list:
    """查找所有实验目录"""
    experiments = []
    if not os.path.isdir(output_dir):
        return experiments
    
    for dn in os.listdir(output_dir):
        exp_path = os.path.join(output_dir, dn)
        if dn.startswith("exp_") and os.path.isdir(exp_path):
            experiments.append(exp_path)
    
    return sorted(experiments)


def compare_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> list:
    """比较两个配置的差异"""
    differences = []
    
    best1 = config1.get("best_config", {})
    best2 = config2.get("best_config", {})
    
    if not best1 or not best2:
        return [["N/A", "Config not available", ""]]
    
    for section in ["feature_config", "model_config", "trade_config"]:
        s1 = best1.get(section, {})
        s2 = best2.get(section, {})
        
        all_keys = set(s1.keys()) | set(s2.keys())
        for key in all_keys:
            v1 = s1.get(key)
            v2 = s2.get(key)
            if v1 != v2:
                differences.append([
                    f"{section}.{key}",
                    str(v1),
                    str(v2),
                ])
    
    return differences


def compare_metrics(metrics1: Dict[str, Any], metrics2: Dict[str, Any]) -> list:
    """比较两个实验的指标，包含验证集指标和 holdout 指标。"""
    comparisons = []

    # 验证集 & 整体指标
    val_keys = [
        "best_val_metric",
        "best_val_final_equity_proxy",
        "final_equity",
        "n_trades",
    ]
    # Holdout 指标（由 main_cli.py 写入 manifest["metrics"]）
    holdout_keys = [
        "holdout_metric",
        "holdout_equity",
    ]

    for key in val_keys + holdout_keys:
        v1 = metrics1.get(key)
        v2 = metrics2.get(key)
        if v1 is None or v2 is None:
            continue

        try:
            diff = v2 - v1
            diff_pct = f"{diff / v1 * 100:+.2f}%" if v1 != 0 else "N/A"
            comparisons.append([
                key,
                f"{v1:.6f}" if isinstance(v1, float) else str(v1),
                f"{v2:.6f}" if isinstance(v2, float) else str(v2),
                f"{diff:+.6f}",
                diff_pct,
            ])
        except TypeError:
            # 非数值型指标（如字符串）跳过差值计算
            comparisons.append([key, str(v1), str(v2), "N/A", "N/A"])

    return comparisons


def compare_seeds(exp1: Dict[str, Any], exp2: Dict[str, Any]) -> list:
    """比较种子和环境配置"""
    comparisons = []
    
    m1 = exp1.get("manifest", {})
    m2 = exp2.get("manifest", {})
    
    comparisons.append(["seed", str(m1.get("seed")), str(m2.get("seed"))])
    comparisons.append(["git_commit", m1.get("git_commit_hash", "unknown")[:12], m2.get("git_commit_hash", "unknown")[:12]])
    comparisons.append(["timestamp", m1.get("created_at"), m2.get("created_at")])
    
    data1 = m1.get("data", {})
    data2 = m2.get("data", {})
    comparisons.append(["data_hash", data1.get("data_hash", "unknown")[:16], data2.get("data_hash", "unknown")[:16]])
    comparisons.append(["data_path", data1.get("data_path", "unknown"), data2.get("data_path", "unknown")])
    comparisons.append(["n_rows", str(data1.get("n_rows")), str(data2.get("n_rows"))])
    
    pv1 = m1.get("package_versions", {})
    pv2 = m2.get("package_versions", {})
    pkg_comparison = []
    all_pkgs = set(pv1.keys()) | set(pv2.keys())
    for pkg in sorted(all_pkgs):
        v1 = pv1.get(pkg, "N/A")
        v2 = pv2.get(pkg, "N/A")
        if v1 != v2:
            pkg_comparison.append([pkg, v1, v2])
    
    return comparisons, pkg_comparison


def main():
    parser = argparse.ArgumentParser(description="Compare two experiment runs")
    parser.add_argument("--exp1", type=str, required=True, help="First experiment directory")
    parser.add_argument("--exp2", type=str, required=True, help="Second experiment directory")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for experiments")
    
    args = parser.parse_args()
    
    if not os.path.isabs(args.exp1):
        args.exp1 = os.path.join(args.output_dir, args.exp1)
    if not os.path.isabs(args.exp2):
        args.exp2 = os.path.join(args.output_dir, args.exp2)
    
    exp1_data = load_experiment_data(args.exp1)
    exp2_data = load_experiment_data(args.exp2)
    
    if not exp1_data:
        print(f"[ERROR] No experiment data found in {args.exp1}")
        sys.exit(1)
    if not exp2_data:
        print(f"[ERROR] No experiment data found in {args.exp2}")
        sys.exit(1)
    
    print("=" * 80)
    print("EXPERIMENT COMPARISON REPORT")
    print("=" * 80)
    print(f"Exp 1: {args.exp1}")
    print(f"Exp 2: {args.exp2}")
    print()
    
    print("-" * 80)
    print("1. SEED & ENVIRONMENT")
    print("-" * 80)
    env_comparison, pkg_comparison = compare_seeds(exp1_data, exp2_data)
    print(tabulate(env_comparison, headers=["Property", "Exp1", "Exp2"], tablefmt="grid"))
    print()
    
    if pkg_comparison:
        print("Package version differences:")
        print(tabulate(pkg_comparison, headers=["Package", "Exp1", "Exp2"], tablefmt="grid"))
        print()
    
    print("-" * 80)
    print("2. METRICS COMPARISON")
    print("-" * 80)
    m1 = exp1_data.get("manifest", {}).get("metrics", {})
    m2 = exp2_data.get("manifest", {}).get("metrics", {})
    metrics_table = compare_metrics(m1, m2)
    if metrics_table:
        print(tabulate(metrics_table, headers=["Metric", "Exp1", "Exp2", "Diff", "Diff %"], tablefmt="grid"))
    else:
        print("No metrics available for comparison")
    print()
    
    print("-" * 80)
    print("3. CONFIG DIFFERENCES")
    print("-" * 80)
    c1 = exp1_data.get("config", {})
    c2 = exp2_data.get("config", {})
    config_diff = compare_configs(c1, c2)
    if config_diff:
        print(tabulate(config_diff, headers=["Config Key", "Exp1 Value", "Exp2 Value"], tablefmt="grid"))
    else:
        print("No config differences found")
    print()
    
    print("=" * 80)
    print("END OF COMPARISON REPORT")
    print("=" * 80)


if __name__ == "__main__":
    main()
