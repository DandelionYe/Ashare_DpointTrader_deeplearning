# run_manifest.py
"""
P1: 实验运行清单管理

提供：
1. manifest.json - 单次实验的完整元数据
2. config.json - 简化的配置文件（用于 replay）
3. CLI replay 功能 - 从历史 manifest 重新运行实验
4. 数据范围、ticker 列表、样本量等落盘记录
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd


def _get_next_experiment_id(output_dir: str) -> int:
    """获取下一个实验 ID"""
    os.makedirs(output_dir, exist_ok=True)
    existing = []
    for dn in os.listdir(output_dir):
        if dn.startswith("exp_") and os.path.isdir(os.path.join(output_dir, dn)):
            try:
                nid = int(dn.split("_")[1])
                existing.append(nid)
            except (IndexError, ValueError):
                pass
    return (max(existing) + 1) if existing else 1


def get_ticker_list(df: pd.DataFrame, data_path: str) -> List[str]:
    """从数据中提取 ticker 列表"""
    if "ticker" in df.columns:
        return df["ticker"].unique().tolist()
    elif "code" in df.columns:
        return df["code"].unique().tolist()
    
    filename = os.path.basename(data_path)
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        ticker_name = filename.rsplit(".", 1)[0]
        return [ticker_name]
    return ["unknown"]


def create_experiment_dir(output_dir: str, experiment_id: int) -> str:
    """创建单次实验的独立目录"""
    exp_dir = os.path.join(output_dir, f"exp_{experiment_id:03d}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "artifacts"), exist_ok=True)
    return exp_dir


def create_manifest(
    experiment_dir: str,
    run_id: int,
    timestamp: str,
    git_commit_hash: str,
    package_versions: Dict[str, str],
    seed: int,
    data_info: Dict[str, Any],
    cli_args: Dict[str, Any],
    best_config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    创建实验 manifest.json
    
    Args:
        experiment_dir: 实验目录
        run_id: 运行 ID
        timestamp: 时间戳
        git_commit_hash: git 提交哈希
        package_versions: 包版本
        seed: 随机种子
        data_info: 数据信息
        cli_args: 命令行参数
        best_config: 最优配置
        metrics: 评估指标
        
    Returns:
        manifest 字典
    """
    manifest = {
        "manifest_version": "1.0",
        "run_id": run_id,
        "experiment_id": run_id,
        "created_at": timestamp,
        "git_commit_hash": git_commit_hash,
        "package_versions": package_versions,
        "seed": seed,
        "data": {
            "data_path": data_info.get("data_path"),
            "data_hash": data_info.get("data_hash"),
            "n_rows": data_info.get("n_rows"),
            "n_columns": data_info.get("n_columns"),
            "date_range": data_info.get("date_range"),
            "tickers": data_info.get("tickers", []),
            "columns": data_info.get("columns", []),
        },
        "cli_args": cli_args,
    }
    
    if best_config:
        manifest["best_config"] = best_config
        
    if metrics:
        manifest["metrics"] = metrics
    
    manifest_path = os.path.join(experiment_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    return manifest


def create_config_json(
    experiment_dir: str,
    manifest: Dict[str, Any],
) -> None:
    """
    创建简化的 config.json（用于 replay）
    """
    config = {
        "seed": manifest.get("seed"),
        "data_path": manifest.get("data", {}).get("data_path"),
        "data_hash": manifest.get("data", {}).get("data_hash"),
        "cli_args": manifest.get("cli_args", {}),
        "best_config": manifest.get("best_config"),
    }
    
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_manifest(experiment_dir: str) -> Optional[Dict[str, Any]]:
    """从实验目录加载 manifest.json"""
    manifest_path = os.path.join(experiment_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(experiment_dir: str) -> Optional[Dict[str, Any]]:
    """从实验目录加载 config.json"""
    config_path = os.path.join(experiment_dir, "config.json")
    if not os.path.exists(config_path):
        return None
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_experiment(output_dir: str) -> Optional[tuple[int, str]]:
    """查找最新的实验目录"""
    if not os.path.isdir(output_dir):
        return None
    
    candidates = []
    for dn in os.listdir(output_dir):
        if dn.startswith("exp_") and os.path.isdir(os.path.join(output_dir, dn)):
            try:
                exp_id = int(dn.split("_")[1])
                manifest_path = os.path.join(output_dir, dn, "manifest.json")
                candidates.append((exp_id, os.path.join(output_dir, dn), manifest_path))
            except (IndexError, ValueError):
                continue
    
    if not candidates:
        return None
    
    sorted_candidates = sorted(candidates, key=lambda x: x[0])
    latest_id, exp_dir, manifest_path = sorted_candidates[-1]
    
    return latest_id, exp_dir


def replay_from_manifest(
    experiment_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    从 manifest 重新运行实验（replay）
    
    Args:
        experiment_dir: 历史实验目录
        output_dir: 输出目录
        
    Returns:
        用于 replay 的配置字典
    """
    manifest = load_manifest(experiment_dir)
    if manifest is None:
        raise FileNotFoundError(f"No manifest.json found in {experiment_dir}")
    
    config = {
        "data_path": manifest.get("data", {}).get("data_path"),
        "data_hash": manifest.get("data", {}).get("data_hash"),
        "seed": manifest.get("seed"),
        "git_commit_hash": manifest.get("git_commit_hash"),
        "package_versions": manifest.get("package_versions"),
        "cli_args": manifest.get("cli_args", {}),
        "best_config": manifest.get("best_config"),
        "replay_from": experiment_dir,
        "original_timestamp": manifest.get("created_at"),
    }
    
    print(f"[INFO] Replaying experiment from {experiment_dir}")
    print(f"[INFO] Original timestamp: {config['original_timestamp']}")
    print(f"[INFO] Original seed: {config['seed']}")
    print(f"[INFO] Data path: {config['data_path']}")
    
    return config


def list_experiments(output_dir: str) -> List[Dict[str, Any]]:
    """列出所有实验"""
    experiments = []
    
    if not os.path.isdir(output_dir):
        return experiments
    
    for dn in os.listdir(output_dir):
        if dn.startswith("exp_") and os.path.isdir(os.path.join(output_dir, dn)):
            manifest = load_manifest(os.path.join(output_dir, dn))
            if manifest:
                experiments.append({
                    "experiment_id": manifest.get("experiment_id"),
                    "directory": os.path.join(output_dir, dn),
                    "created_at": manifest.get("created_at"),
                    "seed": manifest.get("seed"),
                    "git_commit_hash": manifest.get("git_commit_hash"),
                    "data_hash": manifest.get("data", {}).get("data_hash"),
                    "data_path": manifest.get("data", {}).get("data_path"),
                })
    
    return sorted(experiments, key=lambda x: x.get("created_at", ""), reverse=True)


def export_data_version_spec(
    data_path: str,
    df: pd.DataFrame,
    output_dir: str,
    version: str = "1.0",
) -> str:
    """
    导出数据版本标识规范
    
    Args:
        data_path: 数据文件路径
        df: 数据 DataFrame
        output_dir: 输出目录
        version: 版本号
        
    Returns:
        数据版本标识符
    """
    from repro import compute_data_hash
    
    data_hash = compute_data_hash(df)
    ticker_list = get_ticker_list(df, data_path)
    
    date_start = str(df.index.min()) if hasattr(df, "index") else "unknown"
    date_end = str(df.index.max()) if hasattr(df, "index") else "unknown"
    
    spec = {
        "version": version,
        "data_file": os.path.basename(data_path),
        "data_hash": data_hash,
        "tickers": ticker_list,
        "n_rows": len(df),
        "date_range": {
            "start": date_start,
            "end": date_end,
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    
    spec_path = os.path.join(output_dir, "data_version.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    
    data_version_id = f"v{version}_{data_hash[:8]}"
    return data_version_id


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment Manifest Management")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--replay", type=str, help="Replay from experiment directory")
    parser.add_argument("--latest", action="store_true", help="Replay from latest experiment")
    
    args = parser.parse_args()
    
    if args.list:
        experiments = list_experiments(args.output_dir)
        print(f"Found {len(experiments)} experiments:")
        for exp in experiments:
            print(f"  exp_{exp['experiment_id']:03d}: {exp['created_at']}, seed={exp['seed']}, hash={exp.get('git_commit_hash', 'unknown')[:8]}")
    
    elif args.replay:
        config = replay_from_manifest(args.replay, args.output_dir)
        print("\nReplay config:")
        print(json.dumps(config, indent=2))
    
    elif args.latest:
        latest = find_latest_experiment(args.output_dir)
        if latest:
            exp_id, exp_dir = latest
            config = replay_from_manifest(exp_dir, args.output_dir)
            print(f"\nLatest experiment: exp_{exp_id:03d}")
            print(json.dumps(config, indent=2))
        else:
            print("No experiments found")
    
    else:
        parser.print_help()
