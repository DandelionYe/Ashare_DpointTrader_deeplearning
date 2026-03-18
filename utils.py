# utils.py
"""
实验工具模块合集

本模块合并了原有的 repro.py 和 run_manifest.py，提供完整的实验可复现性和清单管理功能。

功能模块:
    P0 - 可复现性工具 (repro.py):
        - 统一 seed 设置入口（torch, numpy, random, pandas）
        - 获取包版本信息
        - 获取 git commit hash
        - 数据哈希计算

    P1 - 实验运行清单管理 (run_manifest.py):
        - manifest.json - 单次实验的完整元数据
        - config.json - 简化的配置文件（用于 replay）
        - CLI replay 功能 - 从历史 manifest 重新运行实验
        - 数据范围、ticker 列表、样本量等落盘记录

使用示例:
    >>> from utils import set_global_seed, ReproducibilityContext, create_manifest
    >>>
    >>> # 设置随机种子
    >>> set_global_seed(42)
    >>>
    >>> # 使用上下文管理器
    >>> with ReproducibilityContext(42) as ctx:
    ...     # 执行实验代码
    ...     pass
    >>>
    >>> # 创建实验清单
    >>> manifest = create_manifest(experiment_dir, run_id, timestamp, ...)

作者：Original authors of repro.py and run_manifest.py
合并日期：2026-03-18
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# P0: 可复现性工具 (来自 repro.py)
# =============================================================================

def get_git_commit_hash() -> str:
    """获取当前 git commit hash（若非 git 仓库则返回 unknown）"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def get_package_versions() -> Dict[str, str]:
    """获取核心依赖包版本"""
    packages = [
        "torch",
        "numpy",
        "pandas",
        "sklearn",
        "scikit-learn",
        "joblib",
        "xgboost",
        "openpyxl",
        "xlsxwriter",
    ]
    versions = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"

    if "sklearn" in versions and versions["sklearn"] == "not_installed":
        try:
            import sklearn
            versions["sklearn"] = sklearn.__version__
        except ImportError:
            pass

    versions["python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return versions


def set_global_seed(seed: int) -> Dict[str, Any]:
    """
    统一设置所有随机种子，确保实验可复现。

    Args:
        seed: 随机种子值

    Returns:
        包含设置信息的字典
    """
    seed = int(seed)

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Pandas (基于 NumPy)
    try:
        pd.options.mode.use_inf_as_na = True
    except Exception:
        pass

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # TensorFlow (如果有)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    # 设置环境变量
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return {
        "seed": seed,
        "python_hashseed": str(seed),
        "torch_deterministic": True,
        "torch_benchmark": False,
    }


def compute_data_hash(df: pd.DataFrame) -> str:
    """计算 DataFrame 内容的 SHA-256 哈希"""
    raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(raw).hexdigest()


def get_data_info(df: pd.DataFrame, data_path: str) -> Dict[str, Any]:
    """获取数据的基本信息"""
    return {
        "data_path": data_path,
        "data_hash": compute_data_hash(df),
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "date_range": {
            "start": str(df.index.min()) if hasattr(df, "index") and df.index.dtype != "object" else "unknown",
            "end": str(df.index.max()) if hasattr(df, "index") and df.index.dtype != "object" else "unknown",
        },
        "columns": list(df.columns),
    }


def generate_run_metadata(
    seed: int,
    data_path: str,
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    生成单次实验的完整元数据。

    Args:
        seed: 随机种子
        data_path: 数据文件路径
        df: 清洗后的数据 DataFrame
        config: 实验配置

    Returns:
        包含所有元数据的字典
    """
    return {
        "run_id": None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "timestamp_unix": time.time(),
        "git_commit_hash": get_git_commit_hash(),
        "package_versions": get_package_versions(),
        "seed": seed,
        "data_info": get_data_info(df, data_path),
        "config_snapshot": config,
    }


class ReproducibilityContext:
    """可复现性上下文管理器"""

    def __init__(self, seed: int):
        self.seed = seed
        self.seed_info: Dict[str, Any] = {}

    def __enter__(self):
        self.seed_info = set_global_seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def export_environment_lock(filepath: str = "requirements-lock.txt") -> None:
    """
    导出当前环境的锁定依赖到文件。

    Args:
        filepath: 输出文件路径
    """
    versions = get_package_versions()

    lines = [
        "# Auto-generated lock file for reproducibility",
        "# Generated at: " + datetime.now().isoformat(timespec="seconds"),
        f"# Python: {versions.get('python', 'unknown')}",
        "",
    ]

    for pkg, ver in sorted(versions.items()):
        if pkg != "python":
            lines.append(f"{pkg}=={ver}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Exported environment lock to {filepath}")


# =============================================================================
# P1: 实验运行清单管理 (来自 run_manifest.py)
# =============================================================================

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


def find_latest_experiment(output_dir: str) -> Optional[Tuple[int, str]]:
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
    # 内部调用：使用本模块的 compute_data_hash 函数
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


# =============================================================================
# 公开 API 导出列表
# =============================================================================

__all__ = [
    # P0: 可复现性工具
    "get_git_commit_hash",
    "get_package_versions",
    "set_global_seed",
    "compute_data_hash",
    "get_data_info",
    "generate_run_metadata",
    "ReproducibilityContext",
    "export_environment_lock",
    # P1: 实验运行清单管理
    "_get_next_experiment_id",
    "get_ticker_list",
    "create_experiment_dir",
    "create_manifest",
    "create_config_json",
    "load_manifest",
    "load_config",
    "find_latest_experiment",
    "replay_from_manifest",
    "list_experiments",
    "export_data_version_spec",
]


# =============================================================================
# CLI 入口 (合并自两个文件的 __main__ 块)
# =============================================================================

if __name__ == "__main__":
    import argparse

    # 创建主解析器
    main_parser = argparse.ArgumentParser(description="实验工具模块合集")
    subparsers = main_parser.add_subparsers(dest="command", help="可用命令")

    # repro 子命令
    repro_parser = subparsers.add_parser("repro", help="可复现性工具")
    repro_parser.add_argument("--info", action="store_true", help="显示环境信息")
    repro_parser.add_argument("--test-seed", type=int, help="测试种子设置")
    repro_parser.add_argument("--export-lock", type=str, help="导出环境锁定文件")

    # manifest 子命令
    manifest_parser = subparsers.add_parser("manifest", help="实验清单管理")
    manifest_parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    manifest_parser.add_argument("--list", action="store_true", help="列出所有实验")
    manifest_parser.add_argument("--replay", type=str, help="从历史实验目录 replay")
    manifest_parser.add_argument("--latest", action="store_true", help="从最新实验 replay")

    args = main_parser.parse_args()

    if args.command == "repro" or args.command is None:
        print("=== Reproducibility Tool ===")
        print(f"Git commit: {get_git_commit_hash()}")
        print(f"Python: {sys.version}")
        print(f"Package versions: {get_package_versions()}")
        if args.test_seed:
            print(f"Test seed setting: {set_global_seed(args.test_seed)}")
        if args.export_lock:
            export_environment_lock(args.export_lock)

    elif args.command == "manifest":
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
            manifest_parser.print_help()

    else:
        main_parser.print_help()
