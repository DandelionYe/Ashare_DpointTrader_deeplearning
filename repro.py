# repro.py
"""
P0: 可复现性工具模块

提供：
1. 统一 seed 设置入口（torch, numpy, random, pandas）
2. 获取包版本信息
3. 获取 git commit hash
4. 数据哈希计算
"""
from __future__ import annotations

import hashlib
import os
import random
import subprocess
import sys
from datetime import datetime
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd


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
    import time
    
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
    import time
    
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


if __name__ == "__main__":
    print("=== Reproducibility Tool ===")
    print(f"Git commit: {get_git_commit_hash()}")
    print(f"Python: {sys.version}")
    print(f"Package versions: {get_package_versions()}")
    print(f"Test seed setting: {set_global_seed(42)}")
