# persistence.py
"""
最优配置的持久化 I/O。
管理 best_so_far.json（全局最优）和 best_pool.json（Top-K 池）。
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from constants import BEST_SO_FAR_FILENAME, BEST_POOL_FILENAME


def config_hash(cfg: Dict[str, object]) -> str:
    """对配置字典做 SHA-256，用于去重和缓存 key。"""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def best_so_far_path(output_dir: str) -> str:
    return os.path.join(output_dir, BEST_SO_FAR_FILENAME)


def best_pool_path(output_dir: str) -> str:
    return os.path.join(output_dir, BEST_POOL_FILENAME)


def load_best_so_far(output_dir: str) -> Optional[Dict[str, object]]:
    path = best_so_far_path(output_dir)
    if not output_dir or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        return blob.get("best_config")
    except Exception:
        return None


def load_best_so_far_metric(output_dir: str) -> Optional[float]:
    path = best_so_far_path(output_dir)
    if not output_dir or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        m = blob.get("best_metric", None)
        return float(m) if m is not None else None
    except Exception:
        return None


def save_best_so_far(
    output_dir: str,
    best_config: Dict[str, object],
    best_metric: float,
) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    blob = {
        "saved_at": pd.Timestamp.now().isoformat(),
        "best_metric": float(best_metric),
        "best_config": best_config,
    }
    with open(best_so_far_path(output_dir), "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)


def load_best_pool(output_dir: str) -> List[Dict[str, object]]:
    path = best_pool_path(output_dir)
    if not output_dir or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        items = blob.get("items", [])
        return items if isinstance(items, list) else []
    except Exception:
        return []


def save_best_pool(output_dir: str, items: List[Dict[str, object]]) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    blob = {
        "saved_at": pd.Timestamp.now().isoformat(),
        "items": items,
    }
    with open(best_pool_path(output_dir), "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)


def update_best_pool(
    output_dir: str,
    candidate_config: Dict[str, object],
    candidate_metric: float,
    top_k: int,
) -> None:
    """将候选配置加入 Top-K 池，按 metric 降序维护，去重。"""
    items = load_best_pool(output_dir)
    cand_hash = config_hash(candidate_config)
    items = [it for it in items if it.get("hash") != cand_hash]
    items.append({"metric": float(candidate_metric), "hash": cand_hash, "config": candidate_config})
    items = sorted(items, key=lambda x: float(x.get("metric", -np.inf)), reverse=True)[: int(top_k)]
    save_best_pool(output_dir, items)