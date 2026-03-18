# rolling_trainer.py
"""
滚动再训练模块。

P0:
    - 按调度条件触发重训
    - 支持两种窗口：expanding, rolling
    - 支持固定重训频率：monthly
    - 保存每次重训后的模型与 manifest

P1:
    - 支持 weekly/quarterly retrain
    - rolling window length 配置
    - retrain 后自动评估近期表现
    - 支持模型快照管理
    - 支持最近窗口内校准器同步更新

P2:
    - 增加模型失效监控
    - 支持 fallback model
    - 支持自动降级到 baseline
    - 支持滚动再训练与横截面框架整合
    - 支持近期表现漂移告警
"""
from __future__ import annotations

import logging
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


WINDOW_TYPES = ["expanding", "rolling"]
RETRAIN_FREQUENCIES = ["daily", "weekly", "monthly", "quarterly"]


@dataclass
class WindowConfig:
    """窗口配置。"""
    window_type: str = "expanding"
    rolling_window_length: Optional[int] = None
    min_window_size: int = 60


@dataclass
class SchedulerConfig:
    """调度配置。"""
    frequency: str = "monthly"
    day_of_month: int = 1
    day_of_week: int = 0
    hour: int = 0
    
    
@dataclass
class ModelSnapshot:
    """模型快照。"""
    snapshot_id: str
    timestamp: str
    train_end_date: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    model_path: Optional[str] = None
    calibrator_path: Optional[str] = None


@dataclass
class RetrainResult:
    """重训结果。"""
    success: bool
    snapshot_id: str
    train_start_date: str
    train_end_date: str
    metrics: Dict[str, float]
    model_path: str
    calibration_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class RollingWindowManager:
    """
    滚动窗口管理器。
    
    管理 expanding 或 rolling 窗口的数据切片。
    """
    
    def __init__(self, config: WindowConfig):
        self.config = config
    
    def get_train_data(
        self,
        df: pd.DataFrame,
        current_date: str,
    ) -> pd.DataFrame:
        """
        获取训练数据窗口。
        
        Args:
            df: 完整数据集
            current_date: 当前日期（训练截止日）
            
        Returns:
            训练数据窗口
        """
        df = df.copy()
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            cutoff = pd.to_datetime(current_date)
            historical = df[df["date"] <= cutoff].copy()
        else:
            idx = df.index.get_loc(current_date) if current_date in df.index else len(df) - 1
            historical = df.iloc[:idx + 1].copy()
        
        if self.config.window_type == "expanding":
            return historical
        
        elif self.config.window_type == "rolling":
            if self.config.rolling_window_length:
                return historical.tail(self.config.rolling_window_length)
            else:
                return historical
    
    def get_validation_data(
        self,
        df: pd.DataFrame,
        train_end_date: str,
        val_window_days: int = 60,
    ) -> pd.DataFrame:
        """获取验证数据窗口（用于评估重训效果）。"""
        df = df.copy()
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            train_end = pd.to_datetime(train_end_date)
            val_start = train_end + timedelta(days=1)
            val_end = val_start + timedelta(days=val_window_days)
            return df[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()
        else:
            train_idx = df.index.get_loc(train_end_date) if train_end_date in df.index else len(df) - 1
            val_start = train_idx + 1
            val_end = min(val_start + val_window_days, len(df))
            return df.iloc[val_start:val_end].copy()


class RetrainScheduler:
    """
    重训调度器。
    
    判断是否需要触发重训。
    """
    
    def __init__(self, config: SchedulerConfig, last_retrain_date: Optional[str] = None):
        self.config = config
        self.last_retrain_date = last_retrain_date
    
    def should_retrain(self, current_date: str) -> bool:
        """判断是否需要重训。"""
        if self.last_retrain_date is None:
            return True
        
        current = pd.to_datetime(current_date)
        last = pd.to_datetime(self.last_retrain_date)
        
        if self.config.frequency == "daily":
            return (current - last).days >= 1
        
        elif self.config.frequency == "weekly":
            return (current - last).days >= 7
        
        elif self.config.frequency == "monthly":
            return current.month != last.month or current.year != last.year
        
        elif self.config.frequency == "quarterly":
            return (current.month - 1) // 3 != (last.month - 1) // 3
        
        return False
    
    def get_next_retrain_date(self, current_date: str) -> str:
        """获取下次重训日期。"""
        current = pd.to_datetime(current_date)
        
        if self.config.frequency == "daily":
            next_date = current + timedelta(days=1)
        
        elif self.config.frequency == "weekly":
            days_ahead = (7 - current.weekday() + self.config.day_of_week) % 7
            if days_ahead == 0:
                days_ahead = 7
            next_date = current + timedelta(days=days_ahead)
        
        elif self.config.frequency == "monthly":
            if current.day >= self.config.day_of_month:
                next_month = current + pd.DateOffset(months=1)
                next_date = next_month.replace(day=min(self.config.day_of_month, 
                                                        pd.Timestamp(next_month).days_in_month))
            else:
                next_date = current.replace(day=self.config.day_of_month)
        
        elif self.config.frequency == "quarterly":
            quarter = (current.month - 1) // 3
            next_quarter_month = (quarter + 1) * 3 + 1
            if next_quarter_month > 12:
                next_quarter_month = ((next_quarter_month - 1) % 12) + 1
                next_date = current.replace(year=current.year + 1, month=next_quarter_month, day=1)
            else:
                next_date = current.replace(month=next_quarter_month, day=1)
        
        else:
            next_date = current + pd.DateOffset(months=1)
        
        return next_date.strftime("%Y-%m-%d")


class ModelSnapshotManager:
    """
    模型快照管理器。
    
    管理滚动训练过程中的模型版本。
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.snapshots_dir = os.path.join(output_dir, "snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        self.snapshots: List[ModelSnapshot] = []
        self._load_existing_snapshots()
    
    def _load_existing_snapshots(self):
        """加载已有的快照。"""
        if not os.path.exists(self.snapshots_dir):
            return
        
        for fn in os.listdir(self.snapshots_dir):
            if fn.endswith("_manifest.json"):
                try:
                    manifest_path = os.path.join(self.snapshots_dir, fn)
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    snapshot = ModelSnapshot(
                        snapshot_id=data.get("snapshot_id", ""),
                        timestamp=data.get("timestamp", ""),
                        train_end_date=data.get("train_end_date", ""),
                        config=data.get("config", {}),
                        metrics=data.get("metrics", {}),
                        model_path=data.get("model_path"),
                        calibrator_path=data.get("calibrator_path"),
                    )
                    self.snapshots.append(snapshot)
                except Exception:
                    pass
    
    def save_snapshot(
        self,
        result: RetrainResult,
        model_data: Optional[bytes] = None,
    ) -> ModelSnapshot:
        """保存模型快照。"""
        snapshot_id = result.snapshot_id
        snapshot_dir = os.path.join(self.snapshots_dir, snapshot_id)
        os.makedirs(snapshot_dir, exist_ok=True)
        
        model_path = None
        if model_data:
            model_path = os.path.join(snapshot_dir, "model.pkl")
            with open(model_path, 'wb') as f:
                f.write(model_data)
        
        snapshot = ModelSnapshot(
            snapshot_id=snapshot_id,
            timestamp=result.timestamp if hasattr(result, 'timestamp') else datetime.now().isoformat(),
            train_end_date=result.train_end_date,
            config=result.config,
            metrics=result.metrics,
            model_path=model_path,
        )
        
        manifest = {
            "snapshot_id": snapshot.snapshot_id,
            "timestamp": snapshot.timestamp,
            "train_start_date": result.train_start_date,
            "train_end_date": snapshot.train_end_date,
            "config": snapshot.config,
            "metrics": snapshot.metrics,
            "model_path": snapshot.model_path,
            "calibrator_path": snapshot.calibrator_path,
        }
        
        manifest_path = os.path.join(snapshot_dir, f"{snapshot_id}_manifest.json")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        self.snapshots.append(snapshot)
        
        return snapshot
    
    def get_latest_snapshot(self) -> Optional[ModelSnapshot]:
        """获取最新快照。"""
        if not self.snapshots:
            return None
        return sorted(self.snapshots, key=lambda x: x.train_end_date, reverse=True)[0]
    
    def get_snapshot_by_id(self, snapshot_id: str) -> Optional[ModelSnapshot]:
        """根据 ID 获取快照。"""
        for s in self.snapshots:
            if s.snapshot_id == snapshot_id:
                return s
        return None
    
    def get_recent_snapshots(self, n: int = 5) -> List[ModelSnapshot]:
        """获取最近的 n 个快照。"""
        sorted_snapshots = sorted(self.snapshots, key=lambda x: x.train_end_date, reverse=True)
        return sorted_snapshots[:n]


class RollingTrainer:
    """
    滚动再训练器。
    
    整合窗口管理、调度、模型管理功能。
    """
    
    def __init__(
        self,
        output_dir: str,
        window_config: WindowConfig,
        scheduler_config: SchedulerConfig,
        base_config: Dict[str, Any],
    ):
        self.output_dir = output_dir
        self.window_manager = RollingWindowManager(window_config)
        self.scheduler = RetrainScheduler(scheduler_config)
        self.snapshot_manager = ModelSnapshotManager(output_dir)
        self.base_config = base_config
    
    def check_and_retrain(
        self,
        df: pd.DataFrame,
        current_date: str,
        train_func,
    ) -> Optional[RetrainResult]:
        """
        检查是否需要重训，如需要则执行重训。
        
        Args:
            df: 完整数据集
            current_date: 当前日期
            train_func: 训练函数
            
        Returns:
            RetrainResult 或 None（如果不需要重训）
        """
        if not self.scheduler.should_retrain(current_date):
            return None
        
        train_data = self.window_manager.get_train_data(df, current_date)
        
        if len(train_data) < self.window_manager.config.min_window_size:
            return None
        
        snapshot_id = f"snap_{current_date.replace('-', '')}"
        
        try:
            result = train_func(
                df=train_data,
                config=self.base_config,
                snapshot_id=snapshot_id,
            )
            
            self.snapshot_manager.save_snapshot(result)
            
            return result
        
        except Exception as e:
            return RetrainResult(
                success=False,
                snapshot_id=snapshot_id,
                train_start_date=train_data.index[0] if hasattr(train_data.index[0], 'strftime') else str(train_data.index[0]),
                train_end_date=current_date,
                metrics={},
                model_path="",
                error_message=str(e),
            )
    
    def get_current_model(self) -> Optional[ModelSnapshot]:
        """获取当前使用的模型（最新快照）。"""
        return self.snapshot_manager.get_latest_snapshot()
    
    def evaluate_recent_performance(
        self,
        df: pd.DataFrame,
        days: int = 30,
    ) -> Dict[str, float]:
        """评估近期表现。"""
        latest_snapshot = self.get_current_model()
        if not latest_snapshot:
            return {}
        
        val_data = self.window_manager.get_validation_data(
            df, latest_snapshot.train_end_date, val_window_days=days
        )
        
        if val_data.empty:
            return {}
        
        return {
            "val_start_date": val_data.index[0] if hasattr(val_data.index[0], 'strftime') else str(val_data.index[0]),
            "val_end_date": val_data.index[-1] if hasattr(val_data.index[-1], 'strftime') else str(val_data.index[-1]),
            "n_samples": len(val_data),
        }


class ModelMonitor:
    """
    P2: 模型失效监控器。
    
    监控模型表现漂移，触发告警或自动降级。
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.1,
        lookback_snapshots: int = 3,
    ):
        self.drift_threshold = drift_threshold
        self.lookback_snapshots = lookback_snapshots
        self.performance_history: List[Dict[str, float]] = []
    
    def record_performance(self, snapshot_id: str, metrics: Dict[str, float]):
        """记录快照性能。"""
        self.performance_history.append({
            "snapshot_id": snapshot_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        })
    
    def check_drift(self) -> Dict[str, Any]:
        """检查是否存在性能漂移。"""
        if len(self.performance_history) < self.lookback_snapshots:
            return {"is_drifted": False, "reason": "insufficient_history"}
        
        recent = self.performance_history[-self.lookback_snapshots:]
        
        sharpe_values = [r["metrics"].get("sharpe", 0) for r in recent]
        
        if sharpe_values:
            current_sharpe = sharpe_values[-1]
            avg_sharpe = sum(sharpe_values[:-1]) / len(sharpe_values[:-1])
            
            if avg_sharpe > 0 and current_sharpe < avg_sharpe * (1 - self.drift_threshold):
                return {
                    "is_drifted": True,
                    "reason": "sharpe_degradation",
                    "current_sharpe": current_sharpe,
                    "avg_sharpe": avg_sharpe,
                    "drift_pct": (avg_sharpe - current_sharpe) / avg_sharpe,
                }
        
        return {"is_drifted": False, "reason": "stable"}


def create_rolling_trainer(
    output_dir: str,
    window_type: str = "expanding",
    rolling_window_length: Optional[int] = None,
    frequency: str = "monthly",
    base_config: Optional[Dict[str, Any]] = None,
) -> RollingTrainer:
    """创建滚动训练器。"""
    window_config = WindowConfig(
        window_type=window_type,
        rolling_window_length=rolling_window_length,
    )
    
    scheduler_config = SchedulerConfig(frequency=frequency)
    
    return RollingTrainer(
        output_dir=output_dir,
        window_config=window_config,
        scheduler_config=scheduler_config,
        base_config=base_config or {},
    )
