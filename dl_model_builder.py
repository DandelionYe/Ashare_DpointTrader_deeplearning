# dl_model_builder.py
"""
PyTorch 深度学习模型构建与训练（P1/P2 修复版）。
支持 MLP、LSTM、GRU、CNN、Transformer 等模型用于时序二分类任务。

P1-1 修复：GPU 推理逐样本循环
    原版 predict_pytorch_model 对序列模型用 Python 循环逐样本调用 GPU kernel，
    1000 个样本 = 1000 次 kernel 启动 + 1000 次 GPU→CPU 搬运，完全抵消 GPU 优势。
    修复后：用 torch.Tensor.unfold 一次性构建所有滑窗序列，单次批量推理完成。

P1-2 修复：DataLoader 全量数据预置 GPU
    原版 create_sequence_dataset 在构建 DataLoader 前将全量数据 .to(device)，
    导致：
        ① pin_memory 失效（数据已在 GPU，页锁定内存无意义）
        ② num_workers 不可用（CUDA 张量在 fork 子进程里会崩溃）
        ③ 显存被全量数据长期占用，而非按批次按需使用
    修复后：数据始终保留在 CPU，训练循环内按批次用 non_blocking=True 异步转移。
    device 参数保留但不再用于数据预置（向后兼容，仅打印弃用提示）。

P1-2 顺带修复：_make_sequences Python 循环
    原版用 Python for 循环逐个滑窗赋值（即使在 GPU 上也是串行）。
    修复后：用 unfold + permute + contiguous 零拷贝向量化构建，无 Python 循环。

P2-2 修复：AMP 混合精度训练 + cuDNN benchmark 未启用
    问题一：RTX 20 系及以上显卡内置 Tensor Core，FP16 吞吐量是 FP32 的 2～3 倍，
            但原版全程 FP32，Tensor Core 完全闲置。
    修复一：在 train_pytorch_model 中加入 torch.amp.autocast + GradScaler，
            仅在 device=cuda 时启用，CPU 训练不受影响。
            GradScaler 防止 FP16 梯度下溢，unscale 后再做梯度裁剪保证数值稳定。

    问题二：torch.backends.cudnn.benchmark 默认为 False，cuDNN 不会自动选择最快
            的卷积/矩阵算法。对于输入尺寸固定的训练场景，开启后首批次会做一次
            算法搜索，此后每批次均使用最优内核。
    修复二：在 model.to(device) 之后，device=cuda 时设置 cudnn.benchmark=True。

P2-3 修复：BCELoss 与 AMP/autocast 不兼容
    原版模型在 forward() 末尾做 sigmoid，再用 nn.BCELoss() 计算损失。
    这在 CUDA AMP/autocast 下会触发：
        RuntimeError: binary_cross_entropy and BCELoss are unsafe to autocast
    修复后：
        ① 所有模型 forward() 统一输出 raw logits（不再做 sigmoid）
        ② 训练损失改为 nn.BCEWithLogitsLoss()
        ③ 推理阶段再用 torch.sigmoid(logits) 转为概率
    这是 PyTorch 官方推荐的二分类标准写法，数值更稳定，也兼容混合精度。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "MLP",
    "LSTM",
    "GRU",
    "CNN1D",
    "Transformer",
    "_get_device",
    "train_pytorch_model",
    "predict_pytorch_model",
    "create_sequence_dataset",
]


# =========================================================
# 设备检测
# =========================================================
def _get_device() -> torch.device:
    """自动检测并返回可用的计算设备（CUDA 或 CPU）。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 序列数据构建工具
# =========================================================
def create_sequence_dataset(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    seq_len: int = 20,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),  # 保留以向后兼容，不再用于预置数据
    shuffle: bool = True,
) -> DataLoader:
    """
    将 DataFrame 转换为 PyTorch DataLoader，支持序列数据。

    修复后：
        数据始终保留在 CPU 端，由训练循环负责按批次搬运到 GPU。
        这是 PyTorch 官方推荐的正确模式，可以同时启用：
            pin_memory=True   — 页锁定内存，GPU 异步拷贝更快
            num_workers=0     — CUDA 环境下必须为 0，避免 fork 崩溃

    Args:
        X: 特征 DataFrame (n_samples, n_features)
        y: 标签 Series（可选）
        seq_len: 序列长度（用于 LSTM/GRU/Transformer/CNN）
        batch_size: 批量大小
        device: 已弃用，保留以向后兼容，实际不再影响数据位置
        shuffle: 是否打乱数据

    Returns:
        DataLoader（数据在 CPU 端，pin_memory=True）
    """
    _ = device  # 向后兼容占位，避免 lint 告警

    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    if y is not None:
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    else:
        y_tensor = None

    if seq_len > 1:
        X_seq, y_seq = _make_sequences(X_tensor, y_tensor, seq_len)
        if y_seq is not None:
            dataset = TensorDataset(X_seq, y_seq)
        else:
            dataset = TensorDataset(X_seq)
    else:
        if y_tensor is not None:
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=0,
    )


def _make_sequences(
    X: torch.Tensor,
    y: Optional[torch.Tensor],
    seq_len: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    将时间序列数据转换为滑动窗口格式。

    修复后：
        用 unfold + permute 零拷贝向量化构建，无 Python 循环。

    Args:
        X: 特征张量 (N, n_features)，应在 CPU 上
        y: 标签张量 (N, 1)，可为 None
        seq_len: 序列长度

    Returns:
        X_seq: (N-seq_len+1, seq_len, n_features)
        y_seq: (N-seq_len+1, 1) 或 None
    """
    n_samples = X.shape[0]
    n_sequences = n_samples - seq_len + 1

    if n_sequences <= 0:
        raise ValueError(
            f"seq_len ({seq_len}) 超过样本数 ({n_samples})，无法构建序列。"
            f"请减小 seq_len 或增加数据量。"
        )

    X_seq = X.unfold(0, seq_len, 1).permute(0, 2, 1).contiguous()
    y_seq = y[seq_len - 1:n_samples] if y is not None else None

    return X_seq, y_seq


def _logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """将 raw logits 转为 [0, 1] 概率。"""
    return torch.sigmoid(logits)


# =========================================================
# 模型定义
# 说明：所有模型 forward() 统一返回 raw logits，不再在内部做 sigmoid。
#       训练时配合 BCEWithLogitsLoss，预测时再显式做 sigmoid。
# =========================================================
class MLP(nn.Module):
    """简单的多层感知机，用于二分类任务（输出 raw logits）。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, dropout_rate: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LSTM(nn.Module):
    """
    LSTM（长短期记忆网络）用于时序二分类。
    支持多层 LSTM、双向 LSTM、Dropout 正则化。
    输出 raw logits。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout_rate: float = 0.3,
        bidirectional: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout_rate)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            last_output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            last_output = h_n[-1, :, :]
        x = self.dropout(last_output)
        x = self.fc(x)
        return x


class GRU(nn.Module):
    """
    GRU（门控循环单元）用于时序二分类。
    比 LSTM 更轻量，参数更少。
    输出 raw logits。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout_rate: float = 0.3,
        bidirectional: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout_rate)
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(gru_output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        if self.bidirectional:
            last_output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            last_output = h_n[-1, :, :]
        x = self.dropout(last_output)
        x = self.fc(x)
        return x


class CNN1D(nn.Module):
    """
    一维卷积神经网络（CNN）用于时序二分类。
    使用多个卷积核尺寸捕捉不同时间尺度的模式。
    输出 raw logits。
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        num_filters: int = 64,
        kernel_sizes: Optional[list] = None,
        output_dim: int = 1,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        _ = seq_len  # 目前结构里未直接使用，保留参数以兼容旧调用

        if kernel_sizes is None:
            kernel_sizes = [2, 3, 5]

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        total_features = num_filters * len(kernel_sizes)
        self.fc = nn.Linear(total_features, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            conv_out = self.relu(conv(x))
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)

        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Transformer(nn.Module):
    """
    Transformer Encoder 用于时序二分类。
    使用自注意力机制捕捉长距离依赖关系。
    输出 raw logits。
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout_rate: float = 0.1,
        output_dim: int = 1,
        max_seq_len: int = 100,
    ):
        super().__init__()

        self.d_model = d_model
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(d_model, output_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    """Transformer 位置编码。"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =========================================================
# 训练函数
# =========================================================
def train_pytorch_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """
    训练 PyTorch 模型（支持 MLP/LSTM/GRU/CNN/Transformer）。

    关键修复：
        1. 数据不再预置到 GPU，而是在训练循环中逐 batch 搬运
        2. CUDA 下启用 AMP 混合精度和 GradScaler
        3. 二分类损失从 BCELoss 改为 BCEWithLogitsLoss
        4. 模型输出 raw logits，避免 BCELoss + sigmoid 在 autocast 下报错

    Args:
        X_train: 训练特征 DataFrame
        y_train: 训练标签 Series
        config: 模型配置字典
        device: 计算设备

    Returns:
        训练好的模型
    """
    model_type = str(config.get("model_type", "mlp")).lower()
    input_dim = X_train.shape[1]

    hidden_dim = int(config.get("hidden_dim", 64))
    d_model = int(config.get("d_model", 64))
    learning_rate = float(config.get("learning_rate", 0.001))
    epochs = int(config.get("epochs", 20))
    batch_size = int(config.get("batch_size", 64))
    dropout_rate = float(config.get("dropout_rate", 0.3))
    seq_len = int(config.get("seq_len", 20))

    if model_type == "mlp":
        model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        train_loader = create_sequence_dataset(X_train, y_train, seq_len=1, batch_size=batch_size)

    elif model_type == "lstm":
        num_layers = int(config.get("num_layers", 2))
        bidirectional = bool(config.get("bidirectional", False))
        model = LSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
        )
        train_loader = create_sequence_dataset(X_train, y_train, seq_len=seq_len, batch_size=batch_size)

    elif model_type == "gru":
        num_layers = int(config.get("num_layers", 2))
        bidirectional = bool(config.get("bidirectional", False))
        model = GRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
        )
        train_loader = create_sequence_dataset(X_train, y_train, seq_len=seq_len, batch_size=batch_size)

    elif model_type == "cnn":
        num_filters = int(config.get("num_filters", 64))
        kernel_sizes = config.get("kernel_sizes", [2, 3, 5])
        model = CNN1D(
            input_dim=input_dim,
            seq_len=seq_len,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            dropout_rate=dropout_rate,
        )
        train_loader = create_sequence_dataset(X_train, y_train, seq_len=seq_len, batch_size=batch_size)

    elif model_type == "transformer":
        nhead = int(config.get("nhead", 4))
        num_layers = int(config.get("num_layers", 2))
        dim_feedforward = int(config.get("dim_feedforward", 128))
        model = Transformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
        )
        train_loader = create_sequence_dataset(X_train, y_train, seq_len=seq_len, batch_size=batch_size)

    else:
        raise ValueError(f"未知的 model_type: {model_type}")

    model = model.to(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 关键修复：AMP/autocast 下必须使用 BCEWithLogitsLoss，而不是 BCELoss
    criterion = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    use_amp: bool = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    model.train()
    best_loss = float("inf")
    patience_counter = 0
    max_patience = 10

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            if len(batch) < 2:
                continue

            X_batch, y_batch = batch

            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                break

    return model


# =========================================================
# 推理函数
# =========================================================
def predict_pytorch_model(
    model: nn.Module,
    X: pd.DataFrame,
    device: torch.device,
    seq_len: int = 20,
) -> pd.Series:
    """
    使用 PyTorch 模型进行批量推理。

    修复后：
        - 模型 forward() 返回 raw logits
        - predict 阶段统一调用 sigmoid 转成概率
        - 序列模型继续使用 unfold 一次性批量构建滑窗

    Args:
        model: 训练好的模型
        X: 特征 DataFrame
        device: 计算设备
        seq_len: 序列长度（用于 LSTM/GRU/CNN/Transformer）

    Returns:
        预测概率 Series（序列模型的 index 从第 seq_len-1 个样本开始）
    """
    model.eval()

    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    with torch.no_grad():
        if isinstance(model, (LSTM, GRU, CNN1D, Transformer)):
            n_samples = X_tensor.shape[0]
            if n_samples - seq_len + 1 <= 0:
                raise ValueError(
                    f"seq_len ({seq_len}) 超过样本数 ({n_samples})，无法进行序列推理。"
                )

            X_seq_all = X_tensor.unfold(0, seq_len, 1).permute(0, 2, 1).contiguous()
            X_seq_all = X_seq_all.to(device, non_blocking=True)

            logits = model(X_seq_all)
            probs = _logits_to_probs(logits).cpu().numpy().flatten()

            valid_indices = X.index[seq_len - 1:]
            return pd.Series(probs, index=valid_indices, name="dpoint")

        #分批推理，固定 batch_size=1024，显存占用上限可控
        else:
            batch_size = 1024
            all_probs = []
            for start in range(0, X_tensor.shape[0], batch_size):
                batch = X_tensor[start : start + batch_size].to(device, non_blocking=True)
                logits = model(batch)
                all_probs.append(_logits_to_probs(logits).cpu())
            probs = torch.cat(all_probs, dim=0).numpy().flatten()
            return pd.Series(probs, index=X.index, name="dpoint")