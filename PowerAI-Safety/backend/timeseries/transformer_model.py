"""电气时序异常检测模型（Transformer 自编码器）。

对应技术方案 5.3 节与 9.5 节（``Industrial-Time-Series-AI``）。

方法：**基于重构误差的无监督异常检测**
    1. 仅用正常工况窗口训练 Transformer 自编码器；
    2. 编码器把 ``(T, C)`` 窗口压缩为单一隐向量（均值池化瓶颈），
       迫使模型学习正常工况的时序模式而非简单复制输入；
    3. 解码器由隐向量重建整个窗口；
    4. 推理时以逐窗口重构 MSE 作为异常分，超过验证集分位阈值判定异常。

选择自编码器而非分类器，是因为现场异常样本稀缺、且异常形态未知，
无监督方法不依赖故障标注，更适合早期落地。

模型权重与环境量纲（Normalizer）分别持久化：``model.pt`` 存 state_dict，
``meta.json`` 存结构超参与归一化统计量。
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少 PyTorch。本项目从 Anaconda base 继承 torch；若未安装，请执行："
        "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc

from backend.timeseries.preprocessing import Normalizer

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def seed_everything(seed: int) -> None:
    """固定随机种子，保证训练与评测可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - 本机为 CPU
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 模型结构
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """正弦位置编码。

    Transformer 本身不含位置信息，而时序数据的先后顺序是关键特征，
    必须显式注入位置编码。
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为 buffer：随模型迁移设备，但不作为可训练参数
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, T, D)"""
        return self.dropout(x + self.pe[:, : x.size(1)])


class TimeseriesAutoencoder(nn.Module):
    """Transformer 自编码器。

    ``(B, T, C) → 线性投影 → +位置编码 → 编码器 → 均值池化瓶颈(Latent)
      → 广播回 T 步 → +位置编码 → 解码器 → 线性输出 → (B, T, C)``
    """

    def __init__(
        self,
        n_channels: int,
        window_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # d_model 必须能被 nhead 整除，否则 MultiheadAttention 报错
        if d_model % nhead != 0:
            adjusted = max(nhead, (d_model // nhead) * nhead)
            logger.warning(
                "d_model(%d) 不能被 nhead(%d) 整除，自动调整为 %d", d_model, nhead, adjusted
            )
            d_model = adjusted

        self.n_channels = n_channels
        self.window_size = window_size
        self.d_model = d_model

        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=window_size + 8, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,       # Pre-LN 在小数据集上更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.latent_proj = nn.Linear(d_model, d_model)
        self.pos_decoder = PositionalEncoding(d_model, max_len=window_size + 8, dropout=dropout)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, n_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, C)`` → 隐向量 ``(B, D)``。"""
        hidden = self.pos_encoder(self.input_proj(x))
        encoded = self.encoder(hidden)
        return torch.tanh(self.latent_proj(encoded.mean(dim=1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        repeated = latent.unsqueeze(1).expand(-1, x.size(1), -1)
        decoded = self.decoder(self.pos_decoder(repeated))
        return self.output_proj(decoded)


# ---------------------------------------------------------------------------
# 训练与推理封装
# ---------------------------------------------------------------------------
@dataclass
class TrainingHistory:
    """训练过程记录，用于报告与问题排查。"""

    epochs: int = 0
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    threshold: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "train_losses": [round(v, 6) for v in self.train_losses],
            "val_losses": [round(v, 6) for v in self.val_losses],
            "best_val_loss": round(self.best_val_loss, 6),
            "threshold": round(self.threshold, 6),
        }


@dataclass
class ModelMetadata:
    """与权重一同持久化的元信息。"""

    channels: List[str]
    window_size: int
    n_channels: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    threshold: float
    normalizer: Dict[str, Any]
    train_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channels": self.channels,
            "window_size": self.window_size,
            "n_channels": self.n_channels,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "threshold": self.threshold,
            "normalizer": self.normalizer,
            "train_metrics": self.train_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        return cls(
            channels=list(data["channels"]),
            window_size=int(data["window_size"]),
            n_channels=int(data["n_channels"]),
            d_model=int(data["d_model"]),
            nhead=int(data["nhead"]),
            num_layers=int(data["num_layers"]),
            dim_feedforward=int(data["dim_feedforward"]),
            dropout=float(data["dropout"]),
            threshold=float(data["threshold"]),
            normalizer=dict(data.get("normalizer") or {}),
            train_metrics=dict(data.get("train_metrics") or {}),
        )


class TransformerAnomalyDetector:
    """Transformer 自编码器异常检测器的训练与推理封装。"""

    def __init__(
        self,
        window_size: int = 96,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self.window_size = int(window_size)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.seed = int(seed)
        self.device = torch.device(device)

        self.model: Optional[TimeseriesAutoencoder] = None
        self.metadata: Optional[ModelMetadata] = None
        self.history = TrainingHistory()

    # -- 构建 -----------------------------------------------------------
    @classmethod
    def from_config(cls, config) -> "TransformerAnomalyDetector":
        """按 ``config.timeseries`` 构建。"""
        model_cfg = config.timeseries.model
        return cls(
            window_size=int(config.timeseries.window_size),
            d_model=int(model_cfg.get("d_model", 64)),
            nhead=int(model_cfg.get("nhead", 4)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 128)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            seed=int(model_cfg.get("seed", 42)),
            device="cpu",
        )

    def build(self, n_channels: int) -> TimeseriesAutoencoder:
        seed_everything(self.seed)
        self.model = TimeseriesAutoencoder(
            n_channels=n_channels,
            window_size=self.window_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)
        return self.model

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    # -- 训练 -----------------------------------------------------------
    def fit(
        self,
        windows: np.ndarray,
        channels: List[str],
        normalizer: Normalizer,
        epochs: int = 40,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        error_percentile: float = 95.0,
        val_ratio: float = 0.2,
        verbose: bool = False,
    ) -> TrainingHistory:
        """在正常工况窗口上训练，并用验证集误差分位确定异常阈值。

        参数
        ----
        windows : ``(N, W, C)`` 归一化后的训练窗口
        channels : 通道名，写入元信息
        normalizer : 已 fit 的归一化器，随模型保存
        error_percentile : 验证集重构误差的该分位数作为阈值
        val_ratio : 验证集比例（从序列末尾切分，避免时序泄漏到训练集）
        """
        if windows.ndim != 3:
            raise ValueError(f"期望形状 (N, W, C)，实际 {windows.shape}")
        n_windows, window_size, n_channels = windows.shape
        if n_windows == 0:
            raise ValueError("训练窗口数为 0，请检查 window_size 与数据长度")
        if window_size != self.window_size:
            logger.warning(
                "窗口长度 %d 与模型配置 %d 不一致，以数据为准", window_size, self.window_size
            )
            self.window_size = window_size

        seed_everything(self.seed)
        model = self.build(n_channels)
        model.train()

        tensor = torch.from_numpy(np.ascontiguousarray(windows)).float()

        # 时序数据不能随机打乱切分，否则验证集与训练集高度相关
        n_val = int(round(n_windows * val_ratio))
        if n_windows - n_val < 8:
            # 样本太少时不做验证集切分，全部用于训练
            n_val = 0
            logger.warning("训练窗口仅 %d 个，跳过验证集切分", n_windows)
        n_train = n_windows - n_val

        train_tensor = tensor[:n_train]
        val_tensor = tensor[n_train:] if n_val > 0 else tensor[:n_train]

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_tensor),
            batch_size=min(batch_size, max(1, n_train)),
            shuffle=True,
            drop_last=False,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )
        criterion = nn.MSELoss()

        history = TrainingHistory()
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction = model(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                # 梯度裁剪：Transformer 在深层易出现梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            scheduler.step()

            train_loss = epoch_loss / max(1, n_batches)
            val_loss = self._reconstruction_error(val_tensor)
            history.train_losses.append(round(train_loss, 6))
            history.val_losses.append(round(val_loss, 6))

            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            if verbose and (epoch % 10 == 0 or epoch == epochs):
                logger.info(
                    "epoch %3d/%d  train=%.5f  val=%.5f", epoch, epochs, train_loss, val_loss
                )

        model.load_state_dict(best_state)
        model.eval()

        # 阈值取验证集误差分位数
        val_errors = self._per_window_errors(val_tensor)
        threshold = float(np.percentile(val_errors, error_percentile)) if len(val_errors) else 0.0
        # 阈值过低会让正常波动被判为异常，给一个下限
        threshold = max(threshold, 1e-6)

        history.epochs = epochs
        history.threshold = threshold
        self.history = history

        self.metadata = ModelMetadata(
            channels=list(channels),
            window_size=window_size,
            n_channels=n_channels,
            d_model=model.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            threshold=threshold,
            normalizer=normalizer.to_dict(),
            train_metrics={
                "n_train_windows": int(n_train),
                "n_val_windows": int(n_val),
                "final_train_loss": history.train_losses[-1] if history.train_losses else None,
                "best_val_loss": round(history.best_val_loss, 6),
                "threshold_percentile": error_percentile,
            },
        )
        return history

    # -- 推理 -----------------------------------------------------------
    def _reconstruction_error(self, tensor: torch.Tensor) -> float:
        if self.model is None or len(tensor) == 0:
            return 0.0
        return float(self._per_window_errors(tensor).mean())

    @torch.no_grad()
    def _per_window_errors(self, tensor: torch.Tensor, batch_size: int = 128) -> np.ndarray:
        """逐窗口重构 MSE，返回 ``(N,)``。"""
        if self.model is None or len(tensor) == 0:
            return np.zeros(0, dtype=np.float32)

        self.model.eval()
        errors: List[np.ndarray] = []
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start:start + batch_size].to(self.device)
            reconstruction = self.model(batch)
            # 对时间与通道维求均值，得到每个窗口一个标量
            mse = ((reconstruction - batch) ** 2).mean(dim=(1, 2))
            errors.append(mse.cpu().numpy())
        return np.concatenate(errors).astype(np.float32)

    def score(self, windows: np.ndarray) -> np.ndarray:
        """对窗口打分，返回逐窗口重构误差。"""
        if self.model is None:
            raise RuntimeError("模型尚未训练或加载")
        if windows.ndim != 3 or len(windows) == 0:
            return np.zeros(0, dtype=np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(windows)).float()
        return self._per_window_errors(tensor)

    @property
    def threshold(self) -> float:
        return float(self.metadata.threshold) if self.metadata else 0.0

    # -- 持久化 ---------------------------------------------------------
    def save(self, directory: PathLike) -> Tuple[Path, Path]:
        """保存为 ``model.pt``（权重）+ ``meta.json``（结构与阈值）。"""
        if self.model is None or self.metadata is None:
            raise RuntimeError("没有可保存的模型，请先训练")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        weights_path = directory / "model.pt"
        meta_path = directory / "meta.json"

        # 分开保存权重与元信息：避免 torch.load 的 weights_only 兼容问题
        torch.save(self.model.state_dict(), weights_path)
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(self.metadata.to_dict(), fh, ensure_ascii=False, indent=2)

        logger.info("模型已保存：%s", directory)
        return weights_path, meta_path

    @classmethod
    def load(cls, directory: PathLike, device: str = "cpu") -> "TransformerAnomalyDetector":
        """从目录加载模型。"""
        directory = Path(directory)
        weights_path = directory / "model.pt"
        meta_path = directory / "meta.json"

        if not weights_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"模型文件不完整，需要同时存在 {weights_path} 与 {meta_path}"
            )

        with meta_path.open("r", encoding="utf-8") as fh:
            metadata = ModelMetadata.from_dict(json.load(fh))

        detector = cls(
            window_size=metadata.window_size,
            d_model=metadata.d_model,
            nhead=metadata.nhead,
            num_layers=metadata.num_layers,
            dim_feedforward=metadata.dim_feedforward,
            dropout=metadata.dropout,
            device=device,
        )
        detector.model = TimeseriesAutoencoder(
            n_channels=metadata.n_channels,
            window_size=metadata.window_size,
            d_model=metadata.d_model,
            nhead=metadata.nhead,
            num_layers=metadata.num_layers,
            dim_feedforward=metadata.dim_feedforward,
            dropout=metadata.dropout,
        ).to(detector.device)

        state = torch.load(weights_path, map_location=detector.device)
        detector.model.load_state_dict(state)
        detector.model.eval()
        detector.metadata = metadata
        return detector

    @property
    def normalizer(self) -> Optional[Normalizer]:
        if self.metadata is None:
            return None
        return Normalizer.from_dict(self.metadata.normalizer)


__all__ = [
    "PositionalEncoding",
    "TimeseriesAutoencoder",
    "TransformerAnomalyDetector",
    "ModelMetadata",
    "TrainingHistory",
    "seed_everything",
]
