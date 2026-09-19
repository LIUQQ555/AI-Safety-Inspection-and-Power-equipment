"""时序预处理：重采样、去噪、归一化、滑动窗口。

对应技术方案 5.3 节的流程::

    原始信号 → 去噪/归一化 → 滑动窗口 → 时域特征 + FFT + STFT → 模型

归一化统计量（均值/标准差）必须随模型一起保存，推理时复用训练集统计量，
否则训练与推理的尺度不一致会导致误报。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from backend.timeseries.signal_loader import SignalData

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# 重采样
# ---------------------------------------------------------------------------
def resample_frame(
    frame: pd.DataFrame,
    interval_min: float,
    aggregation: str = "mean",
) -> pd.DataFrame:
    """按固定间隔重采样（仅当索引为 DatetimeIndex 时生效）。"""
    if not isinstance(frame.index, pd.DatetimeIndex) or interval_min <= 0:
        return frame

    rule = f"{int(round(interval_min))}min"
    resampled = frame.resample(rule).agg(aggregation)
    before = len(resampled)
    resampled = resampled.interpolate(method="linear", limit_direction="both").ffill().bfill()
    if before != len(frame):
        logger.debug("重采样：%d → %d 行（间隔 %s）", len(frame), len(resampled), rule)
    return resampled


# ---------------------------------------------------------------------------
# 去噪
# ---------------------------------------------------------------------------
def moving_average(values: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """沿指定轴做中心移动平均。window<=1 时原样返回。

    使用累积和实现，复杂度 O(N)，边缘用复制填充以保持长度不变。
    """
    if window <= 1:
        return values
    window = int(window)
    if window % 2 == 0:
        window += 1

    pad = window // 2
    pad_width = [(0, 0)] * values.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(values, pad_width, mode="edge")

    kernel = np.ones(window, dtype=np.float64) / window
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), axis, padded
    )


def denoise(values: np.ndarray, window: int = 5) -> np.ndarray:
    """对时序数组去噪，输入形状 ``(T, C)``。"""
    if values.ndim != 2:
        raise ValueError(f"期望形状 (T, C)，实际 {values.shape}")
    return moving_average(values, window, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------
@dataclass
class Normalizer:
    """按通道标准化。fit 得到的统计量需与模型一同持久化。"""

    method: str = "zscore"                     # zscore | minmax | none
    channels: List[str] = field(default_factory=list)
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None
    min_: Optional[np.ndarray] = None
    max_: Optional[np.ndarray] = None

    def fit(self, values: np.ndarray, channels: Sequence[str] = ()) -> "Normalizer":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"期望形状 (T, C)，实际 {values.shape}")
        self.channels = list(channels)

        if self.method == "zscore":
            self.mean_ = values.mean(axis=0)
            # 常值通道标准差为 0，用 1 代替避免除零
            self.std_ = np.where(values.std(axis=0) < 1e-8, 1.0, values.std(axis=0))
        elif self.method == "minmax":
            self.min_ = values.min(axis=0)
            span = values.max(axis=0) - self.min_
            self.max_ = np.where(span < 1e-8, self.min_ + 1.0, values.max(axis=0))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if self.method == "zscore":
            if self.mean_ is None or self.std_ is None:
                raise RuntimeError("Normalizer 尚未 fit")
            return ((values - self.mean_) / self.std_).astype(np.float32)
        if self.method == "minmax":
            if self.min_ is None or self.max_ is None:
                raise RuntimeError("Normalizer 尚未 fit")
            return ((values - self.min_) / (self.max_ - self.min_)).astype(np.float32)
        return values

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """还原到原始量纲，用于把预测/重建结果画回物理单位。"""
        values = np.asarray(values, dtype=np.float32)
        if self.method == "zscore":
            if self.mean_ is None or self.std_ is None:
                raise RuntimeError("Normalizer 尚未 fit")
            return (values * self.std_ + self.mean_).astype(np.float32)
        if self.method == "minmax":
            if self.min_ is None or self.max_ is None:
                raise RuntimeError("Normalizer 尚未 fit")
            return (values * (self.max_ - self.min_) + self.min_).astype(np.float32)
        return values

    def to_dict(self) -> Dict:
        def conv(array: Optional[np.ndarray]) -> Optional[list]:
            return None if array is None else [float(v) for v in np.asarray(array).ravel()]

        return {
            "method": self.method,
            "channels": list(self.channels),
            "mean": conv(self.mean_),
            "std": conv(self.std_),
            "min": conv(self.min_),
            "max": conv(self.max_),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Normalizer":
        normalizer = cls(method=str(data.get("method", "zscore")),
                         channels=list(data.get("channels") or []))

        def conv(key: str) -> Optional[np.ndarray]:
            raw = data.get(key)
            return None if raw is None else np.asarray(raw, dtype=np.float32)

        normalizer.mean_ = conv("mean")
        normalizer.std_ = conv("std")
        normalizer.min_ = conv("min")
        normalizer.max_ = conv("max")
        return normalizer

    def save(self, path: PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: PathLike) -> "Normalizer":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def clip_extremes(values: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """按 sigma 倍标准差裁剪极端值，抑制传感器尖峰对统计量的影响。"""
    if sigma <= 0:
        return values
    mean = values.mean(axis=0)
    std = np.where(values.std(axis=0) < 1e-8, 1.0, values.std(axis=0))
    lower = mean - sigma * std
    upper = mean + sigma * std
    return np.clip(values, lower, upper).astype(np.float32)


# ---------------------------------------------------------------------------
# 滑动窗口
# ---------------------------------------------------------------------------
def make_windows(
    values: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """把 ``(T, C)`` 切为 ``(N, window_size, C)`` 的窗口。

    使用 ``sliding_window_view`` 实现零拷贝切分。若样本数不足一个窗口，
    返回形状为 ``(0, window_size, C)`` 的空数组。
    """
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"期望形状 (T, C)，实际 {values.shape}")
    if window_size <= 0 or stride <= 0:
        raise ValueError(f"window_size 与 stride 必须为正数，收到 {window_size}, {stride}")

    n_samples = values.shape[0]
    if n_samples < window_size:
        return np.empty((0, window_size, values.shape[1]), dtype=values.dtype)

    view = np.lib.stride_tricks.sliding_window_view(values, window_size, axis=0)
    # view 形状为 (T - W + 1, C, W)，调整为 (N, W, C) 后再按 stride 抽样
    view = np.transpose(view, (0, 2, 1))[::stride]
    return np.ascontiguousarray(view)


def window_start_indices(n_samples: int, window_size: int, stride: int) -> np.ndarray:
    """每个窗口在原始序列中的起始下标，用于把窗口映射回时间戳。"""
    if n_samples < window_size:
        return np.empty((0,), dtype=int)
    return np.arange(0, n_samples - window_size + 1, stride, dtype=int)


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
@dataclass
class PreprocessResult:
    """预处理产物。"""

    values_raw: np.ndarray                     # (T, C) 去噪后、未归一化
    values_normalized: np.ndarray              # (T, C) 归一化后
    windows: np.ndarray                        # (N, W, C) 归一化窗口
    channels: List[str]
    normalizer: Normalizer
    start_indices: np.ndarray                  # (N,) 窗口在原始序列中的起点
    timestamps: Optional[pd.DatetimeIndex] = None
    notes: List[str] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return int(self.windows.shape[0])

    def window_time_range(self, window_index: int) -> Tuple[Optional[str], Optional[str]]:
        """窗口对应的起止时间字符串。"""
        if self.timestamps is None or window_index >= len(self.start_indices):
            return None, None
        start = int(self.start_indices[window_index])
        end = start + self.windows.shape[1] - 1
        if end >= len(self.timestamps):
            end = len(self.timestamps) - 1
        return (str(self.timestamps[start]), str(self.timestamps[end]))


def preprocess_signal(
    signal: SignalData,
    window_size: int,
    stride: int,
    denoise_window: int = 5,
    normalize: str = "zscore",
    clip_sigma: float = 5.0,
    channels: Optional[Sequence[str]] = None,
) -> PreprocessResult:
    """对 ``SignalData`` 执行完整预处理流程。"""
    notes: List[str] = []
    used_channels = [c for c in (channels or signal.channels) if c in signal.frame.columns]
    if not used_channels:
        raise ValueError("没有任何可用于分析的通道")

    raw = signal.values(used_channels)
    clipped = clip_extremes(raw, clip_sigma)
    if clip_sigma > 0:
        n_clipped = int((raw != clipped).sum())
        if n_clipped:
            notes.append(f"按 {clip_sigma:.1f}σ 裁剪了 {n_clipped} 个极端值点。")

    denoised = denoise(clipped, denoise_window) if denoise_window > 1 else clipped.astype(np.float32)

    normalizer = Normalizer(method=normalize).fit(denoised, used_channels)
    normalized = normalizer.transform(denoised)

    windows = make_windows(normalized, window_size, stride)
    starts = window_start_indices(len(normalized), window_size, stride)

    if len(windows) == 0:
        notes.append(
            f"样本数 {len(normalized)} 少于窗口长度 {window_size}，无法构造窗口，"
            "时序分支将跳过模型推理。"
        )

    return PreprocessResult(
        values_raw=denoised,
        values_normalized=normalized,
        windows=windows,
        channels=used_channels,
        normalizer=normalizer,
        start_indices=starts,
        timestamps=signal.frame.index if signal.has_time_index else None,
        notes=notes,
    )


__all__ = [
    "Normalizer",
    "PreprocessResult",
    "resample_frame",
    "moving_average",
    "denoise",
    "clip_extremes",
    "make_windows",
    "window_start_indices",
    "preprocess_signal",
]
