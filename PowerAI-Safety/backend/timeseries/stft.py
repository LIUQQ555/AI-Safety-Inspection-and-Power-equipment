"""频域特征提取：FFT、STFT 与谐波分析。

对应技术方案 5.3 节中的「时域特征 + FFT + STFT」，以及第 9.4 节
``AI-Power-Electronics-Diagnostics`` 中使用的频域分析手段。

提供的特征用于两类目的：
    1. 作为时序模型的**附加输入**（幅值谱的顶部若干分量）
    2. 作为**可解释指标**写入报告（主频、谐波总畸变率 THD）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 基础变换
# ---------------------------------------------------------------------------
def fft_spectrum(
    values: np.ndarray,
    axis: int = 0,
    detrend: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """单边幅值谱。

    参数
    ----
    values : 输入信号，沿 ``axis`` 做变换
    detrend : 是否先减去均值（去除直流分量，避免其淹没交流分量）

    返回 ``(频率下标, 幅值)``，频率下标记为 0..N//2。
    """
    values = np.asarray(values, dtype=np.float64)
    if detrend:
        values = values - values.mean(axis=axis, keepdims=True)

    n = values.shape[axis]
    spectrum = np.fft.rfft(values, axis=axis)
    magnitude = np.abs(spectrum) * (2.0 / max(1, n))
    freqs = np.arange(magnitude.shape[axis])
    return freqs, magnitude


def fft_features(
    window: np.ndarray,
    top_k: int = 5,
    detrend: bool = True,
) -> np.ndarray:
    """把一个窗口 ``(W, C)`` 压缩为 ``(C × (top_k + 2))`` 的频域特征向量。

    每个通道提取：
        * 幅值最大的 top_k 个频率分量的幅值
        * 主频位置的归一化频率（能量重心）
        * 谱能量总和的对数
    """
    if window.ndim != 2:
        raise ValueError(f"期望形状 (W, C)，实际 {window.shape}")

    n_samples, n_channels = window.shape
    if n_samples < 4:
        return np.zeros(n_channels * (top_k + 2), dtype=np.float32)

    _, magnitude = fft_spectrum(window, axis=0, detrend=detrend)
    # magnitude: (n_freq, C)
    n_freq = magnitude.shape[0]

    features: List[np.ndarray] = []
    for channel in range(n_channels):
        spec = magnitude[:, channel]
        k = min(top_k, n_freq)
        top_values = np.sort(spec)[::-1][:k]
        if k < top_k:  # 补齐，保证特征维度固定
            top_values = np.concatenate([top_values, np.zeros(top_k - k)])

        total = float(spec.sum())
        if total > 1e-12:
            centroid = float((np.arange(n_freq) * spec).sum() / total) / max(1, n_freq - 1)
        else:
            centroid = 0.0
        log_energy = float(np.log1p(total))

        features.append(np.concatenate([top_values, [centroid, log_energy]]))

    return np.concatenate(features).astype(np.float32)


def stft(
    values: np.ndarray,
    nperseg: int = 32,
    noverlap: int = 16,
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """短时傅里叶变换。

    自行实现而不依赖 scipy，避免额外依赖。使用 Hann 窗。

    返回 ``(freqs, times, magnitude)``，``magnitude`` 形状为
    ``(n_freq, n_frames)``（对于 1 维输入）。
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"stft 期望一维输入，实际 {values.shape}")
    if nperseg <= 0:
        raise ValueError("nperseg 必须为正数")
    if noverlap >= nperseg:
        raise ValueError(f"noverlap({noverlap}) 必须小于 nperseg({nperseg})")

    n = len(values)
    if n < nperseg:
        return np.empty(0), np.empty(0), np.empty((0, 0))

    step = nperseg - noverlap
    n_frames = 1 + (n - nperseg) // step
    window = np.hanning(nperseg)

    frames = np.stack([values[i * step: i * step + nperseg] for i in range(n_frames)])
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, axis=1)
    magnitude = np.abs(spectrum).T  # (n_freq, n_frames)

    freqs = np.arange(magnitude.shape[0])
    times = (np.arange(n_frames) * step + nperseg / 2.0).astype(np.float64)
    return freqs, times, magnitude


def stft_image(
    values: np.ndarray,
    nperseg: int = 32,
    noverlap: int = 16,
) -> Optional[np.ndarray]:
    """生成 STFT 幅值谱图（dB 归一化到 0-255），用于报告插图。

    返回 ``(n_freq, n_frames)`` 的 uint8 图像；输入过短时返回 None。
    """
    _, _, magnitude = stft(values, nperseg=nperseg, noverlap=noverlap)
    if magnitude.size == 0:
        return None

    log_mag = 20.0 * np.log10(magnitude + 1e-10)
    lo, hi = float(log_mag.min()), float(log_mag.max())
    if hi - lo < 1e-8:
        return np.zeros_like(log_mag, dtype=np.uint8)
    normalized = (log_mag - lo) / (hi - lo)
    return (normalized * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 谐波分析
# ---------------------------------------------------------------------------
@dataclass
class HarmonicAnalysis:
    """谐波分析结果。"""

    fundamental_index: int
    fundamental_magnitude: float
    harmonic_magnitudes: List[float]
    thd: float                       # 总谐波畸变率（相对基波，%）
    spectral_energy: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "fundamental_index": self.fundamental_index,
            "fundamental_magnitude": round(self.fundamental_magnitude, 6),
            "thd_percent": round(self.thd, 3),
            "spectral_energy": round(self.spectral_energy, 6),
        }


def harmonic_analysis(
    values: np.ndarray,
    n_harmonics: int = 7,
    fundamental_index: Optional[int] = None,
) -> HarmonicAnalysis:
    """谐波分析。

    基波频率默认取幅值谱中除直流外的最大分量。总谐波畸变率按
    ``THD = sqrt(Σ_{h=2..n} A_h²) / A_1 × 100%`` 计算。
    """
    values = np.asarray(values, dtype=np.float64)
    freqs, magnitude = fft_spectrum(values, axis=0, detrend=True)
    spec = magnitude if magnitude.ndim == 1 else magnitude[:, 0]

    energy = float(np.sum(spec ** 2))

    if len(spec) <= 1:
        return HarmonicAnalysis(0, 0.0, [], 0.0, energy)

    # 跳过直流分量（下标 0）寻找基波
    search = spec[1:]
    if fundamental_index is None:
        if search.size == 0 or search.max() <= 1e-12:
            return HarmonicAnalysis(0, 0.0, [], 0.0, energy)
        fundamental_index = int(np.argmax(search)) + 1

    fundamental = float(spec[fundamental_index])
    harmonics: List[float] = []
    for order in range(2, n_harmonics + 1):
        idx = fundamental_index * order
        if idx < len(spec):
            harmonics.append(float(spec[idx]))
        else:
            harmonics.append(0.0)

    if fundamental > 1e-12:
        thd = float(np.sqrt(np.sum(np.square(harmonics))) / fundamental * 100.0)
    else:
        thd = 0.0

    return HarmonicAnalysis(
        fundamental_index=fundamental_index,
        fundamental_magnitude=fundamental,
        harmonic_magnitudes=harmonics,
        thd=thd,
        spectral_energy=energy,
    )


__all__ = [
    "fft_spectrum",
    "fft_features",
    "stft",
    "stft_image",
    "HarmonicAnalysis",
    "harmonic_analysis",
]
