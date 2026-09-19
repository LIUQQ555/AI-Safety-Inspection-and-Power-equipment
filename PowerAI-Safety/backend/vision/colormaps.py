"""红外伪彩色调色板：生成、反演与可视化。

红外相机通常只输出**伪彩色图像**（JPEG/PNG），像素值不再直接是温度，
而是「温度 → 调色板索引」映射后的 RGB 颜色。要恢复温度，必须把颜色反演
回调色板索引。

本模块的做法：
    1. 用控制点插值生成常见红外调色板（ironbow / rainbow / hot / gray）的 256 级表；
    2. 对输入图像，逐一计算像素到各调色板各级的距离，取总距离最小的调色板；
    3. 用该调色板做最近邻反演，得到 [0, 1] 的归一化值；
    4. 归一化值经线性映射得到温度：``T = t_min + v × (t_max - t_min)``。

**局限（必须知晓）**：
    本方法假设图像使用了上述标准调色板之一，且温度区间由配置给出。
    对于相机私有调色板或未标定的图像，反演得到的是**相对温度**
    （单调性与真实温度一致，绝对值可能整体偏移）。
    若需要精确绝对温度，应使用相机的辐射原始数据（16 位 TIFF/SEQ）
    或随图导出的标定文件。检测结果中的 ``backend`` 字段会标明实际使用的路径。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 调色板控制点（RGB 空间，取值 0-255）
# ---------------------------------------------------------------------------
_PALETTE_STOPS: Dict[str, List[Tuple[float, Tuple[int, int, int]]]] = {
    # 铁红（ironbow）—— FLIR 默认调色板：黑 → 紫 → 红 → 橙 → 黄 → 白
    "ironbow": [
        (0.00, (0, 0, 0)),
        (0.15, (60, 0, 80)),
        (0.35, (140, 0, 110)),
        (0.55, (200, 40, 60)),
        (0.72, (245, 130, 25)),
        (0.86, (250, 205, 70)),
        (1.00, (255, 255, 255)),
    ],
    # 彩虹（rainbow / jet）：蓝 → 青 → 绿 → 黄 → 红
    "rainbow": [
        (0.00, (0, 0, 140)),
        (0.25, (0, 90, 255)),
        (0.50, (0, 220, 180)),
        (0.70, (190, 255, 20)),
        (0.85, (255, 200, 0)),
        (1.00, (200, 0, 0)),
    ],
    # 热金属（hot / white-hot）：黑 → 红 → 黄 → 白
    "hot": [
        (0.00, (0, 0, 0)),
        (0.33, (180, 0, 0)),
        (0.66, (255, 180, 0)),
        (1.00, (255, 255, 255)),
    ],
    # 灰阶
    "gray": [
        (0.00, (0, 0, 0)),
        (1.00, (255, 255, 255)),
    ],
}

PALETTE_NAMES: Tuple[str, ...] = tuple(_PALETTE_STOPS.keys())


@lru_cache(maxsize=8)
def build_palette_table(name: str, levels: int = 256) -> np.ndarray:
    """生成调色板查找表，形状 ``(levels, 3)``，RGB，uint8。

    控制点之间用线性插值。结果按 name+levels 缓存。
    """
    if name not in _PALETTE_STOPS:
        raise KeyError(f"未知调色板 {name!r}，可选：{list(_PALETTE_STOPS)}")

    stops = _PALETTE_STOPS[name]
    positions = np.array([s[0] for s in stops], dtype=np.float64)
    colors = np.array([s[1] for s in stops], dtype=np.float64)

    x = np.linspace(0.0, 1.0, levels)
    table = np.empty((levels, 3), dtype=np.float64)
    for channel in range(3):
        table[:, channel] = np.interp(x, positions, colors[:, channel])
    return np.clip(np.round(table), 0, 255).astype(np.uint8)


@lru_cache(maxsize=8)
def build_all_tables(levels: int = 256) -> np.ndarray:
    """所有调色板拼成一个数组，形状 ``(n_palettes, levels, 3)``。"""
    return np.stack([build_palette_table(name, levels) for name in PALETTE_NAMES], axis=0)


# ---------------------------------------------------------------------------
# 反演
# ---------------------------------------------------------------------------
def _nearest_palette_index(
    pixels_rgb: np.ndarray,
    tables: np.ndarray,
    chunk: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 RGB 像素匹配到最接近的调色板级。

    参数
    ----
    pixels_rgb : (N, 3) uint8
    tables     : (n_palettes, levels, 3) uint8

    返回
    ----
    (palette_idx (N,), level_idx (N,), distance (N,))
    """
    n_palettes, levels, _ = tables.shape
    flat = tables.reshape(n_palettes * levels, 3).astype(np.int32)

    out_palette = np.empty(len(pixels_rgb), dtype=np.int32)
    out_level = np.empty(len(pixels_rgb), dtype=np.int32)
    out_dist = np.empty(len(pixels_rgb), dtype=np.float32)

    for start in range(0, len(pixels_rgb), chunk):
        block = pixels_rgb[start:start + chunk].astype(np.int32)
        # (B, P*L) 的平方欧氏距离
        diff = block[:, None, :] - flat[None, :, :]
        dist = np.einsum("bpc,bpc->bp", diff, diff)
        best = np.argmin(dist, axis=1)
        out_palette[start:start + chunk] = best // levels
        out_level[start:start + chunk] = best % levels
        out_dist[start:start + chunk] = np.sqrt(dist[np.arange(len(block)), best])

    return out_palette, out_level, out_dist


def invert_palette(
    image_bgr: np.ndarray,
    sample_pixels: int = 20000,
    random_seed: int = 42,
    stretch: bool = True,
) -> Tuple[np.ndarray, str, float]:
    """把伪彩色热像图反演为 [0, 1] 的归一化温度场。

    先在随机抽样像素上选出最匹配的调色板，再对全图反演，兼顾速度与精度。

    参数
    ----
    stretch
        为 True 时按图像**实际用到的色阶范围**做拉伸。这是无标定信息时的
        兜底做法：很多热像图只占调色板的一段，直接除以 255 会把整幅图压到
        很窄的温度带上。
        为 False 时按完整调色板（0-255 级）线性映射。**已知温度区间时必须
        用 False** —— 该区间本身就隐含了「色阶 0 ↔ 温度下界，色阶 255 ↔ 温度
        上界」的对应关系，再按用量拉伸会引入系统性偏差
        （例如真实 80 ℃ 被拉伸成 150 ℃）。

    返回 ``(归一化温度场 (H, W) float32, 调色板名, 平均色距)``。
    平均色距可用于判断该图像是否真的符合已知调色板：色距越大越可疑。
    """
    rgb = cv2.cvtColor(_ensure_bgr(image_bgr), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    levels = build_palette_table(PALETTE_NAMES[0]).shape[0]
    tables = build_all_tables(levels)

    flat_rgb = rgb.reshape(-1, 3)

    # 第一步：抽样选调色板
    rng = np.random.default_rng(random_seed)
    n = len(flat_rgb)
    if n > sample_pixels:
        sample_idx = rng.choice(n, size=sample_pixels, replace=False)
        sample = flat_rgb[sample_idx]
    else:
        sample = flat_rgb

    pal_idx, _, dist = _nearest_palette_index(sample, tables)
    counts = np.bincount(pal_idx, minlength=len(PALETTE_NAMES))
    # 以「平均距离最小」为准而非「出现次数最多」，避免大面积单色背景误导
    mean_dist = np.array([
        dist[pal_idx == i].mean() if counts[i] > 0 else np.inf
        for i in range(len(PALETTE_NAMES))
    ])
    best_palette_id = int(np.argmin(mean_dist))
    best_name = PALETTE_NAMES[best_palette_id]

    # 第二步：用选中的调色板做全图反演
    single = build_palette_table(best_name, levels)[None, :, :]
    _, full_level, full_dist = _nearest_palette_index(flat_rgb, single)

    used = full_level.astype(np.float32)
    if stretch:
        # 按图像实际用到的级范围拉伸，而不是硬套 0-255。
        # 原因：许多热像图只占调色板的一段（例如 90-180 级），
        # 若直接除以 255，整幅图会被压缩到很窄的温度带上，温度分辨率极低。
        lo, hi = float(used.min()), float(used.max())
        if hi - lo < 1e-6:
            logger.warning("热像图颜色几乎单一（色阶范围 %.1f-%.1f），温度反演不可靠", lo, hi)
            normalized = np.zeros_like(used)
        else:
            normalized = (used - lo) / (hi - lo)
    else:
        # 已知标定区间：色阶 0 ↔ 温度下界，色阶 255 ↔ 温度上界，直接线性映射。
        lo, hi = 0.0, float(levels - 1)
        normalized = np.clip(used / hi, 0.0, 1.0)

    temperature_field = normalized.reshape(h, w).astype(np.float32)
    mean_color_distance = float(full_dist.mean())

    logger.debug(
        "调色板反演：%s，色阶 %d-%d，平均色距 %.1f",
        best_name, int(lo), int(hi), mean_color_distance,
    )
    return temperature_field, best_name, mean_color_distance


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


# ---------------------------------------------------------------------------
# 正向映射（可视化用）
# ---------------------------------------------------------------------------
def apply_colormap(normalized: np.ndarray, name: str = "ironbow") -> np.ndarray:
    """把 [0, 1] 的归一化场渲染为 BGR 图像（用于生成温度分布示意图）。"""
    table = build_palette_table(name)
    levels = table.shape[0]
    indices = np.clip(np.round(normalized * (levels - 1)), 0, levels - 1).astype(np.int32)
    rgb = table[indices]
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)


def normalized_to_temperature(
    normalized: np.ndarray,
    temp_min_c: float,
    temp_max_c: float,
) -> np.ndarray:
    """归一化场 → 摄氏度温度场。"""
    return (temp_min_c + normalized * (temp_max_c - temp_min_c)).astype(np.float32)


__all__ = [
    "PALETTE_NAMES",
    "build_palette_table",
    "build_all_tables",
    "invert_palette",
    "apply_colormap",
    "normalized_to_temperature",
]
