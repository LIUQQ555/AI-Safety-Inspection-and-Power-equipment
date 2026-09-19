"""合成数据生成器 —— 让整套闭环在零下载的前提下跑起来。

技术方案第十至十二节列出的真实数据集（CPLID / IDDD / thermal-images-equip / ETT）
需要另行下载，见同目录下的 ``prepare_*.py``。但在数据到位之前，
（以及在没有 GPU 的开发机上）系统仍需可运行、可演示、可回归测试，
因此本脚本按已知规律生成三类合成样本：

* **可见光**：绘制变压器外观（油箱 / 套管 / 散热片 / 储油柜），
  按场景叠加渗漏油、锈蚀、破损、悬挂异物等缺陷。
* **红外**：构造温度场，编码为 16 位辐射 PNG（DN 满量程 ↔ 配置温度区间，
  见 ``InfraredDetector.build_temperature_field``），可选另存伪彩色图。
* **时序**：15 分钟采样的三相电压/电流、功率、温度、负荷，按场景注入趋势或突变。

**合成数据只用于打通流程和回归测试，不能用于评估模型精度**，
也不代表真实设备的缺陷形态与温度分布。每个样本的「真值」记录在
``manifest.json`` 中，可用于检查检测链路是否给出方向一致的结论。

用法::

    python datasets/synthesize.py                     # 生成全部场景
    python datasets/synthesize.py --out data/samples  # 指定输出目录
    python datasets/synthesize.py --seed 7            # 换一组随机种子
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.vision import colormaps as cmap  # noqa: E402
from backend.vision import image_processor as ip  # noqa: E402

# 与 config/config.yaml 的 vision.infrared 保持一致。
# 若修改了配置里的温度区间，这里也要同步，否则合成图的绝对温度会对不上。
TEMP_MIN_C = 0.0
TEMP_MAX_C = 150.0

# 设备在画面中的布局（归一化坐标）。
#
# 可见光图与红外图**必须共用这一份定义**：两者要能空间对上，
# 融合阶段才能计算可见光缺陷与红外热点的重合度。此前两处各写一份
# 硬编码坐标，改动时极易只改一边而让两个模态错位。
#
# 取景偏紧是有意为之：现场红外巡检本就是设备特写，画面以设备为主体。
# 这同时让 ``参考温度`` 取全图中位温度时接近设备本体温度——
# 远景图中大面积低温背景会把中位数拉向环境温度，导致 ΔT 偏大。
LAYOUT = {
    # 油箱：左、上、宽、高
    "tank": (0.16, 0.30, 0.68, 0.56),
    # 储油柜
    "conservator": (0.22, 0.22, 0.56, 0.07),
    # 三相套管中心横坐标
    "bushing_x": (0.28, 0.50, 0.72),
    "bushing_top": 0.06,
    "bushing_height": 0.20,
    # 散热片起始横坐标
    "radiator_x": (0.05, 0.85),
    "radiator_top": 0.40,
    "radiator_height": 0.42,
    # 底座
    "base": (0.12, 0.86, 0.76, 0.05),
}

# 场景温度场背景温度（℃），应与 config 的 ambient 语义区分：
# 这里描述的是合成图里画面的环境温度
SCENE_AMBIENT_C = 26.0


# ---------------------------------------------------------------------------
# 场景定义
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    """一个巡检场景：可见光缺陷 + 红外温度 + 时序形态。"""

    name: str
    title: str
    # 可见光缺陷：类名 → 在油箱坐标系中的位置（归一化）与尺寸
    defects: List[str] = field(default_factory=list)
    # 三相套管温度（℃）
    phase_temps: Tuple[float, float, float] = (38.0, 39.5, 40.5)
    # 油箱本体温度（℃）
    tank_temp: float = 34.0
    # 时序形态：正常 / 负荷爬升 / 电流突变
    series: str = "normal"


SCENARIOS: List[Scenario] = [
    Scenario(
        name="normal",
        title="正常运行",
        defects=[],
        phase_temps=(38.0, 39.2, 40.1),
        tank_temp=34.0,
        series="normal",
    ),
    Scenario(
        name="oil_leak",
        title="渗漏油（可见光缺陷，温度轻度偏高）",
        defects=["oil_leak", "oil_stain"],
        phase_temps=(44.0, 45.5, 46.8),
        tank_temp=38.0,
        series="normal",
    ),
    Scenario(
        name="overheat",
        title="接线端子过热（红外严重，可见光无异常）",
        defects=[],
        phase_temps=(52.0, 55.0, 96.5),   # C 相显著过热
        tank_temp=41.0,
        series="load_rise",
    ),
    Scenario(
        name="critical",
        title="多模态一致异常（破损 + 严重过热 + 电流突变）",
        defects=["broken_part", "rust"],
        phase_temps=(58.0, 61.0, 118.0),
        tank_temp=52.0,
        series="current_spike",
    ),
    Scenario(
        name="foreign_object",
        title="悬挂异物（可见光缺陷，温度正常）",
        defects=["foreign_object"],
        phase_temps=(36.5, 37.4, 38.2),
        tank_temp=33.0,
        series="normal",
    ),
]


# ---------------------------------------------------------------------------
# 可见光图像合成
# ---------------------------------------------------------------------------
def _background(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """天空/背景：自上而下的浅灰蓝渐变 + 轻微噪声。"""
    top = np.array([168, 160, 148], dtype=np.float32)     # BGR
    bottom = np.array([112, 108, 102], dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    canvas = top[None, None, :] * (1 - ramp) + bottom[None, None, :] * ramp
    canvas = np.repeat(canvas, width, axis=1)
    canvas += rng.normal(0.0, 3.0, canvas.shape)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _metal_texture(shape: Tuple[int, int], base: int, rng: np.random.Generator,
                   roughness: float = 7.0) -> np.ndarray:
    """灰色金属表面（BGR 三通道）：基色 + 噪声 + 竖向流挂纹理。"""
    patch = np.full(shape, float(base), dtype=np.float32)
    patch += rng.normal(0.0, roughness, shape)
    # 竖向条纹，模拟金属表面的污渍流挂
    stripes = rng.normal(0.0, 1.0, (1, shape[1]))
    stripes = cv2.GaussianBlur(stripes, (1, 9), 0)
    patch += stripes * 4.0
    gray = np.clip(patch, 0, 255).astype(np.uint8)
    return gray[..., None].repeat(3, axis=2)


def _px(box: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int, int, int]:
    """归一化布局 → 像素框 ``(x, y, w, h)``。"""
    nx, ny, nw, nh = box
    return int(w * nx), int(h * ny), int(w * nw), int(h * nh)


def _draw_transformer(canvas: np.ndarray, rng: np.random.Generator) -> Dict[str, Tuple[int, int, int, int]]:
    """按 ``LAYOUT`` 绘制变压器主体，返回各部件的像素框 ``(x, y, w, h)``。"""
    h, w = canvas.shape[:2]
    parts: Dict[str, Tuple[int, int, int, int]] = {}

    # 油箱
    x, y, tw, th = _px(LAYOUT["tank"], w, h)
    tank = (x, y, tw, th)
    canvas[y:y + th, x:x + tw] = _metal_texture((th, tw), 132, rng)
    cv2.rectangle(canvas, (x, y), (x + tw, y + th), (78, 78, 78), 3)
    # 加强筋
    for i in range(1, 5):
        cy = y + int(th * i / 5)
        cv2.line(canvas, (x + 8, cy), (x + tw - 8, cy), (150, 150, 150), 2)
    parts["tank"] = tank

    # 储油柜（油箱上方横置圆筒）
    cx, cy, cw, ch = _px(LAYOUT["conservator"], w, h)
    cons = (cx, cy, cw, ch)
    cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (118, 118, 118), -1)
    cv2.ellipse(canvas, (cx + cw, cy + ch // 2), (10, ch // 2), 0, -90, 90, (118, 118, 118), -1)
    cv2.ellipse(canvas, (cx, cy + ch // 2), (10, ch // 2), 0, 90, 270, (118, 118, 118), -1)
    cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (70, 70, 70), 2)
    parts["conservator"] = cons

    # 三相套管（顶部竖立）
    by = int(h * LAYOUT["bushing_top"])
    bh = int(h * LAYOUT["bushing_height"])
    bw = 18
    for i, cxr in enumerate(LAYOUT["bushing_x"]):
        bx = int(w * cxr) - bw // 2
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (96, 104, 122), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (60, 66, 78), 2)
        # 伞裙
        for k in range(1, 6):
            sy = by + int(bh * k / 6)
            cv2.ellipse(canvas, (bx + bw // 2, sy), (15, 4), 0, 0, 360, (78, 86, 104), -1)
        # 接线端子
        cv2.rectangle(canvas, (bx - 3, by - 10), (bx + bw + 3, by), (168, 168, 168), -1)
        parts[f"bushing_{'abc'[i]}"] = (bx - 3, by - 10, bw + 6, bh + 10)

    # 散热片（两侧竖排）
    ry, rh = int(h * LAYOUT["radiator_top"]), int(h * LAYOUT["radiator_height"])
    for side, x0r in (("left", LAYOUT["radiator_x"][0]), ("right", LAYOUT["radiator_x"][1])):
        x0 = int(w * x0r)
        for k in range(6):
            fx = x0 + k * 7
            cv2.rectangle(canvas, (fx, ry), (fx + 4, ry + rh), (104, 104, 104), -1)
        parts[f"radiator_{side}"] = (x0, ry, 46, rh)

    # 底座
    bx0, by0, bw0, bh0 = _px(LAYOUT["base"], w, h)
    cv2.rectangle(canvas, (bx0, by0), (bx0 + bw0, by0 + bh0), (96, 96, 96), -1)
    return parts


def _add_oil_leak(canvas: np.ndarray, tank: Tuple[int, int, int, int],
                  rng: np.random.Generator) -> None:
    """渗漏油：油箱下部向下的深色不规则流挂 + 地面油渍。"""
    x, y, w, h = tank
    sx = x + int(w * 0.62)
    sy = y + int(h * 0.80)

    # 流挂痕迹
    for k in range(9):
        ex = sx + rng.integers(-14, 22)
        ey = sy + int(h * (0.25 + 0.75 * k / 9)) + rng.integers(-4, 4)
        cv2.line(canvas, (sx + rng.integers(-8, 8), sy + k * 3), (ex, ey), (36, 38, 42), 3)

    # 地面油渍（扁椭圆）
    for k in range(3):
        center = (sx + rng.integers(-40, 60), y + h + 30 + k * 12)
        axes = (int(rng.integers(40, 80)), int(rng.integers(6, 13)))
        overlay = canvas.copy()
        cv2.ellipse(overlay, center, axes, 0, 0, 360, (28, 30, 34), -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)


def _add_oil_stain(canvas: np.ndarray, tank: Tuple[int, int, int, int],
                   rng: np.random.Generator) -> None:
    """油污：油箱表面深色斑块。"""
    x, y, w, h = tank
    overlay = canvas.copy()
    for _ in range(6):
        center = (x + int(rng.integers(20, max(21, w - 20))),
                  y + int(rng.integers(20, max(21, h - 20))))
        axes = (int(rng.integers(18, 46)), int(rng.integers(12, 30)))
        cv2.ellipse(overlay, center, axes, int(rng.integers(0, 180)), 0, 360, (44, 46, 50), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)


def _add_rust(canvas: np.ndarray, tank: Tuple[int, int, int, int],
              rng: np.random.Generator) -> None:
    """锈蚀：橙褐色斑驳区域。"""
    x, y, w, h = tank
    overlay = canvas.copy()
    for _ in range(14):
        center = (x + int(rng.integers(10, max(11, w - 10))),
                  y + int(rng.integers(10, max(11, h - 10))))
        axes = (int(rng.integers(8, 26)), int(rng.integers(6, 20)))
        angle = int(rng.integers(0, 180))
        cv2.ellipse(overlay, center, axes, angle, 0, 360, (52, 96, 158), -1)
    cv2.addWeighted(overlay, 0.68, canvas, 0.32, 0, canvas)
    # 锈迹边缘更亮，模拟蓬松感
    mask = cv2.inRange(canvas, (40, 80, 140), (70, 120, 190))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    edges = cv2.Canny(mask, 40, 120)
    canvas[edges > 0] = (78, 128, 205)


def _add_broken_part(canvas: np.ndarray, parts: Dict[str, Tuple[int, int, int, int]],
                     rng: np.random.Generator) -> None:
    """部件破损：B 相套管中部崩缺，并带裂纹。"""
    bx, by, bw, bh = parts["bushing_b"]
    cy = by + int(bh * 0.45)
    # 崩缺（背景色三角缺口）
    notch = np.array([
        [bx - 2, cy],
        [bx + bw + 2, cy + 6],
        [bx + bw // 2, cy + 26],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [notch], (128, 132, 140))
    # 裂纹
    for k in range(4):
        x0 = bx + int(rng.integers(0, bw))
        cv2.line(canvas, (x0, cy - 14 + k * 4),
                 (x0 + int(rng.integers(-8, 9)), cy + 10 + k * 5), (52, 52, 56), 2)


def _add_foreign_object(canvas: np.ndarray, parts: Dict[str, Tuple[int, int, int, int]],
                        rng: np.random.Generator) -> None:
    """悬挂异物：自导线垂下的深色柔性条带。"""
    ax, ay, aw, ah = parts["bushing_a"]
    x0 = ax + aw // 2
    y0 = ay + ah
    points = [(x0, y0)]
    for k in range(14):
        x0 += int(rng.integers(-6, 7))
        y0 += int(rng.integers(7, 14))
        points.append((x0, y0))
    for a, b in zip(points, points[1:]):
        cv2.line(canvas, a, b, (38, 40, 44), 3)
    # 末端飘带
    tip = points[-1]
    cv2.ellipse(canvas, (tip[0] + 6, tip[1] + 10), (14, 22), 20, 0, 360, (46, 48, 52), -1)


def synthesize_visible(scenario: Scenario, rng: np.random.Generator,
                       size: Tuple[int, int] = (600, 800)) -> np.ndarray:
    """生成一张可见光巡检图像。"""
    h, w = size
    canvas = _background(h, w, rng)
    parts = _draw_transformer(canvas, rng)
    tank = parts["tank"]

    for defect in scenario.defects:
        if defect == "oil_leak":
            _add_oil_leak(canvas, tank, rng)
        elif defect == "oil_stain":
            _add_oil_stain(canvas, tank, rng)
        elif defect == "rust":
            _add_rust(canvas, tank, rng)
        elif defect == "broken_part":
            _add_broken_part(canvas, parts, rng)
        elif defect == "foreign_object":
            _add_foreign_object(canvas, parts, rng)
        else:
            raise ValueError(f"未知缺陷类型：{defect}")

    return canvas


# ---------------------------------------------------------------------------
# 红外图像合成
# ---------------------------------------------------------------------------
def synthesize_temperature_field(scenario: Scenario, rng: np.random.Generator,
                                 size: Tuple[int, int] = (600, 800)) -> np.ndarray:
    """构造温度场（℃），形状 (H, W) float32。

    与可见光图共用同一套坐标，使两个模态的缺陷位置能对应上。
    """
    h, w = size
    ambient = SCENE_AMBIENT_C

    field = np.full((h, w), ambient, dtype=np.float32)
    field += rng.normal(0.0, 0.35, (h, w))
    # 背景缓慢起伏（地面/墙体受日照）
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field += 3.0 * np.exp(-((xx - w * 0.5) ** 2 + (yy - h * 0.95) ** 2) / (2 * (w * 0.45) ** 2))

    # 油箱本体：比环境高，带水平梯度
    x, y, tw, th = _px(LAYOUT["tank"], w, h)
    gradient = np.linspace(-2.0, 2.5, tw, dtype=np.float32)[None, :]
    body = scenario.tank_temp - ambient + gradient
    field[y:y + th, x:x + tw] = ambient + body
    field[y:y + th, x:x + tw] += rng.normal(0.0, 0.5, (th, tw))

    # 储油柜
    cx, cy, cw, ch = _px(LAYOUT["conservator"], w, h)
    field[cy:cy + ch, cx:cx + cw] = scenario.tank_temp + 1.5

    # 散热片
    ry, rh = int(h * LAYOUT["radiator_top"]), int(h * LAYOUT["radiator_height"])
    for x0r in LAYOUT["radiator_x"]:
        x0 = int(w * x0r)
        field[ry:ry + rh, x0:x0 + 46] = scenario.tank_temp + 4.0

    # 三相套管 + 接线端子（热点主要来源）
    by = int(h * LAYOUT["bushing_top"])
    bh = int(h * LAYOUT["bushing_height"])
    bw = 18
    for i, cxr in enumerate(LAYOUT["bushing_x"]):
        bx = int(w * cxr) - bw // 2
        temp = scenario.phase_temps[i]

        # 套管本体：沿轴向衰减（下部接近油箱温度）
        axis = np.linspace(1.0, 0.35, bh, dtype=np.float32)[:, None]
        field[by:by + bh, bx:bx + bw] = ambient + (temp - ambient) * axis
        field[by:by + bh, bx:bx + bw] += rng.normal(0.0, 0.4, (bh, bw))

        # 接线端子：面积小、温度最高的热点
        gx0, gy0, gx1, gy1 = bx - 3, by - 10, bx + bw + 3, by
        gy0 = max(0, gy0)
        field[gy0:gy1, gx0:gx1] = temp
        # 热点周围的热扩散
        spread = np.zeros((h, w), dtype=np.float32)
        spread[gy0:gy1, gx0:gx1] = temp - ambient
        spread = cv2.GaussianBlur(spread, (0, 0), 7.0)
        field = np.maximum(field, ambient + spread)

    return np.clip(field, -20.0, 200.0)


def encode_radiometric(field_c: np.ndarray,
                       temp_min: float = TEMP_MIN_C,
                       temp_max: float = TEMP_MAX_C) -> np.ndarray:
    """温度场 → 16 位辐射 PNG 数据（DN 满量程 ↔ [temp_min, temp_max]）。"""
    normalized = (field_c - temp_min) / (temp_max - temp_min)
    return np.clip(np.round(normalized * 65535.0), 0, 65535).astype(np.uint16)


def encode_pseudo_color(field_c: np.ndarray, palette: str = "ironbow",
                        temp_min: float = TEMP_MIN_C,
                        temp_max: float = TEMP_MAX_C) -> np.ndarray:
    """温度场 → 8 位伪彩色图（BGR）。"""
    normalized = np.clip((field_c - temp_min) / (temp_max - temp_min), 0.0, 1.0)
    return cmap.apply_colormap(normalized, palette)


# ---------------------------------------------------------------------------
# 时序数据合成
# ---------------------------------------------------------------------------
def synthesize_timeseries(scenario: Scenario, rng: np.random.Generator,
                          hours: int = 72, interval_min: int = 15) -> "object":
    """生成 15 分钟采样的三相电气量时序。

    返回 DataFrame，列名使用 ``signal_loader`` 识别的规范英文名。
    """
    import pandas as pd

    n = int(hours * 60 / interval_min)
    steps = np.arange(n, dtype=np.float64)
    # 日负荷曲线：早晚双峰
    hour_of_day = (steps * interval_min / 60.0) % 24.0
    daily = (0.62
             + 0.20 * np.exp(-((hour_of_day - 9.5) ** 2) / (2 * 2.2 ** 2))
             + 0.24 * np.exp(-((hour_of_day - 19.0) ** 2) / (2 * 2.0 ** 2)))
    base_load = 420.0 * daily

    load = base_load.copy()
    current_factor = np.ones(n)
    voltage_offset = np.zeros(n)

    # 电压偏移以 kV 计，数值控制在额定值的百分之几 —— 实际电网中
    # 10 kV 母线电压偏差通常在 ±7% 以内，写成几 kV 是不符合物理的量级。
    if scenario.series == "load_rise":
        # 后 40% 时段负荷持续爬升（对应接触不良/过载的发展过程）
        start = int(n * 0.6)
        ramp = np.linspace(0.0, 1.0, n - start)
        load[start:] *= (1.0 + 0.55 * ramp)
        current_factor[start:] *= (1.0 + 0.55 * ramp)
        voltage_offset[start:] -= 0.28 * ramp         # 大电流引起压降，约 -2.8%

    elif scenario.series == "current_spike":
        # C 相出现间歇性电流突变 + 电压跌落
        for center in (int(n * 0.55), int(n * 0.72), int(n * 0.88)):
            width = max(3, n // 60)
            window = slice(max(0, center - width), min(n, center + width))
            current_factor[window] *= 2.15
            voltage_offset[window] -= 0.52            # 约 -5.2%

    # 三相电压（10 kV 侧线电压，额定 10 kV；此处以 kV 记录）
    rated_voltage = 10.0
    voltage_a = rated_voltage + voltage_offset + rng.normal(0.0, 0.035, n)
    voltage_b = rated_voltage + voltage_offset * 0.95 + rng.normal(0.0, 0.035, n)
    voltage_c = rated_voltage + voltage_offset * 1.20 + rng.normal(0.0, 0.040, n)
    if scenario.series == "current_spike":
        voltage_c -= 0.30    # C 相电压略低，配合电流突变

    # 三相电流（A）。C 相在异常场景下偏高
    base_current = 96.0 * daily
    current_a = base_current * current_factor + rng.normal(0.0, 0.9, n)
    current_b = base_current * current_factor * 0.98 + rng.normal(0.0, 0.9, n)
    current_c = base_current * current_factor * (1.03 if scenario.series == "normal" else 1.06)
    current_c = current_c + rng.normal(0.0, 0.9, n)

    # 功率（kW）与负荷通道
    power = load * 0.93 + rng.normal(0.0, 3.0, n)
    load_channel = load + rng.normal(0.0, 2.5, n)

    # 顶层油温：跟随负荷惯性上升（一阶滞后）
    thermal = np.zeros(n, dtype=np.float64)
    tau = 18.0
    for i in range(1, n):
        target = 28.0 + 26.0 * (load_channel[i] / 520.0)
        thermal[i] = thermal[i - 1] + (target - thermal[i - 1]) / tau
    temperature = thermal + 8.0 + rng.normal(0.0, 0.28, n)

    start_time = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours)
    timestamps = [start_time + timedelta(minutes=int(m) * interval_min) for m in range(n)]

    return pd.DataFrame({
        "timestamp": timestamps,
        "voltage_a": np.round(voltage_a, 4),
        "voltage_b": np.round(voltage_b, 4),
        "voltage_c": np.round(voltage_c, 4),
        "current_a": np.round(current_a, 3),
        "current_b": np.round(current_b, 3),
        "current_c": np.round(current_c, 3),
        "power": np.round(power, 2),
        "load": np.round(load_channel, 2),
        "temperature": np.round(temperature, 2),
    })


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def generate_all(out_dir: Path, seed: int = 42, size: Tuple[int, int] = (600, 800),
                 write_pseudo_color: bool = True) -> Dict[str, object]:
    """为每个场景生成可见光 / 红外 / 时序样本与真值清单。"""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "temperature_range_c": [TEMP_MIN_C, TEMP_MAX_C],
        "note": "合成数据，仅用于打通流程与回归测试，不代表真实缺陷形态与温度分布。",
        "scenarios": [],
    }

    for scenario in SCENARIOS:
        scene_dir = out_dir / scenario.name
        scene_dir.mkdir(parents=True, exist_ok=True)

        visible = synthesize_visible(scenario, rng, size)
        visible_path = scene_dir / f"{scenario.name}_visible.jpg"
        ip.imwrite_unicode(visible_path, visible)

        field = synthesize_temperature_field(scenario, rng, size)
        radiometric_path = scene_dir / f"{scenario.name}_thermal.png"
        ip.imwrite_unicode(radiometric_path, encode_radiometric(field))

        pseudo_path: Optional[Path] = None
        calib_path: Optional[Path] = None
        if write_pseudo_color:
            pseudo_path = scene_dir / f"{scenario.name}_thermal_ironbow.jpg"
            ip.imwrite_unicode(pseudo_path, encode_pseudo_color(field, "ironbow"))
            # 伪彩色图必须带标定文件，否则温度反演只能给出相对分布
            calib_path = scene_dir / f"{scenario.name}_thermal_ironbow.jpg.calib.json"
            calib_path.write_text(
                json.dumps({"temp_min_c": TEMP_MIN_C, "temp_max_c": TEMP_MAX_C},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        frame = synthesize_timeseries(scenario, rng)
        csv_path = scene_dir / f"{scenario.name}_timeseries.csv"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

        entry = {
            "scenario": scenario.name,
            "title": scenario.title,
            "visible": str(visible_path.relative_to(out_dir)),
            "thermal_radiometric": str(radiometric_path.relative_to(out_dir)),
            "thermal_pseudo_color": str(pseudo_path.relative_to(out_dir)) if pseudo_path else None,
            "thermal_calibration": str(calib_path.relative_to(out_dir)) if calib_path else None,
            "timeseries": str(csv_path.relative_to(out_dir)),
            "ground_truth": {
                "visible_defects": scenario.defects,
                "phase_temps_c": list(scenario.phase_temps),
                "max_temp_c": round(float(field.max()), 2),
                "ambient_reference_c": 26.0,
                "series_pattern": scenario.series,
                "expected_thermal_severity": _expected_severity(field.max(), max(scenario.phase_temps) - 26.0),
            },
        }
        manifest["scenarios"].append(entry)  # type: ignore[union-attr]
        print(f"  [{scenario.name}] {scenario.title}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _expected_severity(max_temp: float, delta: float) -> str:
    """按 config 中的阈值给出「预期」热级别，便于对照检测输出。"""
    if max_temp >= 110.0 or delta >= 40.0:
        return "critical"
    if max_temp >= 90.0 or delta >= 20.0:
        return "alarm"
    if max_temp >= 70.0 or delta >= 10.0:
        return "warning"
    return "normal"


def main() -> int:
    parser = argparse.ArgumentParser(description="生成合成巡检样本（可见光/红外/时序）")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "samples"),
                        help="输出目录，默认 data/samples")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--no-pseudo-color", action="store_true",
                        help="不生成伪彩色热像图（只生成 16 位辐射 PNG）")
    args = parser.parse_args()

    out_dir = Path(args.out)
    print(f"生成合成样本 → {out_dir}")
    manifest = generate_all(
        out_dir,
        seed=args.seed,
        size=(args.height, args.width),
        write_pseudo_color=not args.no_pseudo_color,
    )

    count = len(manifest["scenarios"])  # type: ignore[arg-type]
    print(f"\n完成：{count} 个场景，每个场景 1 张可见光 + 1 张红外（+ 伪彩色）+ 1 份时序 CSV")
    print(f"真值清单：{out_dir / 'manifest.json'}")
    print("\n下一步：在页面或 API 中选择同一场景的三个文件执行巡检。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
