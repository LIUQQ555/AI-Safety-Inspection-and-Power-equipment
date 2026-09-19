"""红外热成像检测。

对应技术方案 5.2 节「红外热成像检测模块」。

处理流程（与技术方案一致）::

    红外图像 → 温度场重建 → 热点区域提取 → 温度/热分布特征
            → 阈值判级 → 热缺陷分类 → 热风险分

三条温度重建路径（结果中 ``backend`` 字段标明实际使用哪条）：

    ``radiometric``
        16 位单通道原始数据或随图标定文件。温度精确，判据可直接使用绝对温度法。

    ``pseudo_color_palette``
        8 位伪彩色 JPEG/PNG。通过调色板反演恢复相对温度分布。
        **绝对温度依赖配置中给出的温度区间**，需按实际相机设置填写。

    ``gray_linear``
        8 位灰度热像图。按线性关系映射，精度最低。

判据依据《带电设备红外诊断应用规范》DL/T 664 的两类方法：
    * 绝对温度法：设备表面温度超过阈值
    * 相对温差法：发热部位与环境（或同类设备）的温差 ΔT 超过阈值
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config import Config
from backend.core.schemas import BBox, HotRegion, InfraredResult, ThermalSeverity
from backend.vision import colormaps as cmap
from backend.vision import image_processor as ip

logger = logging.getLogger(__name__)

# 严重程度排序，用于取最严重的一个
_SEVERITY_ORDER = {
    ThermalSeverity.NORMAL: 0,
    ThermalSeverity.WARNING: 1,
    ThermalSeverity.ALARM: 2,
    ThermalSeverity.CRITICAL: 3,
}


def _stretch_to_unit(values: np.ndarray) -> np.ndarray:
    """把数组按自身实际取值范围线性拉伸到 [0, 1]。

    仅在没有任何标定信息时使用（见 ``build_temperature_field`` 的说明）。
    """
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-6:
        logger.warning("图像数值几乎单一（%.1f-%.1f），温度映射不可靠", lo, hi)
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


class InfraredDetector:
    """红外热像检测器。"""

    def __init__(self, config: Config) -> None:
        self.config = config
        cfg = config.vision.infrared

        self.temp_min_c = float(cfg.get("temp_min_c", 0.0))
        self.temp_max_c = float(cfg.get("temp_max_c", 150.0))
        self.ambient_temp_c: Optional[float] = cfg.get("ambient_temp_c", 25.0)
        if self.ambient_temp_c is not None:
            self.ambient_temp_c = float(self.ambient_temp_c)
        self.auto_ambient_percentile = float(cfg.get("auto_ambient_percentile", 10.0))

        self.abs_warning = float(cfg.get("absolute_warning_c", 70.0))
        self.abs_alarm = float(cfg.get("absolute_alarm_c", 90.0))
        self.abs_critical = float(cfg.get("absolute_critical_c", 110.0))

        self.delta_warning = float(cfg.get("delta_warning_c", 10.0))
        self.delta_alarm = float(cfg.get("delta_alarm_c", 20.0))
        self.delta_critical = float(cfg.get("delta_critical_c", 40.0))

        self.imbalance_warning = float(cfg.get("three_phase_imbalance_warning_c", 10.0))
        self.imbalance_alarm = float(cfg.get("three_phase_imbalance_alarm_c", 20.0))

        self.hot_percentile = float(cfg.get("hot_region_percentile", 95.0))
        self.min_region_area = int(cfg.get("min_region_area", 25))
        self.max_regions = int(cfg.get("max_regions", 8))

        severity_cfg = dict(cfg.get("severity_score") or {})
        self.severity_score: Dict[str, float] = {
            "normal": float(severity_cfg.get("normal", 5)),
            "warning": float(severity_cfg.get("warning", 45)),
            "alarm": float(severity_cfg.get("alarm", 75)),
            "critical": float(severity_cfg.get("critical", 95)),
        }

        if self.temp_max_c <= self.temp_min_c:
            raise ValueError(
                f"红外温度区间非法：temp_min_c={self.temp_min_c} >= temp_max_c={self.temp_max_c}"
            )

    # ------------------------------------------------------------------
    # 温度场重建
    # ------------------------------------------------------------------
    def _load_calibration(self, image_path: str) -> Optional[Dict[str, float]]:
        """查找随图导出的标定文件。

        支持两种命名：``xxx.jpg.calib.json`` 与 ``xxx.calib.json``，
        内容形如 ``{"temp_min_c": 20.0, "temp_max_c": 120.0}``。
        """
        if not image_path:
            return None
        base = Path(image_path)
        for candidate in (base.with_suffix(base.suffix + ".calib.json"),
                          base.with_suffix(".calib.json")):
            if candidate.exists():
                try:
                    with candidate.open("r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if "temp_min_c" in data and "temp_max_c" in data:
                        logger.info("使用标定文件：%s", candidate)
                        return {"temp_min_c": float(data["temp_min_c"]),
                                "temp_max_c": float(data["temp_max_c"])}
                except Exception as exc:
                    logger.warning("标定文件解析失败 %s：%s", candidate, exc)
        return None

    def build_temperature_field(
        self,
        image: np.ndarray,
        image_path: str = "",
    ) -> Tuple[np.ndarray, str, List[str]]:
        """把输入图像转换为摄氏温度场 ``(H, W) float32``。

        返回 ``(温度场, backend 标识, 说明列表)``。
        """
        notes: List[str] = []
        calib = self._load_calibration(image_path)
        t_min = calib["temp_min_c"] if calib else self.temp_min_c
        t_max = calib["temp_max_c"] if calib else self.temp_max_c
        if calib:
            notes.append(
                f"检测到标定文件，采用标定温度区间 [{t_min:.1f}, {t_max:.1f}] ℃"
            )

        # 路径一：16 位辐射原始数据
        #
        # 这里**不做按用量拉伸**。16 位原始数据的 DN 满量程是有明确物理意义的
        # （传感器量程 / 导出时设定的显示量程），配置中的温度区间就定义了
        # 「DN 0 ↔ t_min，DN 65535 ↔ t_max」这一对应关系。
        # 若改按观测到的 DN 最小/最大值拉伸，等于把绝对标定丢掉换成相对分布，
        # 配置的温度阈值随即失效。
        if image.ndim == 2 and image.dtype == np.uint16:
            raw = image.astype(np.float32)
            normalized = np.clip(raw / 65535.0, 0.0, 1.0)
            field = cmap.normalized_to_temperature(normalized, t_min, t_max)
            notes.append(
                f"按 16 位辐射原始数据解析温度场（DN 满量程映射至 "
                f"[{t_min:.1f}, {t_max:.1f}] ℃；DN 满量程构成由相机导出设置决定，"
                "若与实际不符请修改 temp_min_c / temp_max_c）。"
            )
            return field, "radiometric", notes

        # 路径二：灰度图
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            gray = image.reshape(image.shape[0], image.shape[1]).astype(np.float32)
            if calib:
                normalized = np.clip(gray / 255.0, 0.0, 1.0)
            else:
                normalized = _stretch_to_unit(gray)
            field = cmap.normalized_to_temperature(normalized, t_min, t_max)
            notes.append(
                "按 8 位灰度热像图解析温度场（线性映射，精度有限；"
                f"{'按标定区间' if calib else '无标定信息，按灰度实际范围拉伸'}）。"
            )
            return field, "gray_linear", notes

        # 路径三：伪彩色图 → 调色板反演
        bgr = ip.ensure_bgr(image)
        gray = ip.to_gray(bgr)
        # 若三通道数值几乎相等，说明其实是灰度图被存成了 RGB
        if float(np.abs(bgr.astype(np.int16) - gray[..., None].astype(np.int16)).mean()) < 2.0:
            normalized = gray.astype(np.float32) / 255.0
            field = cmap.normalized_to_temperature(normalized, t_min, t_max)
            notes.append("图像近似为灰度图，按线性映射解析温度场。")
            return field, "gray_linear", notes

        # 有标定文件时按完整色阶线性映射（标定区间已定义了色阶↔温度对应关系）；
        # 无标定时才按图像实际用到的色阶范围拉伸。
        normalized, palette_name, color_distance = cmap.invert_palette(
            bgr, stretch=calib is None
        )
        field = cmap.normalized_to_temperature(normalized, t_min, t_max)
        notes.append(
            f"伪彩色图像，按 {palette_name} 调色板反演温度场"
            f"（低色距 {color_distance:.1f}；"
            f"{'按标定区间线性映射' if calib else '无标定信息，按图像色阶范围拉伸'}；"
            f"温度区间 [{t_min:.1f}, {t_max:.1f}] ℃）。"
        )
        if color_distance > 60:
            notes.append(
                f"调色板匹配色距偏高（{color_distance:.1f}），该图可能不是标准红外调色板，"
                "温度反演精度有限，建议仅参考相对分布。"
            )
        return field, "pseudo_color_palette", notes

    # ------------------------------------------------------------------
    # 判级
    # ------------------------------------------------------------------
    def _reference_temperature(
        self,
        field: np.ndarray,
        ambient: float,
        three_phase_temps: Optional[List[float]] = None,
    ) -> Tuple[float, str]:
        """确定参考温度 T2（「正常相对应点温度」）。

        DL/T 664 的相对温差法比较的是**发热点与同类正常部位**的温度，
        而不是发热点与环境空气的温度：一台在 25 ℃ 环境里正常运行、
        表面 40 ℃ 的变压器（温升 15 K）完全正常，若拿环境温度当基准，
        所有健康设备都会被判成异常。

        T2 的取值优先级：

        1. ``three_phase`` —— 三相温度中的**最低相**。这是三相设备最可靠的
           参照物：三相工况相同，温度最低的那一相就是「正常相」。
        2. ``warm_median`` —— **偏暖一半像素的中位温度**（见下）。
        3. ``ambient``     —— 兜底，仅在图像数据异常时使用。

        第 2 条为什么不直接用全图中位温度：电力设备红外测温中，设备总是
        画面里偏热的部分，而背景（天空、地面、围墙）偏冷且常常占据大面积。
        全图中位数会被冷背景拉低，于是 ΔT 被系统性放大——一台正常运行的
        变压器也会算出十几 K 的「温升」。取偏暖一半像素的中位数，
        等价于先剔除冷背景再估设备本体温度，对取景宽窄不敏感。

        前提同样是「设备是画面中偏热的部分」，绝大多数电力设备红外巡检
        场景成立；制冷类设备或日照强烈的背景需人工复核。
        ``baseline_source`` 会标明实际用了哪条路径。
        """
        if three_phase_temps and len(three_phase_temps) >= 2:
            return float(min(three_phase_temps)), "three_phase"

        if field.size:
            flat = field.ravel()
            median = float(np.median(flat))
            # 偏暖一半像素的中位温度
            warm = flat[flat >= median]
            baseline = float(np.median(warm)) if warm.size else median
            if baseline > ambient:
                return baseline, "warm_median"

        return float(ambient), "ambient"

    def classify(self, max_temp_c: float, delta_temp_c: float) -> ThermalSeverity:
        """按 DL/T 664 的绝对温度法与相对温差法取较严重者。

        两个方法的含义不同，都在此参与判级：

        * **绝对温度法** ``max_temp_c`` —— 看发热点本身的温度是否越限。
        * **相对温差法** ``delta_temp_c`` —— 看发热点相对**正常相对应点**
          高出多少（见 ``_reference_temperature``）。
        """
        if max_temp_c >= self.abs_critical or delta_temp_c >= self.delta_critical:
            return ThermalSeverity.CRITICAL
        if max_temp_c >= self.abs_alarm or delta_temp_c >= self.delta_alarm:
            return ThermalSeverity.ALARM
        if max_temp_c >= self.abs_warning or delta_temp_c >= self.delta_warning:
            return ThermalSeverity.WARNING
        return ThermalSeverity.NORMAL

    def _score_from_severity(
        self,
        severity: ThermalSeverity,
        hot_regions: List[HotRegion],
    ) -> float:
        base = self.severity_score.get(severity.value, 5.0)
        if severity == ThermalSeverity.NORMAL:
            return base
        # 多个**异常**热点说明问题分布更广，小幅加分。
        # 这里只数达到告警级别的区域：三相设备的三个接线端子本来就都会被
        # 提取成热点，把温度正常的区域也算进来会让分数虚高。
        abnormal = sum(1 for r in hot_regions if r.severity != ThermalSeverity.NORMAL)
        bonus = min(10.0, 3.0 * max(0, abnormal - 1))
        return min(100.0, base + bonus)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray, image_path: str = "") -> InfraredResult:
        started = time.perf_counter()
        h, w = image.shape[:2]

        field, backend, notes = self.build_temperature_field(image, image_path)

        # --- 环境温度 ---
        if self.ambient_temp_c is None:
            ambient = float(np.percentile(field, self.auto_ambient_percentile))
            notes.append(
                f"未配置环境温度，取图像第 {self.auto_ambient_percentile:.0f} 百分位 "
                f"{ambient:.1f} ℃ 作为环境参考温度。"
            )
        else:
            ambient = float(self.ambient_temp_c)

        max_temp = float(field.max())
        mean_temp = float(field.mean())

        # --- 参考温度 T2（正常相对应点）---
        # 先用全图基线提取热点，再从热点/三相数据反过来确定最终参考温度。
        coarse_baseline, _ = self._reference_temperature(field, ambient)
        hot_regions = self._extract_hot_regions(field, coarse_baseline, image_path)
        three_phase_temps, imbalance = self._estimate_three_phase(hot_regions)

        baseline, baseline_source = self._reference_temperature(
            field, ambient, three_phase_temps
        )
        # 参考温度不得高于最高温，否则 ΔT 为负，判级失去意义
        baseline = min(baseline, max_temp)
        delta_temp = max_temp - baseline

        # DL/T 664 相对温差 δ = (T1 − T2) / (T1 − T0) × 100%
        # T1 发热点温度、T2 正常相对应点温度、T0 环境温度。
        # T1 <= T0 时分母无意义（无温升即无过热），取 0。
        if max_temp > ambient:
            relative_ratio = (max_temp - baseline) / (max_temp - ambient) * 100.0
        else:
            relative_ratio = 0.0

        # --- 总体判级：取最严重的热点 ---
        if hot_regions:
            worst = max(hot_regions, key=lambda r: _SEVERITY_ORDER[r.severity])
            severity = worst.severity
        else:
            severity = self.classify(max_temp, delta_temp)

        thermal_score = self._score_from_severity(severity, hot_regions)

        notes.append(
            f"参考温度 T2={baseline:.1f} ℃（来源：{baseline_source}），"
            f"ΔT={delta_temp:.1f} K，相对温差 δ={relative_ratio:.1f}%。"
        )
        if baseline_source == "median":
            notes.append(
                "参考温度取自全图中位温度，前提是被测设备占据画面主体。"
                "若为远景热像图，建议改用三相同比或显式传入三相温度。"
            )

        elapsed = round((time.perf_counter() - started) * 1000, 2)

        return InfraredResult(
            image_path=image_path,
            image_width=w,
            image_height=h,
            backend=backend,
            ambient_temp_c=round(ambient, 2),
            baseline_temp_c=round(baseline, 2),
            baseline_source=baseline_source,
            max_temp_c=round(max_temp, 2),
            mean_temp_c=round(mean_temp, 2),
            delta_temp_c=round(delta_temp, 2),
            relative_delta_ratio=round(relative_ratio, 2),
            thermal_severity=severity,
            thermal_score=round(thermal_score, 2),
            hot_regions=hot_regions,
            three_phase_temps=three_phase_temps,
            three_phase_imbalance_c=imbalance,
            elapsed_ms=elapsed,
            notes=notes,
        )

    def _extract_hot_regions(
        self,
        field: np.ndarray,
        baseline: float,
        image_path: str = "",
    ) -> List[HotRegion]:
        """提取热点连通域。

        ``baseline`` 是参考温度（见 ``_reference_temperature``）。阈值取
        「高分位」与「参考温度 + 半个 warning 温差」中的较大者：
        前者保证总能选出图像里最突出的区域，后者避免整幅温差不大的图像
        被硬切出一个假热点。
        """
        percentile_threshold = float(np.percentile(field, self.hot_percentile))
        floor_threshold = baseline + self.delta_warning * 0.5
        threshold = max(percentile_threshold, floor_threshold)

        mask = (field >= threshold).astype(np.uint8)
        if not mask.any():
            return []

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        regions: List[HotRegion] = []

        for label_id in range(1, count):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < self.min_region_area:
                continue

            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
            bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

            component_mask = labels == label_id
            values = field[component_mask]
            r_max = float(values.max())
            r_mean = float(values.mean())
            r_delta = r_max - baseline
            severity = self.classify(r_max, r_delta)

            regions.append(HotRegion(
                bbox=BBox(float(x), float(y), float(x + bw), float(y + bh)),
                max_temp_c=round(r_max, 2),
                mean_temp_c=round(r_mean, 2),
                delta_temp_c=round(r_delta, 2),
                area_px=area,
                severity=severity,
                score=round(self.severity_score.get(severity.value, 5.0), 2),
            ))

        regions.sort(key=lambda r: (_SEVERITY_ORDER[r.severity], r.max_temp_c), reverse=True)
        return regions[: self.max_regions]

    def _estimate_three_phase(
        self,
        hot_regions: List[HotRegion],
    ) -> Tuple[Optional[List[float]], Optional[float]]:
        """估算三相温度不平衡。

        **这是启发式估计**：假设热像图中面积最大的三个同尺度热点对应 A/B/C 三相。
        真实判断需结合设备实际相序与拍摄角度，现场应用时建议由
        API 显式传入三相温度（见 ``InspectionService.inspect(three_phase_temps=...)``）。

        **不按严重程度筛选候选**：三相比较的意义正是在三相都正常时确认
        「确实都正常」——若只在已经过热的热点里找三相，那么一台三相均温
        偏高的设备反而会被判成不平衡。真正的约束是「三个区域尺度相近」，
        由下面的面积比检查承担。

        返回 ``(三相温度列表, 最大相间温差)``；无法估计时返回 ``(None, None)``。
        """
        candidates = list(hot_regions)
        if len(candidates) < 3:
            return None, None

        # 取面积最大的三个，按水平位置排序作为 A/B/C 相
        top3 = sorted(candidates, key=lambda r: r.area_px, reverse=True)[:3]
        top3.sort(key=lambda r: r.bbox.center[0])

        # 面积相差过大说明不是同尺度三相，放弃估计
        areas = [r.area_px for r in top3]
        if min(areas) / max(areas) < 0.15:
            logger.debug("热点面积差异过大（%s），跳过三相不平衡估计", areas)
            return None, None

        temps = [round(r.max_temp_c, 2) for r in top3]
        imbalance = round(max(temps) - min(temps), 2)
        return temps, imbalance

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------
    def render_temperature_map(self, image: np.ndarray, image_path: str = "") -> np.ndarray:
        """生成温度分布可视化图（用于报告插图）。"""
        field, _, _ = self.build_temperature_field(image, image_path)
        normalized = (field - field.min()) / max(1e-6, float(field.max() - field.min()))
        return cmap.apply_colormap(normalized, "ironbow")


def create_infrared_detector(config: Config) -> InfraredDetector:
    """创建红外检测器。"""
    return InfraredDetector(config)


__all__ = ["InfraredDetector", "create_infrared_detector"]
