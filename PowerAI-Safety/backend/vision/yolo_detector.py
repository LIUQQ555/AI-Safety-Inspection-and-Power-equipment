"""可见光目标检测。

对应技术方案 5.1 节「可见光图像检测模块」。

两种后端：
    * **YoloDetector** —— 基于 Ultralytics YOLO 的真实检测器。加载自训练权重
      （``training/train_yolo.py`` 产出）后，输出设备识别、目标定位与缺陷分类。
    * **HeuristicDetector** —— 未提供权重时的降级实现。基于经典 CV 提取显著
      区域（边缘密度 + 局部对比度），输出**候选区域**而非缺陷判定。

重要：降级实现不会伪造缺陷判定。它的输出中 ``backend`` 字段为
``heuristic_fallback``，``category`` 一律为 ``device``，并在 ``notes`` 中明确
说明「未加载训练权重，不能识别缺陷类型」。这是为了保证系统在任何情况下都
不会给出无依据的安全结论。

许可证提示：Ultralytics 采用 AGPL-3.0。科研验证可用；商业部署需另行确认许可。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.config import Config
from backend.core.schemas import BBox, Detection, Modality, VisibleResult
from backend.vision import image_processor as ip
from backend.vision import labels as lbl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 评分
# ---------------------------------------------------------------------------
def compute_visual_score(
    detections: Sequence[Detection],
    defect_severity: Dict[str, float],
    default_severity: float = 0.5,
) -> Tuple[float, float, List[str]]:
    """由检测结果计算视觉异常分（0-100）。

    计算方式（可解释、可复现）：
        base  = 100 × max(该缺陷类的严重度权重 × 检测置信度)
        bonus = min(15, 5 × (缺陷数量 - 1))       # 多处缺陷叠加
        score = min(100, base + bonus)

    只考虑 ``category == "defect"`` 的检测。无缺陷时为 0 分。

    返回 ``(视觉分, 最高缺陷置信度, 缺陷类别列表)``。
    """
    defects = [d for d in detections if d.category == "defect"]
    if not defects:
        return 0.0, 0.0, []

    best = 0.0
    max_conf = 0.0
    classes: List[str] = []
    for det in defects:
        severity = defect_severity.get(det.label, defect_severity.get(lbl.to_en(det.label), default_severity))
        best = max(best, float(severity) * float(det.confidence))
        max_conf = max(max_conf, float(det.confidence))
        zh = lbl.to_zh(det.label)
        if zh not in classes:
            classes.append(zh)

    bonus = min(15.0, 5.0 * (len(defects) - 1))
    score = min(100.0, best * 100.0 + bonus)
    return round(score, 2), round(max_conf, 4), classes


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------
class BaseVisibleDetector(ABC):
    """可见光检测器接口。"""

    backend: str = "base"

    @abstractmethod
    def detect(self, image: np.ndarray, image_path: str = "") -> VisibleResult:
        """对 BGR 图像执行检测。"""

    @property
    def model_name(self) -> str:
        return self.backend


# ---------------------------------------------------------------------------
# YOLO 实现
# ---------------------------------------------------------------------------
class YoloDetector(BaseVisibleDetector):
    """基于 Ultralytics YOLO 的检测器。"""

    backend = "yolo"

    def __init__(self, config: Config) -> None:
        self.config = config
        vcfg = config.vision.visible

        self.weights_path: Optional[Path] = None
        raw_weights = str(vcfg.get("weights", "") or "")
        if raw_weights:
            candidate = config.resolve("vision", "visible", "weights")
            if candidate.exists():
                self.weights_path = candidate
            else:
                logger.info("未找到自训练权重 %s", candidate)

        self.conf = float(vcfg.get("conf_threshold", 0.25))
        self.iou = float(vcfg.get("iou_threshold", 0.45))
        self.imgsz = int(vcfg.get("imgsz", 640))
        self.device = str(vcfg.get("device", "cpu"))
        self.max_det = int(vcfg.get("max_detections", 50))

        self._defect_classes = {lbl.to_en(c) for c in (vcfg.get("defect_classes") or [])}
        self._device_classes = {lbl.to_en(c) for c in (vcfg.get("device_classes") or [])}
        self._severity: Dict[str, float] = {
            lbl.to_en(k): float(v) for k, v in dict(vcfg.get("defect_severity") or {}).items()
        }

        self._model: Any = None
        self._names: Dict[int, str] = {}

    def load(self) -> bool:
        """加载模型。失败返回 False（调用方应降级到启发式检测器）。"""
        if self.weights_path is None:
            return False
        try:
            from ultralytics import YOLO  # 延迟导入：未安装时不影响其它模块
        except ImportError:
            logger.warning(
                "未安装 ultralytics，无法使用 YOLO 检测器。"
                "安装方式：.venv\\Scripts\\python -m pip install ultralytics"
            )
            return False

        try:
            logger.info("加载 YOLO 权重：%s", self.weights_path)
            self._model = YOLO(str(self.weights_path))
            raw_names = getattr(self._model, "names", {}) or {}
            self._names = {int(k): str(v) for k, v in raw_names.items()}
            logger.info("YOLO 加载完成，类别数 %d：%s", len(self._names), list(self._names.values()))
            return True
        except Exception as exc:
            logger.error("YOLO 权重加载失败：%s", exc)
            self._model = None
            return False

    @property
    def model_name(self) -> str:
        return f"yolov8 ({self.weights_path.name})" if self.weights_path else "yolov8"

    def _category_of(self, class_name: str) -> str:
        """判定类别属于设备还是缺陷。

        优先按配置的 defect_classes 判定；配置未覆盖时按是否在 device_classes 中
        判定；都不在时，依据类别名是否出现在严重度表中做最终判断。
        """
        norm = lbl.to_en(class_name)
        if norm in self._defect_classes:
            return "defect"
        if norm in self._device_classes:
            return "device"
        if norm in self._severity:
            return "defect"
        # 未知类别保守地按缺陷处理，避免漏报
        logger.debug("类别 %s 未在配置中声明，按缺陷处理", class_name)
        return "defect"

    def detect(self, image: np.ndarray, image_path: str = "") -> VisibleResult:
        started = time.perf_counter()
        h, w = image.shape[:2]

        if self._model is None:
            raise RuntimeError("YOLO 模型尚未加载，请先调用 load()")

        try:
            results = self._model.predict(
                source=image,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                max_det=self.max_det,
                verbose=False,
            )
        except Exception as exc:
            logger.error("YOLO 推理失败：%s", exc)
            return VisibleResult(
                image_path=image_path, image_width=w, image_height=h,
                backend=self.backend, model_name=self.model_name,
                visual_score=0.0, elapsed_ms=(time.perf_counter() - started) * 1000,
                notes=[f"YOLO 推理失败：{exc}"],
            )

        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                raw_name = self._names.get(cls_id, f"class_{cls_id}")
                confidence = float(box.conf[0]) if box.conf is not None else 0.0
                xyxy = box.xyxy[0].tolist() if box.xyxy is not None else [0, 0, 0, 0]

                detections.append(Detection(
                    label=lbl.to_en(raw_name),
                    confidence=round(confidence, 4),
                    bbox=BBox.from_list(xyxy),
                    source=Modality.VISIBLE.value,
                    category=self._category_of(raw_name),
                    meta={"class_name": raw_name, "class_name_zh": lbl.to_zh(raw_name), "class_id": cls_id},
                ))

        visual_score, max_conf, defect_classes = compute_visual_score(detections, self._severity)
        defect_count = sum(1 for d in detections if d.category == "defect")
        device_count = sum(1 for d in detections if d.category == "device")

        return VisibleResult(
            image_path=image_path,
            image_width=w,
            image_height=h,
            backend=self.backend,
            model_name=self.model_name,
            detections=detections,
            device_count=device_count,
            defect_count=defect_count,
            defect_classes=defect_classes,
            max_defect_confidence=max_conf,
            visual_score=visual_score,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            notes=[],
        )


# ---------------------------------------------------------------------------
# 启发式降级实现
# ---------------------------------------------------------------------------
class HeuristicDetector(BaseVisibleDetector):
    """无训练权重时的候选区域检测器。

    算法：CLAHE 增强 → Canny 边缘 → 形态学闭运算连通 → 轮廓筛选
          → 按「边缘密度 × 局部对比度」排序取前 N 个区域。

    该检测器只能定位**可能包含设备/部件的显著区域**，不能识别缺陷类型。
    输出中的 category 恒为 "device"，visual_score 恒为 0，
    并在 notes 中显式声明降级状态。
    """

    backend = "heuristic_fallback"

    def __init__(self, config: Config) -> None:
        self.config = config
        hcfg = config.vision.visible.get("heuristic", {}) or {}
        self.min_area_ratio = float(hcfg.get("min_area_ratio", 0.004))
        self.max_area_ratio = float(hcfg.get("max_area_ratio", 0.60))
        self.canny_low = int(hcfg.get("canny_low", 50))
        self.canny_high = int(hcfg.get("canny_high", 150))
        self.max_candidates = int(hcfg.get("max_candidates", 8))

    @property
    def model_name(self) -> str:
        return "heuristic-saliency (untrained)"

    def detect(self, image: np.ndarray, image_path: str = "") -> VisibleResult:
        started = time.perf_counter()
        bgr = ip.ensure_bgr(image)
        h, w = bgr.shape[:2]
        total_area = float(h * w)

        gray = ip.to_gray(bgr)
        enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        blurred = cv2.bilateralFilter(enhanced, 7, 45, 45)

        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Tuple[float, BBox, float]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            ratio = area / total_area
            if ratio < self.min_area_ratio or ratio > self.max_area_ratio:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            bbox = BBox(float(x), float(y), float(x + bw), float(y + bh))

            # 边缘密度：区域内边缘像素占比，反映结构复杂度
            roi_edges = edges[y:y + bh, x:x + bw]
            edge_density = float((roi_edges > 0).sum()) / max(1.0, float(roi_edges.size))

            # 局部对比度：区域内灰度的标准差
            roi_gray = gray[y:y + bh, x:x + bw]
            local_contrast = float(roi_gray.std()) / 128.0

            # 面积适中性：过小的碎块降权
            area_factor = min(1.0, (ratio / 0.05) ** 0.5)

            score = (0.45 * min(1.0, edge_density * 4.0)
                     + 0.40 * min(1.0, local_contrast)
                     + 0.15 * area_factor)
            candidates.append((score, bbox, edge_density))

        candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = candidates[: self.max_candidates]

        detections = [
            Detection(
                label="transformer",
                confidence=round(float(np.clip(score, 0.05, 0.95)), 4),
                bbox=bbox,
                source=Modality.VISIBLE.value,
                category="device",
                meta={"edge_density": round(edge_density, 4), "heuristic": True},
            )
            for score, bbox, edge_density in candidates
        ]

        return VisibleResult(
            image_path=image_path,
            image_width=w,
            image_height=h,
            backend=self.backend,
            model_name=self.model_name,
            detections=detections,
            device_count=len(detections),
            defect_count=0,
            defect_classes=[],
            max_defect_confidence=0.0,
            visual_score=0.0,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            notes=[
                "未加载 YOLO 训练权重，当前使用启发式候选区域检测（heuristic_fallback）。",
                "该模式只能定位显著区域，无法识别缺陷类型，视觉异常分恒为 0。",
                "训练权重：.venv\\Scripts\\python training\\train_yolo.py",
            ],
        )


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
def create_visible_detector(config: Config, force_heuristic: bool = False) -> BaseVisibleDetector:
    """创建可见光检测器。

    优先使用 YOLO；权重缺失、ultralytics 未安装或加载失败时自动降级为
    启发式检测器，保证上层流程始终可以运行。
    """
    if not force_heuristic:
        detector = YoloDetector(config)
        if detector.weights_path is not None and detector.load():
            return detector
        logger.info("YOLO 检测器不可用，降级为启发式检测器")
    return HeuristicDetector(config)


__all__ = [
    "BaseVisibleDetector",
    "YoloDetector",
    "HeuristicDetector",
    "create_visible_detector",
    "compute_visual_score",
]
