"""领域模型定义。

本模块是三个检测分支、融合层、知识层与报告层之间的数据契约。
所有跨模块传递的结构都定义在这里，任何模块不得私自扩展 ad-hoc 字典。

约定：
    * 所有 ``*_score`` 字段统一为 0-100 的浮点数，数值越高越危险。
    * 所有时间字段为本地时区的 ISO 8601 字符串。
    * 所有坐标 bbox 为像素坐标 ``[x1, y1, x2, y2]``，原点在左上角。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------
class Modality(str, Enum):
    """数据模态。"""

    VISIBLE = "visible"
    INFRARED = "infrared"
    ELECTRICAL = "electrical"
    RULE = "rule"
    FUSION = "fusion"

    @property
    def label_zh(self) -> str:
        return {
            "visible": "可见光",
            "infrared": "红外",
            "electrical": "电气时序",
            "rule": "规则",
            "fusion": "融合",
        }[self.value]


class RiskLevel(str, Enum):
    """风险等级，对应技术方案第六节的四档划分。"""

    NORMAL = "normal"
    ATTENTION = "attention"
    ABNORMAL = "abnormal"
    CRITICAL = "critical"

    @property
    def label_zh(self) -> str:
        return {
            "normal": "正常",
            "attention": "关注",
            "abnormal": "异常",
            "critical": "严重异常",
        }[self.value]


class ThermalSeverity(str, Enum):
    """红外热缺陷严重程度。"""

    NORMAL = "normal"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"

    @property
    def label_zh(self) -> str:
        return {
            "normal": "正常",
            "warning": "一般缺陷",
            "alarm": "严重缺陷",
            "critical": "危急缺陷",
        }[self.value]


# ---------------------------------------------------------------------------
# 序列化辅助
# ---------------------------------------------------------------------------
def to_jsonable(obj: Any) -> Any:
    """递归转换为 JSON 可序列化结构。

    与 ``dataclasses.asdict`` 的区别：本函数会把 Enum 转为其 value，
    Path 转为 str，并保留未知类型为 str，避免 json.dumps 抛 TypeError。
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return to_jsonable(obj.to_dict())
    return str(obj)


class SerializableMixin:
    """为 dataclass 提供 ``to_dict``。"""

    def to_dict(self) -> Dict[str, Any]:
        return {k: to_jsonable(v) for k, v in asdict(self).items()}  # type: ignore[call-overload]


# ---------------------------------------------------------------------------
# 几何
# ---------------------------------------------------------------------------
@dataclass
class BBox(SerializableMixin):
    """像素坐标系下的矩形框，原点在左上角。"""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_list(cls, values: List[float]) -> "BBox":
        if len(values) != 4:
            raise ValueError(f"bbox 需要 4 个数值，实际收到 {len(values)} 个：{values}")
        return cls(float(values[0]), float(values[1]), float(values[2]), float(values[3]))

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def iou(self, other: "BBox") -> float:
        """交并比。用于可见光缺陷与红外热点的空间位置匹配。"""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        union = self.area + other.area - inter
        return float(inter / union) if union > 0 else 0.0

    def intersect_ratio(self, other: "BBox") -> float:
        """交集面积占较小框面积的比例。

        对「大框套小框」的场景比 IoU 更合适：可见光上的接线端子框往往
        远小于红外热区框，此时 IoU 会很小，但二者确实指向同一位置。
        """
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        smaller = min(self.area, other.area)
        return float(inter / smaller) if smaller > 0 else 0.0

    def center_distance_ratio(self, other: "BBox") -> float:
        """中心点距离按两框平均对角线长度归一化。0 表示完全同心。"""
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
        diag = ((self.width + other.width) / 2.0) ** 2 + ((self.height + other.height) / 2.0) ** 2
        diag = diag ** 0.5
        return float(dist / diag) if diag > 0 else float("inf")


# ---------------------------------------------------------------------------
# 可见光分支
# ---------------------------------------------------------------------------
@dataclass
class Detection(SerializableMixin):
    """单个检测框。"""

    label: str
    confidence: float
    bbox: BBox
    source: str = Modality.VISIBLE.value
    category: str = "device"  # device | defect
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisibleResult(SerializableMixin):
    """可见光检测结果。"""

    image_path: str
    image_width: int
    image_height: int
    backend: str  # "yolo" | "heuristic_fallback"
    model_name: str
    detections: List[Detection] = field(default_factory=list)
    device_count: int = 0
    defect_count: int = 0
    defect_classes: List[str] = field(default_factory=list)
    max_defect_confidence: float = 0.0
    visual_score: float = 0.0
    annotated_image_path: Optional[str] = None
    elapsed_ms: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """无任何缺陷检出。"""
        return self.defect_count == 0

    @property
    def score_available(self) -> bool:
        """该分支的分数是否可用于融合。

        启发式兜底检测器只能定位设备区域、**无法识别缺陷**，其
        ``visual_score`` 恒为 0。这个 0 表示「没有能力判断」而非
        「判断为正常」。若让它参与融合，会以「视觉 0 分」的名义稀释其它
        模态：即使红外和时序都指向严重，加权后也到不了「严重异常」。
        因此这种情况按**模态缺失**处理——权重重新归一化，报告中标明原因。
        """
        return self.backend != "heuristic_fallback"


# ---------------------------------------------------------------------------
# 红外分支
# ---------------------------------------------------------------------------
@dataclass
class HotRegion(SerializableMixin):
    """一个红外热点区域。"""

    bbox: BBox
    max_temp_c: float
    mean_temp_c: float
    delta_temp_c: float
    area_px: int
    severity: ThermalSeverity = ThermalSeverity.NORMAL
    score: float = 0.0


@dataclass
class InfraredResult(SerializableMixin):
    """红外热成像检测结果。"""

    image_path: str
    image_width: int
    image_height: int
    backend: str  # "radiometric" | "gray_linear" | "pseudo_color_palette"
    # 环境温度 T0，仅作为相对温差 δ 的分母，不直接参与判级
    ambient_temp_c: float
    # 参考温度 T2（正常相对应点温度）。判级用的是它与最高温之差，
    # 而不是「最高温 − 环境温度」——后者会把盛夏里正常运行的设备判成异常。
    # 取值优先级见 InfraredDetector._reference_temperature
    baseline_temp_c: float
    baseline_source: str = "median"  # three_phase | median | ambient
    max_temp_c: float = 0.0
    mean_temp_c: float = 0.0
    # ΔT = max_temp_c − baseline_temp_c，即发热点与正常相对应点的温差
    delta_temp_c: float = 0.0
    # DL/T 664 相对温差 δ = (T1 − T2) / (T1 − T0) × 100%，T1 为发热点温度
    relative_delta_ratio: float = 0.0
    thermal_severity: ThermalSeverity = ThermalSeverity.NORMAL
    thermal_score: float = 0.0
    hot_regions: List[HotRegion] = field(default_factory=list)
    three_phase_temps: Optional[List[float]] = None
    three_phase_imbalance_c: Optional[float] = None
    temperature_map_path: Optional[str] = None
    annotated_image_path: Optional[str] = None
    elapsed_ms: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def hot_region_count(self) -> int:
        """提取到的热点区域总数（含温度正常但相对突出的区域）。"""
        return len(self.hot_regions)

    @property
    def abnormal_region_count(self) -> int:
        """达到告警及以上级别的热点数量。

        判级用的是这个而不是总数：三相设备本来就有三个接线端子，
        三个「温度正常只是比背景热」的区域不代表多点过热。
        """
        return sum(
            1 for r in self.hot_regions
            if r.severity != ThermalSeverity.NORMAL
        )


# ---------------------------------------------------------------------------
# 电气时序分支
# ---------------------------------------------------------------------------
@dataclass
class AnomalyEvent(SerializableMixin):
    """时序异常事件（一个异常窗口）。"""

    start_index: int
    end_index: int
    start_time: Optional[str]
    end_time: Optional[str]
    score: float
    channel: Optional[str] = None
    description: str = ""


@dataclass
class TimeseriesResult(SerializableMixin):
    """电气时序异常检测结果。"""

    csv_path: str
    channels: List[str] = field(default_factory=list)
    n_samples: int = 0
    n_windows: int = 0
    backend: str = "statistical"  # "transformer" | "statistical"
    model_name: str = ""
    is_anomaly: bool = False
    anomaly_ratio: float = 0.0
    electrical_score: float = 0.0
    threshold: float = 0.0
    max_zscore: float = 0.0
    # 业务可解释指标
    load_rise_ratio: float = 0.0
    temp_rise_ratio: Optional[float] = None
    load_trend_slope: float = 0.0
    voltage_deviation_ratio: float = 0.0
    current_imbalance_ratio: float = 0.0
    max_load_ratio: float = 0.0
    peak_to_peak: Dict[str, float] = field(default_factory=dict)
    events: List[AnomalyEvent] = field(default_factory=list)
    plot_path: Optional[str] = None
    elapsed_ms: float = 0.0
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 规则引擎
# ---------------------------------------------------------------------------
@dataclass
class RuleHit(SerializableMixin):
    """一条被触发的规则。"""

    rule_id: str
    name: str
    modality: str
    severity: float
    description: str = ""
    standard: str = ""
    advice: str = ""
    matched: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleResult(SerializableMixin):
    """规则引擎输出。"""

    rule_score: float = 0.0
    hits: List[RuleHit] = field(default_factory=list)
    evaluated_count: int = 0
    skipped_count: int = 0


# ---------------------------------------------------------------------------
# 融合
# ---------------------------------------------------------------------------
@dataclass
class FusionResult(SerializableMixin):
    """多模态融合结果。"""

    risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.NORMAL
    risk_level_name: str = "正常"
    color: str = "#16a34a"
    visual_score: float = 0.0
    thermal_score: float = 0.0
    electrical_score: float = 0.0
    rule_score: float = 0.0
    weights_used: Dict[str, float] = field(default_factory=dict)
    contributions: Dict[str, float] = field(default_factory=dict)
    consistency_bonus: float = 0.0
    modalities_present: List[str] = field(default_factory=list)
    modalities_missing: List[str] = field(default_factory=list)
    # 模态缺失原因：区分「未提供数据」与「提供了但该分支不具备判断能力」，
    # 避免读者把「权重被剔除」误读成「该模态判定为正常」
    missing_reasons: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 知识检索
# ---------------------------------------------------------------------------
@dataclass
class StandardReference(SerializableMixin):
    """一条检索到的标准/规程依据。"""

    doc_id: str
    title: str
    snippet: str
    score: float
    source_path: str = ""
    page: Optional[int] = None


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
@dataclass
class InspectionReport(SerializableMixin):
    """一次完整巡检的汇总结果。"""

    inspection_id: str
    created_at: str
    device_type: str = "transformer"
    device_name: str = ""
    location: str = ""
    operator: str = ""

    visible: Optional[VisibleResult] = None
    infrared: Optional[InfraredResult] = None
    timeseries: Optional[TimeseriesResult] = None

    rules: RuleResult = field(default_factory=RuleResult)
    fusion: FusionResult = field(default_factory=FusionResult)

    # 可见光缺陷与红外热点的最大空间重合度，由 risk.rule_engine 计算
    visual_thermal_iou: float = 0.0

    references: List[StandardReference] = field(default_factory=list)
    llm_analysis: str = ""
    llm_provider: str = "mock"
    report_markdown: str = ""

    elapsed_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def risk_level(self) -> RiskLevel:
        return self.fusion.risk_level


__all__ = [
    "Modality",
    "RiskLevel",
    "ThermalSeverity",
    "to_jsonable",
    "SerializableMixin",
    "BBox",
    "Detection",
    "VisibleResult",
    "HotRegion",
    "InfraredResult",
    "AnomalyEvent",
    "TimeseriesResult",
    "RuleHit",
    "RuleResult",
    "FusionResult",
    "StandardReference",
    "InspectionReport",
]
