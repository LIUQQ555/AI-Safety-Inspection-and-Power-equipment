"""核心公共模块：配置无关的数据契约与通用工具。"""

from backend.core.schemas import (  # noqa: F401
    AnomalyEvent,
    BBox,
    Detection,
    FusionResult,
    HotRegion,
    InfraredResult,
    InspectionReport,
    Modality,
    RiskLevel,
    RuleHit,
    RuleResult,
    StandardReference,
    ThermalSeverity,
    TimeseriesResult,
    VisibleResult,
    to_jsonable,
)

__all__ = [
    "Modality",
    "RiskLevel",
    "ThermalSeverity",
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
    "to_jsonable",
]
