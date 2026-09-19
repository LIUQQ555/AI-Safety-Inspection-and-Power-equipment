"""规则引擎。

对应技术方案第十五节「第二部分：规则判断」。

规则以 YAML 配置驱动（``config/rules.yaml``），支持四种条件类型：

    ``threshold``  字段 运算符 数值           如 delta_temp_c >= 20
    ``contains``   列表字段包含某元素          如 defect_classes 含 oil_leak
    ``any_of``     任一子条件成立
    ``all_of``     全部子条件成立

RuleScore 取所有命中规则 severity 的最大值——**不做累加**。
原因：多条规则往往由同一根因触发（例如温度高同时触发绝对温度与温差两条），
累加会系统性放大评分，使资产看起来比实际更危险。
规则命中条目会完整写入报告，供人工核对。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from backend.core.schemas import (
    InfraredResult,
    RuleHit,
    RuleResult,
    TimeseriesResult,
    VisibleResult,
)
from backend.vision import labels as lbl

logger = logging.getLogger(__name__)

_OPERATORS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


class ConditionError(ValueError):
    """规则条件配置错误。"""


def _evaluate_condition(condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """递归求值单个条件。字段缺失时返回 False（视为不触发）。"""
    ctype = str(condition.get("type", "threshold")).lower()

    if ctype in ("any_of", "all_of"):
        subconditions = condition.get("subconditions") or []
        if not subconditions:
            raise ConditionError(f"{ctype} 条件缺少 subconditions")
        results = [_evaluate_condition(sub, context) for sub in subconditions]
        return any(results) if ctype == "any_of" else all(results)

    field = condition.get("field")
    if not field:
        raise ConditionError(f"条件缺少 field：{condition}")

    if field not in context or context[field] is None:
        return False

    actual = context[field]

    if ctype == "contains":
        expected = condition.get("value")
        if isinstance(actual, (list, tuple, set)):
            return any(str(item) == str(expected) or lbl.to_en(item) == lbl.to_en(expected)
                       for item in actual)
        return str(expected) in str(actual)

    if ctype == "threshold":
        operator = str(condition.get("operator", ">="))
        if operator not in _OPERATORS:
            raise ConditionError(f"不支持的运算符 {operator!r}，可选：{list(_OPERATORS)}")
        expected = condition.get("value")
        try:
            return bool(_OPERATORS[operator](float(actual), float(expected)))
        except (TypeError, ValueError):
            logger.warning("规则条件数值转换失败：%s %s %s", actual, operator, expected)
            return False

    raise ConditionError(f"未知条件类型 {ctype!r}")


class RuleEngine:
    """基于配置的规则引擎。"""

    def __init__(self, config) -> None:
        self.config = config
        self.rules: List[Dict[str, Any]] = []
        for rule in config.rules:
            rule_dict = dict(rule)
            if not rule_dict.get("enabled", True):
                continue
            if not rule_dict.get("id"):
                logger.warning("跳过缺少 id 的规则：%s", rule_dict.get("name"))
                continue
            self.rules.append(rule_dict)

    def evaluate(self, context: Dict[str, Any]) -> RuleResult:
        """对上下文求值，返回命中的规则与综合规则分。"""
        hits: List[RuleHit] = []
        evaluated = 0
        skipped = 0

        for rule in self.rules:
            condition = rule.get("condition") or {}
            try:
                matched = _evaluate_condition(condition, context)
            except ConditionError as exc:
                logger.error("规则 %s 配置错误，已跳过：%s", rule.get("id"), exc)
                skipped += 1
                continue

            evaluated += 1
            if not matched:
                continue

            # 记录命中时参与判断的实际取值，便于人工复核
            matched_fields: Dict[str, Any] = {}
            for field in _collect_fields(condition):
                if field in context:
                    value = context[field]
                    matched_fields[field] = value if isinstance(value, (int, float, str, bool)) else str(value)

            hits.append(RuleHit(
                rule_id=str(rule.get("id")),
                name=str(rule.get("name", rule.get("id"))),
                modality=str(rule.get("modality", "")),
                severity=float(rule.get("severity", 50)),
                description=str(rule.get("description", "")).strip(),
                standard=str(rule.get("standard", "")).strip(),
                advice=str(rule.get("advice", "")).strip(),
                matched=matched_fields,
            ))

        # RuleScore = max(severity)，不累加（理由见模块文档）
        rule_score = max((hit.severity for hit in hits), default=0.0)

        hits.sort(key=lambda hit: hit.severity, reverse=True)
        return RuleResult(
            rule_score=round(float(rule_score), 2),
            hits=hits,
            evaluated_count=evaluated,
            skipped_count=skipped,
        )


def _collect_fields(condition: Dict[str, Any]) -> List[str]:
    """递归收集条件中引用到的字段名。"""
    fields: List[str] = []
    if "field" in condition:
        fields.append(str(condition["field"]))
    for sub in condition.get("subconditions") or []:
        fields.extend(_collect_fields(sub))
    return fields


def compute_visual_thermal_iou(
    visible: Optional[VisibleResult],
    infrared: Optional[InfraredResult],
) -> float:
    """可见光缺陷框与红外热点的最大空间重合度。

    使用 ``intersect_ratio``（交集 / 较小框面积）而非标准 IoU：
    可见光上的缺陷框（如接线端子）通常远小于红外热区框，标准 IoU 会很小，
    但二者确实指向同一位置。

    没有可见光缺陷或没有红外热点时返回 0。
    """
    if visible is None or infrared is None:
        return 0.0
    defects = [d for d in visible.detections if d.category == "defect"]
    if not defects or not infrared.hot_regions:
        return 0.0

    best = 0.0
    for detection in defects:
        for region in infrared.hot_regions:
            best = max(best, detection.bbox.intersect_ratio(region.bbox))
    return round(float(best), 4)


def build_rule_context(
    visible: Optional[VisibleResult],
    infrared: Optional[InfraredResult],
    timeseries: Optional[TimeseriesResult],
    visual_thermal_iou: float = 0.0,
) -> Dict[str, Any]:
    """把三个分支的结果汇总为扁平上下文，供规则求值使用。

    可用字段完整列表见 ``config/rules.yaml`` 顶部注释。
    """
    context: Dict[str, Any] = {
        "visual_thermal_iou": visual_thermal_iou,
        "modalities_present": sum(x is not None for x in (visible, infrared, timeseries)),
    }

    if visible is not None:
        context.update({
            "visual_score": visible.visual_score,
            "defect_count": visible.defect_count,
            "defect_classes": list(visible.defect_classes),
            "max_defect_confidence": visible.max_defect_confidence,
            "device_count": visible.device_count,
            "device_classes": sorted({
                lbl.to_zh(d.label) for d in visible.detections if d.category == "device"
            }),
        })

    if infrared is not None:
        context.update({
            "thermal_score": infrared.thermal_score,
            "max_temp_c": infrared.max_temp_c,
            "mean_temp_c": infrared.mean_temp_c,
            "ambient_temp_c": infrared.ambient_temp_c,
            # ΔT = 发热点温度 − 正常相对应点温度（不是减环境温度，见 infrared_detector）
            "baseline_temp_c": infrared.baseline_temp_c,
            "baseline_source": infrared.baseline_source,
            "delta_temp_c": infrared.delta_temp_c,
            "relative_delta_ratio": infrared.relative_delta_ratio,
            "thermal_severity": infrared.thermal_severity.value,
            # 提取到的热点总数
            "hot_region_count": infrared.hot_region_count,
            # 达到告警及以上级别的热点数 —— 规则应优先用这个
            "abnormal_region_count": infrared.abnormal_region_count,
            "three_phase_imbalance_c": infrared.three_phase_imbalance_c,
        })

    if timeseries is not None:
        context.update({
            "electrical_score": timeseries.electrical_score,
            "is_anomaly": timeseries.is_anomaly,
            "anomaly_ratio": timeseries.anomaly_ratio,
            "load_rise_ratio": timeseries.load_rise_ratio,
            "temp_rise_ratio": timeseries.temp_rise_ratio,
            "load_trend_slope": timeseries.load_trend_slope,
            "voltage_deviation_ratio": timeseries.voltage_deviation_ratio,
            "current_imbalance_ratio": timeseries.current_imbalance_ratio,
            "max_load_ratio": timeseries.max_load_ratio,
            "max_zscore": timeseries.max_zscore,
        })
        if timeseries.peak_to_peak:
            context["peak_to_peak"] = timeseries.peak_to_peak

    # 融合类规则需要的分数在融合前即可确定（它们本身就是三个分支的原始分）
    context.setdefault("visual_score", visible.visual_score if visible else 0.0)
    context.setdefault("thermal_score", infrared.thermal_score if infrared else 0.0)
    context.setdefault("electrical_score", timeseries.electrical_score if timeseries else 0.0)

    return context


__all__ = [
    "RuleEngine",
    "RuleResult",
    "ConditionError",
    "compute_visual_thermal_iou",
    "build_rule_context",
]
