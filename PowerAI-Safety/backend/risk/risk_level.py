"""风险分级。

分级边界全部来自 ``config.fusion.levels``（技术方案第六节：阈值必须是配置项，
不得由大模型生成）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from backend.core.schemas import RiskLevel

# 配置缺失时的兜底分级（与技术方案第六节给出的四档一致）
DEFAULT_LEVELS: List[Dict[str, Any]] = [
    {"name": "正常", "code": "normal", "min": 0, "max": 30, "color": "#16a34a"},
    {"name": "关注", "code": "attention", "min": 30, "max": 60, "color": "#ca8a04"},
    {"name": "异常", "code": "abnormal", "min": 60, "max": 80, "color": "#ea580c"},
    {"name": "严重异常", "code": "critical", "min": 80, "max": 100, "color": "#dc2626"},
]


def get_levels(config) -> List[Dict[str, Any]]:
    """从配置读取分级表，缺失或非法时返回默认分级。"""
    try:
        raw = config.fusion.get("levels")
    except AttributeError:
        return DEFAULT_LEVELS
    if not raw:
        return DEFAULT_LEVELS
    levels = [dict(item) for item in raw]
    levels.sort(key=lambda item: float(item.get("min", 0)))
    return levels or DEFAULT_LEVELS


def classify_risk(score: float, levels: List[Dict[str, Any]]) -> Tuple[RiskLevel, str, str]:
    """按分数返回 ``(RiskLevel, 中文名, 颜色)``。

    区间约定为左闭右开；最高档取到上边界（含 100）。
    """
    score = max(0.0, min(100.0, float(score)))
    last = levels[-1]

    for index, level in enumerate(levels):
        low = float(level.get("min", 0))
        high = float(level.get("max", 100))
        is_last = index == len(levels) - 1
        if (low <= score < high) or (is_last and low <= score <= high):
            code = str(level.get("code", "normal"))
            try:
                enum_value = RiskLevel(code)
            except ValueError:
                enum_value = RiskLevel.NORMAL
            return enum_value, str(level.get("name", enum_value.label_zh)), str(
                level.get("color", "#16a34a")
            )

    code = str(last.get("code", "normal"))
    try:
        enum_value = RiskLevel(code)
    except ValueError:
        enum_value = RiskLevel.NORMAL
    return enum_value, str(last.get("name", enum_value.label_zh)), str(
        last.get("color", "#16a34a")
    )


__all__ = ["DEFAULT_LEVELS", "get_levels", "classify_risk"]
