"""类别标签的中英文映射。

数据集（CPLID / IDDD / thermal-images-equip）以及自训练 YOLO 权重的类别名
可能是英文、拼音或中文。本模块提供统一的双向映射，使检测器、规则引擎与
报告生成器始终使用一致的中文标签。
"""

from __future__ import annotations

from typing import Dict

# 英文规范名 → 中文
CLASS_ZH: Dict[str, str] = {
    # --- 设备类 ---
    "transformer": "变压器",
    "bushing": "套管",
    "terminal": "接线端子",
    "conservator": "储油柜",
    "radiator": "散热器",
    "insulator": "绝缘子",
    "arrester": "避雷器",
    "breaker": "断路器",
    "cable": "电缆",
    "tower": "杆塔",
    "tank": "油箱",
    "cooling_fan": "冷却风扇",
    # --- 缺陷类 ---
    "oil_leak": "渗漏油",
    "rust": "锈蚀",
    "deformation": "变形",
    "foreign_object": "悬挂异物",
    "broken_part": "部件破损",
    "oil_stain": "油污",
    "nest": "鸟巢",
    "kite": "风筝",
    "balloon": "气球",
    "flashover": "闪络痕迹",
    "crack": "裂纹",
    "corrosion": "腐蚀",
    # --- 红外专用 ---
    "hotspot": "热点",
    "hot_spot": "热点",
    "overheat": "过热",
    # --- 绝缘子缺陷（CPLID/IDDD 常用标注）---
    "defect": "缺陷",
    "broken": "破损",
    "flashover_damage": "闪络损伤",
    "self_exploded": "自爆",
    "pollution": "污秽",
}

# 中文（及常见别名）→ 英文规范名
CLASS_EN: Dict[str, str] = {zh: en for en, zh in CLASS_ZH.items()}

# 额外别名，覆盖数据集里的各种写法
_ALIASES: Dict[str, str] = {
    "绝缘子串": "insulator",
    "导线": "cable",
    "鸟窝": "nest",
    "异物": "foreign_object",
    "悬挂物": "foreign_object",
    "漏油": "oil_leak",
    "渗油": "oil_leak",
    "端子": "terminal",
    "接头": "terminal",
    "引线接头": "terminal",
    "oil-leak": "oil_leak",
    "oil leak": "oil_leak",
    "foreign object": "foreign_object",
    "broken part": "broken_part",
    "hot spot": "hotspot",
}
CLASS_EN.update(_ALIASES)


def to_zh(name: str) -> str:
    """英文/别名 → 中文显示名。未知名称原样返回。"""
    if not name:
        return ""
    key = str(name).strip()
    if key in CLASS_ZH:
        return CLASS_ZH[key]
    lowered = key.lower().replace("-", "_").replace(" ", "_")
    if lowered in CLASS_ZH:
        return CLASS_ZH[lowered]
    if key in CLASS_EN:
        return CLASS_ZH.get(CLASS_EN[key], key)
    return key


def to_en(name: str) -> str:
    """中文/别名 → 英文规范名。未知名称做规范化后返回。"""
    if not name:
        return ""
    key = str(name).strip()
    if key in CLASS_EN:
        return CLASS_EN[key]
    lowered = key.lower().replace("-", "_").replace(" ", "_")
    if lowered in CLASS_ZH:
        return lowered
    return lowered


def normalize(name: str) -> str:
    """统一到英文规范名（``to_en`` 的别名，语义更明确）。"""
    return to_en(name)


__all__ = ["CLASS_ZH", "CLASS_EN", "to_zh", "to_en", "normalize"]
