"""模板化大模型提供方（默认，完全离线）。

用途：在没有配置 API Key、也没有 GPU 可跑本地 VLM 的情况下，
让「检测 → 融合 → 检索 → 解释 → 报告」整条链路可以完整跑通并演示。

**这不是一个假实现，而是一个确定性的事实转述器**：
    * 它只复述结构化检测结果中真实存在的字段，不编造任何数值；
    * 所有阈值判据都来自 ``config/*.yaml`` 与规则引擎的命中记录；
    * 输出由事实驱动组装，相同的输入永远得到相同的输出。

它的局限也很明确：**没有语言模型的理解与推理能力**，无法处理
检测结果之外的新情况，行文也较模板化。生产环境应切换到真实 VLM/LLM
（``llm.provider: dashscope`` 或 ``openai``），此时本模块的产物可作为
对照基线（baseline），用于评估真实模型是否带来增益。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from backend.llm.base import LLMProvider, LLMRequest

logger = logging.getLogger(__name__)


def _fmt(value: Any, unit: str = "", digits: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


class MockProvider(LLMProvider):
    """基于规则模板的确定性文本生成。"""

    name = "mock"
    supports_vision = False

    def generate(self, request: LLMRequest) -> str:
        facts = request.facts or {}
        if not facts:
            return (
                "【说明】\n"
                "当前使用模板化生成（MockProvider），但未收到结构化事实，无法生成分析。\n"
                "请通过 LLMRequest.facts 传入检测结果。"
            )

        sections: List[str] = []
        sections.append(self._section_detection(facts))
        sections.append(self._section_analysis(facts))

        evidence = self._section_evidence(facts)
        if evidence:
            sections.append(evidence)

        sections.append(self._section_advice(facts))
        return "\n\n".join(section for section in sections if section)

    # ------------------------------------------------------------------
    def _section_detection(self, facts: Dict[str, Any]) -> str:
        device = facts.get("device_name") or "电力设备"
        lines: List[str] = []

        visible = facts.get("visible") or {}
        if visible:
            defects = visible.get("defect_classes") or []
            if defects:
                lines.append(
                    f"可见光检测在{device}上识别到外观缺陷：{'、'.join(defects)}"
                    f"（最高置信度 {_fmt(visible.get('max_defect_confidence'), '', 2)}）。"
                )
            else:
                count = visible.get("device_count", 0)
                lines.append(
                    f"可见光检测识别到 {count} 个设备/部件目标，未发现外观缺陷。"
                )
            if visible.get("backend") == "heuristic_fallback":
                lines.append(
                    "注意：可见光分支当前运行在启发式降级模式，"
                    "仅能定位显著区域，不具备缺陷分类能力，该结论不构成安全判定依据。"
                )

        infrared = facts.get("infrared") or {}
        if infrared:
            lines.append(
                f"红外检测测得最高温度 {_fmt(infrared.get('max_temp_c'), ' ℃')}，"
                f"相对环境温升 ΔT {_fmt(infrared.get('delta_temp_c'), ' K')}，"
                f"热缺陷等级为{infrared.get('thermal_severity_zh', '—')}。"
            )
            imbalance = infrared.get("three_phase_imbalance_c")
            if imbalance is not None:
                lines.append(f"三相最大温差 {_fmt(imbalance, ' K')}。")

        timeseries = facts.get("timeseries") or {}
        if timeseries:
            backend = timeseries.get("backend")
            lines.append(
                f"电气时序分析（{('Transformer 重构误差模型' if backend == 'transformer' else '统计判据')}）"
                f"给出异常分 {_fmt(timeseries.get('electrical_score'))}，"
                f"负荷相对基线变化 {_fmt((timeseries.get('load_rise_ratio') or 0) * 100, '%')}，"
                f"最大负载率 {_fmt(timeseries.get('max_load_ratio'), '', 2)}。"
            )

        if not lines:
            return "【检测结果】\n未提供任何模态的检测数据。"

        risk_line = (
            f"\n综合风险评分 {_fmt(facts.get('risk_score'), ' 分')}，"
            f"风险等级为**{facts.get('risk_level_name', '—')}**。"
        )
        return "【检测结果】\n" + "\n".join(lines) + risk_line

    # ------------------------------------------------------------------
    def _section_analysis(self, facts: Dict[str, Any]) -> str:
        """归因分析：只在多模态同时给出证据时才断言关联。"""
        paragraphs: List[str] = []

        visible = facts.get("visible") or {}
        infrared = facts.get("infrared") or {}
        timeseries = facts.get("timeseries") or {}

        has_thermal = bool(infrared) and infrared.get("thermal_score", 0) >= 50
        has_load_rise = (timeseries.get("load_rise_ratio") or 0) >= 0.15
        has_defect = bool(visible.get("defect_classes"))
        iou = facts.get("visual_thermal_iou") or 0.0

        if has_thermal and has_load_rise and has_defect and iou > 0.05:
            paragraphs.append(
                f"可见光缺陷位置与红外热点在空间上重合（重合度 {iou:.2f}），"
                "且时序数据显示近期负荷明显升高。三者互相印证，"
                "指向**载流回路在持续高负荷下的接触不良或接触电阻增大**"
                "——负荷升高使发热加剧，发热又进一步抬高接触电阻，形成正反馈。"
            )
        elif has_thermal and has_load_rise:
            paragraphs.append(
                "红外温升与时序负荷上升同时出现，发热与近期负荷增长存在时间上的关联，"
                "但缺少可见光缺陷定位，无法确认具体发热部位。"
            )
        elif has_thermal and has_defect and iou > 0.05:
            paragraphs.append(
                f"红外热点与可见光缺陷位置重合（重合度 {iou:.2f}），"
                "两路数据指向同一部位，缺陷定位可信度较高。"
            )
        elif has_thermal:
            paragraphs.append(
                "红外检测到异常温升。该结论仅由热成像单模态支持，"
                "尚无其它模态数据印证，建议补充可见光检查与负荷数据后复核。"
            )
        elif has_defect:
            paragraphs.append(
                "可见光检测到外观缺陷。该结论仅由图像单模态支持，"
                "建议结合红外测温判断该缺陷是否已引起发热。"
            )
        elif timeseries and timeseries.get("is_anomaly"):
            paragraphs.append(
                "时序数据存在异常模式，但缺少图像模态，无法定位到具体设备部位。"
            )
        else:
            paragraphs.append(
                "各模态均未检出显著异常，设备运行状态在本次检测范围内未发现异常迹象。"
            )

        thermal_notes = infrared.get("notes") or []
        for note in thermal_notes:
            if "调色板" in note or "色距" in note:
                paragraphs.append(f"数据质量提示：{note}")
                break

        if not paragraphs:
            paragraphs.append("无足够数据支撑归因分析。")
        return "【综合分析】\n" + "\n".join(paragraphs)

    # ------------------------------------------------------------------
    def _section_evidence(self, facts: Dict[str, Any]) -> str:
        rules = facts.get("rule_hits") or []
        references = facts.get("references") or []

        lines: List[str] = []
        if rules:
            lines.append("本次判定命中的规则：")
            for hit in rules[:6]:
                matched = hit.get("matched") or {}
                matched_text = "，".join(
                    f"{key}={value}" for key, value in list(matched.items())[:4]
                )
                line = f"  • [{hit.get('rule_id')}] {hit.get('name')}（权重 {hit.get('severity')}）"
                if matched_text:
                    line += f"  实测：{matched_text}"
                lines.append(line)

        if references:
            lines.append("")
            lines.append("知识库检索到的相关条款：")
            for ref in references[:4]:
                location = f"{ref.get('title', '')}"
                if ref.get("page"):
                    location += f" 第{ref['page']}页"
                lines.append(f"  • {location}（相关度 {ref.get('score', 0):.3f}）")

        if not lines:
            return ""
        return "【判定依据】\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    def _section_advice(self, facts: Dict[str, Any]) -> str:
        rules = facts.get("rule_hits") or []
        level = facts.get("risk_level_name", "")

        # 优先使用规则自带的处置建议，按严重度降序去重
        advice: List[str] = []
        seen: set[str] = set()
        for hit in sorted(rules, key=lambda item: item.get("severity", 0), reverse=True):
            text = (hit.get("advice") or "").strip()
            if text and text not in seen:
                seen.add(text)
                advice.append(text)

        timeframe = {
            "正常": "按正常周期巡检即可。",
            "关注": "纳入下次计划性检修，并适当提高红外测温频次。",
            "异常": "建议在两周内安排处理，处理前加强监测。",
            "严重异常": "建议立即安排停电检查或采取降负荷措施，避免缺陷进一步发展。",
        }.get(level, "")

        lines: List[str] = []
        if timeframe:
            lines.append(timeframe)
        for index, text in enumerate(advice[:5], start=1):
            lines.append(f"{index}. {text}")
        if not advice and not timeframe:
            lines.append("未触发任何处置规则，维持正常运维安排。")

        lines.append("")
        lines.append(
            "以上结论由 AI 依据配置阈值与知识库生成，需由运维人员结合现场实际情况复核。"
        )
        return "【处置建议】\n" + "\n".join(lines)


__all__ = ["MockProvider"]
