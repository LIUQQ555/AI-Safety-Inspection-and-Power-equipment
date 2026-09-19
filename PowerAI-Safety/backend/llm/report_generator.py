"""巡检报告生成。

对应技术方案第十八节创新点四「自动生成结构化巡检报告」。

报告同时产出两种形态：
    * **结构化字段**（``InspectionReport`` dataclass）——供前端渲染与入库
    * **Markdown 全文** ——供导出、归档与人工阅读

大模型只负责「四、综合分析」一节。检测数据、规则判定、风险等级、
依据标准全部由本地模块给出，大模型无权修改，也无法凭空引入结论。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.core.schemas import InspectionReport
from backend.llm.base import LLMProvider, LLMRequest

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是一名电力设备状态检修专家，负责协助运维人员理解 AI 巡检系统的检测结果。

工作准则：
1. 你的输入是已经由检测模型和规则引擎计算好的结构化结果，你**不得修改任何数值、阈值或风险等级**。
2. 你的任务是解释、归因和建议，不是重新检测。若数据不足以支撑某个结论，必须明确指出「数据不足」，不得推测。
3. 引用标准时必须使用输入中提供的知识库条款，不要凭记忆引用未提供的标准编号或条款内容。
4. 语言简洁、专业、面向现场运维人员，避免空话。全文控制在 600 字以内。
5. 如输入中标注了某模态处于降级模式（例如启发式检测、统计判据），必须在分析中说明该结论的可靠性受限。
"""


class ReportGenerator:
    """报告生成器。"""

    def __init__(self, config, provider: LLMProvider) -> None:
        self.config = config
        self.provider = provider
        self.title = str(config.report.get("title", "电力设备AI智能巡检报告"))
        self.disclaimer = str(config.report.get("disclaimer", "")).strip()
        self.include_raw_scores = bool(config.report.get("include_raw_scores", True))
        self.max_tokens = int(config.llm.get("max_tokens", 1500))
        self.temperature = float(config.llm.get("temperature", 0.2))

    # ------------------------------------------------------------------
    # 交给大模型的内容
    # ------------------------------------------------------------------
    def build_facts(self, report: InspectionReport) -> Dict[str, Any]:
        """把报告压缩成结构化事实字典。

        MockProvider 直接消费本字典；真实模型则读 prompt。
        两条路径共享同一份事实，保证「输入一致」。
        """
        facts: Dict[str, Any] = {
            "device_type": report.device_type,
            "device_name": report.device_name or self.config.focus_device_name,
            "location": report.location,
            "created_at": report.created_at,
            "risk_score": report.fusion.risk_score,
            "risk_level_name": report.fusion.risk_level_name,
            "visual_thermal_iou": float(report.visual_thermal_iou or 0.0),
        }

        if report.visible is not None:
            visible = report.visible
            facts["visible"] = {
                "backend": visible.backend,
                "model_name": visible.model_name,
                "device_count": visible.device_count,
                "defect_count": visible.defect_count,
                "defect_classes": list(visible.defect_classes),
                "max_defect_confidence": visible.max_defect_confidence,
                "visual_score": visible.visual_score,
            }

        if report.infrared is not None:
            infrared = report.infrared
            facts["infrared"] = {
                "backend": infrared.backend,
                "max_temp_c": infrared.max_temp_c,
                "mean_temp_c": infrared.mean_temp_c,
                "delta_temp_c": infrared.delta_temp_c,
                "ambient_temp_c": infrared.ambient_temp_c,
                "thermal_severity": infrared.thermal_severity.value,
                "thermal_severity_zh": infrared.thermal_severity.label_zh,
                "thermal_score": infrared.thermal_score,
                "hot_region_count": len(infrared.hot_regions),
                "three_phase_imbalance_c": infrared.three_phase_imbalance_c,
                "notes": list(infrared.notes),
            }

        if report.timeseries is not None:
            ts = report.timeseries
            facts["timeseries"] = {
                "backend": ts.backend,
                "electrical_score": ts.electrical_score,
                "is_anomaly": ts.is_anomaly,
                "anomaly_ratio": ts.anomaly_ratio,
                "load_rise_ratio": ts.load_rise_ratio,
                "temp_rise_ratio": ts.temp_rise_ratio,
                "max_load_ratio": ts.max_load_ratio,
                "voltage_deviation_ratio": ts.voltage_deviation_ratio,
                "current_imbalance_ratio": ts.current_imbalance_ratio,
                "n_samples": ts.n_samples,
                "n_windows": ts.n_windows,
            }

        facts["rule_hits"] = [
            {
                "rule_id": hit.rule_id,
                "name": hit.name,
                "severity": hit.severity,
                "description": hit.description,
                "standard": hit.standard,
                "advice": hit.advice,
                "matched": hit.matched,
            }
            for hit in report.rules.hits
        ]

        facts["references"] = [
            {
                "title": ref.title,
                "doc_id": ref.doc_id,
                "snippet": ref.snippet,
                "score": ref.score,
                "page": ref.page,
            }
            for ref in report.references
        ]

        return facts

    def build_prompt(self, facts: Dict[str, Any]) -> str:
        """构造给真实大模型的文本提示。"""
        lines: List[str] = [
            f"设备：{facts.get('device_name')}（{facts.get('device_type')}）",
            f"检测时间：{facts.get('created_at')}",
            f"系统综合风险评分：{facts.get('risk_score')} / 100，等级：{facts.get('risk_level_name')}",
            "",
        ]

        visible = facts.get("visible")
        if visible:
            lines.append(
                f"【可见光】检出设备 {visible['device_count']} 个，缺陷 {visible['defect_count']} 处，"
                f"缺陷类型：{('、'.join(visible['defect_classes']) or '无')}，"
                f"最高置信度 {visible['max_defect_confidence']:.2f}，视觉分 {visible['visual_score']}。"
            )

        infrared = facts.get("infrared")
        if infrared:
            lines.append(
                f"【红外】最高温度 {infrared['max_temp_c']} ℃，环境温度 {infrared['ambient_temp_c']} ℃，"
                f"温升 ΔT {infrared['delta_temp_c']} K，热缺陷等级：{infrared['thermal_severity_zh']}，"
                f"热风险分 {infrared['thermal_score']}，热点数量 {infrared['hot_region_count']}。"
            )

        timeseries = facts.get("timeseries")
        if timeseries:
            lines.append(
                f"【电气时序】异常分 {timeseries['electrical_score']}，"
                f"是否异常：{'是' if timeseries['is_anomaly'] else '否'}，"
                f"负荷相对基线变化 {timeseries['load_rise_ratio']:.1%}，"
                f"最大负载率 {timeseries['max_load_ratio']:.2f}，"
                f"电压偏差 {timeseries['voltage_deviation_ratio']:.2%}，"
                f"三相电流不平衡度 {timeseries['current_imbalance_ratio']:.2%}。"
            )

        if facts.get("visual_thermal_iou"):
            lines.append(f"【空间关联】可见光缺陷与红外热点最大重合度 {facts['visual_thermal_iou']:.2f}。")

        rules = facts.get("rule_hits") or []
        if rules:
            lines.append("")
            lines.append("【已命中规则】")
            for hit in rules:
                matched = hit.get("matched") or {}
                detail = "，".join(f"{k}={v}" for k, v in list(matched.items())[:4])
                lines.append(f"- {hit['name']}（{hit['rule_id']}，权重 {hit['severity']}）{detail}")

        references = facts.get("references") or []
        if references:
            lines.append("")
            lines.append("【知识库检索结果】")
            for ref in references:
                page = f" 第{ref['page']}页" if ref.get("page") else ""
                lines.append(f"- {ref['title']}{page}（相关度 {ref['score']:.3f}）：{ref['snippet'][:200]}")

        lines.append("")
        lines.append(
            "请基于以上事实，按以下结构输出中文分析（不要重复罗列数据，600 字以内）：\n"
            "【综合分析】说明异常的性质与可能原因，以及各模态结论之间的印证关系；\n"
            "【处置建议】给出具体、可执行的下一步动作，并注明所依据的标准条款。"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def generate_analysis(self, report: InspectionReport) -> str:
        """调用大模型生成分析文本。"""
        facts = self.build_facts(report)
        prompt = self.build_prompt(facts)

        images: List[str] = []
        if self.provider.supports_vision:
            # 只把可见光标注图与红外原图交给 VLM，避免一次发送过多图像
            if report.visible and report.visible.annotated_image_path:
                images.append(report.visible.annotated_image_path)
            if report.infrared and report.infrared.image_path:
                images.append(report.infrared.image_path)

        request = LLMRequest(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            images=images,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            facts=facts,
        )

        try:
            text = self.provider.generate(request)
        except Exception as exc:  # 兜底：provider 不应抛异常，但这里再保护一层
            logger.error("大模型生成失败：%s", exc)
            text = f"【大模型调用失败】\n原因：{exc}"

        return (text or "").strip()

    # ------------------------------------------------------------------
    # Markdown 渲染
    # ------------------------------------------------------------------
    def render_markdown(self, report: InspectionReport) -> str:
        """把报告渲染为 Markdown 全文。"""
        lines: List[str] = []
        fusion = report.fusion
        device_name = report.device_name or self.config.focus_device_name

        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"**报告编号：** {report.inspection_id}　　"
                     f"**检测时间：** {report.created_at}")
        lines.append("")
        lines.append(f"**设备：** {device_name}　　"
                     f"**位置：** {report.location or '—'}　　"
                     f"**检测人：** {report.operator or '—'}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # --- 风险结论 ---
        lines.append("## 一、风险结论")
        lines.append("")
        lines.append(f"> **综合风险评分：{fusion.risk_score:.1f} / 100　→　"
                     f"风险等级：{fusion.risk_level_name}**")
        lines.append("")

        if self.include_raw_scores:
            lines.append("| 模态 | 得分（0-100） | 权重 | 加权贡献 |")
            lines.append("| --- | ---: | ---: | ---: |")
            names = {"visual": "可见光检测", "thermal": "红外热成像",
                     "electrical": "电气时序", "rule": "规则判定"}
            scores = {"visual": fusion.visual_score, "thermal": fusion.thermal_score,
                      "electrical": fusion.electrical_score, "rule": fusion.rule_score}
            for key in ("visual", "thermal", "electrical", "rule"):
                if key not in fusion.weights_used:
                    continue
                lines.append(
                    f"| {names[key]} | {scores[key]:.1f} | "
                    f"{fusion.weights_used[key]:.2f} | {fusion.contributions.get(key, 0):.1f} |"
                )
            if fusion.consistency_bonus:
                lines.append(f"| 多模态一致性加成 | — | — | +{fusion.consistency_bonus:.1f} |")
            lines.append(f"| **合计** | | | **{fusion.risk_score:.1f}** |")
            lines.append("")

            if fusion.modalities_missing:
                names_short = {"visual": "可见光", "thermal": "红外", "electrical": "电气时序"}
                missing = "、".join(names_short.get(k, k) for k in fusion.modalities_missing)
                lines.append(f"*说明：本次未提供{missing}数据，其权重已按比例重新分配到其余模态。*")
                lines.append("")

        # --- 检测结果 ---
        lines.append("## 二、检测结果明细")
        lines.append("")

        if report.visible is not None:
            visible = report.visible
            lines.append(f"### 2.1 可见光检测")
            lines.append("")
            lines.append(f"- 检测后端：`{visible.backend}`（{visible.model_name}）")
            lines.append(f"- 检出设备/部件：{visible.device_count} 个")
            lines.append(f"- 检出缺陷：{visible.defect_count} 处")
            if visible.defect_classes:
                lines.append(f"- 缺陷类型：{'、'.join(visible.defect_classes)}")
                lines.append(f"- 最高缺陷置信度：{visible.max_defect_confidence:.2f}")
                lines.append("")
                lines.append("| 缺陷类型 | 置信度 | 位置 (x1, y1, x2, y2) |")
                lines.append("| --- | ---: | --- |")
                for detection in visible.detections:
                    if detection.category != "defect":
                        continue
                    from backend.vision import labels as lbl
                    box = ", ".join(f"{v:.0f}" for v in detection.bbox.to_list())
                    lines.append(f"| {lbl.to_zh(detection.label)} | {detection.confidence:.2f} | {box} |")
            else:
                lines.append("- 未发现外观缺陷")
            for note in visible.notes:
                lines.append(f"- ⚠️ {note}")
            lines.append("")

        if report.infrared is not None:
            infrared = report.infrared
            lines.append("### 2.2 红外热成像检测")
            lines.append("")
            lines.append(f"- 温度解析方式：`{infrared.backend}`")
            lines.append(f"- 环境参考温度：{infrared.ambient_temp_c:.1f} ℃")
            lines.append(f"- 最高温度：{infrared.max_temp_c:.1f} ℃")
            lines.append(f"- 平均温度：{infrared.mean_temp_c:.1f} ℃")
            lines.append(f"- 最大温升 ΔT：{infrared.delta_temp_c:.1f} K")
            lines.append(f"- 热缺陷等级：{infrared.thermal_severity.label_zh}")
            if infrared.three_phase_imbalance_c is not None:
                temps = infrared.three_phase_temps or []
                lines.append(f"- 三相温度：{', '.join(f'{t:.1f} ℃' for t in temps)}")
                lines.append(f"- 三相最大温差：{infrared.three_phase_imbalance_c:.1f} K")
            if infrared.hot_regions:
                lines.append("")
                lines.append("| # | 最高温度 (℃) | 平均温度 (℃) | ΔT (K) | 面积 (px) | 等级 |")
                lines.append("| ---: | ---: | ---: | ---: | ---: | --- |")
                for index, region in enumerate(infrared.hot_regions, start=1):
                    lines.append(
                        f"| {index} | {region.max_temp_c:.1f} | {region.mean_temp_c:.1f} | "
                        f"{region.delta_temp_c:.1f} | {region.area_px} | {region.severity.label_zh} |"
                    )
            for note in infrared.notes:
                lines.append(f"- ⚠️ {note}")
            lines.append("")

        if report.timeseries is not None:
            ts = report.timeseries
            lines.append("### 2.3 电气时序分析")
            lines.append("")
            lines.append(f"- 检测方式：`{ts.backend}`（{ts.model_name}）")
            lines.append(f"- 样本数：{ts.n_samples}，窗口数：{ts.n_windows}")
            lines.append(f"- 通道：{'、'.join(ts.channels)}")
            lines.append(f"- 异常分：{ts.electrical_score:.1f}　判定：{'异常' if ts.is_anomaly else '正常'}")
            lines.append(f"- 负荷相对基线变化：{ts.load_rise_ratio:+.1%}")
            lines.append(f"- 最大负载率：{ts.max_load_ratio:.2f}")
            lines.append(f"- 电压偏差：{ts.voltage_deviation_ratio:.2%}")
            lines.append(f"- 三相电流不平衡度：{ts.current_imbalance_ratio:.2%}")
            lines.append(f"- 滚动 z-score 最大值：{ts.max_zscore:.2f}")
            if ts.events:
                lines.append("")
                lines.append("**检出事件：**")
                for event in ts.events[:8]:
                    span = ""
                    if event.start_time and event.end_time:
                        span = f"（{event.start_time} ~ {event.end_time}）"
                    lines.append(f"- {event.description}{span}")
            for note in ts.notes:
                lines.append(f"- ⚠️ {note}")
            lines.append("")

        # --- 规则判定 ---
        lines.append("## 三、规则判定")
        lines.append("")
        if report.rules.hits:
            lines.append(f"共命中 {len(report.rules.hits)} 条规则，规则分取最高权重 "
                         f"{report.rules.rule_score:.0f}（不做累加，避免同一根因重复计分）。")
            lines.append("")
            lines.append("| 规则 | 名称 | 权重 | 依据标准 | 实测值 |")
            lines.append("| --- | --- | ---: | --- | --- |")
            for hit in report.rules.hits:
                matched = "，".join(f"{k}={v}" for k, v in list(hit.matched.items())[:3]) or "—"
                lines.append(f"| `{hit.rule_id}` | {hit.name} | {hit.severity:.0f} | "
                             f"{hit.standard or '—'} | {matched} |")
        else:
            lines.append("未命中任何规则。")
        lines.append("")

        # --- 依据标准 ---
        lines.append("## 四、知识库依据")
        lines.append("")
        if report.references:
            for index, ref in enumerate(report.references, start=1):
                page = f" 第 {ref.page} 页" if ref.page else ""
                lines.append(f"**{index}. {ref.title}**{page}　"
                             f"<sub>相关度 {ref.score:.3f}　来源：{ref.source_path or '—'}</sub>")
                lines.append("")
                lines.append(f"> {ref.snippet}")
                lines.append("")
        else:
            lines.append("知识库中未检索到相关条款。"
                         "请在 `knowledge/` 目录下放入电力标准、检修规程等文档后重建索引。")
            lines.append("")

        # --- 大模型分析 ---
        lines.append("## 五、综合分析与处置建议")
        lines.append("")
        if report.llm_analysis:
            lines.append(report.llm_analysis)
        else:
            lines.append("未生成分析内容。")
        lines.append("")

        # --- 附录 ---
        lines.append("---")
        lines.append("")
        lines.append("## 附录：本次使用的判定阈值")
        lines.append("")
        lines.append("| 项目 | 取值 |")
        lines.append("| --- | --- |")
        lines.append(f"| 红外绝对温度告警 / 报警 / 危急 (℃) | "
                     f"{self.config.vision.infrared.absolute_warning_c} / "
                     f"{self.config.vision.infrared.absolute_alarm_c} / "
                     f"{self.config.vision.infrared.absolute_critical_c} |")
        lines.append(f"| 红外温升 ΔT 告警 / 报警 / 危急 (K) | "
                     f"{self.config.vision.infrared.delta_warning_c} / "
                     f"{self.config.vision.infrared.delta_alarm_c} / "
                     f"{self.config.vision.infrared.delta_critical_c} |")
        lines.append(f"| 三相温差告警 / 报警 (K) | "
                     f"{self.config.vision.infrared.three_phase_imbalance_warning_c} / "
                     f"{self.config.vision.infrared.three_phase_imbalance_alarm_c} |")
        weights = dict(self.config.fusion.weights)
        lines.append(f"| 融合权重（视觉/红外/时序/规则） | "
                     f"{weights.get('visual')} / {weights.get('thermal')} / "
                     f"{weights.get('electrical')} / {weights.get('rule')} |")
        lines.append("")
        if self.disclaimer:
            lines.append("---")
            lines.append("")
            lines.append(f"> {self.disclaimer}")
        lines.append("")
        lines.append(f"<sub>大模型提供方：`{report.llm_provider}`　"
                     f"总耗时：{report.elapsed_ms:.0f} ms</sub>")

        return "\n".join(lines)


__all__ = ["ReportGenerator", "SYSTEM_PROMPT"]
