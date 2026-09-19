"""多模态结果级融合。

对应技术方案第六节::

    RiskScore = w1 × VisualScore + w2 × ThermalScore
              + w3 × ElectricalScore + w4 × RuleScore

第一版采用**结果级融合**（方案明确建议）：实现简单、稳定、易于解释。
不做特征级融合，因为三个模态的输入空间差异过大，且缺乏带标注的
多模态配对数据来训练融合网络。

两个关键处理：

**权重重归一化**
    现场往往只拿到部分模态（例如只有红外图、没有传感器数据）。
    此时缺失模态的权重会被剔除，并在剩余模态间按比例重新分配，
    否则缺失模态会把总分系统性拉低，产生「漏报」。

**规则权重按需参与**
    规则分只在有规则命中时才计入权重。规则引擎「没有命中」意味着
    「未发现配置中定义的异常模式」，而不是「风险为 0」，
    不应参与加权稀释其它模态的分数。

**一致性加成**
    多个模态同时指向异常时，结论可信度高于单一模态，
    按配置给予小幅加成（幅度可配置，默认 6 分）。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from backend.core.schemas import (
    FusionResult,
    InfraredResult,
    RuleResult,
    TimeseriesResult,
    VisibleResult,
)
from backend.risk.risk_level import classify_risk, get_levels

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "visual": 0.30,
    "thermal": 0.35,
    "electrical": 0.20,
    "rule": 0.15,
}


class RiskFusion:
    """多模态风险融合器。"""

    def __init__(self, config) -> None:
        self.config = config
        raw_weights = dict(config.fusion.get("weights") or {})
        self.weights: Dict[str, float] = {
            key: float(raw_weights.get(key, DEFAULT_WEIGHTS[key]))
            for key in DEFAULT_WEIGHTS
        }
        total = sum(self.weights.values())
        if total <= 0:
            logger.warning("融合权重之和为 0，回退到默认权重")
            self.weights = dict(DEFAULT_WEIGHTS)
            total = sum(self.weights.values())
        # 权重归一到 1，避免用户填写的权重不必和为 1
        self.weights = {key: value / total for key, value in self.weights.items()}

        self.levels = get_levels(config)
        self.consistency_bonus = float(config.fusion.get("consistency_bonus", 6.0))
        self.consistency_threshold = float(config.fusion.get("consistency_threshold", 60.0))

    def fuse(
        self,
        visible: Optional[VisibleResult] = None,
        infrared: Optional[InfraredResult] = None,
        timeseries: Optional[TimeseriesResult] = None,
        rules: Optional[RuleResult] = None,
    ) -> FusionResult:
        """执行融合。三个分支结果至少提供一个。

        「有结果对象」不等于「有可用的判断」：可见光分支在无训练权重时会退回
        启发式检测器，它无法识别缺陷、``visual_score`` 恒为 0。这种 0 是
        *没有能力判断*，不是 *判断为正常*，必须按缺失处理（见
        ``VisibleResult.score_available``），否则会给整体风险分设一个上不去
        的天花板。
        """
        scores: Dict[str, float] = {}
        missing_reasons: Dict[str, str] = {}

        if visible is not None:
            if visible.score_available:
                scores["visual"] = float(visible.visual_score)
            else:
                missing_reasons["visual"] = (
                    f"可见光检测器为兜底实现（{visible.backend}），不具备缺陷识别能力，"
                    "其分数不参与融合。"
                )
        if infrared is not None:
            scores["thermal"] = float(infrared.thermal_score)
        if timeseries is not None:
            scores["electrical"] = float(timeseries.electrical_score)

        if not scores:
            raise ValueError("融合至少需要一个模态的检测结果")

        has_rule_hits = rules is not None and len(rules.hits) > 0
        if has_rule_hits:
            scores["rule"] = float(rules.rule_score)

        # --- 权重重归一化 ---
        active = {key: self.weights[key] for key in scores}
        weight_sum = sum(active.values())
        if weight_sum <= 0:  # pragma: no cover - 构造时已保证
            active = {key: 1.0 / len(scores) for key in scores}
            weight_sum = 1.0
        active = {key: value / weight_sum for key, value in active.items()}

        # --- 加权求和 ---
        contributions = {key: round(active[key] * scores[key], 4) for key in scores}
        risk_score = sum(contributions.values())

        # --- 一致性加成 ---
        abnormal_modalities = [key for key, value in scores.items()
                               if key != "rule" and value >= self.consistency_threshold]
        bonus = 0.0
        if len(abnormal_modalities) >= 3 and self.consistency_bonus > 0:
            bonus = self.consistency_bonus
            risk_score += bonus

        risk_score = round(max(0.0, min(100.0, risk_score)), 2)
        level, level_name, color = classify_risk(risk_score, self.levels)

        present = [key for key in ("visual", "thermal", "electrical") if key in scores]
        missing = [key for key in ("visual", "thermal", "electrical") if key not in scores]

        # 模态缺失时把原因一并带出，避免读者把「权重被剔除」误读成「该模态正常」
        for key in missing:
            missing_reasons.setdefault(key, "未提供该模态数据。")

        return FusionResult(
            risk_score=risk_score,
            risk_level=level,
            risk_level_name=level_name,
            color=color,
            visual_score=round(scores.get("visual", 0.0), 2),
            thermal_score=round(scores.get("thermal", 0.0), 2),
            electrical_score=round(scores.get("electrical", 0.0), 2),
            rule_score=round(scores.get("rule", 0.0), 2),
            weights_used={key: round(value, 4) for key, value in active.items()},
            contributions=contributions,
            consistency_bonus=round(bonus, 2),
            modalities_present=present,
            modalities_missing=missing,
            missing_reasons=missing_reasons,
        )


def create_risk_fusion(config) -> RiskFusion:
    return RiskFusion(config)


def explain_fusion(fusion: FusionResult, names: Optional[Dict[str, str]] = None) -> str:
    """把融合结果渲染成一行可读的算式说明（用于报告）。"""
    names = names or {
        "visual": "视觉", "thermal": "红外",
        "electrical": "时序", "rule": "规则",
    }
    parts: List[str] = []
    for key, contribution in fusion.contributions.items():
        weight = fusion.weights_used.get(key, 0.0)
        score = {
            "visual": fusion.visual_score,
            "thermal": fusion.thermal_score,
            "electrical": fusion.electrical_score,
            "rule": fusion.rule_score,
        }.get(key, 0.0)
        parts.append(f"{names.get(key, key)} {score:.1f}×{weight:.2f}={contribution:.1f}")

    text = " + ".join(parts)
    if fusion.consistency_bonus:
        text += f" + 一致性加成 {fusion.consistency_bonus:.1f}"
    return f"{text} = {fusion.risk_score:.1f}（{fusion.risk_level_name}）"


__all__ = ["RiskFusion", "create_risk_fusion", "explain_fusion", "DEFAULT_WEIGHTS"]
