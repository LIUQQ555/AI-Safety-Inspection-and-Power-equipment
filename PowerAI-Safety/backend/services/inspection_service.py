"""巡检主流程编排。

对应技术方案第十四节的核心检测流程::

    上传图像 → 图像预处理 → YOLO 目标检测
        ↓
    红外/时序分支 → 多模态融合 → 规则判断 → 风险等级
        ↓
    RAG 标准检索 → VLM/LLM 解释 → 结构化报告

**容错原则**：任一分支失败只记录到 ``report.warnings``，不中断整体流程。
例如传感器 CSV 格式异常时，系统仍会用图像模态给出结论，
并在报告中明确标注「时序数据不可用」——而不是返回一个 500 错误，
或者更糟：把缺失模态当作「正常」。
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2

from backend.config import Config, get_config
from backend.core.schemas import (
    InfraredResult,
    InspectionReport,
    TimeseriesResult,
    VisibleResult,
)
from backend.fusion.risk_fusion import RiskFusion
from backend.llm.factory import create_provider
from backend.llm.report_generator import ReportGenerator
from backend.rag.retriever import KnowledgeBase, compose_retrieval_query
from backend.risk.rule_engine import RuleEngine, build_rule_context, compute_visual_thermal_iou
from backend.services.database import Database
from backend.timeseries.anomaly_detector import TimeseriesAnomalyDetector
from backend.timeseries.signal_loader import load_signal_csv
from backend.vision import image_processor as ip
from backend.vision.infrared_detector import InfraredDetector
from backend.vision.yolo_detector import BaseVisibleDetector, create_visible_detector

logger = logging.getLogger(__name__)


def _new_inspection_id() -> str:
    """可读的巡检编号：INS-年月日时分秒-短随机串。"""
    return f"INS-{datetime.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"


class InspectionService:
    """多模态巡检编排器。

    各分支的检测器在构造时初始化一次并复用——YOLO 权重加载开销较大，
    不应每次请求都重新加载。
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()
        self.device_name = self.config.focus_device_name

        self.visible_detector: BaseVisibleDetector = create_visible_detector(self.config)
        self.infrared_detector = InfraredDetector(self.config)
        self.timeseries_detector = TimeseriesAnomalyDetector(self.config)

        self.rule_engine = RuleEngine(self.config)
        self.fusion = RiskFusion(self.config)
        self.knowledge_base = KnowledgeBase(self.config)

        self.provider = create_provider(self.config)
        self.report_generator = ReportGenerator(self.config, self.provider)

        self.database = Database(self.config.resolve("app", "db_path"))
        self.report_dir = self.config.ensure_dir("app", "report_dir")

    # ------------------------------------------------------------------
    # 分支
    # ------------------------------------------------------------------
    def run_visible(self, image_path: Path, warnings: List[str]) -> Optional[VisibleResult]:
        image = ip.imread_unicode(image_path)
        if image is None:
            warnings.append(f"可见光图像读取失败：{image_path.name}")
            return None
        try:
            result = self.visible_detector.detect(image, str(image_path))
        except Exception as exc:
            logger.exception("可见光检测异常")
            warnings.append(f"可见光检测失败：{exc}")
            return None

        # 保存标注图，供报告插图与前端展示
        try:
            annotated = ip.draw_detections(image, result.detections)
            target = self.report_dir / "annotated" / f"{image_path.stem}_visible.jpg"
            if ip.imwrite_unicode(target, annotated):
                result.annotated_image_path = str(target)
        except Exception as exc:
            logger.warning("可见光标注图生成失败：%s", exc)

        return result

    def run_infrared(
        self,
        image_path: Path,
        warnings: List[str],
        three_phase_temps: Optional[Sequence[float]] = None,
    ) -> Optional[InfraredResult]:
        # IMREAD_UNCHANGED 以保留 16 位辐射数据的原始位深
        image = ip.imread_unicode(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            warnings.append(f"红外图像读取失败：{image_path.name}")
            return None
        try:
            result = self.infrared_detector.detect(image, str(image_path))
        except Exception as exc:
            logger.exception("红外检测异常")
            warnings.append(f"红外检测失败：{exc}")
            return None

        # 调用方显式传入三相温度时，优先采用（现场实测值远比图像估算可靠）
        if three_phase_temps and len(three_phase_temps) == 3:
            temps = [float(t) for t in three_phase_temps]
            result.three_phase_temps = [round(t, 2) for t in temps]
            result.three_phase_imbalance_c = round(max(temps) - min(temps), 2)
            result.notes.append("三相温度由调用方提供（实测值），未使用图像估算。")

        try:
            annotated = ip.draw_hot_regions(image, result.hot_regions)
            target = self.report_dir / "annotated" / f"{image_path.stem}_infrared.jpg"
            if ip.imwrite_unicode(target, annotated):
                result.annotated_image_path = str(target)
        except Exception as exc:
            logger.warning("红外标注图生成失败：%s", exc)

        return result

    def run_timeseries(self, csv_path: Path, warnings: List[str]) -> Optional[TimeseriesResult]:
        try:
            signal = load_signal_csv(csv_path)
        except Exception as exc:
            logger.error("时序数据加载失败：%s", exc)
            warnings.append(f"时序数据加载失败：{exc}")
            return None

        try:
            return self.timeseries_detector.analyze(signal)
        except Exception as exc:
            logger.exception("时序分析异常")
            warnings.append(f"时序分析失败：{exc}")
            return None

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def inspect(
        self,
        visible_image: Optional[Path] = None,
        thermal_image: Optional[Path] = None,
        timeseries_csv: Optional[Path] = None,
        device_name: str = "",
        location: str = "",
        operator: str = "",
        three_phase_temps: Optional[Sequence[float]] = None,
        save: bool = True,
    ) -> InspectionReport:
        """执行一次完整巡检。至少需要提供一个模态。"""
        started = time.perf_counter()
        warnings: List[str] = []

        if not any((visible_image, thermal_image, timeseries_csv)):
            raise ValueError("至少需要提供一种数据：可见光图像、红外图像或时序 CSV")

        report = InspectionReport(
            inspection_id=_new_inspection_id(),
            created_at=datetime.now().isoformat(timespec="seconds"),
            device_type=self.config.focus_device,
            device_name=device_name or self.device_name,
            location=location,
            operator=operator,
            llm_provider=self.provider.name,
        )

        # --- 1. 三个分支 ---
        if visible_image is not None:
            report.visible = self.run_visible(Path(visible_image), warnings)
        if thermal_image is not None:
            report.infrared = self.run_infrared(Path(thermal_image), warnings, three_phase_temps)
        if timeseries_csv is not None:
            report.timeseries = self.run_timeseries(Path(timeseries_csv), warnings)

        if all(x is None for x in (report.visible, report.infrared, report.timeseries)):
            raise RuntimeError(
                "所有模态均处理失败，无法生成报告。详情：" + "；".join(warnings)
            )

        # --- 2. 规则判定 ---
        report.visual_thermal_iou = compute_visual_thermal_iou(report.visible, report.infrared)
        context = build_rule_context(
            report.visible, report.infrared, report.timeseries, report.visual_thermal_iou
        )
        try:
            report.rules = self.rule_engine.evaluate(context)
        except Exception as exc:
            logger.exception("规则引擎异常")
            warnings.append(f"规则引擎执行失败：{exc}")

        # --- 3. 多模态融合与风险分级 ---
        try:
            report.fusion = self.fusion.fuse(
                visible=report.visible,
                infrared=report.infrared,
                timeseries=report.timeseries,
                rules=report.rules,
            )
        except Exception as exc:
            logger.exception("风险融合失败")
            warnings.append(f"风险融合失败：{exc}")

        # --- 4. RAG 检索 ---
        try:
            query = compose_retrieval_query(
                device_name=report.device_name,
                defect_classes=report.visible.defect_classes if report.visible else [],
                thermal_severity=(report.infrared.thermal_severity.value
                                  if report.infrared else ""),
                rule_names=[hit.name for hit in report.rules.hits],
            )
            report.references = self.knowledge_base.search(query)
            if not report.references:
                warnings.append(
                    "知识库未检索到相关条款，报告中「依据标准」为空。"
                    "请向 knowledge/ 目录放入标准文档后重建索引。"
                )
        except Exception as exc:
            logger.exception("知识检索失败")
            warnings.append(f"知识检索失败：{exc}")

        # --- 5. 大模型分析 ---
        try:
            report.llm_analysis = self.report_generator.generate_analysis(report)
        except Exception as exc:
            logger.exception("大模型分析失败")
            warnings.append(f"大模型分析失败：{exc}")

        # --- 6. 渲染 Markdown 报告 ---
        try:
            report.report_markdown = self.report_generator.render_markdown(report)
        except Exception as exc:
            logger.exception("报告渲染失败")
            warnings.append(f"报告渲染失败：{exc}")

        report.warnings = warnings
        report.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        # --- 7. 落盘 ---
        if save:
            try:
                self._save_report_files(report)
                self.database.save_report(report)
            except Exception as exc:
                logger.exception("报告保存失败")
                report.warnings.append(f"报告保存失败：{exc}")

        return report

    # ------------------------------------------------------------------
    def _save_report_files(self, report: InspectionReport) -> None:
        """把 Markdown 报告单独落盘，便于直接打开或归档。"""
        directory = self.report_dir / report.inspection_id
        directory.mkdir(parents=True, exist_ok=True)

        if report.report_markdown:
            (directory / "report.md").write_text(report.report_markdown, encoding="utf-8")

        import json
        from backend.core.schemas import to_jsonable
        (directory / "report.json").write_text(
            json.dumps(to_jsonable(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    @staticmethod
    def classify_image(path: Path) -> str:
        """判断图像属于可见光还是红外。

        仅作为「用户只上传一张图且未指明模态」时的辅助判断。
        显式指定模态时以调用方为准。
        """
        image = ip.imread_unicode(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return "unknown"
        is_thermal, _ = ip.looks_like_thermal(image)
        return "infrared" if is_thermal else "visible"

    def rebuild_knowledge_base(self, force: bool = True) -> int:
        """重建知识库索引，返回块数量。"""
        return self.knowledge_base.build(force=force)

    def reload_timeseries_model(self) -> None:
        """训练完成后调用，使新模型立即生效。"""
        self.timeseries_detector.reload_model()

    def health(self) -> Dict[str, Any]:
        """系统状态，供前端与运维检查。"""
        return {
            "device": {
                "type": self.config.focus_device,
                "name": self.device_name,
            },
            "visible_detector": {
                "backend": self.visible_detector.backend,
                "model": self.visible_detector.model_name,
                "trained": self.visible_detector.backend == "yolo",
            },
            "timeseries_model": {
                "loaded": self.timeseries_detector.model is not None,
                "path": str(self.timeseries_detector.model_dir),
            },
            "llm": self.provider.describe(),
            "knowledge_base": self.knowledge_base.stats(),
            "database": {
                "path": str(self.database.path),
                "records": self.database.count(),
            },
        }


__all__ = ["InspectionService", "create_service"]


def create_service(config: Optional[Config] = None) -> InspectionService:
    return InspectionService(config)
