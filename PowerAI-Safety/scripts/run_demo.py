"""命令行演示：对 data/samples 下的合成场景逐个执行完整巡检。

用于在不开前端、不启服务的情况下验证整条链路，也方便排查某个分支的问题::

    python scripts/run_demo.py                       # 跑全部场景
    python scripts/run_demo.py --scenario overheat   # 只跑一个场景
    python scripts/run_demo.py --markdown            # 额外打印完整报告
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_config  # noqa: E402
from backend.fusion.risk_fusion import explain_fusion  # noqa: E402
from backend.services import InspectionService  # noqa: E402

DEFAULT_SAMPLES = PROJECT_ROOT / "data" / "samples"


def _fmt(values, digits: int = 2) -> str:
    if not values:
        return "-"
    return ", ".join(f"{v:.{digits}f}" if isinstance(v, (int, float)) else str(v) for v in values)


def run(samples_dir: Path, scenario_filter: str = "", show_markdown: bool = False) -> int:
    manifest_path = samples_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"未找到 {manifest_path}\n请先运行：python datasets/synthesize.py", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenarios = manifest["scenarios"]
    if scenario_filter:
        scenarios = [s for s in scenarios if s["scenario"] == scenario_filter]
        if not scenarios:
            available = [s["scenario"] for s in manifest["scenarios"]]
            print(f"未找到场景 {scenario_filter!r}，可选：{available}", file=sys.stderr)
            return 2

    service = InspectionService(get_config())
    health = service.health()
    print("=" * 78)
    print("系统状态")
    print("=" * 78)
    print(f"  可见光检测后端 : {health['visible_detector']['backend']}"
          f"（模型 {health['visible_detector']['model']}）")
    print(f"  时序模型       : {'已加载' if health['timeseries_model']['loaded'] else '未训练，使用统计判据'}")
    print(f"  大模型提供方   : {health['llm']['name']} — {health['llm'].get('note', '')}")
    kb = health["knowledge_base"]
    print(f"  知识库         : {kb.get('chunks', 0)} 个文本块")

    for entry in scenarios:
        name = entry["scenario"]
        gt = entry["ground_truth"]

        def path_of(key: str) -> Path | None:
            value = entry.get(key)
            return samples_dir / value if value else None

        report = service.inspect(
            visible_image=path_of("visible"),
            thermal_image=path_of("thermal_radiometric"),
            timeseries_csv=path_of("timeseries"),
            device_name=f"1号主变-{name}",
            location="示范变电站",
            operator="demo",
        )

        print()
        print("=" * 78)
        print(f"场景 {name} —— {entry['title']}")
        print("=" * 78)
        print(f"  真值       : 可见光缺陷={gt['visible_defects'] or '无'}  "
              f"相温={_fmt(gt['phase_temps_c'], 1)}℃  "
              f"预期热级别={gt['expected_thermal_severity']}  "
              f"时序形态={gt['series_pattern']}")

        if report.visible:
            v = report.visible
            labels = [f"{d.label}({d.confidence:.2f})" for d in v.detections]
            print(f"  可见光分支 : backend={v.backend}  VisualScore={v.visual_score}  "
                  f"检出={labels or '无'}")

        if report.infrared:
            ir = report.infrared
            print(f"  红外分支   : backend={ir.backend}  Tmax={ir.max_temp_c}℃  "
                  f"环境={ir.ambient_temp_c}℃  参考T2={ir.baseline_temp_c}℃({ir.baseline_source})")
            print(f"               ΔT={ir.delta_temp_c}K  δ={ir.relative_delta_ratio}%  "
                  f"级别={ir.thermal_severity.value}  ThermalScore={ir.thermal_score}")
            print(f"               热点={ir.hot_region_count}个（其中异常{ir.abnormal_region_count}个）  "
                  f"三相={_fmt(ir.three_phase_temps, 1)}  三相不平衡={ir.three_phase_imbalance_c}K")

        if report.timeseries:
            ts = report.timeseries
            preview = [e.channel or e.description for e in ts.events][:5]
            print(f"  时序分支   : backend={ts.backend}  异常窗口={len(ts.events)}个  "
                  f"占比={ts.anomaly_ratio:.1%}  ElectricalScore={ts.electrical_score}")
            print(f"               负荷上升比={ts.load_rise_ratio:.2f}  "
                  f"温升比={ts.temp_rise_ratio if ts.temp_rise_ratio is None else round(ts.temp_rise_ratio, 2)}  "
                  f"电压偏差={ts.voltage_deviation_ratio:.2%}  "
                  f"电流不平衡={ts.current_imbalance_ratio:.2%}")
            if preview:
                print(f"               异常窗口示例={preview}")

        hits = [f"{h.name}({h.severity:.2f})" for h in report.rules.hits]
        print(f"  规则命中   : {hits or '无'}  →  RuleScore={report.rules.rule_score}")

        fusion = report.fusion
        print(f"  融合结果   : RiskScore={fusion.risk_score}  等级={fusion.risk_level_name}")
        print(f"               {explain_fusion(fusion)}")
        print(f"  知识库条款 : {len(report.references)} 条"
              + (f"  首条={report.references[0].title or report.references[0].source}"
                 if report.references else ""))

        if report.warnings:
            print("  警告       :")
            for warning in report.warnings:
                print(f"      - {warning}")

        if show_markdown and report.report_markdown:
            print()
            print(report.report_markdown)

    print()
    print("=" * 78)
    print(f"演示完成，共 {len(scenarios)} 个场景。报告已保存至 data/reports/")
    print("=" * 78)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="对合成样本执行完整巡检流程")
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES), help="样本目录")
    parser.add_argument("--scenario", default="", help="只运行指定场景")
    parser.add_argument("--markdown", action="store_true", help="打印完整 Markdown 报告")
    args = parser.parse_args()
    return run(Path(args.samples), args.scenario, args.markdown)


if __name__ == "__main__":
    raise SystemExit(main())
