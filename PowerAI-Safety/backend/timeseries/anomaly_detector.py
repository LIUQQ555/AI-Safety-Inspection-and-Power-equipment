"""电气时序异常检测编排。

对应技术方案 5.3 节。

两条推理路径：
    * ``transformer``  —— 加载 ``models/timeseries/`` 下的自训练模型，
      以窗口重构误差为异常分。
    * ``statistical``  —— 无模型时的降级路径，使用滚动 z-score + IQR 判据。

两条路径都会额外计算**业务可解释指标**（负荷增幅、温升增幅、电压偏差、
三相不平衡度等）。这些指标既进入规则引擎参与判断，也写入报告供运维人员
直接阅读——只给一个「异常分 0.83」对现场是没有意义的。

**指标口径说明**（避免误读）：
    * 增幅类指标（``*_rise_ratio``）比较「序列前段基线」与「最近时段」，
      基线取前半段，近期取后四分之一，符合「近期负荷是否升高」的业务问法。
    * 电压偏差以额定电压为基准；未配置额定值时用序列中位数近似，
      此时该指标只反映波动幅度，不代表真实偏差。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.config import Config
from backend.core.schemas import AnomalyEvent, TimeseriesResult
from backend.timeseries import preprocessing as prep
from backend.timeseries.signal_loader import SignalData
from backend.timeseries.transformer_model import TransformerAnomalyDetector

logger = logging.getLogger(__name__)

MODEL_DIR_NAME = "timeseries"


# ---------------------------------------------------------------------------
# 业务指标
# ---------------------------------------------------------------------------
def rise_ratio(values: np.ndarray, baseline_frac: float = 0.5, recent_frac: float = 0.25) -> float:
    """近期相对基线的增幅比。

    ``(近期均值 - 基线均值) / |基线均值|``。基线均值为 0 时返回 0。
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    n = len(values)
    if n < 8:
        return 0.0
    n_baseline = max(2, int(n * baseline_frac))
    n_recent = max(2, int(n * recent_frac))
    baseline = float(values[:n_baseline].mean())
    recent = float(values[-n_recent:].mean())
    if abs(baseline) < 1e-9:
        return 0.0
    return float((recent - baseline) / abs(baseline))


def trend_slope(values: np.ndarray) -> float:
    """线性趋势斜率，按序列均值归一化（单位：每样本的相对变化量）。"""
    values = np.asarray(values, dtype=np.float64).ravel()
    n = len(values)
    if n < 4:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    slope = float(np.polyfit(x, values, 1)[0])
    mean = float(np.abs(values).mean())
    return float(slope / mean) if mean > 1e-9 else 0.0


def voltage_deviation(values: np.ndarray, rated: Optional[float]) -> Tuple[float, float]:
    """最大电压偏差率。返回 ``(偏差率, 所用额定值)``。"""
    values = np.asarray(values, dtype=np.float64).ravel()
    if len(values) == 0:
        return 0.0, 0.0
    reference = float(rated) if rated and rated > 0 else float(np.median(values))
    if abs(reference) < 1e-9:
        return 0.0, reference
    deviation = float(np.max(np.abs(values - reference)) / abs(reference))
    return deviation, reference


def current_imbalance(phase_a: np.ndarray, phase_b: np.ndarray, phase_c: np.ndarray) -> float:
    """三相电流不平衡度。

    按 GB/T 15543 的口径：``max|I_phase - I_avg| / I_avg``。
    取全时段的最大值。
    """
    a = np.asarray(phase_a, dtype=np.float64).ravel()
    b = np.asarray(phase_b, dtype=np.float64).ravel()
    c = np.asarray(phase_c, dtype=np.float64).ravel()
    n = min(len(a), len(b), len(c))
    if n == 0:
        return 0.0
    stacked = np.vstack([a[:n], b[:n], c[:n]])
    average = stacked.mean(axis=0)
    safe = np.where(np.abs(average) < 1e-9, np.nan, average)
    deviation = np.abs(stacked - safe) / np.abs(safe)
    value = np.nanmax(deviation) if np.isfinite(deviation).any() else 0.0
    return float(value)


def rolling_zscore_max(values: np.ndarray, window: int) -> float:
    """滚动 z-score 的最大绝对值。"""
    values = np.asarray(values, dtype=np.float64).ravel()
    n = len(values)
    if n < max(8, window // 4):
        return 0.0
    window = max(4, min(int(window), n))
    max_z = 0.0
    for start in range(0, n - window + 1, max(1, window // 4)):
        segment = values[start:start + window]
        std = float(segment.std())
        if std < 1e-9:
            continue
        z = float(np.max(np.abs(segment - segment.mean())) / std)
        max_z = max(max_z, z)
    return max_z


def expected_max_z(n: int) -> float:
    """纯噪声下，n 个样本的最大 |z| 的期望值。

    这是判读 z-score 时绕不开的一点：**最大偏差本身随样本数增长**。
    对 n 个标准正态样本，``E[max|Z|] ≈ sqrt(2 ln(2n))``——
    n=96 时约为 3.2，n=1000 时约为 3.9。

    因此「z 峰值 > 3 判为异常」是个伪判据：一段完全正常的白噪声序列
    也会稳定地给出 2.5~3.2 的峰值，用固定阈值 3.0 去卡，
    等于把大半个正常序列判成异常。

    正确做法是把峰值换算成「相对噪声预期的倍数」再判读：
    比值 1.0 表示和噪声无异，2.0 表示明显超出噪声水平。
    """
    if n < 2:
        return 1.0
    return float(np.sqrt(2.0 * np.log(2.0 * n)))


@dataclass
class BusinessMetrics:
    """时序业务指标汇总。"""

    load_rise_ratio: float = 0.0
    temp_rise_ratio: Optional[float] = None
    load_trend_slope: float = 0.0
    voltage_deviation_ratio: float = 0.0
    current_imbalance_ratio: float = 0.0
    max_load_ratio: float = 0.0
    max_zscore: float = 0.0
    peak_to_peak: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def compute_business_metrics(signal: SignalData, config: Config) -> BusinessMetrics:
    """从原始信号计算业务可解释指标。"""
    metrics = BusinessMetrics()
    frame = signal.frame

    def column(name: str) -> Optional[np.ndarray]:
        return frame[name].to_numpy(dtype=np.float64) if name in frame.columns else None

    # --- 负荷 ---
    load_values = column("load")
    if load_values is None:
        load_values = column("power")
        if load_values is not None:
            metrics.notes.append("未提供负荷通道，改用功率通道估算负荷增幅。")

    rated_kw = config.timeseries.get("rated_capacity_kw")
    if load_values is not None and len(load_values):
        metrics.load_rise_ratio = round(rise_ratio(load_values), 4)
        metrics.load_trend_slope = round(trend_slope(load_values), 6)

        peak = float(np.max(load_values))
        # 数值 > 2 视为 kW 原始功率，按额定容量换算负载率；否则视为已是标幺值
        if peak > 2.0:
            if rated_kw and float(rated_kw) > 0:
                metrics.max_load_ratio = round(peak / float(rated_kw), 4)
            else:
                metrics.notes.append(
                    "负荷通道数值较大但未配置 rated_capacity_kw，无法换算负载率。"
                )
        else:
            metrics.max_load_ratio = round(peak, 4)

    # --- 温度 ---
    temp_values = column("temperature")
    if temp_values is not None and len(temp_values):
        metrics.temp_rise_ratio = round(rise_ratio(temp_values), 4)

    # --- 电压偏差 ---
    voltage_values = column("voltage")
    if voltage_values is not None and len(voltage_values):
        rated_v = config.timeseries.get("rated_voltage_kv")
        deviation, reference = voltage_deviation(voltage_values, rated_v)
        metrics.voltage_deviation_ratio = round(deviation, 4)
        if not rated_v:
            metrics.notes.append(
                f"未配置 rated_voltage_kv，以序列中位数 {reference:.3f} 作为额定值近似，"
                "电压偏差只反映波动幅度。"
            )

    # --- 三相电流不平衡 ---
    phase_names = signal.phase_values("current")
    if phase_names:
        phases = [frame[name].to_numpy(dtype=np.float64) for name in phase_names]
        metrics.current_imbalance_ratio = round(current_imbalance(*phases), 4)
    elif signal.phase_values("voltage"):
        metrics.notes.append("仅检测到分相电压，缺少分相电流，未计算电流不平衡度。")

    # --- 统计量 ---
    for channel in signal.base_channels:
        values = frame[channel].to_numpy(dtype=np.float64)
        if len(values):
            metrics.peak_to_peak[channel] = round(float(values.max() - values.min()), 4)

    zscore_window = int(config.timeseries.anomaly.get("zscore_window", 96))
    max_z = 0.0
    for channel in signal.base_channels:
        values = frame[channel].to_numpy(dtype=np.float64)
        max_z = max(max_z, rolling_zscore_max(values, zscore_window))
    metrics.max_zscore = round(max_z, 4)

    return metrics


# ---------------------------------------------------------------------------
# 检测器
# ---------------------------------------------------------------------------
class TimeseriesAnomalyDetector:
    """时序异常检测器。"""

    def __init__(self, config: Config, model_dir: Optional[Path] = None) -> None:
        self.config = config
        ts_cfg = config.timeseries

        self.window_size = int(ts_cfg.get("window_size", 96))
        self.stride = int(ts_cfg.get("stride", 24))
        self.denoise_window = int(ts_cfg.get("denoise_window", 5))
        self.normalize = str(ts_cfg.get("normalize", "zscore"))
        self.clip_sigma = float(ts_cfg.get("clip_sigma", 5.0))

        anomaly_cfg = ts_cfg.anomaly
        self.error_percentile = float(anomaly_cfg.get("error_percentile", 95.0))

        # 统计判据的阈值。z-score 用「相对纯噪声预期的倍数」而非绝对值，
        # 原因见 expected_max_z 的说明。
        stat_cfg = dict(anomaly_cfg.get("statistical") or {})
        self.zscore_warn_factor = float(stat_cfg.get("zscore_factor_warning", 1.15))
        self.zscore_critical_factor = float(stat_cfg.get("zscore_factor_critical", 2.0))

        self.model_dir = Path(model_dir) if model_dir else config.ensure_dir("models", "timeseries_dir")
        self._model: Optional[TransformerAnomalyDetector] = None
        self._model_load_attempted = False

    # -- 模型加载 -------------------------------------------------------
    @property
    def model(self) -> Optional[TransformerAnomalyDetector]:
        if not self._model_load_attempted:
            self._model_load_attempted = True
            if (self.model_dir / "meta.json").exists() and (self.model_dir / "model.pt").exists():
                try:
                    self._model = TransformerAnomalyDetector.load(self.model_dir)
                    logger.info("已加载时序模型：%s", self.model_dir)
                except Exception as exc:
                    logger.error("时序模型加载失败，降级为统计检测：%s", exc)
                    self._model = None
        return self._model

    def reload_model(self) -> None:
        """清除模型缓存，使下次分析重新从磁盘加载（训练完成后调用）。"""
        self._model = None
        self._model_load_attempted = False

    # -- 主流程 ---------------------------------------------------------
    def analyze(self, signal: SignalData) -> TimeseriesResult:
        started = time.perf_counter()
        notes: List[str] = list(signal.notes)

        result = TimeseriesResult(
            csv_path=signal.source_path,
            model_name="",
            notes=notes,
        )

        # --- 预处理（只用基础通道，避免分相通道稀释模型输入）---
        try:
            pre = prep.preprocess_signal(
                signal,
                window_size=self.window_size,
                stride=self.stride,
                denoise_window=self.denoise_window,
                normalize=self.normalize,
                clip_sigma=self.clip_sigma,
                channels=signal.base_channels,
            )
        except Exception as exc:
            logger.error("时序预处理失败：%s", exc)
            result.notes.append(f"预处理失败：{exc}")
            result.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            return result

        notes.extend(pre.notes)
        result.channels = pre.channels
        result.n_samples = signal.n_samples
        result.n_windows = pre.n_windows

        # --- 业务指标（用原始量纲，与模型无关）---
        metrics = compute_business_metrics(signal, self.config)
        result.load_rise_ratio = metrics.load_rise_ratio
        result.temp_rise_ratio = metrics.temp_rise_ratio
        result.load_trend_slope = metrics.load_trend_slope
        result.voltage_deviation_ratio = metrics.voltage_deviation_ratio
        result.current_imbalance_ratio = metrics.current_imbalance_ratio
        result.max_load_ratio = metrics.max_load_ratio
        result.max_zscore = metrics.max_zscore
        result.peak_to_peak = metrics.peak_to_peak
        notes.extend(metrics.notes)

        # --- 异常打分 ---
        model = self.model
        if model is not None and pre.n_windows > 0:
            expected = model.metadata.n_channels if model.metadata else None
            if expected is not None and expected != len(pre.channels):
                message = (
                    f"模型训练时使用 {expected} 个通道 {model.metadata.channels}，"
                    f"当前数据为 {len(pre.channels)} 个通道 {pre.channels}，"
                    "通道不匹配，降级为统计检测。"
                )
                logger.warning(message)
                notes.append(message)
                model = None

        if model is not None and pre.n_windows > 0:
            result.backend = "transformer"
            result.model_name = f"Transformer-AE (d_model={model.d_model}, layers={model.num_layers})"
            errors = model.score(pre.windows)
            threshold = model.threshold
            result.threshold = round(float(threshold), 6)
            result.electrical_score, detail = self._score_from_errors(errors, threshold)
            notes.append(detail)

            anomalous = errors > threshold
            result.is_anomaly = bool(np.any(anomalous))
            result.anomaly_ratio = round(float(np.mean(anomalous)), 4)
            result.events = self._build_events(
                pre, errors, anomalous, source="transformer"
            )
        else:
            result.backend = "statistical"
            result.model_name = "interpretable-indicators (untrained fallback)"
            result.electrical_score, detail = self._score_from_statistics(
                metrics, signal.n_samples
            )
            notes.append(detail)

            expected = expected_max_z(signal.n_samples)
            result.threshold = round(expected * self.zscore_warn_factor, 4)
            result.is_anomaly = result.electrical_score > 0
            # 统计路径没有「窗口」概念，用异常分归一化后的比例代替窗口占比，
            # 表示「异常程度占满量程的比例」，与模型路径的 anomaly_ratio 语义相近。
            result.anomaly_ratio = round(min(1.0, result.electrical_score / 100.0), 4)
            result.events = self._events_from_statistics(signal, metrics)

            notes.append(
                "未加载训练好的时序模型，当前使用统计判据（statistical）。"
                "训练模型：.venv\\Scripts\\python training\\train_timeseries.py"
            )

        # --- 输出图表 ---
        result.notes = notes
        result.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return result

    # -- 打分 -----------------------------------------------------------
    def _score_from_errors(
        self, errors: np.ndarray, threshold: float
    ) -> Tuple[float, str]:
        """由重构误差计算 0-100 的电气异常分。"""
        if len(errors) == 0 or threshold <= 0:
            return 0.0, "无可评分窗口。"

        exceed_ratio = float(np.mean(errors > threshold))
        severity_ratio = float(errors.max() / threshold)

        # 广度：受影响窗口的比例（30% 及以上即视为广泛异常）
        breadth = min(1.0, exceed_ratio / 0.30)
        # 幅度：超阈倍数，超过 3 倍阈值即视为满分
        severity = min(1.0, max(0.0, (severity_ratio - 1.0) / 2.0))

        score = 100.0 * (0.50 * breadth + 0.50 * severity)
        detail = (
            f"Transformer 重构误差：阈值 {threshold:.5f}，"
            f"最大 {errors.max():.5f}（{severity_ratio:.2f}× 阈值），"
            f"超阈窗口占比 {exceed_ratio:.1%}。"
        )
        return round(float(np.clip(score, 0.0, 100.0)), 2), detail

    def _score_from_statistics(
        self,
        metrics: BusinessMetrics,
        n_samples: int,
    ) -> Tuple[float, str]:
        """统计判据打分。

        取各指标中**最严重**的一个作为总分，而不是加权求和：这些指标是同一个
        异常的不同侧面（电流突变会同时抬高 z-score 和电流不平衡度），
        相加等于把同一个根因重复计分。规则引擎出于同样的理由取 max 而非 sum。

        每个指标都给定了 warning / critical 两级阈值，中间线性插值。
        低于 warning 一律得 0 —— 正常运行的设备本就该拿到接近 0 的异常分。
        """
        cfg = dict(self.config.timeseries.anomaly.get("statistical") or {})

        def ramp(value: Optional[float], key: str) -> float:
            warning = float(cfg.get(f"{key}_warning", 0.0))
            critical = float(cfg.get(f"{key}_critical", 0.0))
            if value is None or critical <= warning:
                return 0.0
            if value <= warning:
                return 0.0
            if value >= critical:
                return 100.0
            return 100.0 * (value - warning) / (critical - warning)

        # z 峰值换算成「相对纯噪声预期的倍数」
        expected = expected_max_z(n_samples)
        z_factor = metrics.max_zscore / expected if expected > 0 else 0.0

        indicators = [
            ("zscore_factor", z_factor,
             f"滚动 z-score 峰值 {metrics.max_zscore:.2f}（纯噪声预期 {expected:.2f}，"
             f"倍数 {z_factor:.2f}）"),
            ("load_rise", metrics.load_rise_ratio,
             f"负荷增幅 {metrics.load_rise_ratio:+.1%}"),
            ("voltage_deviation", metrics.voltage_deviation_ratio,
             f"电压偏差 {metrics.voltage_deviation_ratio:.2%}"),
            ("current_imbalance", metrics.current_imbalance_ratio,
             f"电流不平衡度 {metrics.current_imbalance_ratio:.2%}"),
            ("load_ratio", metrics.max_load_ratio,
             f"最大负载率 {metrics.max_load_ratio:.1%}"),
        ]

        scores = {key: ramp(value, key) for key, value, _ in indicators}
        top_key = max(scores, key=lambda k: scores[k])
        score = scores[top_key]

        detail = "统计判据：" + "；".join(text for _, _, text in indicators) + "。"
        if score > 0:
            detail += f" 主导指标：{top_key}（{score:.0f} 分）。"
        return round(float(np.clip(score, 0.0, 100.0)), 2), detail

    # -- 事件 -----------------------------------------------------------
    def _build_events(
        self,
        pre: prep.PreprocessResult,
        errors: np.ndarray,
        anomalous: np.ndarray,
        source: str = "transformer",
    ) -> List[AnomalyEvent]:
        """把连续的异常窗口合并为事件。"""
        events: List[AnomalyEvent] = []
        n = len(anomalous)
        index = 0
        while index < n:
            if not anomalous[index]:
                index += 1
                continue
            start = index
            while index < n and anomalous[index]:
                index += 1
            end = index - 1

            window_start = int(pre.start_indices[start])
            window_end = int(pre.start_indices[end]) + pre.windows.shape[1] - 1
            start_time, _ = pre.window_time_range(start)
            _, end_time = pre.window_time_range(end)

            events.append(AnomalyEvent(
                start_index=window_start,
                end_index=window_end,
                start_time=start_time,
                end_time=end_time,
                score=round(float(errors[start:end + 1].max()), 6),
                description=(
                    f"{source} 检出连续 {end - start + 1} 个异常窗口，"
                    f"最大重构误差 {errors[start:end + 1].max():.5f}"
                ),
            ))
        return events

    def _events_from_statistics(
        self, signal: SignalData, metrics: BusinessMetrics
    ) -> List[AnomalyEvent]:
        """统计路径下的事件描述：把超阈的业务指标转成事件条目。

        阈值与 ``_score_from_statistics`` 共用同一份配置，避免两个地方
        各写一套数字而互相矛盾（评分说正常、事件列表却说超限）。
        """
        cfg = dict(self.config.timeseries.anomaly.get("statistical") or {})
        events: List[AnomalyEvent] = []
        last = max(0, signal.n_samples - 1)

        def add(channel: str, score: float, description: str) -> None:
            events.append(AnomalyEvent(
                start_index=0, end_index=last, start_time=None, end_time=None,
                score=round(float(score), 4), channel=channel, description=description,
            ))

        expected = expected_max_z(signal.n_samples)
        z_factor = metrics.max_zscore / expected if expected > 0 else 0.0
        if z_factor >= self.zscore_warn_factor:
            add("zscore_factor", z_factor,
                f"滚动 z-score 峰值 {metrics.max_zscore:.2f} 为纯噪声预期值 "
                f"{expected:.2f} 的 {z_factor:.2f} 倍"
                f"（告警倍数 {self.zscore_warn_factor:.2f}）")

        checks = [
            ("load_rise_ratio", metrics.load_rise_ratio, "load_rise", "负荷较基线显著升高"),
            ("voltage_deviation_ratio", metrics.voltage_deviation_ratio,
             "voltage_deviation", "电压偏差超限"),
            ("current_imbalance_ratio", metrics.current_imbalance_ratio,
             "current_imbalance", "三相电流不平衡"),
            ("max_load_ratio", metrics.max_load_ratio, "load_ratio", "负载率偏高"),
        ]
        for attr, value, key, description in checks:
            limit = float(cfg.get(f"{key}_warning", 0.0))
            if limit > 0 and value >= limit:
                add(key, value, f"{description}（{value:.2%} ≥ 阈值 {limit:.2%}）")

        return events


def create_timeseries_detector(config: Config) -> TimeseriesAnomalyDetector:
    return TimeseriesAnomalyDetector(config)


__all__ = [
    "BusinessMetrics",
    "compute_business_metrics",
    "rise_ratio",
    "trend_slope",
    "voltage_deviation",
    "current_imbalance",
    "rolling_zscore_max",
    "TimeseriesAnomalyDetector",
    "create_timeseries_detector",
]
