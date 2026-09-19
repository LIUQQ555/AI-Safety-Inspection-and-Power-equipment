"""时序信号加载与列名归一化。

对应技术方案 5.3 节。现场导出的 CSV 列名可能是中文、英文或拼音，
本模块负责把它们统一到规范通道名，缺失通道不报错，只记录在 ``missing`` 中，
由上层按可用通道降级处理。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# ---------------------------------------------------------------------------
# 规范通道名
# ---------------------------------------------------------------------------
CANONICAL_CHANNELS: tuple[str, ...] = (
    "voltage", "current", "power", "temperature", "load", "frequency",
)

# A/B/C 分相通道。提供分相数据时才能计算真正意义的三相不平衡度。
PHASE_CHANNELS: tuple[str, ...] = (
    "voltage_a", "voltage_b", "voltage_c",
    "current_a", "current_b", "current_c",
)

ALL_CHANNELS: tuple[str, ...] = CANONICAL_CHANNELS + PHASE_CHANNELS

# 原始列名 → 规范名。匹配时先做小写+去空格/下划线/括号的归一化。
_COLUMN_ALIASES: Dict[str, str] = {
    # 电压
    "voltage": "voltage", "volt": "voltage", "u": "voltage", "v": "voltage",
    "电压": "voltage", "线电压": "voltage", "相电压": "voltage",
    # 电流
    "current": "current", "amp": "current", "ampere": "current",
    "i": "current", "a": "current",
    "电流": "current", "相电流": "current", "线电流": "current",
    # 功率
    "power": "power", "p": "power", "kw": "power", "mw": "power",
    "有功功率": "power", "功率": "power", "有功": "power", "负荷功率": "power",
    # 温度
    "temperature": "temperature", "temp": "temperature", "t": "temperature",
    "油温": "temperature", "绕组温度": "temperature", "温度": "temperature",
    "顶层油温": "temperature", "环境温度": "temperature", "topoil": "temperature",
    # 负荷
    "load": "load", "loadrate": "load", "loadratio": "load", "loading": "load",
    "负荷": "load", "负载": "load", "负载率": "load", "负荷率": "load",
    "容量比": "load",
    # 频率
    "frequency": "frequency", "freq": "frequency", "f": "frequency", "hz": "frequency",
    "频率": "frequency",
    # A/B/C 分相电压
    "voltagea": "voltage_a", "ua": "voltage_a", "uab": "voltage_a",
    "a相电压": "voltage_a", "电压a": "voltage_a",
    "voltageb": "voltage_b", "ub": "voltage_b",
    "b相电压": "voltage_b", "电压b": "voltage_b",
    "voltagec": "voltage_c", "uc": "voltage_c",
    "c相电压": "voltage_c", "电压c": "voltage_c",
    # A/B/C 分相电流
    "currenta": "current_a", "ia": "current_a",
    "a相电流": "current_a", "电流a": "current_a",
    "currentb": "current_b", "ib": "current_b",
    "b相电流": "current_b", "电流b": "current_b",
    "currentc": "current_c", "ic": "current_c",
    "c相电流": "current_c", "电流c": "current_c",
}

_TIME_ALIASES: frozenset[str] = frozenset({
    "timestamp", "time", "datetime", "date", "ts", "recordtime",
    "时间", "时刻", "日期", "采集时间", "记录时间", "时间戳",
})

CHANNEL_UNITS: Dict[str, str] = {
    "voltage": "kV",
    "current": "A",
    "power": "kW",
    "temperature": "℃",
    "load": "p.u.",
    "frequency": "Hz",
}


def _normalize_key(name: Any) -> str:
    """把列名归一化为可比较的键：小写、去空格/下划线/连字符/括号/单位后缀。"""
    text = str(name).strip().lower()
    for ch in (" ", "_", "-", "(", ")", "[", "]", "（", "）", "【", "】", "\t"):
        text = text.replace(ch, "")

    # 先判断整体是否已经是一个已知别名，是则直接返回。
    #
    # 这一步必须放在单位剥离之前。去掉分隔符后 "voltage_a" 变成 "voltagea"，
    # 而单位列表里既有 "a"（安培）也有 "c"，会把结尾的相别字母当成单位吃掉：
    #   voltage_a → voltagea → voltage
    #   voltage_c → voltagec → voltage
    # 结果 A/C 两相被折叠进基础通道 voltage，分相信息丢失、三相不平衡度算不出来，
    # 而且 C 相还会覆盖 A 相且没有任何提示。
    if text in _COLUMN_ALIASES or text in _TIME_ALIASES:
        return text

    # 去掉常见的单位后缀，如 voltage(kv) → voltage
    for unit in ("kv", "v", "a", "ka", "kw", "mw", "kvar", "hz", "℃", "c", "°c", "%"):
        if text.endswith(unit) and len(text) > len(unit):
            stripped = text[: -len(unit)]
            if stripped in _COLUMN_ALIASES or stripped in _TIME_ALIASES:
                return stripped
    return text


@dataclass
class SignalData:
    """归一化后的时序数据。"""

    frame: pd.DataFrame
    source_path: str = ""
    time_column: Optional[str] = None
    mapping: Dict[str, str] = field(default_factory=dict)   # 规范名 → 原始列名
    missing: List[str] = field(default_factory=list)        # 未找到的规范通道
    sampling_interval_min: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    @property
    def channels(self) -> List[str]:
        """实际可用的全部通道（按规范顺序，含分相通道）。"""
        return [c for c in ALL_CHANNELS if c in self.frame.columns]

    @property
    def base_channels(self) -> List[str]:
        """不含分相通道的基础通道，用于时序模型输入。"""
        return [c for c in CANONICAL_CHANNELS if c in self.frame.columns]

    def phase_values(self, quantity: str) -> Optional[List[str]]:
        """返回某物理量（voltage/current）的三个分相列名；不全时返回 None。"""
        names = [f"{quantity}_{phase}" for phase in ("a", "b", "c")]
        return names if all(n in self.frame.columns for n in names) else None

    @property
    def n_samples(self) -> int:
        return int(len(self.frame))

    @property
    def has_time_index(self) -> bool:
        return isinstance(self.frame.index, pd.DatetimeIndex)

    def values(self, channels: Optional[Sequence[str]] = None) -> np.ndarray:
        """返回 ``(T, C)`` 的浮点数组。"""
        cols = list(channels) if channels else self.channels
        if not cols:
            raise ValueError("时序数据不含任何可用通道")
        return self.frame[cols].to_numpy(dtype=np.float32)

    def duration_hours(self) -> Optional[float]:
        if not self.has_time_index or len(self.frame) < 2:
            return None
        return float((self.frame.index[-1] - self.frame.index[0]).total_seconds() / 3600.0)


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
def _read_raw(path: PathLike) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"时序数据文件不存在：{path}")

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    # 依次尝试常见分隔符与编码
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except Exception as exc:
            last_error = exc
            break
    raise ValueError(f"读取时序文件失败 {path}：{last_error}")


def load_signal_frame(
    raw: pd.DataFrame,
    source_path: str = "",
    time_column: Optional[str] = None,
    sampling_interval_min: Optional[float] = None,
) -> SignalData:
    """把原始 DataFrame 归一化为 ``SignalData``。"""
    if raw is None or raw.empty:
        raise ValueError("时序数据为空")

    notes: List[str] = []
    mapping: Dict[str, str] = {}
    normalized_to_raw: Dict[str, str] = {}

    for column in raw.columns:
        key = _normalize_key(column)
        normalized_to_raw.setdefault(key, str(column))

    # 时间列
    detected_time: Optional[str] = None
    if time_column and time_column in raw.columns:
        detected_time = time_column
    else:
        for key, original in normalized_to_raw.items():
            if key in _TIME_ALIASES:
                detected_time = original
                break

    # 通道列
    #
    # 先按规范名归组，再检查是否有多个原始列落到同一个规范名上。
    # 这种情况必须显式提示：静默丢弃一列会让分析结果在无人察觉的情况下出错
    # （例如两个列都叫「电流」时，后面那一列会被悄悄忽略）。
    grouped: Dict[str, List[str]] = {}
    for key, raw_name in normalized_to_raw.items():
        canonical = _COLUMN_ALIASES.get(key)
        if canonical:
            grouped.setdefault(canonical, []).append(raw_name)

    out = pd.DataFrame(index=raw.index)
    for canonical in ALL_CHANNELS:
        candidates = grouped.get(canonical) or []
        if not candidates:
            continue
        original = candidates[0]
        if len(candidates) > 1:
            notes.append(
                f"有多个列映射到通道 {canonical}：{candidates}，仅使用 {original}。"
                "请检查是否存在重复列名或缺少 A/B/C 分相标识。"
            )
            logger.warning("通道 %s 存在多个候选列：%s", canonical, candidates)

        series = pd.to_numeric(raw[original], errors="coerce")
        if series.notna().sum() == 0:
            notes.append(f"列 {original} 无法解析为数值，已跳过。")
            continue
        out[canonical] = series
        mapping[canonical] = original

    if out.empty:
        raise ValueError(
            f"未能识别任何已知通道。文件中的列为：{list(raw.columns)}。"
            f"支持的中文列名示例：电压、电流、功率、温度、负荷、频率。"
        )

    # 只有基础通道会被报告为缺失；分相通道属于可选增强，缺失是正常的
    missing = [c for c in CANONICAL_CHANNELS if c not in out.columns]
    if missing:
        notes.append(f"缺少通道 {missing}，将按可用通道 {list(out.columns)} 分析。")
    for quantity in ("voltage", "current"):
        if all(f"{quantity}_{p}" in out.columns for p in ("a", "b", "c")):
            notes.append(f"检测到 {quantity} 三相通道，将计算三相不平衡度。")

    # 时间索引
    if detected_time is not None:
        parsed = pd.to_datetime(raw[detected_time], errors="coerce")
        if parsed.notna().sum() >= max(2, int(0.8 * len(raw))):
            out.index = pd.DatetimeIndex(parsed)
            if out.index.has_duplicates:
                out = out[~out.index.duplicated(keep="first")]
            out = out.sort_index()
        else:
            notes.append(f"时间列 {detected_time} 解析失败，改用序号索引。")
            detected_time = None

    # 采样间隔
    interval = sampling_interval_min
    if interval is None and isinstance(out.index, pd.DatetimeIndex) and len(out) > 1:
        deltas = out.index.to_series().diff().dropna()
        if not deltas.empty:
            interval = float(deltas.dt.total_seconds().median() / 60.0)
            notes.append(f"由时间戳推断采样间隔为 {interval:.2f} 分钟。")

    # 缺失值插值
    na_counts = out.isna().sum()
    if int(na_counts.sum()) > 0:
        out = out.interpolate(method="linear", limit_direction="both")
        out = out.ffill().bfill()
        detail = {k: int(v) for k, v in na_counts.items() if v > 0}
        notes.append(f"缺失值已线性插值：{detail}")

    return SignalData(
        frame=out,
        source_path=source_path,
        time_column=detected_time,
        mapping=mapping,
        missing=missing,
        sampling_interval_min=interval,
        notes=notes,
    )


def load_signal_csv(
    path: PathLike,
    time_column: Optional[str] = None,
    sampling_interval_min: Optional[float] = None,
) -> SignalData:
    """从 CSV/Excel 加载时序数据。"""
    raw = _read_raw(path)
    logger.info("已加载时序文件 %s，%d 行 %d 列", path, len(raw), raw.shape[1])
    return load_signal_frame(
        raw,
        source_path=str(path),
        time_column=time_column,
        sampling_interval_min=sampling_interval_min,
    )


__all__ = [
    "CANONICAL_CHANNELS",
    "CHANNEL_UNITS",
    "SignalData",
    "load_signal_csv",
    "load_signal_frame",
]
