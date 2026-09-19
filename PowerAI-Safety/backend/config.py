"""配置加载。

本模块是系统所有阈值、权重、路径的唯一入口。

设计原则（技术方案第六节）：
    阈值与权重只能来自 config/*.yaml，不能由大模型生成。大模型只负责
    引用和解释这些数值，任何情况下都不得改写。

用法::

    from backend.config import get_config
    cfg = get_config()
    print(cfg.fusion.weights.thermal)       # 支持点号访问
    p = cfg.path("vision", "visible", "weights")   # 相对路径 → 绝对路径
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover - 环境缺失时的明确提示
    raise ImportError(
        "缺少 PyYAML。请执行：.venv\\Scripts\\python -m pip install PyYAML"
    ) from exc


# 项目根目录：backend/config.py → backend/ → 项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"
DEFAULT_RULES_PATH = CONFIG_DIR / "rules.yaml"


class ConfigError(RuntimeError):
    """配置文件缺失或格式错误。"""


class AttrDict(dict):
    """支持点号访问的嵌套字典。

    缺失键会抛出带完整路径的 AttributeError，便于定位配置项的拼写错误。
    """

    def __init__(self, data: Dict[str, Any], _path: str = "") -> None:
        super().__init__()
        self._path = _path
        for key, value in data.items():
            self[key] = self._wrap(value, f"{_path}.{key}" if _path else str(key))

    @classmethod
    def _wrap(cls, value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return cls(value, path)
        if isinstance(value, list):
            return [cls._wrap(item, f"{path}[{i}]") for i, item in enumerate(value)]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            available = ", ".join(sorted(k for k in self.keys() if not k.startswith("_")))
            location = self._path or "<root>"
            raise AttributeError(
                f"配置项 {location}.{name} 不存在。可用的同级配置项：{available}"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """按 ``a.b.c`` 形式的点号路径取值，路径不存在时返回 default。"""
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"配置文件不存在：{path}")
    try:
        # 必须显式指定 utf-8：Windows 默认使用 GBK，会导致中文配置项乱码
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"配置文件 {path} YAML 解析失败：{exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件 {path} 顶层必须是映射（dict），实际为 {type(data).__name__}")
    return data


class Config(AttrDict):
    """系统配置。除点号访问外，额外提供路径解析与规则加载能力。"""

    def __init__(
        self,
        data: Dict[str, Any],
        source_path: Optional[Path] = None,
        rules_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(data)
        object.__setattr__(self, "_source_path", source_path)
        object.__setattr__(self, "_rules_raw", rules_data or {})

    # -- 路径解析 ---------------------------------------------------------
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def resolve(self, *parts: str) -> Path:
        """把配置中的相对路径解析为绝对路径（相对项目根目录）。"""
        raw = self.get_path(".".join(parts))
        if raw is None:
            raise ConfigError(f"配置项 {'/'.join(parts)} 不存在，无法解析路径")
        path = Path(str(raw))
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    def path(self, *parts: str) -> Path:
        """``resolve`` 的别名，可读性更好。"""
        return self.resolve(*parts)

    def ensure_dir(self, *parts: str) -> Path:
        """解析路径并确保目录存在。"""
        path = self.resolve(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- 规则库 -----------------------------------------------------------
    @property
    def rules(self) -> List[AttrDict]:
        """规则列表（来自 config/rules.yaml）。"""
        raw = self._rules_raw.get("rules", [])
        wrapped = AttrDict({"rules": raw}, "rules")
        return list(wrapped.get("rules", []))

    # -- 便捷访问器 -------------------------------------------------------
    @property
    def confidence_threshold(self) -> float:
        return float(self.get_path("vision.visible.conf_threshold", 0.25))

    @property
    def focus_device(self) -> str:
        return str(self.get_path("device.focus", "transformer"))

    @property
    def focus_device_name(self) -> str:
        return str(self.get_path("device.display_name", "变压器"))

    def snapshot(self) -> Dict[str, Any]:
        """返回纯 dict 快照（用于写入报告、日志或测试断言）。"""
        return copy.deepcopy(dict(self))


# ---------------------------------------------------------------------------
# 模块级单例
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_cached: Optional[Config] = None


def load_config(
    config_path: Optional[Path] = None,
    rules_path: Optional[Path] = None,
) -> Config:
    """从磁盘加载配置（不使用缓存）。"""
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    rl_path = Path(rules_path) if rules_path else DEFAULT_RULES_PATH

    data = _read_yaml(cfg_path)
    rules_data = _read_yaml(rl_path) if rl_path.exists() else {}
    return Config(data, source_path=cfg_path, rules_data=rules_data)


def get_config(force_reload: bool = False) -> Config:
    """获取全局配置单例（线程安全）。"""
    global _cached
    if _cached is None or force_reload:
        with _lock:
            if _cached is None or force_reload:
                _cached = load_config()
    return _cached


def reload_config() -> Config:
    """重新加载配置，用于运行时修改配置后的热更新。"""
    return get_config(force_reload=True)


__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "Config",
    "ConfigError",
    "AttrDict",
    "load_config",
    "get_config",
    "reload_config",
]
