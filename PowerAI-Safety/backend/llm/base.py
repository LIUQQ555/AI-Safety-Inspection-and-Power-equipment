"""大模型接入接口。

技术方案第七节明确了大模型的**职责边界**：

    大模型不直接代替底层检测模型，只承担——
        1. 多模态结果解释（把结构化检测结果翻译成自然语言）
        2. 归因分析（结合知识库说明可能原因）
        3. 处置建议（引用标准给出下一步动作）

因此本模块的接口设计为「接收结构化事实 + 检索到的标准条款，输出文本」，
而不是「接收原始图像，输出检测结论」。风险等级与阈值始终由
``risk_fusion`` 与 ``rule_engine`` 决定，大模型无权改写。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LLMRequest:
    """一次生成请求。"""

    prompt: str
    system: str = ""
    # 待分析的图像路径（VLM 使用；纯文本模型会忽略）
    images: List[str] = field(default_factory=list)
    max_tokens: int = 1500
    temperature: float = 0.2
    # 结构化事实。真实模型只读 prompt；MockProvider 直接读本字段，
    # 从而生成基于事实的确定性文本，而不必反向解析 prompt 字符串。
    facts: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """大模型提供方接口。"""

    name: str = "base"
    supports_vision: bool = False

    @abstractmethod
    def generate(self, request: LLMRequest) -> str:
        """生成文本。实现方不应抛出未捕获异常——失败时返回带说明的降级文本。"""

    def is_available(self) -> bool:
        """提供方是否可用（例如 API Key 是否已配置）。"""
        return True

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "supports_vision": self.supports_vision,
                "available": self.is_available()}


class LLMError(RuntimeError):
    """大模型调用失败。"""


def encode_image_data_url(path: str, max_side: int = 1024) -> Optional[str]:
    """把图像编码为 ``data:image/...;base64,...``，供兼容 OpenAI 的视觉接口使用。

    超过 ``max_side`` 时先缩放，以控制 token 消耗。
    """
    from backend.vision import image_processor as ip
    import numpy as np
    import base64
    import cv2

    image = ip.imread_unicode(path, cv2.IMREAD_COLOR)
    if image is None:
        return None

    image = ip.resize_max_side(image, max_side)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return None

    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


__all__ = ["LLMRequest", "LLMProvider", "LLMError", "encode_image_data_url"]
