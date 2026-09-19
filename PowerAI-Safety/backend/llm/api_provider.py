"""OpenAI 兼容接口的大模型提供方。

一个类同时支持：

    * **阿里云百炼 / 通义千问**（Qwen3-VL 系列）
      ``base_url = https://dashscope.aliyuncs.com/compatible-mode/v1``
    * **OpenAI**
      ``base_url = https://api.openai.com/v1``
    * 任何其它提供 ``/chat/completions`` 的服务（vLLM、Ollama、One-API 等）

因为百炼提供了 OpenAI 兼容模式，接入 Qwen3-VL 与接入 GPT 的代码完全一致，
只需替换 ``base_url`` 与 ``model``。技术方案第七节推荐的 Qwen3-VL 走的就是这条路径。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.llm.base import LLMProvider, LLMRequest, encode_image_data_url

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """调用 OpenAI 兼容的 ``/chat/completions`` 接口。"""

    supports_vision = True

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        provider_name: str = "openai_compatible",
        timeout_s: float = 60.0,
        max_image_side: int = 1024,
        supports_vision: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.name = provider_name
        self.timeout_s = float(timeout_s)
        self.max_image_side = int(max_image_side)
        self.supports_vision = bool(supports_vision)

    def is_available(self) -> bool:
        return bool(self.api_key and self.model and self.base_url)

    # ------------------------------------------------------------------
    def _build_messages(self, request: LLMRequest) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})

        parts: List[Dict[str, Any]] = [{"type": "text", "text": request.prompt}]

        if request.images and self.supports_vision:
            for image_path in request.images:
                data_url = encode_image_data_url(image_path, self.max_image_side)
                if data_url is None:
                    logger.warning("图像编码失败，已跳过：%s", image_path)
                    continue
                parts.append({"type": "image_url", "image_url": {"url": data_url}})
        elif request.images:
            logger.info("当前模型 %s 未启用视觉能力，忽略 %d 张图像",
                        self.model, len(request.images))

        messages.append({"role": "user", "content": parts})
        return messages

    def generate(self, request: LLMRequest) -> str:
        if not self.is_available():
            return self._failure("提供方未正确配置（缺少 API Key / model / base_url）")

        try:
            import requests
        except ImportError:
            return self._failure("未安装 requests 库，无法发起 HTTP 请求")

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": self._build_messages(request),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(url, json=payload, headers=headers,
                                     timeout=self.timeout_s)
        except Exception as exc:
            logger.error("调用 %s 失败：%s", url, exc)
            return self._failure(f"网络请求异常：{exc}")

        if response.status_code != 200:
            detail = response.text[:500]
            logger.error("大模型接口返回 %d：%s", response.status_code, detail)
            return self._failure(f"接口返回 HTTP {response.status_code}：{detail}")

        try:
            body = response.json()
            return str(body["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, ValueError) as exc:
            logger.error("响应解析失败：%s；原始内容：%s", exc, response.text[:500])
            return self._failure(f"响应格式异常：{exc}")

    # ------------------------------------------------------------------
    @staticmethod
    def _failure(reason: str) -> str:
        """调用失败时的显式降级文本。

        刻意保留醒目的错误标记，避免把「模型没调用成功」误读成
        「模型分析后认为没有风险」——这两者对运维决策的影响完全不同。
        """
        return (
            "【大模型调用失败】\n"
            f"原因：{reason}\n"
            "本次报告未包含大模型生成的分析内容。检测结果、规则判定与风险等级"
            "仍由本地模型与配置阈值独立给出，不受影响。"
        )


__all__ = ["OpenAICompatibleProvider"]
