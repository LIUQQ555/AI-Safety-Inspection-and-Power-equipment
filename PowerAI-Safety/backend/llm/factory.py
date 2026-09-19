"""大模型提供方工厂。

按 ``config.llm.provider`` 选择实现：

    ``mock``      模板化生成，完全离线（默认）
    ``dashscope`` 阿里云百炼 / 通义千问（Qwen3-VL），需 DASHSCOPE_API_KEY
    ``openai``    OpenAI 或任何兼容服务，需 OPENAI_API_KEY

API Key 从环境变量读取，**不写入配置文件**。若选了 API 提供方但 Key 缺失，
自动回退到 MockProvider 并告警——保证流程不中断，同时让用户明确知道
当前看到的不是真实模型输出。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from backend.llm.api_provider import OpenAICompatibleProvider
from backend.llm.base import LLMProvider
from backend.llm.mock_provider import MockProvider

logger = logging.getLogger(__name__)


def create_provider(config, provider_name: Optional[str] = None) -> LLMProvider:
    """创建大模型提供方。"""
    name = (provider_name or str(config.llm.get("provider", "mock"))).lower()
    timeout_s = float(config.llm.get("timeout_s", 60))
    max_image_side = int(config.llm.get("max_image_side", 1024))

    if name == "mock":
        return MockProvider()

    section = "dashscope" if name == "dashscope" else "openai"
    if name not in ("dashscope", "openai"):
        logger.warning("未知的 llm.provider=%r，回退到 mock", name)
        return MockProvider()

    provider_cfg = config.llm.get(section) or {}
    api_key_env = str(provider_cfg.get("api_key_env", f"{section.upper()}_API_KEY"))
    api_key = os.environ.get(api_key_env, "").strip()

    if not api_key:
        logger.warning(
            "未设置环境变量 %s，无法使用 %s 提供方，回退到 MockProvider。"
            "配置方式：set %s=sk-xxxx（Windows）",
            api_key_env, name, api_key_env,
        )
        return MockProvider()

    if name == "dashscope":
        # 百炼的 OpenAI 兼容模式，可直接复用同一套调用代码
        model = str(provider_cfg.get("model", "qwen-vl-max"))
        base_url = str(provider_cfg.get("base_url",
                                        "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    else:
        model = str(provider_cfg.get("model", "gpt-4o-mini"))
        base_url = str(provider_cfg.get("base_url", "https://api.openai.com/v1"))

    logger.info("使用大模型提供方：%s（model=%s）", name, model)
    return OpenAICompatibleProvider(
        model=model,
        base_url=base_url,
        api_key=api_key,
        provider_name=name,
        timeout_s=timeout_s,
        max_image_side=max_image_side,
        supports_vision=True,
    )


def describe_providers(config) -> dict:
    """列出各提供方的可用状态，供前端「设置」页展示。"""
    result = {}
    for name, section in (("dashscope", "dashscope"), ("openai", "openai")):
        provider_cfg = config.llm.get(section) or {}
        env_name = str(provider_cfg.get("api_key_env", f"{name.upper()}_API_KEY"))
        result[name] = {
            "model": str(provider_cfg.get("model", "")),
            "api_key_env": env_name,
            "configured": bool(os.environ.get(env_name, "").strip()),
        }
    result["mock"] = {"model": "template-based", "api_key_env": "", "configured": True}
    return result


__all__ = ["create_provider", "describe_providers"]
