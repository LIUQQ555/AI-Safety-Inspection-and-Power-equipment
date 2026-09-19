"""大模型接入与报告生成。"""

from backend.llm.api_provider import OpenAICompatibleProvider  # noqa: F401
from backend.llm.base import LLMError, LLMProvider, LLMRequest  # noqa: F401
from backend.llm.factory import create_provider, describe_providers  # noqa: F401
from backend.llm.mock_provider import MockProvider  # noqa: F401
from backend.llm.report_generator import ReportGenerator  # noqa: F401

__all__ = [
    "LLMRequest",
    "LLMProvider",
    "LLMError",
    "MockProvider",
    "OpenAICompatibleProvider",
    "create_provider",
    "describe_providers",
    "ReportGenerator",
]
