"""多模态融合层。"""

from backend.fusion.risk_fusion import (  # noqa: F401
    DEFAULT_WEIGHTS,
    RiskFusion,
    create_risk_fusion,
    explain_fusion,
)

__all__ = ["RiskFusion", "create_risk_fusion", "explain_fusion", "DEFAULT_WEIGHTS"]
