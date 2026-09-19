"""视觉检测分支：可见光目标检测与红外热成像分析。"""

from backend.vision.infrared_detector import (  # noqa: F401
    InfraredDetector,
    create_infrared_detector,
)
from backend.vision.yolo_detector import (  # noqa: F401
    BaseVisibleDetector,
    HeuristicDetector,
    YoloDetector,
    compute_visual_score,
    create_visible_detector,
)

__all__ = [
    "BaseVisibleDetector",
    "YoloDetector",
    "HeuristicDetector",
    "create_visible_detector",
    "compute_visual_score",
    "InfraredDetector",
    "create_infrared_detector",
]
