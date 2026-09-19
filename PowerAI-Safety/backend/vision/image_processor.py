"""图像预处理与标注工具。

本模块解决三类在 Windows + 中文环境下必须处理的问题：

1. **中文路径**：``cv2.imread`` 在 Windows 上使用 ANSI 代码页解析路径，
   遇到中文文件名会静默返回 None。本模块统一改用 ``np.fromfile`` +
   ``cv2.imdecode`` 读取，``cv2.imencode`` + ``tofile`` 写出。

2. **中文标注**：``cv2.putText`` 依赖 Hershey 矢量字体，无法渲染中文，
   会输出 ``????``。本模块使用 PIL + 系统中文字体绘制文字，
   找不到字体时降级为英文标签并给出提示。

3. **红外图像识别**：区分伪彩色热像图与普通可见光图，
   决定走哪条温度解析路径。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from backend.core.schemas import BBox, Detection

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
Color = Tuple[int, int, int]  # BGR

# ---------------------------------------------------------------------------
# 调色板与绘图样式（BGR）
# ---------------------------------------------------------------------------
PALETTE: List[Color] = [
    (0, 200, 0),      # 绿 —— 正常设备
    (0, 165, 255),    # 橙 —— 关注
    (0, 0, 220),      # 红 —— 缺陷
    (220, 0, 220),    # 品红
    (220, 180, 0),    # 青蓝
    (0, 120, 255),    # 琥珀
]

CATEGORY_COLORS: dict[str, Color] = {
    "device": (0, 200, 0),
    "defect": (0, 0, 230),
}


# ---------------------------------------------------------------------------
# 中文路径安全的读写
# ---------------------------------------------------------------------------
def imread_unicode(path: PathLike, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """读取图像，兼容中文/空格路径。

    返回 BGR 三通道（或按 flags 指定的通道数）数组；失败返回 None 并记录日志。
    """
    path = Path(path)
    if not path.exists():
        logger.error("图像不存在：%s", path)
        return None
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        if buffer.size == 0:
            logger.error("图像文件为空：%s", path)
            return None
        image = cv2.imdecode(buffer, flags)
        if image is None:
            logger.error("无法解码图像（格式不支持或文件损坏）：%s", path)
        return image
    except Exception as exc:  # pragma: no cover - 取决于具体文件
        logger.error("读取图像失败 %s：%s", path, exc)
        return None


def imwrite_unicode(path: PathLike, image: np.ndarray) -> bool:
    """写出图像，兼容中文/空格路径。"""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix if path.suffix else ".png"
        ok, buffer = cv2.imencode(suffix, image)
        if not ok:
            logger.error("图像编码失败：%s", path)
            return False
        buffer.tofile(str(path))
        return True
    except Exception as exc:  # pragma: no cover
        logger.error("写出图像失败 %s：%s", path, exc)
        return False


# ---------------------------------------------------------------------------
# 基础变换
# ---------------------------------------------------------------------------
def to_gray(image: np.ndarray) -> np.ndarray:
    """转为单通道灰度图。"""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """确保图像为 3 通道 BGR。保留原始位深。"""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def to_display_uint8(image: np.ndarray) -> np.ndarray:
    """把任意位深的图像转成可绘制/可保存的 8 位 BGR 图。

    **仅用于可视化**（画框、写文字、存图）。16 位辐射数据的 DN 值本身是
    物理量，画到图上必须先拉伸到 8 位，但这会丢失绝对标定，
    因此绝不能用本函数的输出做温度计算——温度计算走
    ``InfraredDetector.build_temperature_field``。
    """
    if image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3:
        return image

    data = image
    if data.dtype != np.uint8:
        data = data.astype(np.float32)
        lo, hi = float(data.min()), float(data.max())
        data = np.zeros_like(data) if hi - lo < 1e-6 else (data - lo) / (hi - lo)
        data = (data * 255.0).astype(np.uint8)

    if data.ndim == 2:
        return cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    if data.shape[2] == 4:
        return cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    return data


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    """按比例缩放，使最长边不超过 max_side。小于该值时原样返回。"""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side or longest == 0:
        return image
    scale = max_side / float(longest)
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def denoise(image: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """去噪。``bilateral`` 保边效果好，适合后续做边缘/轮廓分析。"""
    if method == "bilateral":
        return cv2.bilateralFilter(ensure_bgr(image), d=7, sigmaColor=45, sigmaSpace=45)
    if method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    if method == "median":
        return cv2.medianBlur(image, 5)
    return image


def crop_bbox(
    image: np.ndarray,
    bbox: BBox,
    padding_ratio: float = 0.08,
) -> np.ndarray:
    """按 bbox 裁剪 ROI，并向外扩充 padding_ratio 比例的边距。

    自动裁剪到图像边界内；若区域非法则返回整图副本。
    """
    h, w = image.shape[:2]
    pad_x = bbox.width * padding_ratio
    pad_y = bbox.height * padding_ratio

    x1 = int(max(0, np.floor(bbox.x1 - pad_x)))
    y1 = int(max(0, np.floor(bbox.y1 - pad_y)))
    x2 = int(min(w, np.ceil(bbox.x2 + pad_x)))
    y2 = int(min(h, np.ceil(bbox.y2 + pad_y)))

    if x2 <= x1 or y2 <= y1:
        logger.warning("非法裁剪区域 %s，返回整图", bbox.to_list())
        return image.copy()
    return image[y1:y2, x1:x2].copy()


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    """CLAHE 自适应直方图均衡，用于提升暗部细节（巡检图常见背光）。"""
    gray = to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# 中文文字绘制
# ---------------------------------------------------------------------------
_FONT_CANDIDATES: Sequence[str] = (
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",    # 黑体
    r"C:\Windows\Fonts\simsun.ttc",    # 宋体
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)

_cached_font_path: Optional[str] = None
_font_searched = False


def find_chinese_font() -> Optional[str]:
    """在系统中查找可用的中文字体，结果缓存。"""
    global _cached_font_path, _font_searched
    if _font_searched:
        return _cached_font_path
    _font_searched = True
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            _cached_font_path = candidate
            logger.debug("使用中文字体：%s", candidate)
            return _cached_font_path
    logger.warning(
        "未找到系统中文字体，标注文字将降级为 ASCII 标签（中文可能显示为方框）。"
    )
    return None


_LATIN_FALLBACK: dict[str, str] = {
    "变压器": "Transformer",
    "套管": "Bushing",
    "接线端子": "Terminal",
    "储油柜": "Conservator",
    "散热器": "Radiator",
    "渗漏油": "OilLeak",
    "锈蚀": "Rust",
    "变形": "Deformation",
    "悬挂异物": "ForeignObj",
    "部件破损": "BrokenPart",
    "油污": "OilStain",
    "热点": "HotSpot",
}


def ascii_label(text: str) -> str:
    """把中文标签映射为 ASCII，供无中文字体时降级使用。"""
    return _LATIN_FALLBACK.get(text, text.encode("ascii", "replace").decode("ascii"))


def draw_text(
    image: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color: Color = (255, 255, 255),
    font_size: int = 16,
    bg_color: Optional[Color] = None,
) -> np.ndarray:
    """在图像上绘制文字，支持中文。

    ``org`` 为文字左下角坐标。找不到中文字体时自动降级为 ASCII 标签 + cv2.putText。
    """
    font_path = find_chinese_font()
    if font_path is None:
        label = ascii_label(text)
        cv2.putText(image, label, (org[0], org[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return image

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:  # pragma: no cover
        cv2.putText(image, ascii_label(text), (org[0], org[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return image

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:  # pragma: no cover
        font = ImageFont.load_default()

    # PIL 使用 RGB，需要转换后转回，避免颜色通道错位
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    try:
        left, top, right, bottom = draw.textbbox(org, text, font=font)
    except Exception:  # 极老版本 PIL
        right, bottom = org[0] + len(text) * font_size, org[1] + font_size
        left, top = org

    if bg_color is not None:
        pad = 3
        draw.rectangle([left - pad, top - pad, right + pad, bottom + pad], fill=bg_color)

    # 注意：PIL 的 fill 需要 RGB，而调用方传入的是 BGR
    draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))

    result = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    np.copyto(image, result)
    return image


# ---------------------------------------------------------------------------
# 标注绘制
# ---------------------------------------------------------------------------
def draw_detections(
    image: np.ndarray,
    detections: Iterable[Detection],
    thickness: int = 2,
    show_confidence: bool = True,
    label_map: Optional[dict[str, str]] = None,
) -> np.ndarray:
    """在图像副本上绘制检测框。返回新图像，不修改入参。"""
    canvas = ensure_bgr(image).copy()
    label_map = label_map or {}

    for det in detections:
        color = CATEGORY_COLORS.get(det.category, PALETTE[0])
        p1 = (int(round(det.bbox.x1)), int(round(det.bbox.y1)))
        p2 = (int(round(det.bbox.x2)), int(round(det.bbox.y2)))
        cv2.rectangle(canvas, p1, p2, color, thickness)

        name = label_map.get(det.label, det.label)
        caption = f"{name} {det.confidence:.2f}" if show_confidence else name

        # 标签画在框上方；顶到边界时改画到框内
        text_y = max(20, p1[1] - 6)
        draw_text(canvas, caption, (p1[0], text_y),
                  color=(255, 255, 255), font_size=15, bg_color=color)

    return canvas


def draw_hot_regions(
    image: np.ndarray,
    regions: Sequence,
    thickness: int = 2,
) -> np.ndarray:
    """在红外图像上绘制热点框与温度。``regions`` 为 HotRegion 序列。

    输入可以是 16 位辐射图，会先拉伸到 8 位再绘制（仅影响显示，不影响温度计算）。
    """
    canvas = to_display_uint8(image).copy()
    for region in regions:
        bbox: BBox = region.bbox
        color = {
            "normal": (0, 200, 0),
            "warning": (0, 200, 255),
            "alarm": (0, 120, 255),
            "critical": (0, 0, 255),
        }.get(getattr(region.severity, "value", str(region.severity)), (0, 0, 255))

        p1 = (int(round(bbox.x1)), int(round(bbox.y1)))
        p2 = (int(round(bbox.x2)), int(round(bbox.y2)))
        cv2.rectangle(canvas, p1, p2, color, thickness)

        caption = f"{region.max_temp_c:.1f}C dT{region.delta_temp_c:.1f}K"
        draw_text(canvas, caption, (p1[0], max(20, p1[1] - 6)),
                  color=(255, 255, 255), font_size=15, bg_color=color)
    return canvas


def side_by_side(images: Sequence[np.ndarray], gap: int = 12) -> np.ndarray:
    """把多张图横向拼接（高度对齐），用于生成对比图。"""
    valid = [ensure_bgr(img) for img in images if img is not None and img.size > 0]
    if not valid:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    target_h = max(img.shape[0] for img in valid)
    resized = []
    for img in valid:
        h, w = img.shape[:2]
        if h != target_h:
            scale = target_h / float(h)
            img = cv2.resize(img, (max(1, int(w * scale)), target_h))
        resized.append(img)
    separator = np.full((target_h, gap, 3), 240, dtype=np.uint8)
    parts: List[np.ndarray] = []
    for idx, img in enumerate(resized):
        if idx:
            parts.append(separator)
        parts.append(img)
    return np.hstack(parts)


# ---------------------------------------------------------------------------
# 红外图像判定
# ---------------------------------------------------------------------------
def looks_like_thermal(image: np.ndarray) -> tuple[bool, str]:
    """判断图像是否是红外热像图，并返回判定依据。

    判据（按可靠性排序）：
      1. 单通道 16 位图 → 极可能是辐射原始数据
      2. 伪彩色调色板特征：像素集中在少数几个色相上，且饱和度极高
      3. 反查表命中：像素颜色落在常见 ironbow / rainbow 调色板上

    返回 ``(是否红外, 依据说明)``。该判定只用于选择解析路径，
    最终以调用方显式指定的模态为准。
    """
    if image.ndim == 2 and image.dtype == np.uint16:
        return True, "16 位单通道灰度图，符合辐射原始数据特征"

    if image.ndim == 2:
        return True, "单通道灰度图，按红外灰度图处理"

    hsv = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    hue = hsv[:, :, 0].astype(np.int32)

    mean_sat = float(sat.mean())
    # 伪彩色热像图饱和度普遍很高（>110/255），可见光照片通常低得多
    if mean_sat < 100:
        return False, f"平均饱和度 {mean_sat:.0f} 偏低，判定为可见光图像"

    # 统计色相分布：伪彩色图的色相集中在有限几档
    hist = np.bincount(hue.ravel(), minlength=180).astype(np.float64)
    hist /= max(1.0, hist.sum())
    top_share = float(np.sort(hist)[-8:].sum())

    if top_share > 0.55:
        return True, f"饱和度 {mean_sat:.0f} 且色相集中度高（前8档占比 {top_share:.0%}），符合伪彩色热像图"
    return False, f"色相分布分散（前8档仅占 {top_share:.0%}），判定为可见光图像"


__all__ = [
    "PALETTE",
    "CATEGORY_COLORS",
    "imread_unicode",
    "imwrite_unicode",
    "to_gray",
    "ensure_bgr",
    "resize_max_side",
    "denoise",
    "crop_bbox",
    "enhance_contrast",
    "find_chinese_font",
    "draw_text",
    "draw_detections",
    "draw_hot_regions",
    "side_by_side",
    "looks_like_thermal",
    "ascii_label",
]
