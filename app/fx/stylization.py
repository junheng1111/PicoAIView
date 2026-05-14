"""
风格化渲染（heavy，互斥）：cartoon / pencil_sketch / oil_painting。
latency_tier=heavy，每段最多 1 个，不与其他 heavy 同段。
"""

import cv2
import numpy as np

from app.fx._registry import register_fx


@register_fx(
    fx_id="cartoon",
    category="stylization",
    latency_tier="heavy",
    params={"line_size": [3, 9], "blur": [5, 25]},
    description="卡通 / Cel-shading：边缘检测 + 色阶量化，副歌反差用（独占）",
)
def apply_cartoon(frame: np.ndarray, line_size: int = 5, blur: int = 9, **_) -> np.ndarray:
    line_size = max(3, line_size | 1)
    blur = max(5, blur | 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, line_size, 5)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    return cv2.bitwise_and(color, color, mask=edges)


@register_fx(
    fx_id="pencil_sketch",
    category="stylization",
    latency_tier="heavy",
    params={"shade_factor": [0.01, 0.1]},
    description="铅笔素描效果，纯净段独占",
)
def apply_pencil_sketch(frame: np.ndarray, shade_factor: float = 0.05, **_) -> np.ndarray:
    gray, _ = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.07,
                                shade_factor=shade_factor)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


@register_fx(
    fx_id="oil_painting",
    category="stylization",
    latency_tier="heavy",
    params={"size": [3, 9], "dyn_ratio": [0.01, 0.1]},
    description="油画效果，桥段独占（~10ms）",
)
def apply_oil_painting(frame: np.ndarray, size: int = 7,
                       dyn_ratio: float = 0.05, **_) -> np.ndarray:
    try:
        import cv2.xphoto as xphoto
        return xphoto.oilPainting(frame, size, dyn_ratio)
    except (AttributeError, cv2.error):
        # xphoto 不可用时退化为 bilateral
        return cv2.bilateralFilter(frame, size | 1, 75, 75)
