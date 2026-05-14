"""
叠加层特效：beat_ring / spectrum_bar / lens_flare / overlay_text。
"""

import math

import cv2
import numpy as np

from app.fx._registry import register_fx


@register_fx(
    fx_id="beat_ring",
    category="overlay",
    latency_tier="light",
    params={"color": "hex", "thickness": [1, 8]},
    description="节拍光圈：在画面中心画扩散圆环",
)
def apply_beat_ring(frame: np.ndarray, color: str = "#ffffff",
                    thickness: int = 3, pulse: float = 1.0, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    r = int(min(w, h) * 0.3 * pulse)
    if r <= 0:
        return frame
    b, g, rv = _hex_bgr(color)
    result = frame.copy()
    cv2.circle(result, (w // 2, h // 2), r, (b, g, rv), thickness, cv2.LINE_AA)
    return result


@register_fx(
    fx_id="spectrum_bar",
    category="overlay",
    latency_tier="light",
    params={"height": [0.05, 0.3], "color": "hex"},
    description="画面底部频谱柱（从 RMS 数组驱动）",
)
def apply_spectrum_bar(frame: np.ndarray, height: float = 0.15,
                       color: str = "#00ffcc", rms_data=None, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    result = frame.copy()
    if rms_data is None or len(rms_data) == 0:
        return result

    n = len(rms_data)
    bar_w = max(1, w // n)
    b, g, rv = _hex_bgr(color)
    for i, v in enumerate(rms_data):
        bh = int(h * height * float(v))
        x1 = i * bar_w
        x2 = x1 + bar_w - 1
        y1 = h - bh
        cv2.rectangle(result, (x1, y1), (x2, h - 1), (b, g, rv), -1)
    return result


@register_fx(
    fx_id="lens_flare",
    category="overlay",
    latency_tier="medium",
    params={"intensity": [0.0, 1.0], "x": [0.0, 1.0], "y": [0.0, 1.0]},
    description="镜头光晕，intensity=强度，x/y=光源归一化坐标",
)
def apply_lens_flare(frame: np.ndarray, intensity: float = 0.6,
                     x: float = 0.7, y: float = 0.3, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    result = frame.astype(np.float32)
    cx, cy = int(x * w), int(y * h)

    # 主光斑
    overlay = np.zeros_like(result)
    cv2.circle(overlay, (cx, cy), int(w * 0.04), (200, 200, 255), -1)
    cv2.GaussianBlur(overlay, (51, 51), 25, dst=overlay)
    result = np.clip(result + overlay * intensity, 0, 255)

    # 小光斑链
    dx, dy = w // 2 - cx, h // 2 - cy
    for i in range(1, 5):
        lx = cx + dx * i // 4
        ly = cy + dy * i // 4
        r = max(3, int(w * 0.01 * (5 - i)))
        spot = np.zeros_like(result)
        cv2.circle(spot, (lx, ly), r, (180, 180, 255), -1)
        result = np.clip(result + spot * intensity * 0.5, 0, 255)

    return result.astype(np.uint8)


@register_fx(
    fx_id="god_rays",
    category="overlay",
    latency_tier="medium",
    params={"intensity": [0.0, 1.0], "color": "hex"},
    description="体积光神光束（径向高斯渐变），副歌情绪感",
)
def apply_god_rays(frame: np.ndarray, intensity: float = 0.5,
                   color: str = "#ffffcc", t: float = 0.0, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    bc, gc, rc = _hex_bgr(color)

    # 从画面顶部向下衰减的渐变层
    y_pos = np.linspace(0, 1, h, dtype=np.float32)[:, np.newaxis]
    grad = np.maximum(0.0, 1.0 - y_pos * 2.5)
    # 添加轻微时间波动
    grad = grad * (0.8 + 0.2 * math.sin(t * 0.7))

    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[:, :, 0] = bc * grad
    overlay[:, :, 1] = gc * grad
    overlay[:, :, 2] = rc * grad

    # 垂直模糊模拟光柱散射
    overlay = cv2.GaussianBlur(overlay, (1, 31), 0)
    return np.clip(frame.astype(np.float32) + overlay * intensity,
                   0, 255).astype(np.uint8)


@register_fx(
    fx_id="light_leak",
    category="overlay",
    latency_tier="light",
    params={"intensity": [0.0, 1.0], "color": "hex"},
    description="漏光效果：角落渐变暖光叠加，复古 / 温暖段",
)
def apply_light_leak(frame: np.ndarray, intensity: float = 0.3,
                     color: str = "#ff8800", t: float = 0.0, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    bc, gc, rc = _hex_bgr(color)

    # 时间波动强度
    strength = intensity * (0.7 + 0.3 * math.sin(t * 0.4))

    # 右上角到左下角的渐变
    y_pos = np.linspace(0, 1, h, dtype=np.float32)[:, np.newaxis]
    x_pos = np.linspace(1, 0, w, dtype=np.float32)[np.newaxis, :]
    grad = np.clip(1.0 - (y_pos + x_pos) * 0.7, 0, 1).astype(np.float32)

    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[:, :, 0] = bc * grad
    overlay[:, :, 1] = gc * grad
    overlay[:, :, 2] = rc * grad

    return np.clip(frame.astype(np.float32) + overlay * strength,
                   0, 255).astype(np.uint8)


@register_fx(
    fx_id="confetti",
    category="overlay",
    latency_tier="medium",
    params={"density": [10, 100], "size": [3, 15]},
    description="彩色粒子雨，副歌庆祝感",
)
def apply_confetti(frame: np.ndarray, density: int = 40, size: int = 6,
                   t: float = 0.0, **_) -> np.ndarray:
    import random as _rnd
    h, w = frame.shape[:2]
    result = frame.copy()
    _rnd.seed(int(t * 20))
    colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255),
              (255, 255, 80), (255, 80, 255), (80, 255, 255)]
    for _ in range(int(density)):
        x = _rnd.randint(0, w - 1)
        # y falls based on time (loop every 2 seconds)
        y = int(((_rnd.random() + t * 0.5) % 1.0) * h)
        color = colors[_rnd.randint(0, len(colors) - 1)]
        cv2.circle(result, (x, y), max(2, int(size) // 2), color, -1)
    return result


def _hex_bgr(color: str):
    h = color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return b, g, r
