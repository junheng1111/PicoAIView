"""
几何变形特效：fisheye / mirror / kaleidoscope / ripple / twirl / whip_pan。
全部 latency_tier=light（remap / 仿射 ~1ms）。
"""

import math

import cv2
import numpy as np

from app.fx._registry import register_fx


def _make_map(h: int, w: int, mapx: np.ndarray, mapy: np.ndarray):
    return cv2.remap(np.zeros((h, w, 3), np.uint8), mapx, mapy,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


@register_fx(
    fx_id="fisheye",
    category="geometry",
    latency_tier="light",
    params={"strength": [0.0, 1.0]},
    description="鱼眼畸变，strength=1 最强",
)
def apply_fisheye(frame: np.ndarray, strength: float = 0.5, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (x - cx) / cx
    ny = (y - cy) / cy
    r = np.sqrt(nx ** 2 + ny ** 2)
    theta = np.arctan2(ny, nx)
    r2 = r * (1.0 + strength * r ** 2)
    mapx = (r2 * np.cos(theta) * cx + cx).astype(np.float32)
    mapy = (r2 * np.sin(theta) * cy + cy).astype(np.float32)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


@register_fx(
    fx_id="mirror",
    category="geometry",
    latency_tier="light",
    params={"axis": ["h", "v", "both"]},
    description="镜像分屏，axis=h(左右) / v(上下) / both(四方)",
)
def apply_mirror(frame: np.ndarray, axis: str = "h", **_) -> np.ndarray:
    h, w = frame.shape[:2]
    if axis == "h":
        left = frame[:, : w // 2]
        return np.hstack([left, left[:, ::-1]])
    if axis == "v":
        top = frame[: h // 2, :]
        return np.vstack([top, top[::-1, :]])
    # both
    q = frame[: h // 2, : w // 2]
    row = np.hstack([q, q[:, ::-1]])
    return np.vstack([row, row[::-1, :]])


@register_fx(
    fx_id="kaleidoscope",
    category="geometry",
    latency_tier="light",
    params={"slices": [4, 12]},
    description="万花筒效果，slices=瓣数",
)
def apply_kaleidoscope(frame: np.ndarray, slices: int = 6, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = x - cx, y - cy
    theta = np.arctan2(ny, nx)
    r = np.sqrt(nx ** 2 + ny ** 2)
    seg = math.pi / max(slices, 2)
    theta_mod = np.abs((theta % (2 * seg)) - seg)
    mapx = (r * np.cos(theta_mod) + cx).astype(np.float32)
    mapy = (r * np.sin(theta_mod) + cy).astype(np.float32)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


@register_fx(
    fx_id="ripple",
    category="geometry",
    latency_tier="light",
    params={"amp": [0.0, 0.05], "freq": [1.0, 20.0]},
    description="水波纹位移，amp=振幅（归一化），freq=频率",
)
def apply_ripple(frame: np.ndarray, amp: float = 0.02, freq: float = 8.0,
                 t: float = 0.0, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = amp * w * np.sin(2 * math.pi * freq * y / h + t * 5.0)
    dy = amp * h * np.cos(2 * math.pi * freq * x / w + t * 5.0)
    mapx = np.clip(x + dx, 0, w - 1).astype(np.float32)
    mapy = np.clip(y + dy, 0, h - 1).astype(np.float32)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


@register_fx(
    fx_id="twirl",
    category="geometry",
    latency_tier="light",
    params={"angle": [-1.0, 1.0], "radius": [0.1, 1.0]},
    description="漩涡扭曲，angle=扭曲角度（弧度），radius=影响半径（归一化）",
)
def apply_twirl(frame: np.ndarray, angle: float = 0.5,
                radius: float = 0.5, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    rad = min(w, h) * radius
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = x - cx, y - cy
    dist = np.sqrt(nx ** 2 + ny ** 2)
    theta_orig = np.arctan2(ny, nx)
    strength = np.clip(1.0 - dist / (rad + 1e-6), 0.0, 1.0)
    theta_new = theta_orig + angle * strength
    mapx = (dist * np.cos(theta_new) + cx).astype(np.float32)
    mapy = (dist * np.sin(theta_new) + cy).astype(np.float32)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


@register_fx(
    fx_id="whip_pan",
    category="geometry",
    latency_tier="medium",
    params={"direction": ["left", "right"], "strength": [0.0, 1.0]},
    description="甩镜转场：水平方向模糊+平移，仅转场瞬间使用",
)
def apply_whip_pan(frame: np.ndarray, direction: str = "right",
                   strength: float = 0.5, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    ksize = max(3, int(strength * 60) | 1)  # 保证奇数
    blurred = cv2.blur(frame, (ksize, 1))
    dx = int(strength * w * 0.3) * (1 if direction == "right" else -1)
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    shifted = cv2.warpAffine(blurred, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return shifted
