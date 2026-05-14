"""
故障美学（glitch）特效：chromatic_aberration / vhs_tracking / scanlines / datamosh。
大部分 latency_tier=light，少数 medium。
"""

import math
import random

import cv2
import numpy as np

from app.fx._registry import register_fx


@register_fx(
    fx_id="chromatic_aberration",
    category="shader_glitch",
    latency_tier="light",
    params={"amount": [0.0, 0.05]},
    description="RGB 色差错位，drop / glitch 风格",
)
def apply_chromatic_aberration(frame: np.ndarray, amount: float = 0.01, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    dx = max(1, int(amount * w))
    b, g, r = cv2.split(frame)
    r_shifted = np.roll(r, dx, axis=1)
    b_shifted = np.roll(b, -dx, axis=1)
    return cv2.merge([b_shifted, g, r_shifted])


@register_fx(
    fx_id="vhs_tracking",
    category="shader_glitch",
    latency_tier="medium",
    params={"intensity": [0.0, 1.0]},
    description="录像带卷曲错位，复古 / 赛博风",
)
def apply_vhs_tracking(frame: np.ndarray, intensity: float = 0.5, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    result = frame.copy()
    num_bands = max(1, int(intensity * 10))
    for _ in range(num_bands):
        y = random.randint(0, h - 1)
        band_h = random.randint(1, max(2, int(h * 0.05)))
        shift = random.randint(-int(w * 0.08 * intensity), int(w * 0.08 * intensity))
        y2 = min(h, y + band_h)
        band = frame[y:y2, :]
        result[y:y2, :] = np.roll(band, shift, axis=1)
    return result


@register_fx(
    fx_id="scanlines",
    category="shader_glitch",
    latency_tier="light",
    params={"spacing": [2, 8], "darkness": [0.0, 0.8]},
    description="CRT 扫描线，8-bit / 赛博风",
)
def apply_scanlines(frame: np.ndarray, spacing: int = 3,
                    darkness: float = 0.4, **_) -> np.ndarray:
    result = frame.copy().astype(np.float32)
    h = frame.shape[0]
    for y in range(0, h, spacing):
        result[y] = result[y] * (1.0 - darkness)
    return np.clip(result, 0, 255).astype(np.uint8)


@register_fx(
    fx_id="roll_bar",
    category="shader_glitch",
    latency_tier="light",
    params={"speed": [0.1, 2.0], "width": [0.02, 0.15]},
    description="行同步错位黑条滚动，故意'坏'画面",
)
def apply_roll_bar(frame: np.ndarray, speed: float = 0.5,
                   width: float = 0.05, t: float = 0.0, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_h = max(1, int(h * width))
    bar_y = int((t * speed * h) % h)
    result = frame.copy()
    y2 = min(h, bar_y + bar_h)
    result[bar_y:y2, :] = np.zeros((y2 - bar_y, w, 3), dtype=np.uint8)
    return result


@register_fx(
    fx_id="datamosh",
    category="shader_glitch",
    latency_tier="medium",
    params={"intensity": [0.0, 1.0]},
    description="伪数据马赛克，块向量混乱，drop 瞬间用",
)
def apply_datamosh(frame: np.ndarray, intensity: float = 0.5, **_) -> np.ndarray:
    from app.fx.time_domain import _RING, _RING_LOCK
    with _RING_LOCK:
        if len(_RING) < 2:
            return frame
        prev = list(_RING)[-2]

    h, w = frame.shape[:2]
    block = max(4, int(16 * intensity))
    result = frame.copy()
    num_blocks = int(intensity * 20)
    for _ in range(num_blocks):
        bx = random.randint(0, max(0, w - block - 1))
        by = random.randint(0, max(0, h - block - 1))
        if prev.shape[0] > by + block and prev.shape[1] > bx + block:
            result[by:by + block, bx:bx + block] = prev[by:by + block, bx:bx + block]
    return result


@register_fx(
    fx_id="glitch_cut",
    category="shader_glitch",
    latency_tier="light",
    params={"intensity": [0.0, 1.0]},
    description="基础 glitch 切割，水平色块错位",
)
def apply_glitch_cut(frame: np.ndarray, intensity: float = 0.3, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    result = frame.copy()
    num_slices = max(1, int(intensity * 15))
    for _ in range(num_slices):
        y = random.randint(0, h - 2)
        sl_h = random.randint(1, max(2, int(h * 0.04)))
        shift = random.randint(-int(w * 0.1), int(w * 0.1))
        y2 = min(h, y + sl_h)
        result[y:y2, :] = np.roll(frame[y:y2, :], shift, axis=1)
    return result


@register_fx(
    fx_id="rgb_shift_strobe",
    category="shader_glitch",
    latency_tier="light",
    params={"amount": [0.0, 0.05]},
    description="RGB 通道动态错位 + 闪烁，高 BPM 段迷幻 glitch",
)
def apply_rgb_shift_strobe(frame: np.ndarray, amount: float = 0.02,
                           t: float = 0.0, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    shift_r = int(amount * w * math.sin(t * 7.3))
    shift_b = int(amount * w * math.cos(t * 5.1))
    b, g, r = cv2.split(frame)
    r_shifted = np.roll(r, shift_r, axis=1)
    b_shifted = np.roll(b, shift_b, axis=1)
    return cv2.merge([b_shifted, g, r_shifted])
