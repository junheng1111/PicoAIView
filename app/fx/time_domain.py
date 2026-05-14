"""
时间域特效：echo_trails（残影拖尾）/ strobe / slit_scan。
依赖帧缓冲 ring buffer（最近 8 帧）。
"""

import collections
from typing import Deque

import cv2
import numpy as np

from app.fx._registry import register_fx

# 全局帧 ring buffer（最多 8 帧）
_RING: Deque[np.ndarray] = collections.deque(maxlen=8)
_RING_LOCK = __import__("threading").Lock()


def push_frame_to_ring(frame: np.ndarray) -> None:
    """FX Compositor 每帧调用，把帧加入 ring buffer。"""
    with _RING_LOCK:
        _RING.append(frame.copy())


@register_fx(
    fx_id="echo_trails",
    category="time",
    latency_tier="medium",
    params={"frames": [2, 8], "decay": [0.0, 1.0]},
    description="残影拖尾：前 N 帧 alpha 衰减叠加，rave / electronic drop 用",
)
def apply_echo_trails(frame: np.ndarray, frames: int = 4,
                      decay: float = 0.6, **_) -> np.ndarray:
    with _RING_LOCK:
        history = list(_RING)

    if not history:
        return frame

    n = min(frames, len(history))
    result = frame.astype(np.float32)
    alpha = decay
    for i in range(n - 1, -1, -1):
        past = history[-(i + 1)].astype(np.float32)
        result = cv2.addWeighted(result, 1.0, past, alpha * (1.0 - decay ** (i + 1)), 0)

    return np.clip(result, 0, 255).astype(np.uint8)


@register_fx(
    fx_id="strobe",
    category="time",
    latency_tier="light",
    params={"rate": [2, 30]},
    description="频闪效果，rate=每秒切黑帧次数",
)
def apply_strobe(frame: np.ndarray, rate: float = 8.0, t: float = 0.0, **_) -> np.ndarray:
    phase = (t * rate) % 1.0
    if phase < 0.5:
        return frame
    return np.zeros_like(frame)


@register_fx(
    fx_id="freeze",
    category="time",
    latency_tier="light",
    params={},
    description="定格：输出最后一帧（需外部控制触发/解除）",
)
def apply_freeze(frame: np.ndarray, frozen: bool = False, **_) -> np.ndarray:
    with _RING_LOCK:
        if frozen and _RING:
            return _RING[-1].copy()
    return frame


@register_fx(
    fx_id="slit_scan",
    category="time",
    latency_tier="heavy",
    params={"speed": [0.1, 2.0]},
    description="Slit-scan：行级帧混合，实验性（独占）",
)
def apply_slit_scan(frame: np.ndarray, speed: float = 1.0, **_) -> np.ndarray:
    with _RING_LOCK:
        if len(_RING) < 2:
            return frame
        history = list(_RING)

    h, w = frame.shape[:2]
    result = np.empty_like(frame)
    n = len(history)
    for row in range(h):
        t_offset = int((row / h) * n * speed) % n
        src = history[t_offset]
        result[row] = src[row] if src.shape[0] > row else frame[row]
    return result
