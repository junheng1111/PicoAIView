"""
数字 PTZ 特效：pan / tilt / zoom_pulse / shake。
全部 latency_tier=light（仿射变换 <1ms）。
"""

import math
import random
import time

import cv2
import numpy as np

from app.fx._registry import register_fx


@register_fx(
    fx_id="pan",
    category="digital_ptz",
    latency_tier="light",
    params={"amp": [0.0, 0.1], "phase": [0.0, 6.28]},
    description="数字横向摇移，amp=画面宽比例，随节拍做正弦运动",
)
def apply_pan(frame: np.ndarray, amp: float = 0.03, phase: float = 0.0,
              t: float = 0.0, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    dx = int(amp * w * math.sin(t * 2 * math.pi + phase))
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


@register_fx(
    fx_id="tilt",
    category="digital_ptz",
    latency_tier="light",
    params={"amp": [0.0, 0.1]},
    description="数字纵向摇移",
)
def apply_tilt(frame: np.ndarray, amp: float = 0.03,
               t: float = 0.0, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    dy = int(amp * h * math.sin(t * 2 * math.pi))
    M = np.float32([[1, 0, 0], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


@register_fx(
    fx_id="zoom_pulse",
    category="digital_ptz",
    latency_tier="light",
    params={"amp": [0.0, 0.2]},
    description="重拍缩放脉冲，amp=缩放幅度（0.1=10%放大）",
)
def apply_zoom_pulse(frame: np.ndarray, amp: float = 0.08, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = 1.0 + amp
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


@register_fx(
    fx_id="shake",
    category="digital_ptz",
    latency_tier="light",
    params={"px": [1, 12]},
    description="onset 抖动，px=最大像素偏移量",
)
def apply_shake(frame: np.ndarray, px: float = 4.0, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    dx = random.randint(-int(px), int(px))
    dy = random.randint(-int(px // 2), int(px // 2))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


@register_fx(
    fx_id="breathe",
    category="digital_ptz",
    latency_tier="light",
    params={"amp": [0.01, 0.08], "period": [2.0, 12.0]},
    description="呼吸感缩放：极缓正弦 zoom，amp=幅度，period=周期秒，intro/bridge/outro 首选",
)
def apply_breathe(frame: np.ndarray, amp: float = 0.03, period: float = 6.0,
                  t: float = 0.0, **_) -> np.ndarray:
    period = max(period, 0.1)
    scale = 1.0 + amp * math.sin(2 * math.pi * t / period)
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0, scale)
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


@register_fx(
    fx_id="slow_push",
    category="digital_ptz",
    latency_tier="light",
    params={"direction": ["in", "out"], "period": [10.0, 60.0], "amp": [0.05, 0.25]},
    description="缓慢推进/拉出：半周期内持续单向缩放，period=推进周期秒，桥段前积累张力",
)
def apply_slow_push(frame: np.ndarray, direction: str = "in", period: float = 30.0,
                    amp: float = 0.12, t: float = 0.0, **_) -> np.ndarray:
    period = max(period, 1.0)
    t_mod = t % period
    phase = 0.0 if direction == "in" else math.pi
    scale = 1.0 + amp * 0.5 * (1.0 - math.cos(2 * math.pi * t_mod / period + phase))
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0, scale)
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


_KEN_BURNS_TARGET_MAP = {
    "lt": (-1, -1), "rt": (1, -1), "lb": (-1, 1), "rb": (1, 1),
    "l":  (-1,  0), "r":  (1,  0), "t":  (0, -1), "b":  (0,  1),
}


@register_fx(
    fx_id="ken_burns",
    category="digital_ptz",
    latency_tier="light",
    params={"target": ["lt", "rt", "lb", "rb", "l", "r", "t", "b"],
            "period": [15.0, 60.0], "zoom": [0.02, 0.15]},
    description="Ken Burns 运镜：缓慢 pan+zoom 组合，电影纪录片质感，verse/intro 用",
)
def apply_ken_burns(frame: np.ndarray, target: str = "rt", period: float = 30.0,
                    zoom: float = 0.08, t: float = 0.0, **_) -> np.ndarray:
    period = max(period, 1.0)
    # 半正弦：0→peak→0，与段落起伏自然契合
    progress = math.sin(math.pi * (t % period) / period)
    tx, ty = _KEN_BURNS_TARGET_MAP.get(target, (1, -1))
    h, w = frame.shape[:2]
    dx = int(tx * zoom * 0.4 * w * progress)
    dy = int(ty * zoom * 0.4 * h * progress)
    scale = 1.0 + zoom * progress
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


@register_fx(
    fx_id="subject_zoom",
    category="digital_ptz",
    latency_tier="light",
    depends_on=["yolov8s"],
    params={"ratio": [0.5, 0.95], "smooth": [0.0, 1.0]},
    description="以检测到的主体为中心做数字裁剪放大，ratio=裁剪比例",
)
def apply_subject_zoom(frame: np.ndarray, ratio: float = 0.7,
                       smooth: float = 0.1, vision_state=None, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = 0.5, 0.5

    if vision_state is not None:
        pb = vision_state.primary_bbox()
        if pb:
            cx = pb[0] + pb[2] / 2.0
            cy = pb[1] + pb[3] / 2.0

    half_w = ratio / 2.0
    half_h = ratio / 2.0
    left   = max(0, int((cx - half_w) * w))
    right  = min(w, int((cx + half_w) * w))
    top    = max(0, int((cy - half_h) * h))
    bottom = min(h, int((cy + half_h) * h))

    if right - left < 4 or bottom - top < 4:
        return frame

    cropped = frame[top:bottom, left:right]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
