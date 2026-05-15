"""
BPU 驱动 AI-aware 特效 ⭐。
每个 fx 的"AI 感知部分"由 BpuRunner 在独立线程提供（VisionState），
FX Compositor 只消费 VisionState 最近帧，延迟来自合成步骤本身。
"""

import cv2
import numpy as np

from app.fx._registry import register_fx

# ---------------------------------------------------------------------------
# 智能跟拍
# ---------------------------------------------------------------------------

@register_fx(
    fx_id="subject_track",
    category="ai_aware",
    latency_tier="light",
    depends_on=["yolov8s"],
    params={"smooth": [0.0, 1.0], "zoom": [0.5, 1.0]},
    description="智能跟拍：以 primary bbox 为中心数字裁剪 + 平滑，副歌主拍",
)
def apply_subject_track(frame: np.ndarray, smooth: float = 0.7,
                        zoom: float = 0.75, vision_state=None, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = 0.5, 0.5

    if vision_state is not None:
        pb = vision_state.primary_bbox()
        if pb:
            cx = pb[0] + pb[2] / 2.0
            cy = pb[1] + pb[3] / 2.0

    half = zoom / 2.0
    left   = max(0, int((cx - half) * w))
    right  = min(w, int((cx + half) * w))
    top    = max(0, int((cy - half) * h))
    bottom = min(h, int((cy + half) * h))

    if right - left < 4 or bottom - top < 4:
        return frame

    cropped = frame[top:bottom, left:right]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# 节拍 bbox 描边
# ---------------------------------------------------------------------------

@register_fx(
    fx_id="beat_bbox",
    category="ai_aware",
    latency_tier="light",
    depends_on=["yolov8s"],
    params={"color": "hex", "thickness": [1, 6]},
    description="主拍时在主体 bbox 外画发光描边",
)
def apply_beat_bbox(frame: np.ndarray, color: str = "#00ff88",
                    thickness: int = 3, vision_state=None, **_) -> np.ndarray:
    if vision_state is None:
        return frame

    h, w = frame.shape[:2]
    result = frame.copy()
    c = _hex_bgr(color)

    for det in vision_state.detections:
        if det.class_id != 0:
            continue
        x, y, bw, bh = det.bbox
        x1, y1 = int(x * w), int(y * h)
        x2, y2 = int((x + bw) * w), int((y + bh) * h)
        cv2.rectangle(result, (x1, y1), (x2, y2), c, thickness, cv2.LINE_AA)

    return result


# ---------------------------------------------------------------------------
# 姿态光剑
# ---------------------------------------------------------------------------

POSE_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),   # 手臂
    (11, 13), (13, 15), (12, 14), (14, 16),  # 腿
    (5, 6), (11, 12),  # 肩/髋
    (5, 11), (6, 12),  # 躯干
]

_POSE_TRAIL: list = []
_MAX_TRAIL = 12


@register_fx(
    fx_id="pose_trail",
    category="ai_aware",
    latency_tier="medium",
    depends_on=["yolov8s_pose"],
    params={"decay": [0.0, 1.0], "color": "hex"},
    description="姿态光剑：手腕/脚踝关键点拖尾粒子，onset 爆发用",
)
def apply_pose_trail(frame: np.ndarray, decay: float = 0.7,
                     color: str = "#ff6600", vision_state=None, **_) -> np.ndarray:
    global _POSE_TRAIL
    h, w = frame.shape[:2]
    result = frame.copy().astype(np.float32)
    c = np.array(_hex_bgr(color), dtype=np.float32)

    # 更新拖尾
    new_pts = []
    if vision_state is not None:
        for pd in vision_state.pose_detections:
            for idx in (9, 10, 15, 16):   # 手腕 + 脚踝
                if idx < len(pd.keypoints):
                    kp = pd.keypoints[idx]
                    if kp.score > 0.3:
                        new_pts.append((kp.x, kp.y, 1.0))  # (x, y, alpha)

    _POSE_TRAIL = [(x, y, a * decay) for x, y, a in _POSE_TRAIL if a * decay > 0.05]
    _POSE_TRAIL.extend(new_pts)
    _POSE_TRAIL = _POSE_TRAIL[-_MAX_TRAIL:]

    # 绘制拖尾
    for x, y, alpha in _POSE_TRAIL:
        px, py = int(x * w), int(y * h)
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(result, (px, py), 6, (c * alpha).tolist(), -1)

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 人像分割相关
# ---------------------------------------------------------------------------

@register_fx(
    fx_id="semantic_bg_blur",
    category="ai_aware",
    latency_tier="medium",
    depends_on=["yolov8s_seg"],
    params={"blur": [0.0, 1.0]},
    description="语义背景模糊：分割 mask 外区域高斯模糊，minimal 风",
)
def apply_semantic_bg_blur(frame: np.ndarray, blur: float = 0.7,
                           vision_state=None, **_) -> np.ndarray:
    # seg_mask 需由 BpuRunner 在副歌段启用 seg 模型时填充
    # 若无 mask，退化为全帧模糊
    ksize = max(3, int(blur * 51) | 1)
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)

    if vision_state is not None and hasattr(vision_state, "seg_mask") \
            and vision_state.seg_mask is not None:
        mask = cv2.resize(vision_state.seg_mask, (frame.shape[1], frame.shape[0]))
        mask3 = cv2.merge([mask, mask, mask])
        return np.where(mask3 > 0, frame, blurred)

    return blurred


@register_fx(
    fx_id="portrait_filter",
    category="ai_aware",
    latency_tier="medium",
    depends_on=["yolov8s_seg"],
    params={"saturation": [0.0, 2.0], "brightness": [0.0, 2.0]},
    description="人像滤镜：前景增彩，背景降饱和，副歌用",
)
def apply_portrait_filter(frame: np.ndarray, saturation: float = 1.5,
                          brightness: float = 1.2, vision_state=None, **_) -> np.ndarray:
    # 增强整体前景感（无 mask 时全帧处理）
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * brightness, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# 多人聚焦
# ---------------------------------------------------------------------------

_FOCUS_IDX: int = 0
_LAST_SWITCH: float = 0.0


@register_fx(
    fx_id="multi_person_focus",
    category="ai_aware",
    latency_tier="light",
    depends_on=["yolov8s"],
    params={"switch_on": ["downbeat"]},
    description="多人聚焦：downbeat 循环切主角，暗化其他人",
)
def apply_multi_person_focus(frame: np.ndarray, vision_state=None,
                              t: float = 0.0, **_) -> np.ndarray:
    global _FOCUS_IDX, _LAST_SWITCH
    if vision_state is None:
        return frame

    h, w = frame.shape[:2]
    persons = [d for d in vision_state.detections if d.class_id == 0]
    if len(persons) < 2:
        return frame

    # 每次触发时切换（外部 t 参数可驱动切换）
    _FOCUS_IDX = _FOCUS_IDX % len(persons)
    focus = persons[_FOCUS_IDX]
    result = (frame.astype(np.float32) * 0.4).astype(np.uint8)

    x, y, bw, bh = focus.bbox
    x1, y1 = max(0, int(x * w)), max(0, int(y * h))
    x2, y2 = min(w, int((x + bw) * w)), min(h, int((y + bh) * h))
    result[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return result


# ---------------------------------------------------------------------------
# 人群密度反应
# ---------------------------------------------------------------------------

@register_fx(
    fx_id="crowd_density_react",
    category="ai_aware",
    latency_tier="light",
    depends_on=["yolov8s"],
    params={"max_count": [1, 20]},
    description="人群密度驱动其他 fx 强度（元数据，自身无视觉变化）",
)
def apply_crowd_density_react(frame: np.ndarray, vision_state=None,
                               max_count: int = 8, **_) -> np.ndarray:
    # 此 fx 本身不修改画面，density 信息通过 vision_state.subject_count 传递
    # 配合其他 fx 参数动态缩放（由 FX Compositor 读取 density 后注入 amp）
    return frame


# ---------------------------------------------------------------------------
# 运动反应
# ---------------------------------------------------------------------------

@register_fx(
    fx_id="motion_react",
    category="ai_aware",
    latency_tier="light",
    depends_on=["yolov8s"],
    params={"intensity_mul": [0.5, 3.0]},
    description="舞步触发：motion_intensity 驱动饱和度/亮度，越嗨越鲜艳",
)
def apply_motion_react(frame: np.ndarray, intensity_mul: float = 1.5,
                       vision_state=None, **_) -> np.ndarray:
    intensity = 0.0
    if vision_state is not None:
        intensity = float(getattr(vision_state, "motion_intensity", 0.0))

    if intensity < 0.05:
        return frame

    sat_boost = 1.0 + intensity * intensity_mul
    bright_boost = 1.0 + intensity * 0.3
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_boost, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * bright_boost, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# 伪景深
# ---------------------------------------------------------------------------

@register_fx(
    fx_id="depth_dof",
    category="ai_aware",
    latency_tier="medium",
    depends_on=["yolov8s"],
    params={"strength": [0.0, 1.0], "focus_plane": [0.0, 1.0]},
    description="伪景深：以主体为焦平面，上下区域渐进模糊，桥段 / 抒情段",
)
def apply_depth_dof(frame: np.ndarray, strength: float = 0.5,
                    focus_plane: float = 0.5, vision_state=None, **_) -> np.ndarray:
    h, w = frame.shape[:2]

    # 用主体 bbox 中心确定焦平面
    focus_y = focus_plane
    if vision_state is not None:
        pb = (vision_state.primary_bbox()
              if hasattr(vision_state, "primary_bbox") else None)
        if pb:
            focus_y = pb[1] + pb[3] / 2.0

    ksize = max(3, int(strength * 25) | 1)
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)

    y_coords = np.linspace(0, 1, h, dtype=np.float32)[:, np.newaxis]
    dist = np.abs(y_coords - focus_y)
    blend = np.clip(dist * 3.5 * strength, 0.0, 1.0)[:, :, np.newaxis]

    result = (frame.astype(np.float32) * (1.0 - blend) +
              blurred.astype(np.float32) * blend)
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 柔和主体漂移（AI-aware 舒缓运镜）
# ---------------------------------------------------------------------------

_DRIFT_CX: float = 0.5
_DRIFT_CY: float = 0.5


@register_fx(
    fx_id="subject_drift",
    category="ai_aware",
    latency_tier="light",
    depends_on=["yolov8s"],
    params={"smooth": [0.9, 0.99], "amp": [0.02, 0.12]},
    description="柔和主体漂移：超低 EMA 跟随主体做 pan，比 subject_track 慢 10 倍，全程持续挂",
)
def apply_subject_drift(frame: np.ndarray, smooth: float = 0.97,
                        amp: float = 0.06, vision_state=None, **_) -> np.ndarray:
    global _DRIFT_CX, _DRIFT_CY
    h, w = frame.shape[:2]

    target_cx, target_cy = 0.5, 0.5
    if vision_state is not None:
        pb = (vision_state.primary_bbox()
              if hasattr(vision_state, "primary_bbox") else None)
        if pb:
            target_cx = pb[0] + pb[2] / 2.0
            target_cy = pb[1] + pb[3] / 2.0

    alpha = 1.0 - smooth
    _DRIFT_CX = smooth * _DRIFT_CX + alpha * target_cx
    _DRIFT_CY = smooth * _DRIFT_CY + alpha * target_cy

    # 将主体中心缓慢拉向画面中心方向（负偏移 = 向画面中心平移）
    dx = int((_DRIFT_CX - 0.5) * amp * w)
    dy = int((_DRIFT_CY - 0.5) * amp * h)
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


# ---------------------------------------------------------------------------
# 粒子跟手势（DESIGN.md §4.2 pose_particles）
# ---------------------------------------------------------------------------

_PART_LIST: list = []   # [(x, y, vx, vy, alpha)]
_MAX_PARTS: int = 120


@register_fx(
    fx_id="pose_particles",
    category="ai_aware",
    latency_tier="medium",
    depends_on=["yolov8s_pose"],
    params={"rate": [0.5, 3.0], "decay": [0.80, 0.98], "color": "hex"},
    description="粒子跟手势：从腕/踝关键点发射上浮粒子，drop/onset 触发，动感十足",
)
def apply_pose_particles(frame: np.ndarray, rate: float = 1.5, decay: float = 0.92,
                         color: str = "#ff4488", vision_state=None, **_) -> np.ndarray:
    global _PART_LIST
    import random
    h, w = frame.shape[:2]
    result = frame.copy().astype(np.float32)
    c = np.array(_hex_bgr(color), dtype=np.float32)

    # 物理更新
    updated = []
    for x, y, vx, vy, alpha in _PART_LIST:
        x += vx
        y += vy
        vy -= 0.0015          # 上浮
        alpha *= decay
        if alpha > 0.04 and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            updated.append((x, y, vx, vy, alpha))
    _PART_LIST = updated

    # 从关键点发射新粒子（手腕 9/10、脚踝 15/16）
    if vision_state is not None and len(_PART_LIST) < _MAX_PARTS:
        for pd in vision_state.pose_detections:
            for idx in (9, 10, 15, 16):
                if idx >= len(pd.keypoints):
                    continue
                kp = pd.keypoints[idx]
                if kp.score < 0.3:
                    continue
                for _ in range(max(1, int(rate))):
                    vx = random.uniform(-0.007, 0.007)
                    vy = random.uniform(-0.011, -0.003)
                    _PART_LIST.append((kp.x, kp.y, vx, vy, 1.0))

    # 渲染
    for x, y, vx, vy, alpha in _PART_LIST:
        px, py = int(x * w), int(y * h)
        if 0 <= px < w and 0 <= py < h:
            radius = max(2, int(alpha * 7))
            cv2.circle(result, (px, py), radius, (c * alpha).tolist(), -1, cv2.LINE_AA)

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _hex_bgr(color: str):
    h = color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return b, g, r
