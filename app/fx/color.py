"""
调色特效：brightness_curve / saturation_flash / hue_rotate / 3d_lut / duotone / vignette。
全部 latency_tier=light（查找表操作 <1ms）。
"""

import numpy as np
import cv2

from app.fx._registry import register_fx


@register_fx(
    fx_id="brightness_curve",
    category="color",
    latency_tier="light",
    params={"k": [0.0, 2.0]},
    description="亮度乘法，k<1 变暗，k>1 提亮，常用于 RMS 驱动",
)
def apply_brightness_curve(frame: np.ndarray, k: float = 1.0, **_) -> np.ndarray:
    k = max(0.0, k)
    if abs(k - 1.0) < 1e-3:
        return frame
    lut = np.clip(np.arange(256, dtype=np.float32) * k, 0, 255).astype(np.uint8)
    return cv2.LUT(frame, lut)


@register_fx(
    fx_id="flash",
    category="color",
    latency_tier="light",
    params={"amp": [0.0, 1.0]},
    description="白色闪光叠加，amp=强度 0-1",
)
def apply_flash(frame: np.ndarray, amp: float = 0.4, **_) -> np.ndarray:
    amp = max(0.0, min(1.0, amp))
    white = np.full_like(frame, 255)
    return cv2.addWeighted(frame, 1.0 - amp, white, amp, 0)


@register_fx(
    fx_id="saturation_flash",
    category="color",
    latency_tier="light",
    params={"amp": [0.0, 1.0]},
    description="饱和度爆闪，amp=0 为灰度，amp=1 为最大饱和",
)
def apply_saturation_flash(frame: np.ndarray, amp: float = 0.6, **_) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + amp * 2.0), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


@register_fx(
    fx_id="hue_rotate",
    category="color",
    latency_tier="light",
    params={"deg": [0, 180]},
    description="色相旋转 deg 度",
)
def apply_hue_rotate(frame: np.ndarray, deg: float = 30.0, **_) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(deg / 2)) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


@register_fx(
    fx_id="invert",
    category="color",
    latency_tier="light",
    params={},
    description="反色（负片效果）",
)
def apply_invert(frame: np.ndarray, **_) -> np.ndarray:
    return cv2.bitwise_not(frame)


@register_fx(
    fx_id="grayscale",
    category="color",
    latency_tier="light",
    params={"strength": [0.0, 1.0]},
    description="灰度混合，strength=1 为完全灰度",
)
def apply_grayscale(frame: np.ndarray, strength: float = 1.0, **_) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    strength = max(0.0, min(1.0, strength))
    return cv2.addWeighted(frame, 1.0 - strength, gray_bgr, strength, 0)


@register_fx(
    fx_id="vignette",
    category="color",
    latency_tier="light",
    params={"strength": [0.0, 1.0], "radius": [0.3, 1.0]},
    description="暗角效果，聚焦感",
)
def apply_vignette(frame: np.ndarray, strength: float = 0.5,
                   radius: float = 0.7, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    Y, X = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    mask = 1.0 - np.clip((dist - radius) / (1.0 - radius + 1e-6), 0.0, 1.0) * strength
    mask = mask[..., np.newaxis].astype(np.float32)
    return np.clip(frame.astype(np.float32) * mask, 0, 255).astype(np.uint8)


@register_fx(
    fx_id="duotone",
    category="color",
    latency_tier="light",
    params={"color_shadow": "hex", "color_highlight": "hex"},
    description="双色调效果，阴影/高光各映射到一个颜色",
)
def apply_duotone(frame: np.ndarray,
                  color_shadow: str = "#0d1b8a",
                  color_highlight: str = "#f5c518", **_) -> np.ndarray:
    def hex_to_bgr(h: str):
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)

    shadow_bgr = np.array(hex_to_bgr(color_shadow), dtype=np.float32) / 255.0
    high_bgr   = np.array(hex_to_bgr(color_highlight), dtype=np.float32) / 255.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    result = np.zeros_like(frame, dtype=np.float32)
    for c in range(3):
        result[..., c] = shadow_bgr[c] * (1.0 - gray) + high_bgr[c] * gray
    return np.clip(result * 255, 0, 255).astype(np.uint8)


@register_fx(
    fx_id="3d_lut",
    category="color",
    latency_tier="light",
    params={"lut": ["cinematic_warm", "teal_orange", "bleach_bypass",
                    "vintage_film", "cold_blue"]},
    description="3D LUT 电影级调色，从 app/fx/lut3d/ 目录加载 .cube 文件",
)
def apply_3d_lut(frame: np.ndarray, lut: str = "cinematic_warm", **_) -> np.ndarray:
    """软件模拟 3D LUT（36³ lattice 三线性插值）。
    若 .cube 文件不存在则返回原帧。
    """
    import os

    cube_path = os.path.join(
        os.path.dirname(__file__), "lut3d", f"{lut}.cube"
    )
    if not os.path.isfile(cube_path):
        return frame

    lut_table = _load_cube(cube_path)
    if lut_table is None:
        return frame

    # 三线性插值
    f = frame.astype(np.float32) / 255.0
    size = lut_table.shape[0]
    idx = f * (size - 1)
    i0 = np.floor(idx).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, size - 1)
    frac = idx - i0

    # 简化：最近邻 fallback（速度快，质量稍差）
    r_idx = np.clip(np.round(f[..., 2] * (size - 1)).astype(np.int32), 0, size - 1)
    g_idx = np.clip(np.round(f[..., 1] * (size - 1)).astype(np.int32), 0, size - 1)
    b_idx = np.clip(np.round(f[..., 0] * (size - 1)).astype(np.int32), 0, size - 1)

    out = lut_table[b_idx, g_idx, r_idx]  # (H, W, 3) RGB
    return np.clip(out[..., ::-1] * 255, 0, 255).astype(np.uint8)


@register_fx(
    fx_id="bloom",
    category="shader_pixel",
    latency_tier="medium",
    params={"threshold": [0.5, 0.95], "intensity": [0.0, 1.0]},
    description="Bloom / Glow：高亮区域泛光叠加，副歌爆点用",
)
def apply_bloom(frame: np.ndarray, threshold: float = 0.7,
                intensity: float = 0.5, **_) -> np.ndarray:
    bright = frame.astype(np.float32)
    thresh = threshold * 255.0
    mask = np.clip(bright - thresh, 0, None)
    glow = cv2.GaussianBlur(mask.astype(np.uint8), (15, 15), 0)
    return np.clip(frame.astype(np.float32) + glow.astype(np.float32) * intensity,
                   0, 255).astype(np.uint8)


@register_fx(
    fx_id="film_grain",
    category="shader_pixel",
    latency_tier="light",
    params={"amount": [0.0, 0.3]},
    description="胶片噪点：叠加随机 grain，复古 MV 感",
)
def apply_film_grain(frame: np.ndarray, amount: float = 0.08, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    mag = int(amount * 60) + 1
    noise = np.random.randint(-mag, mag + 1, (h, w, 1), dtype=np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@register_fx(
    fx_id="posterize",
    category="shader_pixel",
    latency_tier="light",
    params={"levels": [2, 8]},
    description="色阶量化（Posterize）：减少颜色层次，8-bit 风格",
)
def apply_posterize(frame: np.ndarray, levels: int = 4, **_) -> np.ndarray:
    levels = max(2, min(8, int(levels)))
    step = 256 // levels
    lut = np.array([min(255, (i // step) * step + step // 2)
                    for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, lut)


@register_fx(
    fx_id="pixelate",
    category="shader_pixel",
    latency_tier="light",
    params={"block_size": [4, 48]},
    description="像素化马赛克，block_size=像素块大小",
)
def apply_pixelate(frame: np.ndarray, block_size: int = 16, **_) -> np.ndarray:
    h, w = frame.shape[:2]
    bs = max(4, int(block_size))
    sw, sh = max(1, w // bs), max(1, h // bs)
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


@register_fx(
    fx_id="color_cycling",
    category="color",
    latency_tier="light",
    params={"speed": [0.1, 3.0]},
    description="色相循环动画，speed=旋转速度（圈/秒），高 BPM 段迷幻感",
)
def apply_color_cycling(frame: np.ndarray, speed: float = 1.0,
                        t: float = 0.0, **_) -> np.ndarray:
    deg = (t * speed * 30.0) % 180.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(deg / 2)) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


@register_fx(
    fx_id="motion_blur",
    category="shader_pixel",
    latency_tier="medium",
    params={"amount": [0.0, 1.0], "angle": ["h", "v"]},
    description="方向运动模糊，angle=h(水平)/v(垂直)，加速感 / whip pan 辅助",
)
def apply_motion_blur(frame: np.ndarray, amount: float = 0.3,
                      angle: str = "h", **_) -> np.ndarray:
    ksize = max(3, int(amount * 40) | 1)
    if angle == "v":
        return cv2.blur(frame, (1, ksize))
    return cv2.blur(frame, (ksize, 1))


_LUT_CACHE: dict = {}


def _load_cube(path: str):
    if path in _LUT_CACHE:
        return _LUT_CACHE[path]
    try:
        size = None
        data = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("LUT_3D_SIZE"):
                    size = int(line.split()[-1])
                elif line and not line.startswith("#") and not line.startswith("TITLE"):
                    try:
                        vals = [float(x) for x in line.split()]
                        if len(vals) == 3:
                            data.append(vals)
                    except ValueError:
                        pass
        if size is None or len(data) != size ** 3:
            return None
        arr = np.array(data, dtype=np.float32).reshape(size, size, size, 3)
        _LUT_CACHE[path] = arr
        return arr
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 主题滤镜（category="theme"）—— 持续背景色彩风格，编排中作为全曲基底
# ---------------------------------------------------------------------------

@register_fx(
    fx_id="theme_warm",
    category="theme",
    latency_tier="light",
    params={"strength": [0.0, 1.0]},
    description="暖色调主题：金橙色调，RnB/流行适用",
)
def apply_theme_warm(frame: np.ndarray, strength: float = 0.6, **_) -> np.ndarray:
    strength = max(0.0, min(1.0, strength))
    # 蓝通道压低，红/绿通道微提
    lut_r = np.clip(np.arange(256) * (1.0 + 0.12 * strength), 0, 255).astype(np.uint8)
    lut_g = np.clip(np.arange(256) * (1.0 + 0.04 * strength), 0, 255).astype(np.uint8)
    lut_b = np.clip(np.arange(256) * (1.0 - 0.18 * strength), 0, 255).astype(np.uint8)
    b, g, r = cv2.split(frame)
    return cv2.merge([cv2.LUT(b, lut_b), cv2.LUT(g, lut_g), cv2.LUT(r, lut_r)])


@register_fx(
    fx_id="theme_cool",
    category="theme",
    latency_tier="light",
    params={"strength": [0.0, 1.0]},
    description="冷色调主题：青蓝色调，EDM/电子适用",
)
def apply_theme_cool(frame: np.ndarray, strength: float = 0.6, **_) -> np.ndarray:
    strength = max(0.0, min(1.0, strength))
    lut_r = np.clip(np.arange(256) * (1.0 - 0.15 * strength), 0, 255).astype(np.uint8)
    lut_g = np.clip(np.arange(256) * (1.0 + 0.03 * strength), 0, 255).astype(np.uint8)
    lut_b = np.clip(np.arange(256) * (1.0 + 0.20 * strength), 0, 255).astype(np.uint8)
    b, g, r = cv2.split(frame)
    return cv2.merge([cv2.LUT(b, lut_b), cv2.LUT(g, lut_g), cv2.LUT(r, lut_r)])


@register_fx(
    fx_id="theme_vintage",
    category="theme",
    latency_tier="light",
    params={"strength": [0.0, 1.0]},
    description="复古胶片主题：褪色+暖黄+暗角，民谣/indie 适用",
)
def apply_theme_vintage(frame: np.ndarray, strength: float = 0.7, **_) -> np.ndarray:
    strength = max(0.0, min(1.0, strength))
    # 饱和度降低 + 暖黄偏移
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= (1.0 - 0.35 * strength)   # 降饱和
    hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 - 0.05 * strength) + 8 * strength, 0, 255)
    bgr = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    # 暖黄叠加
    warm = np.full_like(bgr, (30, 60, 90), dtype=np.uint8)  # BGR: 橙黄
    bgr = cv2.addWeighted(bgr, 1.0, warm, 0.08 * strength, 0)
    # 轻暗角
    h, w = bgr.shape[:2]
    Y, X = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((X - w/2) / (w/2))**2 + ((Y - h/2) / (h/2))**2)
    vig = 1.0 - np.clip((dist - 0.6) / 0.5, 0.0, 1.0) * 0.4 * strength
    return np.clip(bgr.astype(np.float32) * vig[..., np.newaxis], 0, 255).astype(np.uint8)


@register_fx(
    fx_id="theme_neon",
    category="theme",
    latency_tier="light",
    params={"strength": [0.0, 1.0]},
    description="霓虹赛博主题：高饱和+深暗+洋红/青，电子/夜店适用",
)
def apply_theme_neon(frame: np.ndarray, strength: float = 0.7, **_) -> np.ndarray:
    strength = max(0.0, min(1.0, strength))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + 0.6 * strength), 0, 255)   # 超饱和
    hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 - 0.25 * strength), 0, 255)  # 压暗背景
    bgr = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    # 青色/洋红叠加
    tint = np.zeros_like(bgr, dtype=np.float32)
    tint[..., 0] = 40 * strength   # B channel boost
    tint[..., 2] = 20 * strength   # R channel boost
    return np.clip(bgr.astype(np.float32) + tint, 0, 255).astype(np.uint8)


@register_fx(
    fx_id="theme_cinematic",
    category="theme",
    latency_tier="light",
    params={"strength": [0.0, 1.0]},
    description="电影感主题：teal-orange + 裁黑边暗角，商业MV适用",
)
def apply_theme_cinematic(frame: np.ndarray, strength: float = 0.6, **_) -> np.ndarray:
    strength = max(0.0, min(1.0, strength))
    # Teal (阴影) + Orange (高光) 经典电影色调
    f32 = frame.astype(np.float32)
    lum = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # 阴影偏青：蓝绿微提，红压低
    shadow_mask = (1.0 - lum)[..., np.newaxis]
    f32[..., 0] += 12 * strength * shadow_mask[..., 0]   # B
    f32[..., 1] += 8  * strength * shadow_mask[..., 0]   # G
    f32[..., 2] -= 15 * strength * shadow_mask[..., 0]   # R
    # 高光偏橙：红微提，蓝压低
    high_mask = lum[..., np.newaxis]
    f32[..., 0] -= 10 * strength * high_mask[..., 0]
    f32[..., 2] += 14 * strength * high_mask[..., 0]
    # 暗角
    h, w = frame.shape[:2]
    Y, X = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((X - w/2) / (w/2))**2 + ((Y - h/2) / (h/2))**2)
    vig = 1.0 - np.clip((dist - 0.55) / 0.5, 0.0, 1.0) * 0.5 * strength
    f32 *= vig[..., np.newaxis]
    return np.clip(f32, 0, 255).astype(np.uint8)

