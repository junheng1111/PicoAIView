"""
粉晕综合滤镜 (pink_halo)。

六层叠加：
  1. 整体粉色调校   — 提红压蓝绿，LUT 查表，<1ms
  2. 三层脉冲晕圈   — 浅粉/中粉/深玫红，各自独立频率呼吸
  3. 旋转弧线光迹   — 两条反向弧，柔化模糊成光带
  4. 20粒子轨道     — 绕中心旋转，明暗闪烁
  5. 中心核心发光   — 随脉冲膨胀收缩的高斯光斑
  6. 暗角压边       — 四周压暗，人物自然突出

性能设计：
  ① LUT 色调校色（3×256 entry），uint8→uint8，最后一次 astype
  ② 晕圈 / 核心 / 暗角在半分辨率 (H/2, W/2) 操作
  ③ 粒子直接绘制在 float32 overlay，无额外 astype
  ④ 弧线用独立 uint8 buffer blur 后一次转换
  ⑤ 核心用双基底线性插值，无帧内 exp()
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.fx._registry import register_fx

# ── 粒子系统（固定种子）───────────────────────────────────────────────────
_RNG          = np.random.default_rng(42)
_N            = 20
_P_BASE_ANGLE = _RNG.uniform(0, 2 * math.pi, _N).astype(np.float32)
_P_RADIUS_N   = _RNG.uniform(0.16, 0.40, _N).astype(np.float32)    # × R
_P_OMEGA      = _RNG.uniform(-0.55, 0.55, _N).astype(np.float32)   # rad/s
_P_FLICK_F    = _RNG.uniform(1.8, 5.5, _N).astype(np.float32)
_P_FLICK_PH   = _RNG.uniform(0, 2 * math.pi, _N).astype(np.float32)
_P_SIZE       = _RNG.integers(2, 5, _N)

# 粒子颜色 (BGR): 浅粉 / 玫瑰粉 / 深玫红
_P_COLORS = np.array(
    [[195.0, 140.0, 255.0],
     [155.0,  80.0, 255.0],
     [ 60.0,  20.0, 220.0]], dtype=np.float32
)
_P_CI = _RNG.integers(0, 3, _N)

# ── 帧尺寸缓存 ─────────────────────────────────────────────────────────────
_CACHE: dict = {}


def _make_lut(scale: float, bias: float) -> np.ndarray:
    return np.clip(np.arange(256, dtype=np.float32) * scale + bias,
                   0, 255).astype(np.uint8)


def _build_cache(h: int, w: int) -> dict:
    lh, lw = h // 2, w // 2
    cx, cy = lw / 2.0, lh / 2.0
    R = min(lw, lh) / 2.0

    xs = (np.arange(lw, dtype=np.float32) - cx) / R
    ys = (np.arange(lh, dtype=np.float32) - cy) / R
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    dist  = np.sqrt(XX ** 2 + YY ** 2)
    dist2 = XX ** 2 + YY ** 2

    def soft_ring(r: float, sigma: float) -> np.ndarray:
        d = np.abs(dist - r)
        return np.exp(-(d * d) / (2.0 * sigma * sigma)).astype(np.float32)

    r0 = soft_ring(0.28, 0.030)    # 内环 浅粉
    r1 = soft_ring(0.52, 0.040)    # 中环 中粉
    r2 = soft_ring(0.78, 0.055)    # 外环 深玫红

    # 各通道预乘颜色值（消除帧内 3D broadcasting）
    rings = {
        "r0b": r0 * 200.0, "r0g": r0 * 150.0, "r0r": r0 * 255.0,
        "r1b": r1 * 155.0, "r1g": r1 *  80.0, "r1r": r1 * 255.0,
        "r2b": r2 *  70.0, "r2g": r2 *  22.0, "r2r": r2 * 210.0,
    }

    # 核心发光：两极端 sigma 基底
    core_lo = np.exp(-dist2 / (2.0 * 0.070 ** 2)).astype(np.float32)
    core_hi = np.exp(-dist2 / (2.0 * 0.110 ** 2)).astype(np.float32)

    # 暗角（预缓存 3ch，直接乘）
    vig1 = np.clip(1.0 - dist * 0.52, 0.18, 1.0).astype(np.float32)
    vig3 = np.stack([vig1, vig1, vig1], axis=2)

    # 色调 LUT（全分辨率用）
    lut_b = _make_lut(0.86, -8.0)
    lut_g = _make_lut(0.90, -4.0)
    lut_r = _make_lut(1.13, 20.0)

    return {
        "lh": lh, "lw": lw, "R": R, "cx": cx, "cy": cy,
        "dist2": dist2,
        "core_lo": core_lo, "core_hi": core_hi, "core_d": core_hi - core_lo,
        "vig3": vig3,
        "lut_b": lut_b, "lut_g": lut_g, "lut_r": lut_r,
        **rings,
    }


def _get_cache(h: int, w: int) -> dict:
    key = (h, w)
    if key not in _CACHE:
        _CACHE[key] = _build_cache(h, w)
    return _CACHE[key]


@register_fx(
    fx_id="pink_halo",
    category="overlay",
    latency_tier="medium",
    params={
        "intensity":   [0.0, 1.0],
        "pulse_speed": [0.3, 3.0],
    },
    description=(
        "粉晕综合滤镜：粉色调 + 三层脉冲晕圈(浅粉/中粉/深玫红) "
        "+ 旋转弧线光迹 + 20粒子轨道 + 中心核心发光 + 暗角压边"
    ),
)
def apply_pink_halo(
    frame: np.ndarray,
    intensity:   float = 0.85,
    pulse_speed: float = 1.0,
    t:           float = 0.0,
    **_,
) -> np.ndarray:
    h, w = frame.shape[:2]
    c    = _get_cache(h, w)
    lh, lw = c["lh"], c["lw"]
    R, cx, cy = c["R"], c["cx"], c["cy"]
    intensity = float(np.clip(intensity, 0.0, 1.0))
    tf = float(t)

    # ── 1. 色调校色（LUT，uint8→uint8，按通道）──────────────────────────
    # 最后统一 astype，减少中间 float 分配
    col_b = cv2.LUT(frame[:, :, 0], c["lut_b"])   # (H,W) uint8
    col_g = cv2.LUT(frame[:, :, 1], c["lut_g"])
    col_r = cv2.LUT(frame[:, :, 2], c["lut_r"])

    # ── 2-5 在半分辨率上构建 overlay ────────────────────────────────────
    ov = np.zeros((lh, lw, 3), dtype=np.float32)

    # ── 2. 三层脉冲晕圈（独立频率，逐通道 in-place）─────────────────────
    s0 = pulse_speed * 1.00;  b0 = (0.45 + 0.55 * math.sin(tf * s0))        * 0.55
    s1 = pulse_speed * 1.41;  b1 = (0.45 + 0.55 * math.sin(tf * s1 + 1.05)) * 0.45
    s2 = pulse_speed * 0.71;  b2 = (0.45 + 0.55 * math.sin(tf * s2 + 2.09)) * 0.35

    ov[:, :, 0] = c["r0b"] * b0 + c["r1b"] * b1 + c["r2b"] * b2
    ov[:, :, 1] = c["r0g"] * b0 + c["r1g"] * b1 + c["r2g"] * b2
    ov[:, :, 2] = c["r0r"] * b0 + c["r1r"] * b1 + c["r2r"] * b2

    # ── 3. 旋转弧线（独立 uint8 buffer，blur 后加入 ov）──────────────────
    arc = np.zeros((lh, lw, 3), dtype=np.uint8)
    icx, icy = int(cx), int(cy)
    cv2.ellipse(arc, (icx, icy), (int(R * 0.62),) * 2,
                int(tf * 42.0) % 360, 0, 75, (185, 125, 255), 3, cv2.LINE_AA)
    cv2.ellipse(arc, (icx, icy), (int(R * 0.55), int(R * 0.45)),
                int(-tf * 37.0 + 180.0) % 360, 0, 55, (55, 18, 200), 2, cv2.LINE_AA)
    cv2.GaussianBlur(arc, (9, 9), 0, dst=arc)   # in-place blur
    # 直接 add（允许 unsafe 类型提升，避免显式 astype）
    ov += arc * (0.80 / 255.0)   # arc uint8 → scale to [0, ~1] float, add to ov

    # ── 4. 粒子轨道（直接在 float32 ov 上绘制，无额外 astype）────────────
    angles = _P_BASE_ANGLE + _P_OMEGA * tf
    radii  = _P_RADIUS_N * R
    px_all = (cx + radii * np.cos(angles)).astype(np.int32)
    py_all = (cy + radii * np.sin(angles)).astype(np.int32)
    flick  = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(_P_FLICK_F * tf + _P_FLICK_PH))

    for i in range(_N):
        px, py = int(px_all[i]), int(py_all[i])
        if not (0 <= px < lw and 0 <= py < lh):
            continue
        bc = _P_COLORS[_P_CI[i]]
        f  = float(flick[i])
        cv2.circle(ov, (px, py), int(_P_SIZE[i]),
                   (float(bc[0] * f), float(bc[1] * f), float(bc[2] * f)),
                   -1, cv2.LINE_AA)

    # ── 5. 中心核心发光（插值，无帧内 exp）────────────────────────────────
    pc   = 0.5 + 0.5 * math.sin(tf * pulse_speed * 1.80)
    core = c["core_lo"] + c["core_d"] * pc
    br   = 0.55 + 0.45 * pc
    ov[:, :, 0] += core * (230.0 * br)
    ov[:, :, 1] += core * (200.0 * br)
    ov[:, :, 2] += core * (255.0 * br)

    # ── 6. 暗角（半分辨率 in-place 乘）────────────────────────────────────
    ov *= c["vig3"]

    # ── 7. 上采样 overlay → 全分辨率，叠加 ──────────────────────────────
    ov_full = cv2.resize(ov, (w, h), interpolation=cv2.INTER_LINEAR)

    # 合并调色帧 + overlay：一次性合成，避免多个临时数组
    # col_b/g/r 是 uint8，ov_full 是 float32，output 也是 float32
    out = np.empty((h, w, 3), dtype=np.float32)
    out[:, :, 0] = col_b
    out[:, :, 1] = col_g
    out[:, :, 2] = col_r
    out += ov_full * intensity
    np.clip(out, 0.0, 255.0, out=out)

    return out.astype(np.uint8)
