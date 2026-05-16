"""
FX Compositor：把 ChoreoPlan（段落策略）+ 节拍时间轴 + VisionState 三层融合成最终画面。

调用顺序（每帧）：
  compositor.process(frame, audio_clock) → processed_frame
"""

from __future__ import annotations

import bisect
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.fx._registry import FX_REGISTRY, apply_fx
from app.fx.time_domain import push_frame_to_ring


class FxCompositor:
    """
    执行当前时间点对应的 FX 链路。

    ChoreoTrack 格式（由 expand_plan_to_track 或 rule fallback 生成）：
    每个 entry: {"t": float, "fx": str, "continuous"?: bool, "t_end"?: float, **params}
    """

    def __init__(self):
        self._track: List[Dict[str, Any]] = []
        self._vision_state = None
        self._manual_overrides: Dict[str, Any] = {}   # fx_id → {"amp": ...}

        # 按时间排序的 beat events（非 continuous）
        self._beat_events: List[Dict] = []
        self._beat_times: List[float] = []

        # 持续型 fx（按时间区间）
        self._continuous_fx: List[Dict] = []

        # RMS 实时值（由 Orchestrator 更新）
        self.current_rms: float = 0.0

        # 节拍脉冲标记（beat_tick → 持续 0.1s 的闪光/缩放）
        self._last_beat_t: float = -999.0
        self._beat_pulse_dur: float = 0.10   # 脉冲持续时间 (s)

        # 帧缓冲 ring（时间域特效用）
        self._last_ring_push: float = 0.0

        # subject_zoom 视口（归一化：left, top, width, height），供 bbox 坐标变换使用
        # (0, 0, 1, 1) 表示全帧，无缩放
        self._viewport: tuple = (0.0, 0.0, 1.0, 1.0)

        # 主题滤镜（持续背景层）：fx_id → params
        self._active_theme: str = ""
        self._theme_params: Dict[str, Any] = {}

        # 常驻实时层配置（可通过 set_live_config 覆盖）
        self._live_config: Dict[str, Any] = {
            "subject_zoom":     True,
            "subject_zoom_amp": 1.0,   # 0.0 = 关闭，1.0 = 默认强度
            "zoom_pulse":       True,
            "zoom_pulse_amp":   1.0,
            "brightness":       True,
            "brightness_amp":   1.0,
        }

    def load_track(self, track: List[Dict[str, Any]]) -> None:
        """加载 ChoreoTrack，分类为 beat events 和 continuous fx。"""
        self._beat_events = []
        self._continuous_fx = []

        for entry in track:
            if entry.get("continuous"):
                self._continuous_fx.append(entry)
            else:
                self._beat_events.append(entry)

        self._beat_events.sort(key=lambda x: x["t"])
        self._beat_times = [e["t"] for e in self._beat_events]

    def set_vision_state(self, vs) -> None:
        self._vision_state = vs

    def set_live_config(self, updates: Dict[str, Any]) -> None:
        """更新常驻实时层参数，只更新传入的 key。"""
        self._live_config.update(updates)

    def get_live_config(self) -> Dict[str, Any]:
        return dict(self._live_config)

    def set_theme(self, theme_id: str, params: Optional[Dict[str, Any]] = None) -> None:
        """设置持续背景主题滤镜。theme_id='' 清除主题。"""
        self._active_theme = theme_id or ""
        self._theme_params = params or {}

    def get_viewport(self) -> tuple:
        """返回当前 subject_zoom 视口 (left, top, w, h) 归一化坐标。"""
        return self._viewport

    def override_fx(self, fx_id: str, params: Dict[str, Any]) -> None:
        self._manual_overrides[fx_id] = params

    def clear_override(self, fx_id: str) -> None:
        self._manual_overrides.pop(fx_id, None)

    def clear_all_overrides(self) -> None:
        self._manual_overrides.clear()

    def get_overrides(self) -> Dict[str, Any]:
        return dict(self._manual_overrides)

    def process(self, frame: np.ndarray, audio_clock: float) -> np.ndarray:
        """
        主处理入口。audio_clock = 当前音频时间（秒）。
        """
        if frame is None or frame.size == 0:
            return frame

        # 更新帧缓冲（时间域特效用）
        now = time.time()
        if now - self._last_ring_push >= 0.033:
            push_frame_to_ring(frame)
            self._last_ring_push = now

        result = frame.copy()
        vs = self._vision_state

        # 0. 主题滤镜（最底层，背景色彩风格）
        # 优先级：段落级 theme_set（区间最窄者）> 全曲 theme_set > set_theme() 全局设置
        active_theme, active_theme_params = self._resolve_theme(audio_clock)
        if active_theme:
            result = apply_fx(active_theme, result, vision_state=vs,
                              t=audio_clock, **active_theme_params)

        # 重置每帧视口（无 subject_zoom 时为全帧）
        self._viewport = (0.0, 0.0, 1.0, 1.0)

        # 1. 持续型 FX（时间区间匹配）
        for entry in self._continuous_fx:
            t0 = entry.get("t", 0.0)
            t1 = entry.get("t_end", float("inf"))
            if t0 <= audio_clock < t1:
                fx_id = entry.get("fx", "")
                params = {k: v for k, v in entry.items()
                          if k not in ("t", "t_end", "fx", "continuous")}
                # RMS 驱动：把当前 rms 注入 k 参数
                if entry.get("fx") == "brightness_curve":
                    params["k"] = 0.7 + self.current_rms * 0.8

                params["t"] = audio_clock
                result = apply_fx(fx_id, result, vision_state=vs, **params)

        # 2. 瞬态 FX（最近触发的 beat/onset 事件，窗口 ≤ 0.08s）
        WIN = 0.08
        idx = bisect.bisect_right(self._beat_times, audio_clock)
        for i in range(max(0, idx - 5), min(len(self._beat_events), idx + 2)):
            entry = self._beat_events[i]
            if abs(entry["t"] - audio_clock) <= WIN:
                fx_id = entry.get("fx", "")
                params = {k: v for k, v in entry.items()
                          if k not in ("t", "fx", "transition")}
                params["t"] = audio_clock
                result = apply_fx(fx_id, result, vision_state=vs, **params)

        # 3. 手动覆盖（前端 ws 调参）
        for fx_id, params in list(self._manual_overrides.items()):
            if params.get("amp", 1.0) > 0:
                result = apply_fx(fx_id, result, vision_state=vs, t=audio_clock,
                                  **params)

        # 4. 常驻实时 FX 层（无需加载 choreo，节奏自动驱动）
        result = self._apply_live_fx(result, audio_clock, vs)

        return result

    def _resolve_theme(self, audio_clock: float):
        """按音频时间找出当前应用的主题。
        策略：区间越窄优先级越高（段落 theme_set < 全曲 theme_set < set_theme 全局）。
        返回 (theme_id, params_dict)。
        """
        active_theme  = self._active_theme
        active_params = self._theme_params
        best_dur = float("inf")

        for entry in self._continuous_fx:
            if entry.get("fx") != "theme_set":
                continue
            t0 = entry.get("t", 0.0)
            t1 = entry.get("t_end", float("inf"))
            if t0 <= audio_clock < t1:
                dur = t1 - t0
                if dur < best_dur:
                    best_dur = dur
                    active_theme  = entry.get("theme_id", "")
                    active_params = {k: v for k, v in entry.items()
                                     if k not in ("t", "t_end", "fx", "continuous", "theme_id")}

        return active_theme, active_params

    def mark_beat(self, t: float) -> None:
        """Orchestrator 节拍触发时调用，记录节拍时间用于脉冲 FX。"""
        self._last_beat_t = t

    def _apply_live_fx(self, frame: np.ndarray, audio_clock: float, vs) -> np.ndarray:
        """
        常驻实时特效层（节奏驱动）：
        - subject_zoom：按人脸/身体中心做数字变焦（跟随主体）
        - zoom_pulse：节拍脉冲缩放，RMS 越大脉冲越强
        - brightness：亮度随 RMS 微调（呼吸感）
        - pan：轻微横向漂移（慢节奏感）
        不依赖 ChoreoTrack，始终生效。
        """
        cfg    = self._live_config
        rms    = self.current_rms
        result = frame

        # a) subject_zoom
        if cfg.get("subject_zoom", True) and cfg.get("subject_zoom_amp", 1.0) > 0:
            if vs and vs.subject_count > 0:
                base_amp = 0.05 + rms * 0.20
                zoom_amp = base_amp * cfg["subject_zoom_amp"]
                ratio = 0.7
                pb = vs.primary_bbox()
                cx = (pb[0] + pb[2] / 2.0) if pb else 0.5
                cy = (pb[1] + pb[3] / 2.0) if pb else 0.5
                vl = min(max(0.0, cx - ratio / 2.0), 1.0 - ratio)
                vt = min(max(0.0, cy - ratio / 2.0), 1.0 - ratio)
                self._viewport = (vl, vt, ratio, ratio)
                result = apply_fx("subject_zoom", result, vision_state=vs,
                                  t=audio_clock, amp=zoom_amp)

        # b) zoom_pulse
        if cfg.get("zoom_pulse", True) and cfg.get("zoom_pulse_amp", 1.0) > 0:
            pulse_age = audio_clock - self._last_beat_t
            if 0.0 <= pulse_age < self._beat_pulse_dur:
                pulse_strength = (1.0 - pulse_age / self._beat_pulse_dur)
                pulse_amp = rms * 0.15 * pulse_strength * cfg["zoom_pulse_amp"]
                if pulse_amp > 0.01:
                    result = apply_fx("zoom_pulse", result, vision_state=vs,
                                      t=audio_clock, amp=pulse_amp)

        # c) brightness_curve
        if cfg.get("brightness", True) and cfg.get("brightness_amp", 1.0) > 0:
            dev = rms * 0.30 * cfg["brightness_amp"]   # 偏离 1.0 的幅度
            brightness_k = 1.0 + (dev - 0.15)          # 等效 0.85~1.15 at amp=1
            if abs(brightness_k - 1.0) > 0.02:
                result = apply_fx("brightness_curve", result, vision_state=vs,
                                  t=audio_clock, k=brightness_k)

        return result
