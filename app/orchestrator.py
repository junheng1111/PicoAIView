"""
Orchestrator：asyncio EventBus。
统一调度 BeatEngine 节拍事件、视觉事件、FX 命令、时钟同步。§3 架构核心。
"""

from __future__ import annotations

import asyncio
import bisect
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from app.media.compositor import FxCompositor
from app.media.camera import CameraSource
from app.vision.bpu_runner import BpuRunner


# COCO 17关键点骨架连接（左侧绿，右侧蓝，躯干白）
_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # 头部
    (5, 7), (7, 9),                              # 左臂
    (6, 8), (8, 10),                             # 右臂
    (5, 6), (5, 11), (6, 12), (11, 12),          # 躯干
    (11, 13), (13, 15),                          # 左腿
    (12, 14), (14, 16),                          # 右腿
]
_SKEL_COLOR = [
    (0, 255, 128), (0, 255, 128), (0, 255, 128), (0, 255, 128),
    (0, 255, 0),   (0, 255, 0),
    (255, 128, 0), (255, 128, 0),
    (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
    (0, 200, 255), (0, 200, 255),
    (128, 0, 255), (128, 0, 255),
]
_KPT_COLOR = [
    (0,255,128),(0,255,128),(0,255,128),(0,255,128),(0,255,128),
    (0,255,0),(255,128,0),
    (0,255,0),(255,128,0),
    (0,255,0),(255,128,0),
    (0,200,255),(128,0,255),
    (0,200,255),(128,0,255),
    (0,200,255),(128,0,255),
]



_MODEL_INPUT_W = 640   # YOLO 模型输入宽度
_MODEL_INPUT_H = 640   # YOLO 模型输入高度


def _draw_pose_overlay(frame: np.ndarray, vs,
                       src_w: int = 1280, src_h: int = 720) -> np.ndarray:
    """在帧上绘制 pose 骨架 + 关键点。

    坐标映射链：
      YOLO 输入 (640×640, 拉伸自 src_w×src_h)
        → kp.x = model_x / 640  ≡ src_x / src_w
        → kp.y = model_y / 640  ≡ src_y / src_h
      显示帧 (disp_w × disp_h, 等比缩放自 src_w×src_h)
        → x_disp = kp.x × disp_w  (拉伸因子抵消，等比缩放因子保留)
        → y_disp = kp.y × disp_h
    """
    if not vs or not vs.pose_detections:
        return frame

    disp_h, disp_w = frame.shape[:2]
    VIS_TH = 0.3

    for pose in vs.pose_detections:
        kpts = pose.keypoints

        # 归一化坐标 → 显示像素（kp.x/kp.y 已等价于 src 归一化坐标）
        kpx = [
            (int(kp.x * disp_w), int(kp.y * disp_h), kp.score)
            for kp in kpts
        ]

        # 骨架连线
        for idx, (a, b) in enumerate(_SKELETON):
            if a >= len(kpx) or b >= len(kpx):
                continue
            ax, ay, av = kpx[a]
            bx, by, bv = kpx[b]
            if av < VIS_TH or bv < VIS_TH:
                continue
            if not (0 <= ax < disp_w and 0 <= ay < disp_h
                    and 0 <= bx < disp_w and 0 <= by < disp_h):
                continue
            cv2.line(frame, (ax, ay), (bx, by), _SKEL_COLOR[idx], 2, cv2.LINE_AA)

        # 关键点圆点
        for j, (kx, ky, vis) in enumerate(kpx):
            if vis < VIS_TH or not (0 <= kx < disp_w and 0 <= ky < disp_h):
                continue
            color = _KPT_COLOR[j] if j < len(_KPT_COLOR) else (255, 255, 255)
            cv2.circle(frame, (kx, ky), 4, color, -1, cv2.LINE_AA)

    return frame


def _render_frame(frame: np.ndarray, compositor, t: float, vs) -> np.ndarray:
    """渲染一帧：缩放 → pose 骨架 → FX Compositor → HUD。"""
    src_h, src_w = frame.shape[:2]
    if src_w > 640:
        proc = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
    else:
        proc = frame.copy()

    if vs:
        # 传入源帧尺寸，供坐标映射注释用（实际计算直接用 disp_w/disp_h）
        proc = _draw_pose_overlay(proc, vs, src_w=src_w, src_h=src_h)

    try:
        processed = compositor.process(proc, t)
    except Exception:
        processed = proc

    if vs and vs.subject_count > 0:
        ph2, pw2 = processed.shape[:2]
        cv2.putText(processed, f"Persons: {vs.subject_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 64), 2)

    return processed


def _render_and_encode(frame: np.ndarray, compositor, t: float, vs):
    """渲染 + JPEG 编码，在同一线程池调用中完成，避免二次 to_thread 开销。
    返回 (rendered_ndarray, jpeg_bytes)。
    """
    rendered = _render_frame(frame, compositor, t, vs)
    _, jpg = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return rendered, jpg.tobytes()


class Orchestrator:
    """
    asyncio 事件总线 + 运行时协调器。

    事件类型（内部）：
    - beat.tick         主拍
    - beat.downbeat     重拍
    - onset             onset
    - energy            RMS 值更新
    - vision.subject    视觉主体变化
    - fx.cmd            特效命令
    - clock.sync        音频时钟同步
    """

    def __init__(self):
        self.camera: Optional[CameraSource] = None
        self.bpu_runner: Optional[BpuRunner] = None
        self.compositor: Optional[FxCompositor] = None

        self._audio_clock: float = 0.0
        self._audio_clock_ts: float = 0.0  # 本地时间戳（用于漂移补偿）
        self._clock_drift: float = 0.0

        self._feat: Dict[str, Any] = {}
        self._track: List[Dict] = []
        self._beats: List[float] = []
        self._downbeats: List[float] = []
        self._onsets: List[float] = []

        self._session_running: bool = False
        self._handlers: Dict[str, List[Callable]] = {}

        # 最新合成帧 + 预编码 JPEG（供 MJPEG 流消费）
        self._processed_frame: Optional[np.ndarray] = None
        self._processed_jpeg:  Optional[bytes] = None
        self._processed_ts:    float = 0.0
        self._processed_lock = threading.Lock()

        # asyncio 任务
        self._status_task: Optional[asyncio.Task] = None
        self._beat_task:   Optional[asyncio.Task] = None
        self._vision_task: Optional[asyncio.Task] = None

        # 专用渲染线程（完全脱离 asyncio 事件循环）
        self._render_thread: Optional[threading.Thread] = None
        self._render_running: bool = False

        self._last_fps_ts: float = 0.0
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def get_processed_frame(self) -> Optional[np.ndarray]:
        with self._processed_lock:
            return self._processed_frame.copy() if self._processed_frame is not None else None

    def get_processed_jpeg(self):
        """返回 (jpeg_bytes, timestamp)，MJPEG 流用 timestamp 判断是否有新帧。"""
        with self._processed_lock:
            return self._processed_jpeg, self._processed_ts

    def setup(self, camera: CameraSource, bpu_runner: BpuRunner,
              compositor: FxCompositor) -> None:
        self.camera = camera
        self.bpu_runner = bpu_runner
        self.compositor = compositor

    def load_session(self, music_id: str, track: List[Dict],
                     feat: Dict[str, Any]) -> None:
        self._track = track
        self._feat = feat
        self._beats = feat.get("beats", [])
        self._downbeats = feat.get("downbeats", [])
        self._onsets = feat.get("onsets", [])
        if self.compositor:
            self.compositor.load_track(track)
        # 重置音频时钟：等待客户端第一次上报前先停在 0
        self._audio_clock = 0.0
        self._audio_clock_ts = time.monotonic()
        self._clock_drift = 0.0
        self._session_running = True
        print(f"[Orchestrator] session loaded music_id={music_id} "
              f"beats={len(self._beats)} track_events={len(track)}")

    def stop_session(self) -> None:
        self._session_running = False

    # ------------------------------------------------------------------
    # 时钟同步（§6.2）
    # ------------------------------------------------------------------

    def update_audio_clock(self, t: float) -> None:
        """客户端通过 WS 上报 audio.currentTime。"""
        now = time.monotonic()
        self._audio_drift_correction(t, now)
        self._audio_clock = t
        self._audio_clock_ts = now

    def _audio_drift_correction(self, reported_t: float, local_ts: float) -> None:
        """简单 PI 补偿：把客户端报告时间与本地估算时间对齐。"""
        estimated = self._audio_clock + (local_ts - self._audio_clock_ts)
        err = reported_t - estimated
        self._clock_drift += 0.1 * err   # I 项（积分，慢漂移）

    def current_audio_clock(self) -> float:
        """读取当前音频时间（含漂移补偿）。"""
        elapsed = time.monotonic() - self._audio_clock_ts
        return self._audio_clock + elapsed + self._clock_drift

    def current_active_fx(self) -> str:
        """当前激活的 FX 名（取最近一个 beat event）。供 status 推送与机械臂导演复用。"""
        if not (self._session_running and self.compositor):
            return ""
        t = self.current_audio_clock()
        bt = self.compositor._beat_times
        if not bt:
            return ""
        idx = bisect.bisect_right(bt, t) - 1
        if 0 <= idx < len(self.compositor._beat_events):
            return self.compositor._beat_events[idx].get("fx", "")
        return ""

    # ------------------------------------------------------------------
    # EventBus
    # ------------------------------------------------------------------

    def on(self, event: str, fn: Callable) -> None:
        self._handlers.setdefault(event, []).append(fn)

    async def emit(self, event: str, data: Any = None) -> None:
        for fn in self._handlers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(fn):
                    await fn(data)
                else:
                    fn(data)
            except Exception as e:
                print(f"[Orchestrator] emit '{event}' handler error: {e}")

    # ------------------------------------------------------------------
    # 后台任务
    # ------------------------------------------------------------------

    def start_background_tasks(self) -> None:
        self._status_task = asyncio.ensure_future(self._status_loop())
        self._beat_task   = asyncio.ensure_future(self._beat_loop())
        self._vision_task = asyncio.ensure_future(self._vision_loop())

        # 独立渲染线程：完全脱离 asyncio 事件循环，避免 to_thread 竞争
        self._render_running = True
        self._render_thread = threading.Thread(
            target=self._render_thread_fn, daemon=True, name="render"
        )
        self._render_thread.start()

    async def _status_loop(self) -> None:
        """每秒推送 status 到 WS。"""
        from app.api.ws import broadcast_status

        while True:
            await asyncio.sleep(1.0)
            try:
                fps = self._fps()
                vs = self.bpu_runner.vision_state() if self.bpu_runner else None
                bpu_stats = {
                    "util": round(vs.motion_intensity if vs else 0.0, 2),
                    "subjects": vs.subject_count if vs else 0,
                }
                # 当前激活的 FX 名（取最近一个 beat event）
                active_fx = self.current_active_fx()
                await broadcast_status(fps, 0.0, self.current_audio_clock(),
                                       bpu_stats, active_fx,
                                       self.compositor._active_theme if self.compositor else "")
            except Exception:
                pass

    async def _beat_loop(self) -> None:
        """节拍触发。"""
        from app.api.ws import broadcast_beat

        last_beat_idx = 0
        last_down_idx = 0

        while True:
            await asyncio.sleep(0.015)   # 15ms 轮询（渲染已移至独立线程）
            if not self._session_running:
                continue

            t = self.current_audio_clock()

            # 主拍
            idx = bisect.bisect_left(self._beats, t)
            if idx > last_beat_idx and idx <= len(self._beats):
                last_beat_idx = idx
                await self.emit("beat.tick", {"t": t})
                await broadcast_beat(t, "beat")
                if self.compositor:
                    self.compositor.current_rms = self._rms_at(t)
                    self.compositor.mark_beat(t)   # 触发脉冲 FX

            # 重拍
            idx2 = bisect.bisect_left(self._downbeats, t)
            if idx2 > last_down_idx and idx2 <= len(self._downbeats):
                last_down_idx = idx2
                await self.emit("beat.downbeat", {"t": t})
                await broadcast_beat(t, "down")

    async def _vision_loop(self) -> None:
        """视觉事件：把 BpuRunner state 推给 Compositor + WS。"""
        from app.api.ws import broadcast_vision

        while True:
            await asyncio.sleep(0.033)
            if not self.bpu_runner:
                continue

            vs = self.bpu_runner.vision_state()
            if self.compositor:
                self.compositor.set_vision_state(vs)

            # 异步广播
            try:
                await broadcast_vision(vs)
            except Exception:
                pass

    def _render_thread_fn(self) -> None:
        """独立渲染线程：读帧 → BPU 推帧 → FX Compositor → JPEG 编码。
        完全使用 time.sleep，不依赖 asyncio，避免阻塞事件循环。
        """
        FRAME_INTERVAL = 0.040  # 25fps 上限

        while self._render_running:
            frame_start = time.monotonic()

            if not self.camera or not self.compositor:
                time.sleep(0.01)
                continue

            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            if self.bpu_runner:
                self.bpu_runner.push_frame(frame)

            t = self.current_audio_clock()
            vs = self.bpu_runner.vision_state() if self.bpu_runner else None
            if vs and not self._session_running:
                self.compositor.current_rms = vs.motion_intensity

            try:
                processed, jpg_bytes = _render_and_encode(frame, self.compositor, t, vs)
                with self._processed_lock:
                    self._processed_frame = processed
                    self._processed_jpeg  = jpg_bytes
                    self._processed_ts    = time.monotonic()
            except Exception:
                with self._processed_lock:
                    self._processed_frame = frame

            self._frame_count += 1

            elapsed = time.monotonic() - frame_start
            wait = max(0.001, FRAME_INTERVAL - elapsed)
            time.sleep(wait)

    def _fps(self) -> float:
        now = time.monotonic()
        elapsed = now - self._last_fps_ts
        if elapsed >= 1.0:
            fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_ts = now
            return fps
        return 0.0

    def _rms_at(self, t: float) -> float:
        rms_data = self._feat.get("rms", [])
        if not rms_data:
            return 0.5
        # 二分查找最近 rms 值
        times = [r[0] for r in rms_data]
        idx = bisect.bisect_left(times, t)
        idx = min(idx, len(rms_data) - 1)
        return float(rms_data[idx][1])
