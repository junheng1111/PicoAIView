"""
Orchestrator：asyncio EventBus。
统一调度 BeatEngine 节拍事件、视觉事件、FX 命令、时钟同步。§3 架构核心。
"""

from __future__ import annotations

import asyncio
import bisect
import time
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from app.media.compositor import FxCompositor
from app.media.camera import CameraSource
from app.vision.bpu_runner import BpuRunner


def _render_frame(frame: np.ndarray, compositor, t: float, vs) -> np.ndarray:
    """在线程池中执行帧渲染（CPU 密集，不阻塞 asyncio 事件循环）。"""
    # 降分辨率加速处理：640×360 比 1280×720 快 4x，对显示质量影响极小
    h, w = frame.shape[:2]
    if w > 640:
        proc = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
    else:
        proc = frame.copy()

    try:
        processed = compositor.process(proc, t)
    except Exception:
        processed = proc

    # 叠加 YOLO 检测框
    if vs and vs.detections:
        ph, pw = processed.shape[:2]
        for det in vs.detections:
            x, y, bw, bh = det.bbox
            x1, y1 = int(x * pw), int(y * ph)
            x2, y2 = int((x + bw) * pw), int((y + bh) * ph)
            color = (0, 255, 64) if det.label == "person" else (0, 165, 255)
            cv2.rectangle(processed, (x1, y1), (x2, y2), color, 2)
            cv2.putText(processed, f"{det.label} {det.confidence:.2f}",
                        (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if vs.subject_count > 0:
            cv2.putText(processed, f"Persons: {vs.subject_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 64), 2)

    return processed


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

        # 最新合成帧（供 MJPEG 流消费）
        self._processed_frame: Optional[np.ndarray] = None
        self._processed_lock = __import__('threading').Lock()

        # 状态推送任务
        self._status_task: Optional[asyncio.Task] = None
        self._beat_task: Optional[asyncio.Task] = None
        self._vision_task: Optional[asyncio.Task] = None
        self._frame_task: Optional[asyncio.Task] = None

        self._last_fps_ts: float = 0.0
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def get_processed_frame(self) -> Optional[np.ndarray]:
        """返回最近一帧合成后的图像（YOLO框 + FX），供 MJPEG 流使用。"""
        with self._processed_lock:
            return self._processed_frame.copy() if self._processed_frame is not None else None

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
        self._frame_task  = asyncio.ensure_future(self._frame_loop())

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
                active_fx = ""
                if self._session_running and self.compositor:
                    t = self.current_audio_clock()
                    bt = self.compositor._beat_times
                    if bt:
                        import bisect as _bisect
                        idx = _bisect.bisect_right(bt, t) - 1
                        if 0 <= idx < len(self.compositor._beat_events):
                            active_fx = self.compositor._beat_events[idx].get("fx", "")
                await broadcast_status(fps, 0.0, self.current_audio_clock(),
                                       bpu_stats, active_fx)
            except Exception:
                pass

    async def _beat_loop(self) -> None:
        """节拍触发。"""
        from app.api.ws import broadcast_beat

        last_beat_idx = 0
        last_down_idx = 0

        while True:
            await asyncio.sleep(0.005)   # 5ms 轮询
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

    async def _frame_loop(self) -> None:
        """主帧处理循环：读帧 → BPU → FX Compositor（线程池执行，不阻塞事件循环）。"""
        FRAME_INTERVAL = 0.040  # 25fps 上限

        while True:
            frame_start = time.monotonic()

            if not self.camera or not self.compositor:
                await asyncio.sleep(0.01)
                continue

            frame = self.camera.get_frame()
            if frame is None:
                await asyncio.sleep(0.005)
                continue

            # 推帧给 BPU 异步推理（仅入队，不阻塞）
            if self.bpu_runner:
                self.bpu_runner.push_frame(frame)

            t = self.current_audio_clock()
            vs = self.bpu_runner.vision_state() if self.bpu_runner else None
            if vs and not self._session_running:
                self.compositor.current_rms = vs.motion_intensity

            compositor = self.compositor
            try:
                # FX 处理移到线程池，避免阻塞 asyncio 事件循环
                processed = await asyncio.to_thread(
                    _render_frame, frame, compositor, t, vs
                )
                with self._processed_lock:
                    self._processed_frame = processed
            except Exception:
                with self._processed_lock:
                    self._processed_frame = frame

            self._frame_count += 1

            # 控制帧率
            elapsed = time.monotonic() - frame_start
            wait = max(0.001, FRAME_INTERVAL - elapsed)
            await asyncio.sleep(wait)

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
