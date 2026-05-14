"""视觉事件发布 —— 将 VisionState 变化转换为命名事件，发布到 Orchestrator EventBus。"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from app.vision.bpu_runner import VisionState


class VisionEventEmitter:
    """
    在后台周期性读取 VisionState，与上一帧对比后发布事件：
    - vision.subject_count_change
    - vision.motion_burst      (motion_intensity 突增到阈值以上)
    - vision.pose_available    (有姿态数据)
    """

    MOTION_BURST_TH = 0.4

    def __init__(self, bus_publish: Callable, poll_hz: float = 15.0):
        self._publish = bus_publish
        self._interval = 1.0 / poll_hz
        self._prev_count: int = 0
        self._prev_motion: float = 0.0
        self._task: Optional[asyncio.Task] = None

    def start(self, bpu_runner) -> None:
        self._runner = bpu_runner
        self._task = asyncio.ensure_future(self._loop())

    async def _loop(self) -> None:
        import asyncio as _asyncio
        while True:
            await _asyncio.sleep(self._interval)
            vs = self._runner.vision_state()

            if vs.subject_count != self._prev_count:
                await self._publish("vision.subject_count_change",
                                    {"count": vs.subject_count})
                self._prev_count = vs.subject_count

            if (vs.motion_intensity >= self.MOTION_BURST_TH
                    and self._prev_motion < self.MOTION_BURST_TH):
                await self._publish("vision.motion_burst",
                                    {"intensity": vs.motion_intensity})

            self._prev_motion = vs.motion_intensity

            if vs.pose_detections:
                await self._publish("vision.pose_available",
                                    {"count": len(vs.pose_detections)})
