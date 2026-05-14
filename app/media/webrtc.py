"""
WebRTC 推流：基于 aiortc 的单向视频推流，+ 双向 DataChannel。
参考 §7.3。
"""

from __future__ import annotations

import asyncio
import fractions
import time
from typing import Optional

import av
import cv2
import numpy as np

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from aiortc.contrib.media import MediaBlackhole
    _AIORTC = True
except ImportError:
    _AIORTC = False
    RTCPeerConnection = None

SAMPLE_RATE = 8000
AUDIO_CODEC = "opus"


class CameraVideoTrack:
    """自定义 VideoStreamTrack，从 FX Compositor 输出帧。"""

    kind = "video"

    def __init__(self, compositor, camera):
        self._compositor = compositor
        self._camera = camera
        self._pts = 0
        self._clock_base = fractions.Fraction(1, 90000)

    if _AIORTC:
        from aiortc import VideoStreamTrack as _Base
        # 动态继承（避免 import error）
        _aiortc_base = _Base

    async def recv(self):
        frame_bgr = self._camera.get_frame()
        if frame_bgr is None:
            # 黑帧
            frame_bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # FX 由外部 audio_clock 驱动，此处直接取最近处理帧
        # （Compositor.process 在 pipeline 线程里调用，这里直接读）
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        av_frame.pts = self._pts
        av_frame.time_base = fractions.Fraction(1, 30)
        self._pts += 1
        return av_frame


class WebRTCSession:
    """管理单个 WebRTC PeerConnection。"""

    def __init__(self, compositor, camera):
        self._compositor = compositor
        self._camera = camera
        self._pc: Optional[object] = None

    async def handle_offer(self, sdp: str, sdp_type: str) -> dict:
        """处理 SDP Offer，返回 answer。"""
        if not _AIORTC:
            return {"error": "aiortc not installed"}

        self._pc = RTCPeerConnection()

        # 添加视频 track
        track = _make_video_track(self._compositor, self._camera)
        self._pc.addTrack(track)

        offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
        await self._pc.setRemoteDescription(offer)
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)

        return {
            "sdp": self._pc.localDescription.sdp,
            "type": self._pc.localDescription.type,
        }

    async def close(self) -> None:
        if self._pc:
            await self._pc.close()


def _make_video_track(compositor, camera):
    """创建 VideoStreamTrack（aiortc 要求继承）。"""
    if not _AIORTC:
        return None

    from aiortc import VideoStreamTrack as _VSTBase

    class _Track(_VSTBase):
        def __init__(self):
            super().__init__()
            self._compositor = compositor
            self._camera = camera
            self._pts = 0

        async def recv(self):
            await asyncio.sleep(1.0 / 30)
            frame_bgr = self._camera.get_frame()
            if frame_bgr is None:
                frame_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)

            # 缩放到推流分辨率
            frame_bgr = cv2.resize(frame_bgr, (1280, 720))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            av_frame.pts = self._pts
            av_frame.time_base = fractions.Fraction(1, 30)
            self._pts += 1
            return av_frame

    return _Track()
