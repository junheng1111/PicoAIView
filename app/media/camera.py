"""
摄像头采集模块：MIPI CSI（hobot_vio）→ v4l2（OpenCV）→ 合成测试源 自动切换。
参考 preview_no/detector.py 的 _read_stream() 实现。
"""

from __future__ import annotations

import math
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

# 确保 hobot_vio 可导入（同 bpu_runner 路径策略）
_HOBOT_SYS_PATH = "/usr/local/lib/python3.10/dist-packages"
if _HOBOT_SYS_PATH not in sys.path:
    sys.path.append(_HOBOT_SYS_PATH)


class CameraSource:
    """
    帧读取线程。
    优先 hobot_vio (RDK X5 MIPI CSI)，其次 OpenCV v4l2，最后合成测试帧。
    """

    def __init__(self, width: int = 1280, height: int = 720,
                 fps: int = 30, device: int = 0):
        self.width = width
        self.height = height
        self.fps = fps
        self.device = device

        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._backend: str = "unknown"

    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="camera_source")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest.copy() if self._latest is not None else None

    @property
    def backend(self) -> str:
        return self._backend

    # ------------------------------------------------------------------

    def _run(self) -> None:
        """后台线程：持续尝试打开真实摄像头，失败时用合成帧填充，成功后切换到真实帧。"""
        while self._running:
            if self._try_hobot():
                # hobot_vio 循环结束（相机断开），回到外层继续重试
                print("[Camera] hobot_vio 循环退出，5s 后重试...")
                self._backend = "reconnecting"
                time.sleep(5)
                continue
            # hobot_vio 不可用，尝试 OpenCV
            if self._try_opencv():
                print("[Camera] OpenCV 循环退出，5s 后重试...")
                self._backend = "reconnecting"
                time.sleep(5)
                continue
            # 两者都失败，用合成帧填充并等待重试
            if self._backend != "synthetic":
                print("[Camera] 所有摄像头失败，使用合成帧，15s 后重试真实摄像头...")
            self._backend = "synthetic"
            self._run_synthetic_one_pass(duration=15.0)  # 产出合成帧约 15s 后重试真实摄像头

    def _try_hobot(self) -> bool:
        """尝试 hobot_vio，成功则进入循环，返回 True；失败返回 False。"""
        try:
            from hobot_vio import libsrcampy as srcampy  # type: ignore
        except ImportError:
            return False

        cam = srcampy.Camera()  # 只创建一次，避免重试时 handle 泄漏
        ret = -1
        for attempt in range(5):
            if not self._running:
                return False
            ret = cam.open_cam(0, -1, self.fps,
                               [self.width], [self.height],
                               self.height, self.width)
            if ret == 0:
                break
            print(f"[Camera] hobot_vio open_cam 尝试 {attempt+1}/5 失败 ret={ret}，等待重试...")
            time.sleep(5)   # 给 csi2 驱动更多时间释放
        if ret != 0:
            print(f"[Camera] hobot_vio open_cam 全部失败，改用 OpenCV")
            return False

        self._backend = "hobot_vio"
        print(f"[Camera] hobot_vio 已打开 {self.width}×{self.height} @{self.fps}fps")

        while self._running:
            try:
                raw = cam.get_img(2, self.width, self.height)  # type=2 → NV12
                if raw is None or len(raw) == 0:
                    time.sleep(0.01)
                    continue
                nv12 = np.frombuffer(raw, dtype=np.uint8).reshape(
                    self.height * 3 // 2, self.width)
                bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
                # IMX219 在 RDK X5 上竖直方向翻转
                bgr = cv2.flip(bgr, 0)
                with self._lock:
                    self._latest = bgr
            except Exception as e:
                print(f"[Camera] hobot_vio 读帧失败: {e}")
                time.sleep(0.05)

        cam.close_cam()
        return True

    def _try_opencv(self) -> bool:
        """尝试 OpenCV v4l2，成功则进入循环，返回 True；失败返回 False。"""
        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._backend = "opencv"
        print(f"[Camera] OpenCV v4l2 已打开 device={self.device}")

        fail_count = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                if fail_count >= 30:  # ~1.5s 连续失败 → 相机断开
                    print("[Camera] OpenCV 连续读帧失败，断开重连...")
                    break
                time.sleep(0.05)
                continue
            fail_count = 0
            with self._lock:
                self._latest = frame

        cap.release()
        return True

    def _run_synthetic_one_pass(self, duration: float = 15.0) -> None:
        """合成测试帧，持续 duration 秒后返回，外层循环将重新尝试真实摄像头。"""
        w, h = self.width, self.height
        interval = 1.0 / self.fps
        deadline = time.time() + duration
        t = 0
        while self._running and time.time() < deadline:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            r = int((math.sin(t * 0.03) + 1) * 40 + 20)
            frame[:, :, 0] = r
            frame[:, :, 1] = int((math.cos(t * 0.02) + 1) * 30 + 10)
            frame[:, :, 2] = int((math.sin(t * 0.04 + 1) + 1) * 40 + 20)
            ts = time.strftime("%H:%M:%S")
            cv2.putText(frame, "PicoClaw", (w // 2 - 140, h // 2 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (160, 160, 160), 3)
            cv2.putText(frame, "No Camera - Synthetic", (w // 2 - 220, h // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2)
            cv2.putText(frame, ts, (w // 2 - 60, h // 2 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 100), 2)
            with self._lock:
                self._latest = frame
            t += 1
            time.sleep(interval)

    def _synthetic_loop(self) -> None:
        """合成测试帧（无任何物理摄像头时的 fallback，确保 MJPEG 流有内容）。"""
        self._backend = "synthetic"
        w, h = self.width, self.height
        interval = 1.0 / self.fps
        t = 0
        while self._running:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # 动态渐变背景
            r = int((math.sin(t * 0.03) + 1) * 40 + 20)
            frame[:, :, 0] = r
            frame[:, :, 1] = int((math.cos(t * 0.02) + 1) * 30 + 10)
            frame[:, :, 2] = int((math.sin(t * 0.04 + 1) + 1) * 40 + 20)
            # 文字信息
            ts = time.strftime("%H:%M:%S")
            cv2.putText(frame, "PicoClaw", (w // 2 - 140, h // 2 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (160, 160, 160), 3)
            cv2.putText(frame, "No Camera - Synthetic", (w // 2 - 220, h // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2)
            cv2.putText(frame, ts, (w // 2 - 60, h // 2 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 100), 2)
            with self._lock:
                self._latest = frame
            t += 1
            time.sleep(interval)
