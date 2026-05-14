import cv2
import threading
import time
import atexit
import numpy as np

from hobot_vio import libsrcampy as srcampy
from rdk_infer import RdkYoloV8
from config import MODEL_BIN, WIDTH, HEIGHT


_RDK_ENGINE = None
_RDK_ENGINE_LOCK = threading.Lock()


def _get_rdk_engine():
    global _RDK_ENGINE
    with _RDK_ENGINE_LOCK:
        if _RDK_ENGINE is None:
            _RDK_ENGINE = RdkYoloV8(MODEL_BIN)
        return _RDK_ENGINE


class AIDetector:
    def __init__(self):
        self.latest_frame = None
        self.latest_results = []
        self.running = True
        self._cleaned = False
        self._cam = None

        self.infer_engine = _get_rdk_engine()

        threading.Thread(target=self._read_stream, daemon=True).start()
        threading.Thread(target=self._infer_loop, daemon=True).start()

        atexit.register(self.cleanup)

    def _read_stream(self):
        cam = srcampy.Camera()
        # 参数：pipeline=0, 分辨率宽, 高, 格式(0=NV12)
        # open_cam(pipe_id, video_index, fps, width, height)
        # IMX219 接在 mipi_host:0，对应 video_index=0，支持分辨率 1920x1080
        cam_w, cam_h = 1920, 1080
        # 正确 API: open_cam(pipe_id, video_index, fps, [width_list], [height_list], sensor_h, sensor_w)
        ret = cam.open_cam(0, -1, 30, [cam_w], [cam_h], cam_h, cam_w)
        if ret != 0:
            print(f"[Camera] open_cam 失败，返回码: {ret}")
            return
        self._cam = cam
        print(f"[Camera] 已打开摄像头 {cam_w}x{cam_h}")

        while self.running:
            try:
                # 正确 API: get_img(type, width, height)  type=2 → NV12
                raw = cam.get_img(2, cam_w, cam_h)
                if raw is None or len(raw) == 0:
                    time.sleep(0.01)
                    continue
                # NV12 → BGR
                nv12 = np.frombuffer(raw, dtype=np.uint8).reshape(cam_h * 3 // 2, cam_w)
                bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
                self.latest_frame = bgr
            except Exception as e:
                print(f"[Camera] 读帧失败: {e}")
                time.sleep(0.05)

        cam.close_cam()

    def _infer_loop(self):
        last_ts = 0.0
        while self.running:
            frame = self.latest_frame
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            if now - last_ts < 0.05:  # ~20fps
                time.sleep(0.005)
                continue
            last_ts = now

            try:
                dets = self.infer_engine.infer(frame, only_person=False, conf_th=0.25)
                self.latest_results = self.infer_engine.to_view_objects(dets)
            except Exception as e:
                if int(time.time()) % 5 == 0:
                    print(f"[RdkInfer] error: {e}")
                time.sleep(0.05)

    def get_data(self):
        return self.latest_frame, self.latest_results

    def cleanup(self):
        if getattr(self, "_cleaned", False):
            return
        self._cleaned = True
        print("正在清理资源...")
        self.running = False
        if self._cam:
            try:
                self._cam.close_cam()
            except Exception:
                pass
        print("DEBUG: 资源清理完成.")
