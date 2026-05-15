"""
RDK X5 BPU 推理封装。
支持 YOLOv8 检测 / 姿态估计 / 分割，异步在独立线程跑，FX Compositor 只消费 VisionState。

依赖：hobot_dnn (pyeasy_dnn) —— 通过 system-site-packages 复用 RDK X5 系统包。
"""

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# 确保系统 hobot_dnn 可导入：将系统 site-packages 追加到末尾
# 追加而非插入头部，使 venv 的 numpy 1.x 优先于系统 numpy 2.x
_HOBOT_SYS_PATH = "/usr/local/lib/python3.10/dist-packages"
if _HOBOT_SYS_PATH not in sys.path:
    sys.path.append(_HOBOT_SYS_PATH)

try:
    from hobot_dnn import pyeasy_dnn as dnn  # RDK X5 BPU Python 绑定
    _DNN_AVAILABLE = True
    print("[BPU] hobot_dnn 加载成功")
except (ImportError, AttributeError) as _e:
    _DNN_AVAILABLE = False
    dnn = None  # 开发机 fallback：推理不可用，返回空结果
    print(f"[BPU] hobot_dnn 不可用: {_e}")

COCO_PERSON_CLASS_ID = 0
COCO_NUM_CLASSES = 80


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # 归一化 [x, y, w, h]
    class_id: int


@dataclass(frozen=True)
class Keypoint:
    x: float   # 归一化
    y: float
    score: float


@dataclass(frozen=True)
class PoseDetection:
    detection: Detection
    keypoints: List[Keypoint]  # 17 COCO keypoints


@dataclass
class VisionState:
    """FX Compositor 读取的视觉状态快照（最近一帧推理结果）。"""
    timestamp: float = 0.0
    detections: List[Detection] = field(default_factory=list)
    pose_detections: List[PoseDetection] = field(default_factory=list)
    # seg_mask: Optional[np.ndarray] = None  # (H, W) uint8，按需启用
    primary_id: int = -1          # ByteTracker 主体 track_id
    subject_count: int = 0
    motion_intensity: float = 0.0  # 0–1，由帧差估计

    def primary_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """返回主体归一化 bbox，无主体时返回 None。"""
        if not self.detections:
            return None
        # 返回面积最大的 person bbox 作为主体
        persons = [d for d in self.detections if d.class_id == COCO_PERSON_CLASS_ID]
        if not persons:
            return None
        return max(persons, key=lambda d: d.bbox[2] * d.bbox[3]).bbox


# ---------------------------------------------------------------------------
# 单模型推理器（hobot_dnn 封装）
# ---------------------------------------------------------------------------

class _BpuModel:
    """单个 .bin 模型的推理封装，线程安全。"""

    def __init__(self, bin_path: str):
        self.bin_path = bin_path
        self._lock = threading.Lock()
        self._models = None
        self._input_h = 640
        self._input_w = 640
        self._ready = False
        self._last_error_ts: float = 0.0
        self._init_lock = threading.Lock()

    # 模型加载超时（秒）：dnn.load() 是 C++ 阻塞调用，BPU 驱动上下文残留时会永久挂死
    LOAD_TIMEOUT = 12.0

    def _ensure_ready(self) -> bool:
        if self._ready:
            return True
        if not _DNN_AVAILABLE:
            return False
        now = time.time()
        if self._last_error_ts != 0.0 and now - self._last_error_ts < 30.0:
            return False
        with self._init_lock:
            if self._ready:
                return True

            result: list = [None]
            error:  list = [None]

            def _do_load():
                try:
                    result[0] = dnn.load([self.bin_path])
                except Exception as e:
                    error[0] = e

            t = threading.Thread(target=_do_load, daemon=True, name="bpu_load")
            t.start()
            t.join(timeout=self.LOAD_TIMEOUT)

            if t.is_alive():
                # dnn.load() 卡住 → BPU 驱动上下文被上次进程占用未释放
                print(f"[BpuModel] ⚠ 加载超时 ({self.LOAD_TIMEOUT}s): {self.bin_path}")
                print("[BpuModel] BPU 驱动上下文未释放，服务继续但无 BPU 推理。")
                print("[BpuModel] 解决方法: bash kill.sh 后重试；若仍卡请 reboot")
                self._last_error_ts = time.time()
                return False

            if error[0] is not None:
                self._last_error_ts = time.time()
                print(f"[BpuModel] 初始化失败 {self.bin_path}: {error[0]}")
                return False

            self._models = result[0]
            try:
                props = self._models[0].inputs[0].properties
                if hasattr(props, "valid_shape"):
                    shape = props.valid_shape
                    if len(shape) == 4:
                        self._input_h = int(shape[1]) if shape[1] > 3 else int(shape[2])
                        self._input_w = int(shape[2]) if shape[1] > 3 else int(shape[3])
            except Exception:
                pass
            self._ready = True
            print(f"[BpuModel] 加载成功: {self.bin_path}  输入 {self._input_h}×{self._input_w}")
            return True

    def _preprocess_nv12(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR → NV12，RDK X5 Bayese 系列模型要求 NV12 输入。"""
        resized = cv2.resize(frame_bgr, (self._input_w, self._input_h),
                             interpolation=cv2.INTER_LINEAR)
        yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV_I420)
        h, w = self._input_h, self._input_w
        y_plane = yuv[:h, :]
        uv_i420 = yuv[h:, :].reshape(-1, 2)
        uv_nv12 = uv_i420[:, [0, 1]].reshape(h // 2, w)
        nv12 = np.vstack([y_plane, uv_nv12]).astype(np.uint8)
        return nv12[np.newaxis, ...]

    def forward(self, frame_bgr: np.ndarray):
        """执行推理，返回原始 outputs 列表，推理失败返回 None。

        锁只用于保护 _models 引用的获取，推理本身在锁外执行，
        确保 close() 可以在推理进行中安全运行而不死锁。
        """
        with self._lock:
            if not self._ensure_ready() or self._models is None:
                return None
            models_ref = self._models   # 取引用后立即释放锁

        inp = self._preprocess_nv12(frame_bgr)
        try:
            return models_ref[0].forward(inp)
        except Exception as e:
            print(f"[BpuModel] forward 失败 {self.bin_path}: {e}")
            return None

    def close(self) -> None:
        """显式释放 BPU 模型资源（必须在推理线程停止后调用）。"""
        with self._lock:
            if self._models is not None:
                try:
                    # horizon_dnn 没有显式 close()，靠 del 触发 C++ 析构释放 BPU 上下文
                    del self._models
                except Exception:
                    pass
                self._models = None
                self._ready = False
                print(f"[BpuModel] 已释放: {self.bin_path}")


# ---------------------------------------------------------------------------
# YOLOv8 原始 6-head 后处理（无 on-chip NMS）
# ---------------------------------------------------------------------------

_COCO_NAMES: List[str] = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]


def _dfl_decode(box_dfl: np.ndarray) -> np.ndarray:
    """DFL 分布 → ltrb 距离（格子单位）。box_dfl: (N, 64) → (N, 4)。"""
    x = box_dfl.reshape(-1, 4, 16)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    s = e / e.sum(axis=-1, keepdims=True)
    return (s * np.arange(16, dtype=np.float32)).sum(axis=-1)


def _nms_cpu(boxes: np.ndarray, scores: np.ndarray, iou_th: float) -> List[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = x1[order[1:]].clip(min=x1[i])
        yy1 = y1[order[1:]].clip(min=y1[i])
        xx2 = x2[order[1:]].clip(max=x2[i])
        yy2 = y2[order[1:]].clip(max=y2[i])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_th]
    return keep


def _parse_detections(
    outputs, conf_th: float = 0.25, only_person: bool = False,
    iou_th: float = 0.45, input_hw: Tuple[int, int] = (640, 640),
) -> List[Detection]:
    """解析 YOLOv8 原始 6-head 输出（无 on-chip NMS）。

    outputs[0/2/4]: (1, H, W, 80) 类别 logits  (P3/P4/P5)
    outputs[1/3/5]: (1, H, W, 64) DFL box       (P3/P4/P5)
    """
    if outputs is None or len(outputs) < 6:
        return []

    ih, iw = input_hw
    all_boxes:  List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_cls:    List[np.ndarray] = []

    for ci, bi in ((0, 1), (2, 3), (4, 5)):
        cls_buf = np.array(outputs[ci].buffer).squeeze()  # (H, W, 80)
        box_buf = np.array(outputs[bi].buffer).squeeze()  # (H, W, 64)
        if cls_buf.ndim != 3 or box_buf.ndim != 3:
            continue

        H, W = cls_buf.shape[:2]
        stride = iw // W   # 8 / 16 / 32

        cls_prob  = 1.0 / (1.0 + np.exp(-np.clip(cls_buf.reshape(-1, 80), -20, 20)))
        max_score = cls_prob.max(axis=1)
        mask = max_score > conf_th
        if not mask.any():
            continue

        cls_ids = cls_prob.argmax(axis=1)[mask]
        scores  = max_score[mask]

        if only_person:
            p_mask  = cls_ids == COCO_PERSON_CLASS_ID
            cls_ids = cls_ids[p_mask]
            scores  = scores[p_mask]
            mask_idx = np.where(mask)[0][p_mask]
        else:
            mask_idx = np.where(mask)[0]

        if scores.size == 0:
            continue

        ltrb = _dfl_decode(box_buf.reshape(-1, 64)[mask_idx]) * stride

        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        cx = ((xs.reshape(-1) + 0.5) * stride)[mask_idx]
        cy = ((ys.reshape(-1) + 0.5) * stride)[mask_idx]

        x1 = np.clip((cx - ltrb[:, 0]) / iw, 0.0, 1.0)
        y1 = np.clip((cy - ltrb[:, 1]) / ih, 0.0, 1.0)
        x2 = np.clip((cx + ltrb[:, 2]) / iw, 0.0, 1.0)
        y2 = np.clip((cy + ltrb[:, 3]) / ih, 0.0, 1.0)

        all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        all_scores.append(scores)
        all_cls.append(cls_ids)

    if not all_boxes:
        return []

    boxes  = np.concatenate(all_boxes,  axis=0)
    scores = np.concatenate(all_scores, axis=0)
    cls_id = np.concatenate(all_cls,    axis=0)

    dets: List[Detection] = []
    for c in np.unique(cls_id):
        idx  = np.where(cls_id == c)[0]
        keep = _nms_cpu(boxes[idx], scores[idx], iou_th)
        for k in keep:
            i    = idx[k]
            x1, y1, x2, y2 = boxes[i]
            w_box = float(x2 - x1)
            h_box = float(y2 - y1)
            if w_box <= 0 or h_box <= 0:
                continue
            label = _COCO_NAMES[c] if c < len(_COCO_NAMES) else f"class_{c}"
            dets.append(Detection(
                label=label, confidence=float(scores[i]),
                bbox=(float(x1), float(y1), w_box, h_box),
                class_id=int(c),
            ))

    return sorted(dets, key=lambda d: -d.confidence)


def _to_hwc(arr: np.ndarray) -> np.ndarray:
    """将任意 batch 维布局的 buffer 规整为 (H, W, C)。
    支持 (1,H,W,C)、(H,W,C)、(1,H,W)、(H,W) 等常见 hobot_dnn 输出形状。
    """
    # 去掉 batch=1 维（只去首维）
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]          # → (H, W, C)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]          # → (W, C) — 不常见，但安全
    # 如果 cls head 是 (H, W)，补回 channel 维
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]   # → (H, W, 1)
    return arr


def _parse_pose(
    outputs,
    conf_th: float = 0.20,
    iou_th:  float = 0.45,
    input_hw: Tuple[int, int] = (640, 640),
    _debug: bool = False,
) -> List[PoseDetection]:
    """解析 YOLOv8-pose 原始 9-head 输出（无 on-chip NMS）。

    outputs[0/3/6]: (1, H, W, 1)  person 置信度 logit   (P3/P4/P5)
    outputs[1/4/7]: (1, H, W, 64) DFL box encoding
    outputs[2/5/8]: (1, H, W, 51) 17 关键点 × 3 (x_off, y_off, vis_logit)
    关键点解码：kpt_x = (col + x_off) * stride  (像素)
    """
    if outputs is None or len(outputs) < 9:
        if outputs is not None:
            print(f"[_parse_pose] ⚠ 输出 {len(outputs)} 个 head，需要 9 个，无法解析。"
                  f" shapes: {[np.array(o.buffer).shape for o in outputs]}")
        return []

    if _debug:
        print(f"[_parse_pose] 诊断: {len(outputs)} heads")
        for i, o in enumerate(outputs):
            buf = np.array(o.buffer)
            kpt_note = " ← kpt head (x/y offset range)" if (i % 3 == 2) else ""
            print(f"  [{i}] raw_shape={buf.shape}  max={buf.max():.4f}  min={buf.min():.4f}{kpt_note}")

    ih, iw = input_hw
    all_boxes:  List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_kpts:   List[np.ndarray] = []

    for ci, bi, ki, stride in ((0, 1, 2, 8), (3, 4, 5, 16), (6, 7, 8, 32)):
        cls_buf = _to_hwc(np.array(outputs[ci].buffer))   # (H, W, 1)
        box_buf = _to_hwc(np.array(outputs[bi].buffer))   # (H, W, 64)
        kpt_buf = _to_hwc(np.array(outputs[ki].buffer))   # (H, W, 51)

        if box_buf.ndim != 3 or box_buf.shape[-1] != 64:
            if _debug:
                print(f"  scale stride={stride}: unexpected box shape {box_buf.shape}, skip")
            continue

        H, W = box_buf.shape[:2]
        scores = 1.0 / (1.0 + np.exp(-np.clip(cls_buf.reshape(-1), -20, 20)))

        if _debug:
            kraw_all = kpt_buf.reshape(-1, 17, 3)
            print(f"  stride={stride:2d}: HW=({H},{W})  max_score={scores.max():.4f}  "
                  f"above_{conf_th}={scores[scores > conf_th].size}  "
                  f"kraw_xy=[{kraw_all[..., :2].min():.2f}, {kraw_all[..., :2].max():.2f}]")
        mask   = scores > conf_th
        if not mask.any():
            continue

        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        col_m = xs.reshape(-1)[mask]
        row_m = ys.reshape(-1)[mask]
        cx_m  = (col_m + 0.5) * stride
        cy_m  = (row_m + 0.5) * stride

        ltrb = _dfl_decode(box_buf.reshape(-1, 64)[mask]) * stride
        x1 = np.clip((cx_m - ltrb[:, 0]) / iw, 0.0, 1.0)
        y1 = np.clip((cy_m - ltrb[:, 1]) / ih, 0.0, 1.0)
        x2 = np.clip((cx_m + ltrb[:, 2]) / iw, 0.0, 1.0)
        y2 = np.clip((cy_m + ltrb[:, 3]) / ih, 0.0, 1.0)

        kraw = kpt_buf.reshape(-1, 17, 3)[mask]        # (M, 17, 3)
        # kraw[:, :, 0/1] 是相对于检测格子角点的偏移（格子单位，非像素）
        # 例：kraw_x=5 stride=8 → 向右偏移 5 格 = 40像素
        # 锚点中心用 col+0.5 / row+0.5 表示格子中心（而非角点）
        kx = np.clip((col_m[:, None] + 0.5 + kraw[:, :, 0]) * stride / iw, 0.0, 1.0)
        ky = np.clip((row_m[:, None] + 0.5 + kraw[:, :, 1]) * stride / ih, 0.0, 1.0)
        kv = 1.0 / (1.0 + np.exp(-np.clip(kraw[:, :, 2], -20, 20)))

        all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        all_scores.append(scores[mask])
        all_kpts.append(np.stack([kx, ky, kv], axis=-1))   # (M, 17, 3)

    if not all_boxes:
        return []

    boxes  = np.concatenate(all_boxes,  axis=0)
    scores = np.concatenate(all_scores, axis=0)
    kpts   = np.concatenate(all_kpts,   axis=0)

    keep = _nms_cpu(boxes, scores, iou_th)
    result: List[PoseDetection] = []
    for k in keep:
        x1, y1, x2, y2 = boxes[k]
        w_box = float(x2 - x1)
        h_box = float(y2 - y1)
        if w_box <= 0 or h_box <= 0:
            continue
        det = Detection(
            label="person", confidence=float(scores[k]),
            bbox=(float(x1), float(y1), w_box, h_box),
            class_id=COCO_PERSON_CLASS_ID,
        )
        keypoints = [
            Keypoint(x=float(kpts[k, j, 0]),
                     y=float(kpts[k, j, 1]),
                     score=float(kpts[k, j, 2]))
            for j in range(17)
        ]
        result.append(PoseDetection(detection=det, keypoints=keypoints))

    return sorted(result, key=lambda p: -p.detection.confidence)


# ---------------------------------------------------------------------------
# BPU 资源调度器：管理多个模型，独立线程持续推理
# ---------------------------------------------------------------------------

class BpuRunner:
    """
    常驻：yolov8s_detect（检测）+ yolov8s_pose（姿态）。
    按需加载：yolov8s_seg（分割）、fast_depth（深度）、scrfd_2.5g（人脸）。

    FX Compositor 通过 vision_state() 读取最新 VisionState，不阻塞。
    """

    MODEL_DETECT = "detect"
    MODEL_POSE   = "pose"
    MODEL_SEG    = "seg"
    MODEL_DEPTH  = "depth"
    MODEL_FACE   = "face"

    def __init__(self, model_dir: str = "models"):
        self._model_dir = model_dir
        self._models: Dict[str, _BpuModel] = {}
        self._state = VisionState()
        self._state_lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 帧差估计 motion
        self._prev_gray: Optional[np.ndarray] = None

        # 活跃模型集合（运行时可动态变更）
        self._active_models: set = {self.MODEL_POSE}
        self._model_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def start(self, detect_bin: Optional[str] = None, pose_bin: str = "",
              seg_bin: Optional[str] = None,
              depth_bin: Optional[str] = None,
              face_bin: Optional[str] = None) -> None:
        """初始化并启动后台推理线程。"""
        def _add(key, path):
            if path:
                self._models[key] = _BpuModel(path)

        _add(self.MODEL_DETECT, detect_bin)
        _add(self.MODEL_POSE,   pose_bin)
        _add(self.MODEL_SEG,    seg_bin)
        _add(self.MODEL_DEPTH,  depth_bin)
        _add(self.MODEL_FACE,   face_bin)

        self._running = True
        self._thread = threading.Thread(target=self._infer_loop, daemon=True,
                                        name="bpu_runner")
        self._thread.start()
        print("[BpuRunner] 后台推理线程已启动")

    def stop(self) -> None:
        """停止推理线程并显式释放所有 BPU 模型资源。"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)   # 等待推理线程退出
        # 推理线程已停止，安全地释放所有模型
        for key, model in list(self._models.items()):
            model.close()
        self._models.clear()
        print("[BpuRunner] 所有 BPU 资源已释放")

    def vision_state(self) -> VisionState:
        """线程安全地读取最新 VisionState（FX Compositor 调用）。"""
        with self._state_lock:
            return self._state

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        """主线程把摄像头帧推入，推理线程消费。"""
        self._latest_frame = frame_bgr

    def loaded_models(self) -> list:
        """返回已注册的模型 key 列表（不论 DNN 是否可用）。"""
        return list(self._models.keys())

    def enable_model(self, model_key: str) -> None:
        with self._model_lock:
            self._active_models.add(model_key)

    def disable_model(self, model_key: str) -> None:
        with self._model_lock:
            self._active_models.discard(model_key)

    # ------------------------------------------------------------------
    # 内部推理循环
    # ------------------------------------------------------------------

    _latest_frame: Optional[np.ndarray] = None

    _debug_log_interval: float = 5.0   # 每隔 N 秒打印一次检测统计

    def _infer_loop(self) -> None:
        last_ts = 0.0
        last_log_ts = 0.0
        frame_count = 0
        det_count = 0
        first_inference = True   # 首次成功推理时打印诊断

        while self._running:
            frame = self._latest_frame
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            if now - last_ts < 0.033:   # 最多 30fps
                time.sleep(0.005)
                continue
            last_ts = now

            with self._model_lock:
                active = set(self._active_models)

            dets: List[Detection] = []
            poses: List[PoseDetection] = []

            # 姿态（优先；pose 结果里已含 person bbox，无需再跑 detect）
            if self.MODEL_POSE in active and self.MODEL_POSE in self._models:
                outputs = self._models[self.MODEL_POSE].forward(frame)
                if outputs is None:
                    if now - last_log_ts > self._debug_log_interval:
                        print("[BpuRunner] pose forward() 返回 None，推理失败")
                        last_log_ts = now
                else:
                    if first_inference:
                        first_inference = False
                        poses = _parse_pose(outputs, _debug=True)
                    else:
                        poses = _parse_pose(outputs)
                    dets  = [p.detection for p in poses]

            frame_count += 1
            det_count += len(poses)
            if now - last_log_ts > self._debug_log_interval:
                fps_est = frame_count / max(1, now - (last_log_ts or now - 1))
                print(f"[BpuRunner] {fps_est:.1f} fps | "
                      f"检测到 pose: {len(poses)} 人 | "
                      f"累计 {det_count}/{frame_count} 帧有人")
                last_log_ts = now
                frame_count = 0
                det_count = 0

            # motion intensity（帧差均值）
            motion = 0.0
            gray = cv2.cvtColor(
                cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY
            ).astype(np.float32)
            if self._prev_gray is not None:
                diff = np.abs(gray - self._prev_gray).mean()
                motion = float(min(1.0, diff / 30.0))
            self._prev_gray = gray

            persons = [d for d in dets if d.class_id == COCO_PERSON_CLASS_ID]
            new_state = VisionState(
                timestamp=now,
                detections=dets,
                pose_detections=poses,
                primary_id=0,
                subject_count=len(persons),
                motion_intensity=motion,
            )

            with self._state_lock:
                self._state = new_state

    # ------------------------------------------------------------------
    # 序列化（给 WebSocket 下行推送）
    # ------------------------------------------------------------------

    @staticmethod
    def serialize_state(vs: VisionState) -> Dict[str, Any]:
        return {
            "t": vs.timestamp,
            "subjects": vs.subject_count,
            "primary_id": vs.primary_id,
            "motion": round(vs.motion_intensity, 3),
            "detections": [
                {
                    "label": d.label,
                    "confidence": round(d.confidence, 3),
                    "bbox": [round(v, 4) for v in d.bbox],
                    "class_id": d.class_id,
                }
                for d in vs.detections
            ],
            "poses": [
                {
                    "confidence": round(p.detection.confidence, 3),
                    "bbox": [round(v, 4) for v in p.detection.bbox],
                    "keypoints": [
                        [round(kp.x, 4), round(kp.y, 4), round(kp.score, 3)]
                        for kp in p.keypoints
                    ],
                }
                for p in vs.pose_detections
            ],
        }
