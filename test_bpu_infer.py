"""
独立 BPU 推理测试：验证 hobot_dnn + YOLOv8 原始输出后处理是否正确。
用法：venv/bin/python test_bpu_infer.py [图片路径]
"""
import sys
import time
import numpy as np

sys.path.append('/usr/local/lib/python3.10/dist-packages')
import cv2

print(f"[env] python: {sys.executable}")
print(f"[env] numpy:  {np.__version__}")

try:
    from hobot_dnn import pyeasy_dnn as dnn
    print("[BPU] hobot_dnn 加载成功")
except Exception as e:
    print(f"[BPU] hobot_dnn 加载失败: {e}")
    sys.exit(1)

MODEL_PATH = "models/yolov8s_detect_bayese_640x640_nv12.bin"
IMG_PATH   = sys.argv[1] if len(sys.argv) > 1 else None
CONF_TH    = 0.25
IOU_TH     = 0.45

COCO_NAMES = [
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

# ── YOLOv8 后处理 ────────────────────────────────────────────────────────────

def _dfl_decode(box_dfl: np.ndarray) -> np.ndarray:
    """DFL (Distribution Focal Loss) → ltrb 距离。
    box_dfl: (N, 64)  → N 个锚点，每个坐标用 16-bin softmax 分布编码
    返回: (N, 4) ltrb（单位：像素格，乘以 stride 后才是真实像素）
    """
    N = box_dfl.shape[0]
    x = box_dfl.reshape(N, 4, 16)
    # softmax
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    s = e / e.sum(axis=-1, keepdims=True)
    # weighted sum → 期望距离 (0..15 bins)
    bins = np.arange(16, dtype=np.float32)
    return (s * bins).sum(axis=-1)   # (N, 4)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_th: float) -> list:
    """简单 CPU NMS，boxes: (N,4) xyxy 归一化。"""
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
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


def parse_yolov8_outputs(outputs, conf_th=0.25, iou_th=0.45, input_hw=(640,640)):
    """
    解析 6-head YOLOv8 原始输出（无 on-chip NMS）：
      outputs[0]: (1, H0, W0, 80)  类别 logits  P3
      outputs[1]: (1, H0, W0, 64)  DFL box      P3
      outputs[2]: (1, H1, W1, 80)  类别 logits  P4
      outputs[3]: (1, H1, W1, 64)  DFL box      P4
      outputs[4]: (1, H2, W2, 80)  类别 logits  P5
      outputs[5]: (1, H2, W2, 64)  DFL box      P5
    返回: list of (label, conf, x1_norm, y1_norm, x2_norm, y2_norm, cls_id)
    """
    ih, iw = input_hw
    all_boxes, all_scores, all_cls = [], [], []

    scale_pairs = [(outputs[0], outputs[1]),
                   (outputs[2], outputs[3]),
                   (outputs[4], outputs[5])]

    for cls_out, box_out in scale_pairs:
        cls_buf = np.array(cls_out.buffer).squeeze()   # (H, W, 80)
        box_buf = np.array(box_out.buffer).squeeze()   # (H, W, 64)

        H, W = cls_buf.shape[:2]
        stride = iw // W   # 8, 16, 32

        # 类别分数
        cls_prob = _sigmoid(cls_buf.reshape(-1, cls_buf.shape[-1]))  # (N, 80)
        max_score = cls_prob.max(axis=1)       # (N,)
        mask = max_score > conf_th
        if not mask.any():
            continue

        cls_ids = cls_prob.argmax(axis=1)[mask]
        scores  = max_score[mask]
        box_sel = box_buf.reshape(-1, 64)[mask]   # (M, 64)

        # DFL → ltrb (格子单位) → xyxy 归一化
        ltrb = _dfl_decode(box_sel) * stride          # (M, 4) 像素单位

        # 生成锚点中心坐标（像素）
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        cx = (xs.reshape(-1) + 0.5) * stride   # (N,)
        cy = (ys.reshape(-1) + 0.5) * stride
        cx, cy = cx[mask], cy[mask]

        x1 = (cx - ltrb[:, 0]) / iw
        y1 = (cy - ltrb[:, 1]) / ih
        x2 = (cx + ltrb[:, 2]) / iw
        y2 = (cy + ltrb[:, 3]) / ih

        all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        all_scores.append(scores)
        all_cls.append(cls_ids)

    if not all_boxes:
        return []

    boxes  = np.concatenate(all_boxes,  axis=0)
    scores = np.concatenate(all_scores, axis=0)
    cls_id = np.concatenate(all_cls,    axis=0)

    # 每类单独 NMS，再合并
    results = []
    for c in np.unique(cls_id):
        idx = np.where(cls_id == c)[0]
        keep = _nms(boxes[idx], scores[idx], iou_th)
        for k in keep:
            i = idx[k]
            label = COCO_NAMES[c] if c < len(COCO_NAMES) else f"cls{c}"
            x1, y1, x2, y2 = boxes[i].clip(0, 1)
            results.append((label, float(scores[i]), x1, y1, x2, y2, int(c)))

    results.sort(key=lambda r: -r[1])
    return results


# ── 1. 加载模型 ───────────────────────────────────────────────────────────────
print(f"\n[1] 加载模型: {MODEL_PATH}")
t0 = time.time()
models = dnn.load([MODEL_PATH])
print(f"    耗时: {(time.time()-t0)*1000:.1f} ms，模型数: {len(models)}")
model = models[0]

print(f"    输出数量: {len(model.outputs)}")
# 打印每个输出 shape（通过推理一次来确认）

# ── 2. 准备输入 ───────────────────────────────────────────────────────────────
H_IN, W_IN = 640, 640

if IMG_PATH:
    print(f"\n[2] 读取图片: {IMG_PATH}")
    bgr = cv2.imread(IMG_PATH)
    if bgr is None:
        print("    读取失败，改用纯灰帧")
        bgr = np.full((480, 640, 3), 128, dtype=np.uint8)
    else:
        print(f"    原始尺寸: {bgr.shape}")
else:
    print("\n[2] 未指定图片，使用 128-灰帧（480×640）")
    bgr = np.full((480, 640, 3), 128, dtype=np.uint8)

def bgr_to_nv12(frame, h=640, w=640):
    r = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    yuv = cv2.cvtColor(r, cv2.COLOR_BGR2YUV_I420)
    y   = yuv[:h, :]
    uv  = yuv[h:, :].reshape(-1, 2).reshape(h//2, w)
    return np.vstack([y, uv]).astype(np.uint8)[np.newaxis]

nv12 = bgr_to_nv12(bgr, H_IN, W_IN)
print(f"    NV12 shape: {nv12.shape}")

# ── 3. 推理 ───────────────────────────────────────────────────────────────────
print("\n[3] 推理 ...")
t1 = time.time()
outputs = model.forward(nv12)
print(f"    耗时: {(time.time()-t1)*1000:.1f} ms，输出头数: {len(outputs)}")
for i, o in enumerate(outputs):
    buf = np.array(o.buffer)
    print(f"    output[{i}] shape={buf.shape}  dtype={buf.dtype}  "
          f"min={buf.min():.3f}  max={buf.max():.3f}")

# ── 4. 后处理 ─────────────────────────────────────────────────────────────────
print(f"\n[4] YOLOv8 后处理 (conf≥{CONF_TH}, iou≤{IOU_TH}) ...")
t2 = time.time()
dets = parse_yolov8_outputs(outputs, CONF_TH, IOU_TH, (H_IN, W_IN))
print(f"    耗时: {(time.time()-t2)*1000:.1f} ms，检测到 {len(dets)} 个目标:")
for d in dets:
    label, conf, x1, y1, x2, y2, _ = d
    print(f"    {label:15s} conf={conf:.3f}  [{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}]")

if not dets:
    print("    （无目标——若为灰帧属正常，请传含人物图片验证）")

# ── 5. 可视化 ─────────────────────────────────────────────────────────────────
if dets:
    orig = bgr.copy()
    oh, ow = orig.shape[:2]
    for label, conf, x1, y1, x2, y2, _ in dets:
        px1, py1 = int(x1*ow), int(y1*oh)
        px2, py2 = int(x2*ow), int(y2*oh)
        color = (0, 255, 64) if label == "person" else (0, 165, 255)
        cv2.rectangle(orig, (px1, py1), (px2, py2), color, 2)
        cv2.putText(orig, f"{label} {conf:.2f}", (px1, max(py1-6,14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    out_path = "/tmp/bpu_test_result.jpg"
    cv2.imwrite(out_path, orig)
    print(f"\n[5] 结果图已保存: {out_path}")

print("\n[done] BPU 推理测试完成")
