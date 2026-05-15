"""
快速诊断 YOLOv8-pose Bayese 模型输出格式。
用法：
    cd /root/pico_view
    source venv/bin/activate
    python test_model.py
"""
import sys, time
import numpy as np

sys.path.append('/usr/local/lib/python3.10/dist-packages')
from hobot_dnn import pyeasy_dnn as dnn

MODEL = "models/yolov8s_pose_bayese_640x640_nv12.bin"

print(f"加载模型: {MODEL}")
t0 = time.time()
models = dnn.load([MODEL])
m = models[0]
print(f"加载耗时 {time.time()-t0:.2f}s")

# ── 输入信息 ──────────────────────────────────────────────────────────
print(f"\n=== 输入 ({len(m.inputs)} 个) ===")
for i, inp in enumerate(m.inputs):
    p = inp.properties
    print(f"  [{i}] type={p.tensor_type}  dtype={p.dtype}  shape={p.shape}")

# ── 输出信息 ──────────────────────────────────────────────────────────
print(f"\n=== 输出 ({len(m.outputs)} 个) ===")
for i, out in enumerate(m.outputs):
    p = out.properties
    print(f"  [{i}] type={p.tensor_type}  dtype={p.dtype}  shape={p.shape}")

# ── 构造 dummy NV12 输入并推理 ─────────────────────────────────────────
print("\n=== 推理（dummy 黑帧 640×640 NV12）===")
H, W = 640, 640
y_plane  = np.zeros((H, W),     dtype=np.uint8)
uv_plane = np.full((H//2, W), 128, dtype=np.uint8)   # 128 = 中性 UV
nv12     = np.vstack([y_plane, uv_plane])[np.newaxis, ...]  # (1, 960, 640)
print(f"  输入 ndarray shape = {nv12.shape}")

t1 = time.time()
outputs = m.forward(nv12)
print(f"  推理耗时 {(time.time()-t1)*1000:.1f} ms，输出 {len(outputs)} 个 tensor")

# ── 输出详细 ─────────────────────────────────────────────────────────
print("\n=== 输出 buffer 详细 ===")
for i, o in enumerate(outputs):
    buf = np.array(o.buffer)
    print(f"  [{i}] shape={buf.shape}  dtype={buf.dtype}  "
          f"min={buf.min():.3f}  max={buf.max():.3f}  mean={buf.mean():.3f}")
    # 找 non-zero 元素数量
    nz = np.count_nonzero(buf)
    print(f"       non-zero={nz}/{buf.size}")

# ── 猜测格式 ─────────────────────────────────────────────────────────
print("\n=== 格式推断 ===")
n = len(outputs)
if n == 9:
    print("  → 标准 YOLOv8-pose 6-head: 3 scales × (cls, box_dfl, kpt)")
    print("     _parse_pose 应能正常工作")
elif n == 6:
    print("  → 可能是 YOLOv8-pose 3-head (no kpt separate): 3 scales × (cls+kpt, box_dfl)")
    print("     需要调整 _parse_pose 索引")
elif n == 3:
    print("  → 可能是 post-processed 输出: [boxes, scores, keypoints]")
    print("     需要完全重写 _parse_pose")
else:
    print(f"  → 未知格式（{n} 个输出），请人工检查 shape")

# ── 对真实相机帧推理（可选）─────────────────────────────────────────────
print("\n=== 真实帧推理（使用 /dev/video0）===")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_sq = cv2.resize(frame, (W, H))
            yuv = cv2.cvtColor(frame_sq, cv2.COLOR_BGR2YUV_I420)
            y_p  = yuv[:H, :]
            uv_i = yuv[H:, :].reshape(-1, 2)
            uv_n = uv_i[:, [0, 1]].reshape(H//2, W)
            nv12_real = np.vstack([y_p, uv_n]).astype(np.uint8)[np.newaxis, ...]
            t2 = time.time()
            outs_real = m.forward(nv12_real)
            print(f"  推理耗时 {(time.time()-t2)*1000:.1f} ms")
            for i, o in enumerate(outs_real):
                buf = np.array(o.buffer)
                nz = np.count_nonzero(buf)
                mx = buf.max()
                print(f"  [{i}] shape={buf.shape}  max={mx:.3f}  non-zero={nz}/{buf.size}")
        else:
            print("  读帧失败")
    else:
        print("  摄像头打不开，跳过")
except Exception as e:
    print(f"  异常: {e}")

print("\n完成。根据输出格式确认 bpu_runner._parse_pose 是否需要调整。")
