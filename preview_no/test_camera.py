from hobot_vio import libsrcampy as srcampy
import numpy as np, cv2, time, sys

CAM_W, CAM_H = 1920, 1080

cam = srcampy.Camera()
# 正确 API: open_cam(pipe_id, video_index, fps, [width_list], [height_list], sensor_h, sensor_w)
ret = cam.open_cam(0, -1, 30, [CAM_W], [CAM_H], CAM_H, CAM_W)
sys.stderr.write(f'open_cam -> {ret}\n')
sys.stderr.flush()

if ret == 0:
    raw = None
    for attempt in range(30):
        # 正确 API: get_img(type, width, height)  type=2 → NV12
        raw = cam.get_img(2, CAM_W, CAM_H)
        if raw is not None and len(raw) > 0:
            break
        sys.stderr.write(f'  get_img 尝试 {attempt+1}/30 -> None，再等 0.1s\n')
        sys.stderr.flush()
        time.sleep(0.1)

    if raw is not None and len(raw) > 0:
        nv12 = np.frombuffer(raw, dtype=np.uint8).reshape(CAM_H * 3 // 2, CAM_W)
        bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        cv2.imwrite('/tmp/test_frame.jpg', bgr)
        sys.stderr.write(f'get_img -> OK, len={len(raw)}, frame shape={bgr.shape}\n')
        sys.stderr.write('已保存到 /tmp/test_frame.jpg\n')
    else:
        sys.stderr.write(f'get_img -> 失败，始终返回 None\n')
    sys.stderr.flush()
    cam.close_cam()
else:
    sys.stderr.write(f'open_cam 失败，返回码: {ret}\n')
    sys.stderr.flush()