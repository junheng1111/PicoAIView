"""
FastAPI 应用入口。§11 目录结构 app/main.py。

启动方式：
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.orchestrator import Orchestrator
from app.media.camera import CameraSource
from app.media.compositor import FxCompositor
from app.vision.bpu_runner import BpuRunner

# 全局 orchestrator（通过 get_orchestrator() 访问）
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Optional[Orchestrator]:
    return _orchestrator


# ---------------------------------------------------------------------------
# 配置（从环境变量读取）
# ---------------------------------------------------------------------------

MODEL_DIR   = os.getenv("PICOCLAW_MODEL_DIR",   "models")
DETECT_BIN  = os.getenv("PICOCLAW_DETECT_BIN",  f"{MODEL_DIR}/yolov8s_detect_bayese_640x640_nv12.bin")
# 只有检测模型存在；姿态/分割模型按需添加到 models/ 后自动生效
_pose_default = f"{MODEL_DIR}/yolov8s_pose_bayese_640x640_nv12.bin"
POSE_BIN    = os.getenv("PICOCLAW_POSE_BIN",
               _pose_default if os.path.exists(_pose_default) else None)
SEG_BIN     = os.getenv("PICOCLAW_SEG_BIN",     None)
DEPTH_BIN   = os.getenv("PICOCLAW_DEPTH_BIN",   None)
FACE_BIN    = os.getenv("PICOCLAW_FACE_BIN",    None)

CAM_WIDTH   = int(os.getenv("PICOCLAW_CAM_W",   "1920"))
CAM_HEIGHT  = int(os.getenv("PICOCLAW_CAM_H",   "1080"))
CAM_FPS     = int(os.getenv("PICOCLAW_CAM_FPS", "30"))

# LLM 默认使用本地 OpenClaw Gateway (MiniMax-M2.7)
os.environ.setdefault("PICOCLAW_LLM", "openclaw")


# ---------------------------------------------------------------------------
# Lifespan（启动 / 关闭）
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator

    # 初始化摄像头
    camera = CameraSource(width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS)
    camera.start()

    # 初始化 BPU 推理
    bpu = BpuRunner(model_dir=MODEL_DIR)
    bpu.start(
        detect_bin=DETECT_BIN,
        pose_bin=POSE_BIN,
        seg_bin=SEG_BIN,
        depth_bin=DEPTH_BIN,
        face_bin=FACE_BIN,
    )

    # 初始化 FX Compositor
    compositor = FxCompositor()

    # 初始化 Orchestrator
    orch = Orchestrator()
    orch.setup(camera, bpu, compositor)
    orch.start_background_tasks()
    _orchestrator = orch

    print("[PicoClaw] 系统启动完成，等待请求...")
    yield

    # 正常关闭
    camera.stop()
    bpu.stop()
    print("[PicoClaw] 系统已关闭")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PicoClaw — 地瓜机器人 RDK X5 AI 视觉律动系统",
    version="0.4.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST 路由
from app.api.routes import router as api_router
from app.api.rtc    import router as rtc_router
from app.api.ws     import ws_control_endpoint

app.include_router(api_router)
app.include_router(rtc_router)


@app.websocket("/ws/control")
async def ws_endpoint(websocket: WebSocket):
    await ws_control_endpoint(websocket)


# 静态资源（CSS / JS）
_web_dir = os.path.join(os.path.dirname(__file__), "..", "web")
app.mount("/static", StaticFiles(directory=_web_dir), name="static")


# 前端入口
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def frontend_index():
    return FileResponse(os.path.join(_web_dir, "index.html"))


# 前端静态文件（生产构建后挂载）
_web_dist = os.path.join(os.path.dirname(__file__), "..", "web", "dist")
if os.path.isdir(_web_dist):
    app.mount("/dist", StaticFiles(directory=_web_dist, html=True), name="frontend")


# ---------------------------------------------------------------------------
# 开发期直接运行
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
