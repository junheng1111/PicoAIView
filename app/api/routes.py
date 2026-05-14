"""
FastAPI REST 路由。§7.1。
挂载在 app.main 的 FastAPI app 上。
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import urllib.parse

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.beat.analyzer import analyze as beat_analyze
from app.ai_choreographer.service import orchestrate
from app.ai_choreographer.catalog import get_catalog
from app.fx._registry import FX_REGISTRY

router = APIRouter(prefix="/api")

# 简单内存存储（生产替换为 Redis / SQLite）
_music_store: Dict[str, Dict] = {}
_choreo_store: Dict[str, Dict] = {}
_task_store: Dict[str, str] = {}   # task_id → "running" | "done" | "error"

UPLOAD_DIR = Path(os.getenv("PICOCLAW_UPLOAD_DIR", "/tmp/picoclaw_uploads"))
CACHE_DIR  = Path(os.getenv("PICOCLAW_CACHE_DIR",  "/tmp/picoclaw_cache"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 音乐上传 + 分析
# ---------------------------------------------------------------------------

@router.put("/music/upload")
async def upload_music_raw(request: Request):
    """快速上传：直接读取原始字节，绕过 python-multipart 解析（ARM 上快 37x）。
    Header: X-Filename: 文件名（可选）
    Body: 原始音频字节（application/octet-stream）
    """
    music_id = uuid.uuid4().hex
    filename = urllib.parse.unquote(request.headers.get("x-filename", "audio.mp3"))
    ext = Path(filename).suffix or ".mp3"
    dest = UPLOAD_DIR / f"{music_id}{ext}"

    # 流式写入，避免一次性读入大文件
    written = 0
    with open(dest, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
            written += len(chunk)

    _music_store[music_id] = {"path": str(dest), "filename": filename,
                               "status": "uploaded", "size": written}
    return {"music_id": music_id, "size": written}


@router.get("/music/{music_id}/audio")
async def serve_audio(music_id: str):
    """将上传的音频文件提供给浏览器播放。"""
    if music_id not in _music_store:
        raise HTTPException(404, "music_id not found")
    path = _music_store[music_id].get("path", "")
    if not path or not Path(path).exists():
        raise HTTPException(404, "audio file not found")
    media_type = "audio/mpeg"
    ext = Path(path).suffix.lower()
    if ext == ".wav":
        media_type = "audio/wav"
    elif ext in (".flac", ".ogg"):
        media_type = "audio/ogg"
    return FileResponse(path, media_type=media_type,
                        headers={"Accept-Ranges": "bytes"})


@router.post("/music/{music_id}/analyze")
async def analyze_music(music_id: str, background_tasks: BackgroundTasks):
    """异步触发 BeatEngine 分析。"""
    if music_id not in _music_store:
        raise HTTPException(404, "music_id not found")

    task_id = uuid.uuid4().hex
    _task_store[task_id] = "running"

    async def _run():
        try:
            info = _music_store[music_id]
            # 在线程池中执行 CPU 密集型 librosa 分析，避免阻塞 asyncio 事件循环
            feat = await asyncio.to_thread(
                beat_analyze, info["path"], str(CACHE_DIR)
            )
            _music_store[music_id]["feat"] = feat
            _music_store[music_id]["status"] = "analyzed"
            _task_store[task_id] = "done"
        except Exception as e:
            import traceback
            traceback.print_exc()
            _task_store[task_id] = f"error: {e}"

    background_tasks.add_task(_run)
    return {"task_id": task_id}


@router.get("/music/{music_id}")
async def get_music(music_id: str):
    if music_id not in _music_store:
        raise HTTPException(404, "music_id not found")
    info = dict(_music_store[music_id])
    info.pop("path", None)   # 不暴露本地路径
    feat = info.get("feat", {})
    return {
        "music_id": music_id,
        "filename": info.get("filename"),
        "status": info.get("status"),
        "tempo": feat.get("tempo"),
        "duration": feat.get("duration"),
        "beat_count": len(feat.get("beats", [])),
        "segments": feat.get("segments", []),
        "rms_summary": feat.get("rms_summary", []),
    }


# ---------------------------------------------------------------------------
# AI Choreographer
# ---------------------------------------------------------------------------

class ChoreoAIRequest(BaseModel):
    style: Optional[str] = "energetic"
    model: Optional[str] = None
    force_refresh: bool = False


@router.post("/music/{music_id}/choreo/ai")
async def trigger_ai_choreo(music_id: str, req: ChoreoAIRequest,
                             background_tasks: BackgroundTasks):
    """触发 LLM Choreographer 生成 ChoreoPlan。"""
    if music_id not in _music_store:
        raise HTTPException(404, "music_id not found")
    if _music_store[music_id].get("status") != "analyzed":
        raise HTTPException(400, "请先调用 /analyze")

    task_id = uuid.uuid4().hex
    _task_store[task_id] = "running"

    async def _run():
        try:
            feat = _music_store[music_id]["feat"]
            plan, track, source = await orchestrate(
                music_id=music_id,
                feat=feat,
                style=req.style or "energetic",
                force_refresh=req.force_refresh,
                model_override=req.model,
            )
            _choreo_store[music_id] = {
                "source": source,
                "plan": plan.model_dump() if plan else None,
                "track": track,
                "style": req.style,
            }
            _task_store[task_id] = "done"
        except Exception as e:
            _task_store[task_id] = f"error: {e}"

    background_tasks.add_task(_run)
    return {"task_id": task_id}


@router.get("/music/{music_id}/choreo")
async def get_choreo(music_id: str):
    """获取当前编排。"""
    if music_id not in _choreo_store:
        raise HTTPException(404, "choreo not found")
    data = _choreo_store[music_id]
    return {
        "source": data.get("source"),
        "style": data.get("style"),
        "plan": data.get("plan"),
        "track_count": len(data.get("track", [])),
    }


@router.put("/music/{music_id}/choreo")
async def put_choreo(music_id: str, body: Dict[str, Any]):
    """上传自定义编排（source=manual）。"""
    _choreo_store[music_id] = {"source": "manual", "track": body.get("track", []),
                                "plan": body.get("plan")}
    return {"ok": True}


@router.post("/music/{music_id}/choreo/regenerate")
async def regenerate_choreo(music_id: str, req: ChoreoAIRequest,
                             background_tasks: BackgroundTasks):
    req.force_refresh = True
    return await trigger_ai_choreo(music_id, req, background_tasks)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class SessionStartRequest(BaseModel):
    music_id: str
    mode: str = "auto"   # rhythm | manual | auto


_active_session: Dict = {}


@router.post("/session/start")
async def session_start(req: SessionStartRequest):
    """启动演出 session，通知 Orchestrator。"""
    from app.main import get_orchestrator
    orch = get_orchestrator()
    if orch:
        choreo = _choreo_store.get(req.music_id, {})
        orch.load_session(
            music_id=req.music_id,
            track=choreo.get("track", []),
            feat=_music_store.get(req.music_id, {}).get("feat", {}),
        )
    _active_session.update({"music_id": req.music_id, "mode": req.mode})
    return {"ok": True, "session": _active_session}


@router.post("/session/stop")
async def session_stop():
    from app.main import get_orchestrator
    orch = get_orchestrator()
    if orch:
        orch.stop_session()
    _active_session.clear()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    from app.main import get_orchestrator
    orch = get_orchestrator()
    bpu_ok = "degraded"
    camera_ok = "degraded"
    models: list = []
    if orch:
        # BPU: ok if infer loop has produced at least one state update
        bpu_ok = "ok" if (orch.bpu_runner and
                          orch.bpu_runner.vision_state().timestamp > 0) else "degraded"
        # Camera: ok as soon as any frame (real or synthetic) is available
        camera_ok = "ok" if (orch.camera and
                              orch.camera.get_frame() is not None) else "degraded"
        if orch.bpu_runner:
            models = orch.bpu_runner.loaded_models()

    # LLM provider info
    import os
    llm_backend = os.getenv("PICOCLAW_LLM", "openclaw")
    llm_reachable = False
    try:
        import httpx
        r = httpx.get("http://localhost:18789/v1/models", timeout=1.0)
        llm_reachable = r.status_code < 500
    except Exception:
        pass

    return {
        "bpu": bpu_ok,
        "camera": camera_ok,
        "models": models,
        "fx_count": len(FX_REGISTRY),
        "llm_provider": llm_backend,
        "llm_reachable": llm_reachable,
    }


# ---------------------------------------------------------------------------
# MJPEG 实时视频流（使用 Orchestrator 合成帧：FX + YOLO）
# ---------------------------------------------------------------------------

def _encode_jpeg(frame: np.ndarray) -> bytes:
    """在线程池中执行 JPEG 编码（避免阻塞事件循环）。"""
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return jpg.tobytes()


async def _mjpeg_frames(orch):
    """MJPEG multipart generator：使用 Orchestrator 已合成的帧（含 FX + YOLO 框）。
    帧由 _frame_loop 已降至 640×360，JPEG 编码移到线程池避免阻塞事件循环。
    """
    placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "PicoClaw", (160, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (80, 80, 80), 3)
    cv2.putText(placeholder, "Initializing...", (190, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 60), 2)

    while True:
        display = None
        if orch:
            display = orch.get_processed_frame()

        if display is None:
            display = placeholder.copy()

        # JPEG 编码移到线程池，避免阻塞 asyncio 事件循环（ARM 上编码 ~5ms）
        jpg_bytes = await asyncio.to_thread(_encode_jpeg, display)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n"
               + jpg_bytes + b"\r\n")
        await asyncio.sleep(0.040)   # ~25fps，与帧处理速率匹配


@router.get("/video/stream")
async def video_stream():
    """MJPEG 实时流，视频流 + YOLO检测框overlay。"""
    from app.main import get_orchestrator
    orch = get_orchestrator()
    return StreamingResponse(
        _mjpeg_frames(orch),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Task status
# ---------------------------------------------------------------------------

@router.get("/task/{task_id}")
async def task_status(task_id: str):
    status = _task_store.get(task_id)
    if status is None:
        raise HTTPException(404, "task not found")
    return {"task_id": task_id, "status": status}
