"""
FastAPI REST 路由。§7.1。
挂载在 app.main 的 FastAPI app 上。
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import urllib.parse

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.beat.analyzer import analyze as beat_analyze
from app.ai_choreographer.service import orchestrate
from app.ai_choreographer.catalog import get_catalog
from app.fx._registry import FX_REGISTRY

router = APIRouter(prefix="/api")

# 内存存储（与磁盘 JSON 双写同步）
_music_store: Dict[str, Dict] = {}
_choreo_store: Dict[str, Dict] = {}
_task_store: Dict[str, str] = {}   # task_id → "running" | "done" | "error"

UPLOAD_DIR = Path(os.getenv("PICOCLAW_UPLOAD_DIR", "/tmp/picoclaw_uploads"))
CACHE_DIR  = Path(os.getenv("PICOCLAW_CACHE_DIR",  "/tmp/picoclaw_cache"))
# 持久化目录：音乐元数据 + 编排 JSON
DATA_DIR   = Path(os.getenv("PICOCLAW_DATA_DIR",   "/tmp/picoclaw_data"))
CHOREO_DIR = DATA_DIR / "choreos"
FEAT_DIR   = DATA_DIR / "feats"
MUSIC_DB   = DATA_DIR / "music_library.json"

for _d in (UPLOAD_DIR, CACHE_DIR, DATA_DIR, CHOREO_DIR, FEAT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 持久化辅助
# ---------------------------------------------------------------------------

def _save_music_library() -> None:
    """把音乐元数据（不含 feat 大数据）写入 music_library.json。"""
    snapshot = {}
    for mid, info in _music_store.items():
        entry = {k: v for k, v in info.items() if k != "feat"}
        snapshot[mid] = entry
    tmp = MUSIC_DB.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    tmp.replace(MUSIC_DB)


def _load_music_library() -> None:
    """启动时从 music_library.json 恢复内存状态。"""
    if not MUSIC_DB.exists():
        return
    try:
        with open(MUSIC_DB, encoding="utf-8") as f:
            data = json.load(f)
        for mid, info in data.items():
            if Path(info.get("path", "")).exists():
                _music_store[mid] = info
                # 尝试恢复 feat（beat 分析结果）
                feat_path = FEAT_DIR / f"{mid}.json"
                if feat_path.exists():
                    try:
                        with open(feat_path, encoding="utf-8") as ff:
                            _music_store[mid]["feat"] = json.load(ff)
                    except Exception:
                        pass
    except Exception as e:
        print(f"[routes] 加载音乐库失败: {e}")


def _save_feat(music_id: str, feat: Dict) -> None:
    """把 beat 分析结果保存为 feats/<music_id>.json。"""
    try:
        feat_path = FEAT_DIR / f"{music_id}.json"
        tmp = feat_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(feat, f, ensure_ascii=False)
        tmp.replace(feat_path)
    except Exception as e:
        print(f"[routes] 保存 feat 失败: {e}")


def _save_choreo(music_id: str, data: Dict) -> None:
    """把编排保存为 choreos/<music_id>.json。"""
    try:
        path = CHOREO_DIR / f"{music_id}.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except Exception as e:
        print(f"[routes] 保存编排失败: {e}")


def _load_choreo_from_disk(music_id: str) -> Optional[Dict]:
    """从磁盘加载编排（若内存无缓存时使用）。"""
    path = CHOREO_DIR / f"{music_id}.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


# 启动时恢复
_load_music_library()


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
    _save_music_library()
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
            # 持久化：feat 单独存储（大文件），库元数据内只存摘要
            _save_feat(music_id, feat)
            _music_store[music_id]["tempo"]      = feat.get("tempo")
            _music_store[music_id]["duration"]   = feat.get("duration")
            _music_store[music_id]["beat_count"] = len(feat.get("beats", []))
            _save_music_library()
            _task_store[task_id] = "done"
        except Exception as e:
            import traceback
            traceback.print_exc()
            _task_store[task_id] = f"error: {e}"

    background_tasks.add_task(_run)
    return {"task_id": task_id}


@router.get("/music/library")
async def list_music_library():
    """返回所有已存储的音乐，供前端下拉框使用。"""
    items = []
    for mid, info in _music_store.items():
        feat = info.get("feat", {})
        has_choreo = (mid in _choreo_store
                      or (CHOREO_DIR / f"{mid}.json").exists())
        items.append({
            "music_id": mid,
            "filename": info.get("filename", ""),
            "status": info.get("status", "uploaded"),
            "tempo": info.get("tempo") or feat.get("tempo"),
            "duration": info.get("duration") or feat.get("duration"),
            "has_choreo": has_choreo,
        })
    items.sort(key=lambda x: x["filename"].lower())
    return {"items": items}


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
        "tempo": info.get("tempo") or feat.get("tempo"),
        "duration": info.get("duration") or feat.get("duration"),
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
            choreo_data = {
                "source": source,
                "plan": plan.model_dump() if plan else None,
                "track": track,
                "style": req.style,
                "theme": (plan.model_dump() or {}).get("theme", "") if plan else "",
            }
            _choreo_store[music_id] = choreo_data
            _save_choreo(music_id, choreo_data)   # 持久化
            _task_store[task_id] = "done"
        except Exception as e:
            _task_store[task_id] = f"error: {e}"

    background_tasks.add_task(_run)
    return {"task_id": task_id}


@router.get("/music/{music_id}/choreo")
async def get_choreo(music_id: str):
    """获取当前编排（内存无则从磁盘加载）。"""
    data = _choreo_store.get(music_id)
    if not data:
        data = _load_choreo_from_disk(music_id)
        if not data:
            raise HTTPException(404, "choreo not found")
        _choreo_store[music_id] = data   # 内存缓存
    return {
        "source": data.get("source"),
        "style": data.get("style"),
        "theme": data.get("theme", ""),
        "plan": data.get("plan"),
        "track_count": len(data.get("track", [])),
    }


@router.put("/music/{music_id}/choreo")
async def put_choreo(music_id: str, body: Dict[str, Any]):
    """上传自定义编排（source=manual）并持久化。"""
    data = {"source": "manual", "track": body.get("track", []),
            "plan": body.get("plan"), "theme": body.get("theme", ""),
            "style": body.get("style", "")}
    _choreo_store[music_id] = data
    _save_choreo(music_id, data)
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
    """启动演出 session，通知 Orchestrator并应用主题。"""
    from app.main import get_orchestrator
    orch = get_orchestrator()
    if orch:
        # 内存无则尝试从磁盘加载编排
        choreo = _choreo_store.get(req.music_id)
        if not choreo:
            choreo = _load_choreo_from_disk(req.music_id) or {}
            if choreo:
                _choreo_store[req.music_id] = choreo
        orch.load_session(
            music_id=req.music_id,
            track=choreo.get("track", []),
            feat=_music_store.get(req.music_id, {}).get("feat", {}),
        )
        # 应用编排中的主题滤镜
        if orch.compositor:
            theme_id = choreo.get("theme", "")
            orch.compositor.set_theme(theme_id)
    _active_session.update({"music_id": req.music_id, "mode": req.mode})
    return {"ok": True, "session": _active_session}


@router.post("/session/stop")
async def session_stop():
    from app.main import get_orchestrator
    orch = get_orchestrator()
    if orch:
        orch.stop_session()
        if orch.compositor:
            orch.compositor.set_theme("")   # 清除主题
    _active_session.clear()
    return {"ok": True}


# ---------------------------------------------------------------------------
# 主题 + FX 目录接口
# ---------------------------------------------------------------------------

class ThemeSetRequest(BaseModel):
    theme_id: str = ""
    strength: float = 0.6


@router.post("/session/theme")
async def set_session_theme(req: ThemeSetRequest):
    """实时切换当前主题滤镜。"""
    from app.main import get_orchestrator
    orch = get_orchestrator()
    if orch and orch.compositor:
        orch.compositor.set_theme(
            req.theme_id,
            {"strength": req.strength} if req.theme_id else {}
        )
        return {"ok": True, "theme": req.theme_id}
    return {"ok": False, "reason": "compositor not ready"}


@router.get("/fx/themes")
async def list_themes():
    """返回所有 category='theme' 的 FX 列表。"""
    themes = [
        {"id": meta.fx_id, "description": meta.description}
        for meta in FX_REGISTRY.values()
        if meta.category == "theme"
    ]
    return {"themes": themes}


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
