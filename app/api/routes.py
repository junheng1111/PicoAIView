"""
FastAPI REST 路由。§7.1。
挂载在 app.main 的 FastAPI app 上。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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
# 编排按 choreo_id 存储，与 music_id 解耦；同一首歌可有多套编排
_choreo_store: Dict[str, Dict] = {}   # choreo_id → choreo data (含 music_id 字段)
_task_store: Dict[str, str] = {}      # task_id → "running" | "done" | "error"

# 持久化目录：位于项目根目录 data/，重启不丢失
_PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR   = Path(os.getenv("PICOCLAW_DATA_DIR",   str(_PROJECT_ROOT / "data")))
UPLOAD_DIR = Path(os.getenv("PICOCLAW_UPLOAD_DIR", str(DATA_DIR / "uploads")))
CACHE_DIR  = Path(os.getenv("PICOCLAW_CACHE_DIR",  str(DATA_DIR / "cache")))
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
    """启动时从 music_library.json 和 choreos/ 目录恢复内存状态。"""
    if MUSIC_DB.exists():
        try:
            with open(MUSIC_DB, encoding="utf-8") as f:
                data = json.load(f)
            for mid, info in data.items():
                if Path(info.get("path", "")).exists():
                    _music_store[mid] = info
                    feat_path = FEAT_DIR / f"{mid}.json"
                    if feat_path.exists():
                        try:
                            with open(feat_path, encoding="utf-8") as ff:
                                _music_store[mid]["feat"] = json.load(ff)
                        except Exception:
                            pass
        except Exception as e:
            print(f"[routes] 加载音乐库失败: {e}")

    # 扫描 choreos/ 目录，按 choreo_id 恢复（与 music_id 解耦）
    for p in CHOREO_DIR.glob("*.json"):
        try:
            with open(p, encoding="utf-8") as f:
                choreo = json.load(f)
            cid = choreo.get("choreo_id", p.stem)
            _choreo_store[cid] = choreo
            # 把 choreo_id 追加进对应音乐的列表（若已在 _music_store 里）
            mid = choreo.get("music_id", "")
            if mid and mid in _music_store:
                ids = _music_store[mid].setdefault("choreo_ids", [])
                if cid not in ids:
                    ids.append(cid)
        except Exception:
            pass


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


def _save_choreo(choreo_id: str, data: Dict) -> None:
    """把编排保存为 choreos/<choreo_id>.json（data 中必须含 choreo_id 和 music_id 字段）。"""
    try:
        path = CHOREO_DIR / f"{choreo_id}.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except Exception as e:
        print(f"[routes] 保存编排失败: {e}")


def _choreo_ids_for(music_id: str) -> List[str]:
    """返回某首歌的所有 choreo_id 列表（按创建时间排序）。"""
    return _music_store.get(music_id, {}).get("choreo_ids", [])


def _latest_choreo(music_id: str) -> Optional[Dict]:
    """返回某首歌最新一套编排（内存）；无则返回 None。"""
    ids = _choreo_ids_for(music_id)
    if not ids:
        return None
    return _choreo_store.get(ids[-1])


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

    同文件（SHA256 相同）会自动去重：返回已有 music_id，不重复写磁盘。
    """
    music_id = uuid.uuid4().hex
    filename = urllib.parse.unquote(request.headers.get("x-filename", "audio.mp3"))
    ext = Path(filename).suffix or ".mp3"
    dest = UPLOAD_DIR / f"{music_id}{ext}"

    hasher = hashlib.sha256()
    written = 0
    with open(dest, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
            hasher.update(chunk)
            written += len(chunk)

    file_hash = hasher.hexdigest()

    # 去重：相同 SHA256 → 返回已有记录，删除刚写入的重复文件
    for mid, info in _music_store.items():
        if info.get("sha256") == file_hash:
            dest.unlink(missing_ok=True)
            return {"music_id": mid, "size": info.get("size", written),
                    "deduplicated": True, "filename": info.get("filename")}

    _music_store[music_id] = {
        "path": str(dest), "filename": filename,
        "status": "uploaded", "size": written,
        "sha256": file_hash, "choreo_ids": [],
    }
    _save_music_library()
    return {"music_id": music_id, "size": written, "deduplicated": False}


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
        choreo_ids = info.get("choreo_ids", [])
        items.append({
            "music_id": mid,
            "filename": info.get("filename", ""),
            "status": info.get("status", "uploaded"),
            "tempo": info.get("tempo") or feat.get("tempo"),
            "duration": info.get("duration") or feat.get("duration"),
            "choreo_count": len(choreo_ids),
            "choreo_ids": choreo_ids,
            "has_choreo": len(choreo_ids) > 0,
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
    """触发 LLM Choreographer 生成新 ChoreoPlan（每次生成一套新编排，不覆盖旧的）。"""
    if music_id not in _music_store:
        raise HTTPException(404, "music_id not found")
    if _music_store[music_id].get("status") != "analyzed":
        raise HTTPException(400, "请先调用 /analyze")

    choreo_id = uuid.uuid4().hex
    task_id   = uuid.uuid4().hex
    _task_store[task_id] = "running"

    async def _run():
        try:
            feat = _music_store[music_id]["feat"]
            audio_path = _music_store[music_id].get("path")
            plan, track, source = await orchestrate(
                music_id=music_id,
                feat=feat,
                style=req.style or "energetic",
                force_refresh=req.force_refresh,
                model_override=req.model,
                audio_path=audio_path,
            )
            choreo_data = {
                "choreo_id": choreo_id,
                "music_id":  music_id,
                "source":    source,
                "plan":      plan.model_dump() if plan else None,
                "track":     track,
                "style":     req.style,
                "theme":     (plan.model_dump() or {}).get("theme", "") if plan else "",
            }
            _choreo_store[choreo_id] = choreo_data
            _save_choreo(choreo_id, choreo_data)
            # 追加 choreo_id 进音乐记录
            ids = _music_store[music_id].setdefault("choreo_ids", [])
            if choreo_id not in ids:
                ids.append(choreo_id)
            _save_music_library()
            _task_store[task_id] = "done"
        except Exception as e:
            _task_store[task_id] = f"error: {e}"

    background_tasks.add_task(_run)
    return {"task_id": task_id, "choreo_id": choreo_id}


@router.get("/music/{music_id}/choreos")
async def list_choreos(music_id: str):
    """列出某首歌的所有编排摘要（最新在后）。"""
    if music_id not in _music_store:
        raise HTTPException(404, "music_id not found")
    result = []
    for cid in _choreo_ids_for(music_id):
        d = _choreo_store.get(cid)
        if d:
            result.append({
                "choreo_id": cid,
                "source":    d.get("source"),
                "style":     d.get("style"),
                "theme":     d.get("theme", ""),
                "track_count": len(d.get("track", [])),
            })
    return {"choreos": result}


@router.get("/music/{music_id}/choreo")
async def get_choreo(music_id: str):
    """获取最新一套编排（向后兼容旧接口）。"""
    data = _latest_choreo(music_id)
    if not data:
        raise HTTPException(404, "choreo not found")
    return {
        "choreo_id":   data.get("choreo_id"),
        "source":      data.get("source"),
        "style":       data.get("style"),
        "theme":       data.get("theme", ""),
        "plan":        data.get("plan"),
        "track_count": len(data.get("track", [])),
    }


@router.get("/choreo/{choreo_id}")
async def get_choreo_by_id(choreo_id: str):
    """按 choreo_id 获取具体编排详情。"""
    data = _choreo_store.get(choreo_id)
    if not data:
        raise HTTPException(404, "choreo not found")
    return data


@router.put("/music/{music_id}/choreo")
async def put_choreo(music_id: str, body: Dict[str, Any]):
    """上传自定义编排（source=manual），生成新 choreo_id 并持久化。"""
    if music_id not in _music_store:
        raise HTTPException(404, "music_id not found")
    choreo_id = uuid.uuid4().hex
    data = {
        "choreo_id": choreo_id,
        "music_id":  music_id,
        "source":    "manual",
        "track":     body.get("track", []),
        "plan":      body.get("plan"),
        "theme":     body.get("theme", ""),
        "style":     body.get("style", ""),
    }
    _choreo_store[choreo_id] = data
    _save_choreo(choreo_id, data)
    ids = _music_store[music_id].setdefault("choreo_ids", [])
    if choreo_id not in ids:
        ids.append(choreo_id)
    _save_music_library()
    return {"ok": True, "choreo_id": choreo_id}


@router.post("/music/{music_id}/choreo/regenerate")
async def regenerate_choreo(music_id: str, req: ChoreoAIRequest,
                             background_tasks: BackgroundTasks):
    """生成新编排（不覆盖旧编排，旧的仍可访问）。"""
    req.force_refresh = True
    return await trigger_ai_choreo(music_id, req, background_tasks)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class SessionStartRequest(BaseModel):
    music_id:  str
    choreo_id: Optional[str] = None   # 不填则使用该音乐最新编排
    mode:      str = "auto"            # rhythm | manual | auto


_active_session: Dict = {}


@router.post("/session/start")
async def session_start(req: SessionStartRequest):
    """启动演出 session，通知 Orchestrator 并应用主题。
    choreo_id 指定具体编排；不填则使用该音乐最新编排。
    """
    if req.music_id not in _music_store:
        raise HTTPException(404, "music_id not found")

    # 确定使用哪套编排
    choreo: Dict = {}
    used_choreo_id = req.choreo_id
    if used_choreo_id:
        choreo = _choreo_store.get(used_choreo_id, {})
    else:
        choreo = _latest_choreo(req.music_id) or {}
        used_choreo_id = choreo.get("choreo_id", "")

    from app.main import get_orchestrator
    orch = get_orchestrator()
    if orch:
        orch.load_session(
            music_id=req.music_id,
            track=choreo.get("track", []),
            feat=_music_store.get(req.music_id, {}).get("feat", {}),
        )
        if orch.compositor:
            orch.compositor.set_theme(choreo.get("theme", ""))

    _active_session.update({
        "music_id":  req.music_id,
        "choreo_id": used_choreo_id,
        "mode":      req.mode,
    })
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


class LiveFxRequest(BaseModel):
    subject_zoom:     Optional[bool]  = None
    subject_zoom_amp: Optional[float] = None   # 0.0~2.0，1.0=默认
    zoom_pulse:       Optional[bool]  = None
    zoom_pulse_amp:   Optional[float] = None
    brightness:       Optional[bool]  = None
    brightness_amp:   Optional[float] = None


@router.post("/session/live_fx")
async def set_live_fx(req: LiveFxRequest):
    """实时调整常驻基础层（zoom_pulse / brightness / subject_zoom）。"""
    from app.main import get_orchestrator
    orch = get_orchestrator()
    if not (orch and orch.compositor):
        return {"ok": False, "reason": "compositor not ready"}

    updates: Dict[str, Any] = {}
    for field in ("subject_zoom", "subject_zoom_amp",
                  "zoom_pulse", "zoom_pulse_amp",
                  "brightness", "brightness_amp"):
        val = getattr(req, field)
        if val is not None:
            updates[field] = val
    orch.compositor.set_live_config(updates)
    return {"ok": True, "live_config": orch.compositor.get_live_config()}


@router.get("/session/live_fx")
async def get_live_fx():
    """读取当前常驻层配置。"""
    from app.main import get_orchestrator
    orch = get_orchestrator()
    if orch and orch.compositor:
        return {"ok": True, "live_config": orch.compositor.get_live_config()}
    return {"ok": False}


@router.get("/fx/themes")
async def list_themes():
    """返回所有 category='theme' 的 FX 列表。"""
    themes = [
        {"id": meta.fx_id, "description": meta.description}
        for meta in FX_REGISTRY.values()
        if meta.category == "theme"
    ]
    return {"themes": themes}


@router.get("/fx/catalog")
async def get_fx_catalog():
    """返回完整 FX 目录（供前端 FX 测试区使用）。"""
    from app.fx._registry import build_fx_catalog
    return {"catalog": build_fx_catalog(include_excluded=False)}


# ── FX 测试覆盖（手动叠加，不依赖 choreo）────────────────────────────

class FxTestRequest(BaseModel):
    fx_id:     str
    intensity: float = 0.8    # 0.0~1.0，映射到 intensity/amp/strength


@router.post("/session/fx_test")
async def fx_test_on(req: FxTestRequest):
    """激活单个 FX 覆盖（立刻在实时画面中叠加，不需要 session）。"""
    from app.main import get_orchestrator
    orch = get_orchestrator()
    if not (orch and orch.compositor):
        return {"ok": False, "reason": "compositor not ready"}
    v = float(max(0.0, min(1.0, req.intensity)))
    orch.compositor.override_fx(req.fx_id, {
        "intensity": v, "amp": v, "strength": v,
    })
    return {"ok": True, "fx_id": req.fx_id, "intensity": v}


@router.delete("/session/fx_test/{fx_id}")
async def fx_test_off(fx_id: str):
    """关闭单个 FX 覆盖。"""
    from app.main import get_orchestrator
    orch = get_orchestrator()
    if orch and orch.compositor:
        orch.compositor.clear_override(fx_id)
    return {"ok": True}


@router.post("/session/fx_test/clear")
async def fx_test_clear():
    """清除全部 FX 测试覆盖。"""
    from app.main import get_orchestrator
    orch = get_orchestrator()
    if orch and orch.compositor:
        orch.compositor.clear_all_overrides()
    return {"ok": True}


@router.get("/session/fx_test")
async def fx_test_list():
    """获取当前所有激活的覆盖 FX。"""
    from app.main import get_orchestrator
    orch = get_orchestrator()
    if orch and orch.compositor:
        return {"ok": True, "overrides": orch.compositor.get_overrides()}
    return {"ok": False, "overrides": {}}


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

def _make_placeholder_jpg() -> bytes:
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, "PicoClaw", (160, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (80, 80, 80), 3)
    cv2.putText(img, "Initializing...", (190, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 60), 2)
    _, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpg.tobytes()


async def _mjpeg_frames(orch):
    """MJPEG multipart generator。

    _frame_loop 已在线程池中完成渲染+编码，存入 _processed_jpeg。
    这里只做轮询（10ms）+ 推送，无 CPU 开销，不再额外 to_thread。
    收到新帧（timestamp 变化）立即发出，消除固定 40ms 等待带来的相位抖动。
    """
    placeholder_jpg = _make_placeholder_jpg()
    last_ts: float = -1.0

    while True:
        jpg_bytes: Optional[bytes] = None
        ts: float = 0.0

        if orch:
            jpg_bytes, ts = orch.get_processed_jpeg()

        if jpg_bytes is None:
            jpg_bytes = placeholder_jpg

        if ts != last_ts:
            last_ts = ts
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + jpg_bytes + b"\r\n")

        await asyncio.sleep(0.010)  # 10ms 轮询，新帧到来时最多 10ms 延迟


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
