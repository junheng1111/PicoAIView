"""
WebSocket `/ws/control`：§7.2。
下行：status / beat / vision
上行：audio_clock / fx_override / manual
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Set

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState


class ConnectionManager:
    def __init__(self):
        self._connections: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def broadcast(self, msg: dict) -> None:
        dead = set()
        for ws in self._connections:
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(json.dumps(msg))
            except Exception:
                dead.add(ws)
        self._connections -= dead


manager = ConnectionManager()


async def ws_control_endpoint(websocket: WebSocket):
    """WebSocket handler，挂在 /ws/control。"""
    await manager.connect(websocket)
    from app.main import get_orchestrator

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")
            orch = get_orchestrator()

            if msg_type == "audio_clock":
                if orch:
                    orch.update_audio_clock(float(msg.get("t", 0.0)))

            elif msg_type == "fx_override":
                if orch and orch.compositor:
                    fx_id = msg.get("fx", "")
                    amp = float(msg.get("amp", 0.0))
                    if amp <= 0:
                        orch.compositor.clear_override(fx_id)
                    else:
                        orch.compositor.override_fx(fx_id, {"amp": amp})

            elif msg_type == "manual":
                # 手动命令：snapshot 等
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


async def broadcast_beat(t: float, kind: str) -> None:
    await manager.broadcast({"type": "beat", "t": t, "kind": kind})


async def broadcast_status(fps: float, lat_ms: float,
                           audio_clock: float, bpu_stats: dict,
                           active_fx: str = "", active_theme: str = "") -> None:
    await manager.broadcast({
        "type": "status",
        "fps": round(fps, 1),
        "lat_ms": round(lat_ms),
        "audio_clock": round(audio_clock, 3),
        "audio_t": round(audio_clock, 3),
        "subjects": bpu_stats.get("subjects", 0),
        "fx": active_fx,
        "theme": active_theme,
        "bpu": bpu_stats,
    })


async def broadcast_vision(vs) -> None:
    from app.vision.bpu_runner import BpuRunner
    await manager.broadcast({
        "type": "vision",
        "t": round(vs.timestamp, 3),
        "subjects": vs.subject_count,
        "primary_id": vs.primary_id,
    })
