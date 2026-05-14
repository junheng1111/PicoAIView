"""WebRTC SDP 路由。"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.media.webrtc import WebRTCSession

router = APIRouter()

_sessions: dict = {}


class RTCOffer(BaseModel):
    sdp: str
    type: str
    session_id: str = "default"


@router.post("/rtc/offer")
async def rtc_offer(offer: RTCOffer):
    from app.main import get_orchestrator
    orch = get_orchestrator()

    if not orch:
        return {"error": "orchestrator not ready"}

    sid = offer.session_id
    if sid not in _sessions:
        _sessions[sid] = WebRTCSession(orch.compositor, orch.camera)

    answer = await _sessions[sid].handle_offer(offer.sdp, offer.type)
    return answer
