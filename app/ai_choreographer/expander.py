"""
Plan → Track 展开：把段落级 ChoreoPlan 展开成时间点级 ChoreoTrack。§5.3.5。
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.ai_choreographer.schema import ChoreoPlan, SegmentPlan, FxCommand


def expand_plan_to_track(plan: ChoreoPlan, feat: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    把 LLM 输出的段落级 plan 展开成逐时间点的 ChoreoTrack。
    每个 entry：{"t": float, "fx": str, **params}
    """
    beats = feat.get("beats", [])
    downbeats = feat.get("downbeats", [])
    onsets = feat.get("onsets", [])

    track: List[Dict[str, Any]] = []

    for seg in plan.segments:
        t0, t1 = seg.t_start, seg.t_end

        for fx_cmd in seg.fx:
            trigger = fx_cmd.trigger
            params = dict(fx_cmd.params)

            if trigger == "continuous":
                # 整段持续，发一个事件即可，FX Compositor 按时间范围匹配
                track.append({"t": t0, "t_end": t1, "fx": fx_cmd.id,
                               "continuous": True, **params})

            elif trigger == "beat":
                for t in beats:
                    if t0 <= t < t1:
                        track.append({"t": t, "fx": fx_cmd.id, **params})

            elif trigger == "downbeat":
                for t in downbeats:
                    if t0 <= t < t1:
                        track.append({"t": t, "fx": fx_cmd.id, **params})

            elif trigger == "onset":
                for t in onsets:
                    if t0 <= t < t1:
                        track.append({"t": t, "fx": fx_cmd.id, **params})

            elif trigger == "rms":
                # RMS 驱动的连续曲线
                rms_in_seg = [(rt, rv) for rt, rv in feat.get("rms", [])
                               if t0 <= rt < t1]
                track.append({"t": t0, "t_end": t1, "fx": fx_cmd.id,
                               "continuous": True, "rms_data": rms_in_seg, **params})

        # 转场
        if seg.transition_out:
            trans = seg.transition_out
            track.append({
                "t": t1 - trans.duration,
                "fx": trans.id,
                "duration": trans.duration,
                "transition": True,
            })

    return sorted(track, key=lambda x: x["t"])
