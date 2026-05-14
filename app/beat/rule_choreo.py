"""
规则版编排（fallback）：§5.2 实现。
仅当 LLM API 不可用时使用。
"""

from typing import Any, Dict, List


def build_choreo_rulebased(feat: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deterministic fallback。不依赖 LLM。"""
    track: List[Dict[str, Any]] = []

    for t in feat.get("beats", []):
        track.append({"t": t, "fx": "pan", "v": "sin", "amp": 0.03})

    for t in feat.get("downbeats", []):
        track.append({"t": t, "fx": "zoom_pulse", "amp": 0.08})
        track.append({"t": t, "fx": "flash", "amp": 0.3})

    for t in feat.get("onsets", []):
        track.append({"t": t, "fx": "shake", "px": 4})

    # RMS → 亮度曲线（连续，贯穿全曲）
    if feat.get("rms"):
        track.append({"t": 0.0, "t_end": float("inf"), "fx": "brightness_curve",
                      "continuous": True, "k": 1.0})

    return sorted(track, key=lambda x: x["t"])
