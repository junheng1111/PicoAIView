"""
BeatEngine：离线分析音乐文件，输出 beats / downbeats / onsets / RMS + segments。
§5.1 + §5.1 增量（段落切分）实现。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import librosa
    _LIBROSA = True
except ImportError:
    _LIBROSA = False


def _rms_in_range(rms: np.ndarray, times: np.ndarray,
                  t0: float, t1: float) -> float:
    mask = (times >= t0) & (times < t1)
    vals = rms[mask]
    return float(vals.mean()) if len(vals) > 0 else 0.0


def _auto_label(rms_val: float) -> str:
    if rms_val < 0.25:
        return "low"
    if rms_val < 0.6:
        return "mid"
    return "high"


def analyze(path: str, cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    离线分析音乐文件。
    返回 dict：tempo / beats / downbeats / onsets / rms / segments / duration / path_hash
    若 librosa 不可用，返回带 "error" 键的空骨架。
    """
    if not _LIBROSA:
        return {"error": "librosa not available", "tempo": 120.0,
                "beats": [], "downbeats": [], "onsets": [], "rms": [],
                "segments": [], "duration": 0.0}

    # 缓存
    path_hash = hashlib.sha1(open(path, "rb").read(65536)).hexdigest()[:16]
    if cache_dir:
        cache_path = Path(cache_dir) / f"{path_hash}.beats.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

    import time as _time
    t0 = _time.time()
    print(f"[BeatEngine] 开始加载: {path}")
    y, sr = librosa.load(path, sr=11025, mono=True)
    duration = float(len(y) / sr)
    print(f"[BeatEngine] 加载完成 {duration:.1f}s, 耗时 {_time.time()-t0:.1f}s, 开始节拍分析...")

    t1 = _time.time()
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    print(f"[BeatEngine] 节拍完成 {_time.time()-t1:.1f}s, 开始 onset/rms...")

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time")
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.times_like(rms, sr=sr)
    print(f"[BeatEngine] onset/rms 完成, 开始段落切分...")

    # 归一化 RMS
    rms_max = float(rms.max()) if rms.max() > 0 else 1.0
    rms_norm = rms / rms_max

    downbeats = beats[::4] if len(beats) >= 4 else beats

    # 每 4s 一格的 RMS 摘要（给 LLM）
    rms_summary = []
    t = 0.0
    while t < duration:
        rms_summary.append(round(_rms_in_range(rms_norm, rms_times, t, t + 4.0), 3))
        t += 4.0

    # 段落切分
    segments = _detect_segments(y, sr, rms_norm, rms_times, duration)

    result = {
        "path_hash": path_hash,
        "tempo": float(np.squeeze(tempo)) if hasattr(tempo, "__len__") else float(tempo),
        "duration": duration,
        "beats": beats.tolist(),
        "downbeats": downbeats.tolist(),
        "onsets": onsets.tolist(),
        "rms": list(zip(rms_times.tolist(), rms_norm.tolist())),
        "rms_summary": rms_summary,
        "segments": segments,
    }

    print(f"[BeatEngine] 全部完成，总耗时 {_time.time()-t0:.1f}s")
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f)

    return result


def _detect_segments(y, sr, rms_norm, rms_times, duration: float) -> List[Dict]:
    """
    用 chroma + recurrence matrix 做结构切分（§5.1 增量）。
    输出：[{"t_start", "t_end", "energy", "label"}, ...]
    """
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        # 尝试 agglomerative 切分
        try:
            bounds = librosa.segment.agglomerative(chroma, k=None)
        except TypeError:
            # 某些 librosa 版本不支持 k=None，指定一个合理数量
            k = min(8, max(2, int(duration / 30)))
            bounds = librosa.segment.agglomerative(chroma, k=k)

        bound_times = librosa.frames_to_time(bounds, sr=sr)
        # 确保包含 0 和结尾
        bound_times = np.unique(np.concatenate([[0.0], bound_times, [duration]]))
    except Exception:
        # fallback：均等分段
        step = max(15.0, duration / 8)
        bound_times = np.arange(0.0, duration + step, step)
        bound_times[-1] = duration

    segment_labels = ["intro", "verse1", "chorus1", "verse2", "chorus2",
                      "bridge", "chorus3", "outro"]

    segments = []
    for i, (t0, t1) in enumerate(zip(bound_times[:-1], bound_times[1:])):
        rms_seg = _rms_in_range(rms_norm, rms_times, t0, t1)
        label = segment_labels[i] if i < len(segment_labels) else f"segment{i}"
        segments.append({
            "t_start": round(float(t0), 3),
            "t_end": round(float(t1), 3),
            "energy": round(rms_seg, 3),
            "label": _auto_label(rms_seg),
            "name": label,
        })

    return segments
