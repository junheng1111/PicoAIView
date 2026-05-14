"""
System + User prompt 模板。§5.3.3。
"""

import json
from typing import Any, Dict, List

SYSTEM_PROMPT = """\
你是一名 MV / Live 视觉导演 AI，名叫 PicoClawDirector。
你的任务：根据一段音乐的客观特征，从给定的特效库中挑选并编排镜头方案。

【硬约束 — 违反则输出会被判 invalid 并重试】
1. 输出必须严格符合 ChoreoPlan schema（由 tool schema 约束）。
2. 只能使用 fx_catalog 中列出的 fx id，禁止编造。
3. params 必须落在 fx_catalog 给出的取值范围内。
4. 单段最多挂 4 个 fx，且延迟分级满足：
   - latency_tier=light  不限数量
   - latency_tier=medium ≤ 2 个 / 段
   - latency_tier=heavy  ≤ 1 个 / 段
5. 风格化类（category=stylization）每段最多 1 个，与所有 heavy 互斥。
6. BPU 重型推理（depends_on 含 seg / fast_depth / face_landmark）：
   - 不能在相邻两段都使用同一模型；连续使用同一重型模型不超过 60 秒。
7. 排除清单中的 fx（标 excluded=true）不能出现在 plan 中。

【软偏好】
1. 副歌 / drop 段：优先选 ai_aware 类 fx + 1 个 glitch shader。
2. intro / outro：调色（3d_lut、duotone、vignette）+ 1 个轻几何。
3. bridge：时间域（echo_trails）+ depth_dof，制造反差。
4. 静音段（energy=low）：fx 总数 ≤ 1。
"""


def build_user_prompt(feat: Dict[str, Any], fx_catalog: List[Dict],
                      style: str = "energetic") -> str:
    bpm = feat.get("tempo", 120)
    duration = feat.get("duration", 0)
    beats = feat.get("beats", [])
    downbeats = feat.get("downbeats", [])
    onsets = feat.get("onsets", [])
    onset_density = round(len(onsets) / max(duration, 1), 2)
    rms_summary = feat.get("rms_summary", [])
    segments = feat.get("segments", [])

    segments_json = json.dumps(
        [{"t": [s["t_start"], s["t_end"]], "name": s.get("name", ""),
          "energy": s["label"]}
         for s in segments],
        ensure_ascii=False, indent=None
    )

    catalog_json = json.dumps(fx_catalog, ensure_ascii=False, indent=None)

    return f"""\
## 音频特征
- 时长: {duration:.1f}s
- BPM: {bpm:.1f}
- 主拍数: {len(beats)}, downbeat 数: {len(downbeats)}, onset 平均密度: {onset_density}/s
- RMS 曲线（每 4s 一格 mean，0-1 归一化）: {rms_summary}
- 段落（自动检测）: {segments_json}

## 可用特效库（自动从 FX_REGISTRY 序列化）
{catalog_json}

## 用户风格倾向
"{style}"

## 任务
为整首曲目输出 ChoreoPlan，重点考虑：
- 段落能量与 fx 强度匹配
- 转场点（段落边界）的 transition_out
- continuous 类 fx 用整段时间，trigger=beat/downbeat/onset 类只标 trigger 字段
- music_id 字段填 "{feat.get("path_hash", "unknown")}"
"""


# ChoreoPlan JSON Schema（用于 Claude tool / OpenAI json_schema）
CHOREO_PLAN_SCHEMA = {
    "type": "object",
    "required": ["version", "music_id", "style", "segments"],
    "properties": {
        "version": {"type": "string"},
        "music_id": {"type": "string"},
        "style": {"type": "string"},
        "director_notes": {"type": "string"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "t_start", "t_end", "fx"],
                "properties": {
                    "name": {"type": "string"},
                    "t_start": {"type": "number"},
                    "t_end": {"type": "number"},
                    "fx": {
                        "type": "array",
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "required": ["id", "trigger"],
                            "properties": {
                                "id": {"type": "string"},
                                "trigger": {
                                    "type": "string",
                                    "enum": ["beat", "downbeat", "onset",
                                             "continuous", "rms"],
                                },
                                "params": {"type": "object"},
                            },
                        },
                    },
                    "transition_out": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "duration": {"type": "number"},
                        },
                    },
                },
            },
        },
    },
}
