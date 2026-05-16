"""
AI Choreographer 编排入口。§5.3.1 Pipeline。
orchestrate(music_id, feat, style) → (ChoreoPlan, ChoreoTrack, source)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from app.ai_choreographer.schema import ChoreoPlan
from app.ai_choreographer.provider import make_provider, LLMProvider
from app.ai_choreographer.prompts import SYSTEM_PROMPT, build_user_prompt, CHOREO_PLAN_SCHEMA
from app.ai_choreographer.catalog import get_catalog
from app.ai_choreographer.expander import expand_plan_to_track
from app.ai_choreographer import cache as choreo_cache
from app.beat.rule_choreo import build_choreo_rulebased

_provider: Optional[LLMProvider] = None


def _get_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        _provider = make_provider()
    return _provider


async def orchestrate(
    music_id: str,
    feat: Dict[str, Any],
    style: str = "energetic",
    available_models: Optional[List[str]] = None,
    force_refresh: bool = False,
    model_override: Optional[str] = None,
    audio_path: Optional[str] = None,
) -> Tuple[Optional[ChoreoPlan], List[Dict], str]:
    """
    主编排入口。返回 (plan, track, source)
    source = "ai" | "rule"
    """
    model_name = model_override or os.getenv("PICOCLAW_LLM_MODEL", "claude-sonnet-4-6")

    # 缓存命中
    if not force_refresh:
        cached = choreo_cache.get_plan(music_id, style, model_name)
        if cached:
            try:
                plan = ChoreoPlan.model_validate(cached)
                track = expand_plan_to_track(plan, feat)
                return plan, track, "ai"
            except Exception:
                pass

    # 调用 LLM
    catalog = get_catalog(available_models)
    user_prompt = build_user_prompt(feat, catalog, style)
    schema = CHOREO_PLAN_SCHEMA

    provider = _get_provider()
    print(f"[AIChoreographer] 调用 LLM provider={provider.__class__.__name__} style={style!r} audio={audio_path}")
    plan_dict = None
    try:
        plan_dict = await provider.complete_json(SYSTEM_PROMPT, user_prompt, schema,
                                                 audio_path=audio_path)
        print(f"[AIChoreographer] LLM 返回成功，segments={len(plan_dict.get('segments', []))}")
    except Exception as e:
        print(f"[AIChoreographer] LLM 调用失败，走 fallback: {e}")

    if plan_dict is not None:
        try:
            plan_dict["music_id"] = music_id
            plan = ChoreoPlan.model_validate(plan_dict)
            track = expand_plan_to_track(plan, feat)
            choreo_cache.set_plan(music_id, style, model_name, plan_dict)
            print(f"[AIChoreographer] ✓ AI 编排完成 source=ai track_len={len(track)}")
            return plan, track, "ai"
        except Exception as e:
            print(f"[AIChoreographer] Plan 校验失败: {e}")
            print(f"[AIChoreographer] 原始 plan_dict keys: {list(plan_dict.keys()) if plan_dict else 'None'}")
            segs = plan_dict.get('segments') if plan_dict else None
            print(f"[AIChoreographer] segments type={type(segs).__name__} sample={str(segs)[:300] if segs else 'None'}")

    # Fallback：规则版
    print(f"[AIChoreographer] ⚠ 使用规则 fallback source=rule")
    track = build_choreo_rulebased(feat)
    return None, track, "rule"
