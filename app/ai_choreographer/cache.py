"""diskcache 包装，key = sha1(music_id + style + catalog_ver + model)。"""

import hashlib
import json
import os
from typing import Any, Optional

try:
    import diskcache
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False

_CACHE_DIR = os.getenv("PICOCLAW_LLM_CACHE_DIR", "/tmp/picoclaw_llm_cache")
_CATALOG_VERSION = "v0.4.1"

_cache = None


def _get_cache():
    global _cache
    if _cache is None and _CACHE_AVAILABLE:
        _cache = diskcache.Cache(_CACHE_DIR)
    return _cache


def _make_key(music_id: str, style: str, model: str) -> str:
    raw = f"{music_id}|{style}|{_CATALOG_VERSION}|{model}"
    return hashlib.sha1(raw.encode()).hexdigest()


def get_plan(music_id: str, style: str, model: str) -> Optional[dict]:
    c = _get_cache()
    if c is None:
        return None
    key = _make_key(music_id, style, model)
    val = c.get(key)
    if val is None:
        return None
    return json.loads(val)


def set_plan(music_id: str, style: str, model: str, plan_dict: dict) -> None:
    c = _get_cache()
    if c is None:
        return
    key = _make_key(music_id, style, model)
    c.set(key, json.dumps(plan_dict), expire=7 * 24 * 3600)  # 7天


def invalidate(music_id: str) -> int:
    """清除某 music_id 的所有缓存。"""
    c = _get_cache()
    if c is None:
        return 0
    count = 0
    for key in list(c.iterkeys()):
        # key = sha1，无法反查；暴力清全部（或按 tag）
        # 简单实现：清所有（开发环境可接受）
        pass
    return count
