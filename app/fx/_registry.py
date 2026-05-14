"""
FX 注册表：装饰器注册 + fx_catalog 序列化（供 LLM Choreographer 使用）。

使用方式：
    from app.fx._registry import register_fx, FX_REGISTRY, build_fx_catalog

    @register_fx(
        fx_id="pan",
        category="digital_ptz",
        latency_tier="light",
        depends_on=[],
        params={"amp": [0.0, 0.1]},
        description="数字左右摇移，amp=0.1 表示最大偏移画面宽度 10%",
    )
    def apply_pan(frame, amp=0.03, **_):
        ...
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional

# latency_tier 映射到约束
TIER_BUDGET = {
    "light":  0,   # 无限叠加
    "medium": 2,   # 每段 ≤2
    "heavy":  1,   # 每段 ≤1
}

EXCLUDED_FX: List[str] = ["style_transfer", "super_resolution"]


class FxMeta:
    __slots__ = ("fx_id", "category", "latency_tier", "depends_on",
                 "params", "description", "fn")

    def __init__(
        self,
        fx_id: str,
        category: str,
        latency_tier: str,
        depends_on: List[str],
        params: Dict[str, Any],
        description: str,
        fn: Callable,
    ):
        self.fx_id = fx_id
        self.category = category
        self.latency_tier = latency_tier
        self.depends_on = depends_on
        self.params = params
        self.description = description
        self.fn = fn

    def catalog_entry(self) -> Dict[str, Any]:
        return {
            "id": self.fx_id,
            "category": self.category,
            "latency_tier": self.latency_tier,
            "depends_on": self.depends_on,
            "params": self.params,
            "description": self.description,
        }


# 全局注册表：fx_id → FxMeta
FX_REGISTRY: Dict[str, FxMeta] = {}


def register_fx(
    fx_id: str,
    category: str,
    latency_tier: str = "light",
    depends_on: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    description: str = "",
):
    """装饰器：把函数注册进 FX_REGISTRY。"""
    if depends_on is None:
        depends_on = []
    if params is None:
        params = {}

    def decorator(fn: Callable) -> Callable:
        if fx_id in EXCLUDED_FX:
            return fn  # 排除清单中的 fx 不注册

        meta = FxMeta(
            fx_id=fx_id,
            category=category,
            latency_tier=latency_tier,
            depends_on=depends_on,
            params=params,
            description=description,
            fn=fn,
        )
        FX_REGISTRY[fx_id] = meta

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def build_fx_catalog(include_excluded: bool = False) -> List[Dict[str, Any]]:
    """将 FX_REGISTRY 序列化为 LLM 可读的 fx_catalog 列表。"""
    entries = [meta.catalog_entry() for meta in FX_REGISTRY.values()]
    if include_excluded:
        for ex_id in EXCLUDED_FX:
            entries.append({"id": ex_id, "excluded": True})
    return entries


def apply_fx(fx_id: str, frame, vision_state=None, **params):
    """按 fx_id 执行特效，找不到时原样返回 frame。"""
    meta = FX_REGISTRY.get(fx_id)
    if meta is None:
        return frame
    try:
        if meta.depends_on and vision_state is not None:
            return meta.fn(frame, vision_state=vision_state, **params)
        return meta.fn(frame, **params)
    except Exception as e:
        print(f"[FX] apply_fx '{fx_id}' 失败: {e}")
        return frame
