"""
从 FX_REGISTRY 反射出 fx_catalog 给 LLM。
可选：过滤掉 BPU 当前不可用的 depends_on 模型。
"""

from typing import Any, Dict, List, Optional

from app.fx._registry import FX_REGISTRY, EXCLUDED_FX, build_fx_catalog


def get_catalog(available_models: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    返回 LLM 可用的 fx_catalog。

    available_models: 当前 BPU 已加载/可用的模型列表，
                      如 ["yolov8s", "yolov8s_pose"]。
                      若为 None，则包含所有 fx（开发模式）。
    """
    catalog = build_fx_catalog(include_excluded=False)

    if available_models is not None:
        av_set = set(available_models)
        filtered = []
        for entry in catalog:
            deps = entry.get("depends_on", [])
            # 无依赖 or 所有依赖都可用
            if not deps or all(d in av_set for d in deps):
                filtered.append(entry)
        return filtered

    return catalog
