"""
ChoreoPlan Pydantic 模型（输出契约）。§5.3.4。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator


class FxCommand(BaseModel):
    id: str
    trigger: str = "beat"   # beat | downbeat | onset | continuous | rms
    params: Dict[str, Any] = {}


class Transition(BaseModel):
    id: str = "cut"
    duration: float = 0.1


class SegmentPlan(BaseModel):
    name: str
    t_start: float
    t_end: float
    fx: List[FxCommand] = []
    transition_out: Optional[Transition] = None

    @field_validator("fx")
    @classmethod
    def validate_fx_count(cls, v):
        if len(v) > 4:
            raise ValueError(f"单段最多 4 个 fx，当前 {len(v)}")
        return v


class ChoreoPlan(BaseModel):
    version: str = "1.0"
    music_id: str
    style: str = "energetic"
    director_notes: str = ""
    segments: List[SegmentPlan]
