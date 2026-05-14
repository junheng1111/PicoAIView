"""简单的基于 IoU 的多目标跟踪（ByteTrack 简化版），用于多人场景主体切换。"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.vision.bpu_runner import Detection, COCO_PERSON_CLASS_ID


@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    age: int = 0          # 连续匹配帧数
    miss: int = 0         # 连续丢失帧数


def _iou(a: Tuple, b: Tuple) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class SimpleTracker:
    """基于 IoU 贪心匹配的多目标跟踪器。"""

    def __init__(self, iou_threshold: float = 0.3, max_miss: int = 5):
        self._iou_th = iou_threshold
        self._max_miss = max_miss
        self._tracks: List[Track] = []
        self._next_id = 0

    def update(self, detections: List[Detection]) -> List[Track]:
        persons = [d for d in detections if d.class_id == COCO_PERSON_CLASS_ID]

        # 构建代价矩阵
        if self._tracks and persons:
            cost = np.zeros((len(self._tracks), len(persons)))
            for i, t in enumerate(self._tracks):
                for j, d in enumerate(persons):
                    cost[i, j] = 1.0 - _iou(t.bbox, d.bbox)

            matched_t, matched_d = set(), set()
            # 贪心匹配
            flat = np.argsort(cost.ravel())
            for idx in flat:
                i, j = divmod(int(idx), len(persons))
                if i in matched_t or j in matched_d:
                    continue
                if cost[i, j] < (1.0 - self._iou_th):
                    self._tracks[i].bbox = persons[j].bbox
                    self._tracks[i].confidence = persons[j].confidence
                    self._tracks[i].age += 1
                    self._tracks[i].miss = 0
                    matched_t.add(i)
                    matched_d.add(j)

            # 未匹配检测 → 新 track
            for j, d in enumerate(persons):
                if j not in matched_d:
                    self._tracks.append(Track(
                        track_id=self._next_id,
                        bbox=d.bbox,
                        confidence=d.confidence,
                    ))
                    self._next_id += 1

            # 未匹配 track → 增加 miss
            for i, t in enumerate(self._tracks):
                if i not in matched_t:
                    t.miss += 1
        else:
            # 全部新增
            for d in persons:
                self._tracks.append(Track(
                    track_id=self._next_id,
                    bbox=d.bbox,
                    confidence=d.confidence,
                ))
                self._next_id += 1

        # 清理长时间丢失的 track
        self._tracks = [t for t in self._tracks if t.miss <= self._max_miss]
        return list(self._tracks)

    def primary_track(self) -> Optional[Track]:
        """返回面积最大的 track（主体）。"""
        active = [t for t in self._tracks if t.miss == 0]
        if not active:
            return None
        return max(active, key=lambda t: t.bbox[2] * t.bbox[3])
