"""IoU 跟踪器：帧间人脸框匹配，实现"识别按需触发"。"""

from dataclasses import dataclass


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    person_id: str | None = None
    similarity: float = 0.0
    missing_frames: int = 0
    needs_recognition: bool = True


class IoUTracker:
    """
    基于 IoU 的简单多目标跟踪。

    每帧传入检测框列表，返回更新后的 track 列表。
    新框分配新 track_id 并标记需要识别；
    沿用框复用上次身份，不重新识别。
    """

    def __init__(self, iou_threshold: float = 0.5, max_missing: int = 15):
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.tracks: list[Track] = []
        self._next_id = 0

    def update(self, boxes: list[tuple[int, int, int, int]]) -> list[Track]:
        """传入新一帧的检测框列表，返回所有活跃 track。"""
        matched_track_indices: set[int] = set()
        matched_box_indices: set[int] = set()

        # IoU 贪心匹配
        for ti, track in enumerate(self.tracks):
            best_iou = 0.0
            best_bi = -1
            for bi, box in enumerate(boxes):
                if bi in matched_box_indices:
                    continue
                iou = self._iou(track.bbox, box)
                if iou > best_iou:
                    best_iou = iou
                    best_bi = bi
            if best_iou >= self.iou_threshold and best_bi >= 0:
                track.bbox = boxes[best_bi]
                track.missing_frames = 0
                track.needs_recognition = False
                matched_track_indices.add(ti)
                matched_box_indices.add(best_bi)

        # 未匹配的 track 计 missing
        for ti, track in enumerate(self.tracks):
            if ti not in matched_track_indices:
                track.missing_frames += 1

        # 新框创建新 track
        for bi, box in enumerate(boxes):
            if bi not in matched_box_indices:
                self.tracks.append(
                    Track(
                        track_id=self._next_id,
                        bbox=box,
                        needs_recognition=True,
                    )
                )
                self._next_id += 1

        # 移除消失过久的 track
        self.tracks = [t for t in self.tracks if t.missing_frames < self.max_missing]

        return list(self.tracks)

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)
