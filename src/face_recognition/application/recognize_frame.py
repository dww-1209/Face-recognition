"""RecognizeFrame 用例：编排单帧的检测→跟踪→按需识别流水线。"""

import numpy as np

from face_recognition.application.template_matrix import TemplateMatrixService
from face_recognition.domain.entities import DetectedFace
from face_recognition.domain.interfaces import FacePipeline
from face_recognition.infrastructure.iou_tracker import IoUTracker, Track


class RecognizeFrame:
    """实时帧识别用例：串起 pipeline / tracker / matrix，产出已识别的 tracks。

    和 M1 RecognizeFace 的区别：
      - RecognizeFace：单张图 → 单个 RecognitionResult（CLI/评估用）
      - RecognizeFrame：单帧 → 多 track 列表（实时摄像头用）
        多了"跟踪 + 按需识别 + 重识别节流"逻辑
    """

    def __init__(
        self,
        pipeline: FacePipeline,
        tracker: IoUTracker,
        template_matrix: TemplateMatrixService,
        threshold: float = 0.30,
        recheck_interval: int = 30,
    ):
        self.pipeline = pipeline
        self.tracker = tracker
        self.matrix = template_matrix
        self.threshold = threshold
        self.recheck_interval = recheck_interval
        self._frame_count = 0

    def process_frame(self, frame: np.ndarray) -> list[Track]:
        """处理一帧：检测 → 跟踪 → 按需识别 → 返回 tracks。"""
        self._frame_count += 1

        # 第 1 步：检测 + 编码（带 bbox）
        faces: list[DetectedFace] = self.pipeline.detect_and_encode(frame)
        boxes = [f.bbox for f in faces]

        # 第 2 步：IoU 跟踪——把检测框匹配到已有 track
        tracks = self.tracker.update(boxes)

        # 第 3 步：按需识别——只对新 track 或到达重识别窗口的 track 跑矩阵查询
        for t in tracks:
            if not t.needs_recognition:
                continue
            # 找到这个 track 对应的 DetectedFace（通过 bbox）
            df = self._find_face_for_track(t, faces)
            if df is None:
                continue

            pid, sim = self.matrix.query(df.encoding.vector)
            if sim >= self.threshold:
                t.person_id = pid
                t.similarity = sim
            else:
                t.person_id = None
                t.similarity = sim
            t.needs_recognition = False

        # 第 4 步：定期重识别——到达 recheck_interval 的帧触发所有 track 重识别
        if self._frame_count % self.recheck_interval == 0:
            for t in tracks:
                if t.missing_frames > 0:
                    continue  # 丢失中的 track 不重识别
                df = self._find_face_for_track(t, faces)
                if df is None:
                    continue
                pid, sim = self.matrix.query(df.encoding.vector)
                if sim >= self.threshold:
                    t.person_id = pid
                    t.similarity = sim
                else:
                    t.person_id = None
                    t.similarity = sim

        return tracks

    @staticmethod
    def _find_face_for_track(
        track: Track, faces: list[DetectedFace]
    ) -> DetectedFace | None:
        """通过 bbox 匹配找到 track 对应的 DetectedFace。"""
        for f in faces:
            if f.bbox == track.bbox:
                return f
        return None
