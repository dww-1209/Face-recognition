"""InsightFace buffalo_l 一站式管线：检测 + 关键点对齐 + 编码。"""

import logging

import numpy as np

from face_recognition.domain.entities import DetectedFace, FaceEncoding
from face_recognition.domain.errors import MultipleFacesError, NoFaceError

logger = logging.getLogger(__name__)


class InsightFacePipeline:
    """封装 InsightFace buffalo_l，实现 FacePipeline 协议。"""

    def __init__(
        self,
        pack: str = "buffalo_l",
        ctx_id: int = 0,
        det_size: tuple[int, int] = (640, 640),
    ):
        import insightface

        self.pack = pack
        self.model_version = pack
        self.app = insightface.app.FaceAnalysis(
            name=pack,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        logger.info(
            f"InsightFace {pack} 加载完成 (ctx_id={ctx_id}, det_size={det_size})"
        )

    def encode(self, image: np.ndarray) -> list[FaceEncoding]:
        """
        返回图中所有人脸的编码（已 L2 归一化）。
        BGR 格式输入，输出按检测置信度降序排列。
        """
        faces = self.app.get(image)
        return [
            FaceEncoding(
                vector=self._normalize(f.embedding.astype(np.float32)),
                model_version=self.pack,
            )
            for f in faces
        ]

    def encode_single(self, image: np.ndarray) -> FaceEncoding:
        """要求图中恰好 1 张脸。"""
        faces = self.app.get(image)
        if len(faces) == 0:
            raise NoFaceError("未检测到人脸")
        if len(faces) > 1:
            raise MultipleFacesError(count=len(faces))
        return FaceEncoding(
            vector=self._normalize(faces[0].embedding.astype(np.float32)),
            model_version=self.pack,
        )

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """L2 归一化：InsightFace 原始 embedding 未归一化，需手动归一化。"""
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            raise ValueError("向量范数接近零")
        return vector / norm

    def detect_and_encode(self, image: np.ndarray) -> list[DetectedFace]:
        """同 encode，但保留 InsightFace 的 bbox（int4 像素坐标）。"""
        faces = self.app.get(image)
        out: list[DetectedFace] = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int).tolist()
            enc = FaceEncoding(
                vector=self._normalize(f.embedding.astype(np.float32)),
                model_version=self.pack,
            )
            out.append(DetectedFace(bbox=(x1, y1, x2, y2), encoding=enc))
        return out
