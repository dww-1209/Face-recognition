"""识别用例：对单张图片进行开放集人脸识别。"""

import logging

import numpy as np

from face_recognition.domain.entities import FaceEncoding, RecognitionResult
from face_recognition.domain.interfaces import FacePipeline, PersonRepository

logger = logging.getLogger(__name__)


class RecognizeFace:
    """识别用例：给定图片，返回最匹配的库内人员或 None（库外陌生人）。"""

    def __init__(
        self,
        pipeline: FacePipeline,
        repository: PersonRepository,
        threshold: float = 0.45,
    ):
        self.pipeline = pipeline
        self.repository = repository
        self.threshold = threshold
        # 缓存模板矩阵和 person_id 列表（注册后需刷新）
        self._templates_matrix: np.ndarray | None = None
        self._person_ids: list[str] | None = None

    def refresh_cache(self) -> None:
        """刷新模板矩阵缓存（增删人员后调用）。"""
        self._templates_matrix, self._person_ids = (
            self.repository.all_templates_matrix()
        )

    def _ensure_cache(self) -> None:
        if self._templates_matrix is None or self._person_ids is None:
            self.refresh_cache()

    def execute(self, image: np.ndarray) -> RecognitionResult:
        """
        对单张图片执行识别。

        Args:
            image: BGR 格式图像（np.ndarray）

        Returns:
            RecognitionResult，person_id 为 None 表示库外陌生人
        """
        encodings = self.pipeline.encode(image)
        if not encodings:
            return RecognitionResult(
                person_id=None,
                similarity=0.0,
                threshold=self.threshold,
            )

        # 取第一张脸的编码
        query = encodings[0]
        return self._match(query)

    def execute_single(self, image: np.ndarray) -> RecognitionResult:
        """
        要求图中恰好一张脸，否则抛异常。
        """
        query = self.pipeline.encode_single(image)
        return self._match(query)

    def _match(self, query: FaceEncoding) -> RecognitionResult:
        """矩阵乘法一次得到所有模板的相似度，取最大值。"""
        self._ensure_cache()
        assert self._templates_matrix is not None
        assert self._person_ids is not None

        if self._templates_matrix.shape[0] == 0:
            return RecognitionResult(
                person_id=None,
                similarity=0.0,
                threshold=self.threshold,
            )

        # (M, 512) @ (512,) → (M,)
        similarities = self._templates_matrix @ query.vector
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= self.threshold:
            return RecognitionResult(
                person_id=self._person_ids[best_idx],
                similarity=best_sim,
                threshold=self.threshold,
            )
        else:
            return RecognitionResult(
                person_id=None,
                similarity=best_sim,
                threshold=self.threshold,
            )
