from datetime import UTC, datetime

import numpy as np

from face_recognition.application.strategies.base import encodings_to_matrix, l2_normalize
from face_recognition.domain.entities import FaceEncoding, Template


class MeanAllStrategy:
    """策略 2：全部编码取算术平均（再 L2 归一化），生成 1 个模板。"""

    name = "mean_all"

    def build(self, encodings: list[FaceEncoding]) -> list[Template]:
        if not encodings:
            raise ValueError("编码列表为空，无法计算均值")
        matrix = encodings_to_matrix(encodings)  # (N, 512)
        mean = np.mean(matrix, axis=0)
        mean = l2_normalize(mean)
        return [
            Template(
                encoding=FaceEncoding(vector=mean.astype(np.float32)),
                source="mean_all",
                created_at=datetime.now(UTC),
            )
        ]
