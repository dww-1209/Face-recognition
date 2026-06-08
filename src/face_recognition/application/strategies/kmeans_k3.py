from datetime import UTC, datetime

import numpy as np
from sklearn.cluster import KMeans

from face_recognition.application.strategies.base import encodings_to_matrix, l2_normalize
from face_recognition.domain.entities import FaceEncoding, Template


class KMeansK3Strategy:
    """策略 4：KMeans 聚类 K=3，每簇质心作为模板（共 3 个模板）。"""

    name = "kmeans_k3"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def build(self, encodings: list[FaceEncoding]) -> list[Template]:
        if len(encodings) < 3:
            # 少于 3 张则全部存为独立模板
            return [
                Template(
                    encoding=e,
                    source=f"kmeans_fallback_raw_{i}",
                    created_at=datetime.now(UTC),
                )
                for i, e in enumerate(encodings)
            ]

        matrix = encodings_to_matrix(encodings)  # (N, 512)
        kmeans = KMeans(n_clusters=3, random_state=self.seed, n_init=10)
        kmeans.fit(matrix)

        templates = []
        for i, centroid in enumerate(kmeans.cluster_centers_):
            centroid = l2_normalize(centroid)
            count = int(np.sum(kmeans.labels_ == i))
            templates.append(
                Template(
                    encoding=FaceEncoding(vector=centroid.astype(np.float32)),
                    source=f"kmeans_centroid_{i}_n{count}",
                    created_at=datetime.now(UTC),
                )
            )
        return templates
