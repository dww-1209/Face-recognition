from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

_EMBED_DIM = 512
_NORM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class FaceEncoding:
    """ArcFace 输出的 512 维向量，已 L2 归一化。"""

    vector: np.ndarray  # shape=(512,), dtype=float32, ||v||=1
    model_version: str = "buffalo_l"

    def __post_init__(self) -> None:
        if self.vector.ndim != 1 or self.vector.shape[0] != _EMBED_DIM:
            raise ValueError(f"期望 {_EMBED_DIM} 维向量，收到 shape={self.vector.shape}")
        if self.vector.dtype != np.float32:
            object.__setattr__(self, "vector", self.vector.astype(np.float32))
        norm = float(np.linalg.norm(self.vector))
        if abs(norm - 1.0) > _NORM_TOLERANCE:
            raise ValueError(f"向量未 L2 归一化: ||v||={norm:.6f}，期望 1.0±{_NORM_TOLERANCE}")

    def cosine_similarity(self, other: "FaceEncoding") -> float:
        return float(np.dot(self.vector, other.vector))


@dataclass(frozen=True)
class Template:
    """单条模板向量（一个 Person 可能有多条 Template）。"""

    encoding: FaceEncoding
    source: str  # "mean" / "subset_0_mean" / "kmeans_centroid_1" / "raw_0001.jpg"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class Person:
    """库内人员（领域聚合根）。"""

    person_id: str
    display_name: str
    templates: tuple[Template, ...]

    @property
    def template_count(self) -> int:
        return len(self.templates)


@dataclass(frozen=True)
class RecognitionResult:
    """识别用例的输出。"""

    person_id: str | None  # None 表示未识别（库外陌生人）
    similarity: float
    threshold: float
    matched_template_source: str | None = None  # 匹配到哪个模板


@dataclass(frozen=True)
class DetectedFace:
    """一张人脸在某帧中的位置 + 编码。

    实时场景使用：bbox 用于画框，encoding 用于查询模板矩阵。
    M1 单图注册用 FaceEncoding 足够，不必关心 bbox。
    """
    # bbox 形式：(x1, y1, x2, y2) 整数像素坐标，左上 + 右下
    # 用 tuple 而非 list：四个值一旦定下就不变，配合 frozen 保证不可变
    bbox: tuple[int, int, int, int]
    encoding: FaceEncoding
