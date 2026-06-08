from typing import Protocol

import numpy as np

from face_recognition.domain.entities import FaceEncoding, Person, Template


class FacePipeline(Protocol):
    """一站式人脸管线：检测 + 对齐 + 编码。"""

    model_version: str

    def encode(self, image: np.ndarray) -> list[FaceEncoding]:
        """返回图中所有人脸的编码（可能 0 个或多个）。"""
        ...

    def encode_single(self, image: np.ndarray) -> FaceEncoding:
        """要求图中恰好 1 张脸；0 张或多张抛 NoFaceError / MultipleFacesError。"""
        ...


class PersonRepository(Protocol):
    """人员向量库的抽象。"""

    def add(self, person: Person) -> None:
        """添加人员（如已存在则先删旧再插新，幂等）。"""
        ...

    def get(self, person_id: str) -> Person | None:
        """按 ID 查询人员。"""
        ...

    def remove(self, person_id: str) -> None:
        """按 ID 删除人员。"""
        ...

    def list_all(self) -> list[Person]:
        """列出所有库内人员。"""
        ...

    def all_templates_matrix(self) -> tuple[np.ndarray, list[str]]:
        """返回 (M, 512) 矩阵 + 长度 M 的 person_id 列表。"""
        ...


class TemplateStrategy(Protocol):
    """模板生成策略的统一接口。"""

    name: str

    def build(self, encodings: list[FaceEncoding]) -> list[Template]:
        """从一组编码生成模板列表。"""
        ...
