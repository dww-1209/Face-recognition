"""
模板生成策略协议。

5 个策略的统一接口，定义在 application 层以便各策略实现复用。
实际协议定义在 domain/interfaces.py 的 TemplateStrategy。
本模块提供公共辅助函数。
"""

import numpy as np

from face_recognition.domain.entities import FaceEncoding


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2 归一化，返回单位向量。"""
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        raise ValueError("向量范数接近零，无法归一化")
    return vector / norm


def encodings_to_matrix(encodings: list[FaceEncoding]) -> np.ndarray:
    """将编码列表转为 (N, 512) 矩阵。"""
    return np.stack([e.vector for e in encodings])
