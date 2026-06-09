from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from face_recognition.domain.entities import FaceEncoding
from face_recognition.domain.errors import NoFaceError
from face_recognition.evaluation.embedder import (
    encode_image_paths,
    encode_lfw_images,
)
from face_recognition.evaluation.lfw_loader import LfwImage
from face_recognition.evaluation.types import EvalEncoding


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_encode_image_paths_returns_one_eval_encoding_per_path(tmp_path: Path):
    """encode_image_paths 应该跳过无脸/读取失败，返回有效编码列表。"""
    # 造 3 个**真实可解码**的 jpg —— b"fake" 无法被 cv2.imread 解码,会让
    # encode_image_paths 在 `if img is None: continue` 处全部跳过,
    # 测试就会得到空列表(假阴性绿灯,看似过实则没测到主流程)。
    # 内容随便填个 10×10 黑图就行,pipeline 后面被 mock 不看像素。
    import cv2

    paths = []
    blank = np.zeros((10, 10, 3), dtype=np.uint8)
    for i in range(3):
        p = tmp_path / f"alice_{i}.jpg"
        cv2.imwrite(str(p), blank)
        paths.append(p)

    # mock pipeline：前两张正常返回向量，第三张抛 NoFaceError 模拟"无脸"
    # —— M1 约定 FacePipeline.encode_single 检测不到脸时**抛异常**而非返回 None
    pipeline = MagicMock()
    fe1 = FaceEncoding(vector=_unit_vec(0), model_version="buffalo_l")
    fe2 = FaceEncoding(vector=_unit_vec(1), model_version="buffalo_l")
    # side_effect 接列表时按调用顺序逐项产出；元素若是 Exception 实例则被 raise
    # 这是 MagicMock 的特殊语义（M1 测试已用过同款套路）
    pipeline.encode_single.side_effect = [fe1, fe2, NoFaceError("无脸")]

    result = encode_image_paths(pipeline, paths, person_id="alice")
    assert len(result) == 2  # 第三张被跳过
    assert all(isinstance(r, EvalEncoding) for r in result)
    assert all(r.person_id == "alice" for r in result)


def test_encode_lfw_images_uses_person_name_as_id():
    """LFW 路径下 person_id 来自 person_name，image_path 留 'lfw://<name>'。"""
    pipeline = MagicMock()
    pipeline.encode_single.return_value = FaceEncoding(
        vector=_unit_vec(0), model_version="buffalo_l"
    )
    lfw_imgs = [
        LfwImage(image=np.zeros((250, 250, 3), dtype=np.uint8), person_name="George_W_Bush"),
    ]
    result = encode_lfw_images(pipeline, lfw_imgs)
    assert len(result) == 1
    assert result[0].person_id == "George_W_Bush"
    assert result[0].image_path.startswith("lfw://")


def test_encode_lfw_images_skips_no_face():
    """无脸 LFW 图（罕见但要防御）：抛 NoFaceError 时跳过。

    注意：encode_lfw_images 有 10% 跳过率守卫——如果超过 10% 的图编码失败会抛
    RuntimeError（防止系统性 bug 静默污染评估指标）。所以测试里造 10 张图、只有
    1 张失败（10% 刚好未超阈值），验证单张失败被安静跳过。
    """
    pipeline = MagicMock()
    ok = FaceEncoding(vector=_unit_vec(0), model_version="buffalo_l")
    # 10 张图：第一张抛异常，后 9 张正常 → 跳过率 10% ≤ 阈值 10%
    pipeline.encode_single.side_effect = [NoFaceError("无脸")] + [ok] * 9
    lfw_imgs = [
        LfwImage(image=np.zeros((250, 250, 3), dtype=np.uint8), person_name=f"P{i}")
        for i in range(10)
    ]
    result = encode_lfw_images(pipeline, lfw_imgs)
    assert len(result) == 9  # 1 张被跳过
