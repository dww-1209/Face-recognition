import numpy as np
import pytest

from face_recognition.domain.entities import (
    FaceEncoding,
    Person,
    RecognitionResult,
    Template,
)


def _unit_vector(seed: int = 0) -> np.ndarray:
    """生成 L2 归一化随机 512 维向量。"""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_face_encoding_requires_512_dim():
    with pytest.raises(ValueError):
        FaceEncoding(vector=np.zeros(128, dtype=np.float32), model_version="buffalo_l")


def test_face_encoding_requires_l2_normalized():
    with pytest.raises(ValueError):
        FaceEncoding(vector=np.ones(512, dtype=np.float32), model_version="buffalo_l")


def test_face_encoding_cosine_similarity_self_is_one():
    enc = FaceEncoding(vector=_unit_vector(0), model_version="buffalo_l")
    assert enc.cosine_similarity(enc) == pytest.approx(1.0, abs=1e-6)


def test_face_encoding_cosine_similarity_orthogonal_is_zero():
    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(512, dtype=np.float32)
    v2[1] = 1.0
    e1 = FaceEncoding(vector=v1, model_version="buffalo_l")
    e2 = FaceEncoding(vector=v2, model_version="buffalo_l")
    assert e1.cosine_similarity(e2) == pytest.approx(0.0, abs=1e-6)


def test_face_encoding_is_frozen():
    enc = FaceEncoding(vector=_unit_vector(0), model_version="buffalo_l")
    with pytest.raises(Exception):
        enc.model_version = "other"


def test_person_templates_must_be_tuple():
    enc = FaceEncoding(vector=_unit_vector(0), model_version="buffalo_l")
    tpl = Template(encoding=enc, source="test")
    person = Person(person_id="alice", display_name="Alice", templates=(tpl,))
    assert isinstance(person.templates, tuple)


def test_recognition_result_unknown_has_none_person_id():
    r = RecognitionResult(person_id=None, similarity=0.3, threshold=0.45)
    assert r.person_id is None
    assert r.similarity < r.threshold
