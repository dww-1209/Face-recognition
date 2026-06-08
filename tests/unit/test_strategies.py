from collections.abc import Callable

import numpy as np

from face_recognition.application.strategies.all_vectors import AllVectorsStrategy
from face_recognition.application.strategies.kmeans_k3 import KMeansK3Strategy
from face_recognition.application.strategies.mean_all import MeanAllStrategy
from face_recognition.application.strategies.random_one import RandomOneStrategy
from face_recognition.domain.entities import FaceEncoding


def _make_encodings(n: int) -> list[FaceEncoding]:
    """生成 n 个确定性编码。"""
    encs = []
    for seed in range(n):
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(512).astype(np.float32)
        v /= np.linalg.norm(v)
        encs.append(FaceEncoding(vector=v, model_version="test"))
    return encs


class TestRandomOne:
    def test_returns_one_template(self):
        encs = _make_encodings(20)
        tpls = RandomOneStrategy(seed=42).build(encs)
        assert len(tpls) == 1

    def test_is_deterministic(self):
        encs = _make_encodings(20)
        a = RandomOneStrategy(seed=42).build(encs)
        b = RandomOneStrategy(seed=42).build(encs)
        assert np.allclose(a[0].encoding.vector, b[0].encoding.vector)

    def test_template_is_unit_vector(self):
        encs = _make_encodings(20)
        tpls = RandomOneStrategy(seed=42).build(encs)
        assert abs(np.linalg.norm(tpls[0].encoding.vector) - 1.0) < 1e-3


class TestMeanAll:
    def test_returns_one_template(self):
        encs = _make_encodings(20)
        tpls = MeanAllStrategy().build(encs)
        assert len(tpls) == 1

    def test_mean_is_unit_vector(self):
        encs = _make_encodings(50)
        tpls = MeanAllStrategy().build(encs)
        assert abs(np.linalg.norm(tpls[0].encoding.vector) - 1.0) < 1e-3


class TestKMeansK3:
    def test_returns_three_templates(self):
        encs = _make_encodings(20)
        tpls = KMeansK3Strategy(seed=42).build(encs)
        assert len(tpls) == 3

    def test_centroids_are_unit_vectors(self):
        encs = _make_encodings(20)
        tpls = KMeansK3Strategy(seed=42).build(encs)
        for tpl in tpls:
            assert abs(np.linalg.norm(tpl.encoding.vector) - 1.0) < 1e-3

    def test_fewer_than_3_fallback(self):
        encs = _make_encodings(2)
        tpls = KMeansK3Strategy(seed=42).build(encs)
        assert len(tpls) == 2


class TestAllVectors:
    def test_returns_n_templates(self):
        encs = _make_encodings(20)
        tpls = AllVectorsStrategy().build(encs)
        assert len(tpls) == 20

    def test_each_is_unit_vector(self):
        encs = _make_encodings(5)
        tpls = AllVectorsStrategy().build(encs)
        for tpl in tpls:
            assert abs(np.linalg.norm(tpl.encoding.vector) - 1.0) < 1e-3
