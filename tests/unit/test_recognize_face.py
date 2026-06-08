import numpy as np

from face_recognition.application.recognize_face import RecognizeFace
from face_recognition.domain.entities import FaceEncoding, Person, RecognitionResult, Template


class FakePipeline:
    model_version = "test"

    def encode(self, image):
        rng = np.random.default_rng(42)
        v = rng.standard_normal(512).astype(np.float32)
        v /= np.linalg.norm(v)
        return [FaceEncoding(vector=v, model_version="test")]

    def encode_single(self, image):
        return self.encode(image)[0]


class FakeRepository:
    def __init__(self, templates_matrix=None, person_ids=None):
        self._matrix = (
            templates_matrix
            if templates_matrix is not None
            else np.empty((0, 512), dtype=np.float32)
        )
        self._ids = person_ids if person_ids is not None else []

    def add(self, person): pass
    def get(self, person_id): return None
    def remove(self, person_id): pass
    def list_all(self): return []

    def all_templates_matrix(self):
        return self._matrix, self._ids


class TestRecognizeFace:
    def test_empty_library_returns_unknown(self):
        pipeline = FakePipeline()
        repo = FakeRepository()
        recognizer = RecognizeFace(pipeline, repo, threshold=0.45)
        recognizer.refresh_cache()

        img = np.zeros((112, 112, 3), dtype=np.uint8)
        result = recognizer.execute(img)
        assert result.person_id is None

    def test_known_person_above_threshold(self):
        # 构造一个与查询向量完全一致的模板
        query_vec = np.zeros(512, dtype=np.float32)
        query_vec[0] = 1.0
        tpl_matrix = np.array([query_vec], dtype=np.float32)

        pipeline = FakePipeline()
        repo = FakeRepository(tpl_matrix, ["alice"])
        recognizer = RecognizeFace(pipeline, repo, threshold=0.45)
        recognizer.refresh_cache()

        # 直接测试 _match
        result = recognizer._match(FaceEncoding(vector=query_vec, model_version="test"))
        assert result.person_id == "alice"
        assert result.similarity > 0.9

    def test_below_threshold_returns_unknown(self):
        query_vec = np.zeros(512, dtype=np.float32)
        query_vec[0] = 1.0
        unrelated_vec = np.zeros(512, dtype=np.float32)
        unrelated_vec[1] = 1.0
        tpl_matrix = np.array([unrelated_vec], dtype=np.float32)

        pipeline = FakePipeline()
        repo = FakeRepository(tpl_matrix, ["bob"])
        recognizer = RecognizeFace(pipeline, repo, threshold=0.8)
        recognizer.refresh_cache()

        result = recognizer._match(FaceEncoding(vector=query_vec, model_version="test"))
        assert result.person_id is None
