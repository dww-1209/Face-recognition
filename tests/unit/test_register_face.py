from collections.abc import Callable
from pathlib import Path

from face_recognition.application.register_face import RegisterFace
from face_recognition.application.strategies.mean_all import MeanAllStrategy
from face_recognition.domain.entities import Person


class FakePipeline:
    model_version = "test"

    def encode(self, image):
        import numpy as np

        rng = np.random.default_rng(42)
        v = rng.standard_normal(512).astype(np.float32)
        v /= np.linalg.norm(v)
        from face_recognition.domain.entities import FaceEncoding

        return [FaceEncoding(vector=v, model_version="test")]

    def encode_single(self, image):
        return self.encode(image)[0]


class FakeRepository:
    def __init__(self):
        self.persons: dict[str, Person] = {}

    def add(self, person: Person) -> None:
        self.persons[person.person_id] = person

    def get(self, person_id: str) -> Person | None:
        return self.persons.get(person_id)

    def remove(self, person_id: str) -> None:
        self.persons.pop(person_id, None)

    def list_all(self) -> list[Person]:
        return list(self.persons.values())

    def all_templates_matrix(self):
        import numpy as np
        return np.empty((0, 512), dtype=np.float32), []


class TestRegisterFace:
    def test_register_from_directory(self, tmp_path: Path):
        import cv2
        import numpy as np

        person_dir = tmp_path / "alice"
        person_dir.mkdir()
        img = np.zeros((112, 112, 3), dtype=np.uint8)
        cv2.imwrite(str(person_dir / "img1.jpg"), img)
        cv2.imwrite(str(person_dir / "img2.jpg"), img)

        pipeline = FakePipeline()
        repo = FakeRepository()
        strategy = MeanAllStrategy()
        use_case = RegisterFace(pipeline, repo, strategy)

        stats = use_case.execute(str(tmp_path))
        assert stats["success"] == 1
        assert repo.get("alice") is not None
        assert repo.get("alice").template_count == 1
