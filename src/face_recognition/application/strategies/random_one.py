import random
from datetime import UTC, datetime

from face_recognition.domain.entities import FaceEncoding, Template


class RandomOneStrategy:
    """策略 1：从 N 张编码中随机选 1 张作为唯一模板。"""

    name = "random_one"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def build(self, encodings: list[FaceEncoding]) -> list[Template]:
        if not encodings:
            raise ValueError("编码列表为空，无法选择随机模板")
        rng = random.Random(self.seed)
        chosen = rng.choice(encodings)
        return [
            Template(
                encoding=chosen,
                source=f"random_one_seed{self.seed}",
                created_at=datetime.now(UTC),
            )
        ]
