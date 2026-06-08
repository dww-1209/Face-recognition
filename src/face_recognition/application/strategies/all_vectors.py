from datetime import UTC, datetime

from face_recognition.domain.entities import FaceEncoding, Template


class AllVectorsStrategy:
    """策略 5：全部编码直接存为模板（多向量检索，每人 N 个模板）。"""

    name = "all_vectors"

    def build(self, encodings: list[FaceEncoding]) -> list[Template]:
        if not encodings:
            raise ValueError("编码列表为空")
        return [
            Template(
                encoding=e,
                source=f"raw_{i:04d}",
                created_at=datetime.now(UTC),
            )
            for i, e in enumerate(encodings)
        ]
