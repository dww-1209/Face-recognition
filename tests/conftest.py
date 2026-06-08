from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from face_recognition.domain.entities import FaceEncoding, Template


@pytest.fixture
def make_encoding() -> Callable[[int], FaceEncoding]:
    """生成确定性、L2 归一化的 FaceEncoding 工厂。"""

    def _make(seed: int) -> FaceEncoding:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(512).astype(np.float32)
        v /= np.linalg.norm(v)
        return FaceEncoding(vector=v, model_version="test")

    return _make


@pytest.fixture
def make_template(
    make_encoding: Callable[[int], FaceEncoding],
) -> Callable[[int, str], Template]:
    def _make(seed: int, source: str = "test") -> Template:
        return Template(
            encoding=make_encoding(seed),
            source=source,
            created_at=datetime(2026, 1, 1),
        )

    return _make


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_face.db"
