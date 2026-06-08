"""端到端集成测试：注册 → 识别 全链路（用真模型 + 真 SQLite）。"""

from pathlib import Path

import numpy as np

from face_recognition.application.recognize_face import RecognizeFace
from face_recognition.application.register_face import RegisterFace
from face_recognition.application.strategies.kmeans_k3 import KMeansK3Strategy
from face_recognition.infrastructure.insightface_pipeline import InsightFacePipeline
from face_recognition.infrastructure.sqlite_repository import SqliteRepository


def _load_test_image() -> np.ndarray:
    """用 InsightFace 自带 sample 图（确定性，跨平台）。"""
    import insightface

    model_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
    sample = model_dir / "t1.jpg"
    if not sample.exists():
        # fallback: 用项目中任意一张 LFW 图
        lfw_dir = Path("data/lfw_subset/train/Ariel Sharon")
        if lfw_dir.is_dir():
            imgs = sorted(lfw_dir.glob("*.jpg"))
            if imgs:
                import cv2
                return cv2.imread(str(imgs[0]))
    import cv2
    return cv2.imread(str(sample))


def test_register_then_recognize_same_person(tmp_path: Path):
    """注册一张图 → 用同一张图识别 → 应该认出自己。"""
    pipeline = InsightFacePipeline(pack="buffalo_l", ctx_id=0, det_size=(640, 640))
    repo = SqliteRepository(tmp_path / "test.db")
    strategy = KMeansK3Strategy(seed=42)
    register = RegisterFace(pipeline, repo, strategy)

    img = _load_test_image()
    assert img is not None, "测试图加载失败"

    # 手动注册：编码 → 生成模板 → 入库
    enc = pipeline.encode_single(img)
    templates = strategy.build([enc])
    from face_recognition.domain.entities import Person
    repo.add(Person(person_id="test_user", display_name="Test", templates=tuple(templates)))

    # 识别
    recognizer = RecognizeFace(pipeline, repo, threshold=0.30)
    recognizer.refresh_cache()
    result = recognizer.execute_single(img)

    assert result.person_id == "test_user"
    assert result.similarity > 0.5  # 同一张图相似度应很高
