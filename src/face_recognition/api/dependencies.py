"""
依赖装配：所有"具体实现 → 用例"的注入都在这里。
CLI 和 Server 共用同一份装配逻辑。
"""

import logging

from face_recognition.application.recognize_face import RecognizeFace
from face_recognition.application.register_face import RegisterFace
from face_recognition.application.strategies.all_vectors import AllVectorsStrategy
from face_recognition.application.strategies.kmeans_k3 import KMeansK3Strategy
from face_recognition.application.strategies.manual_three import ManualThreeStrategy
from face_recognition.application.strategies.mean_all import MeanAllStrategy
from face_recognition.application.strategies.random_one import RandomOneStrategy
from face_recognition.domain.interfaces import TemplateStrategy
from face_recognition.infrastructure.config_loader import AppConfig, load_config
from face_recognition.infrastructure.insightface_pipeline import InsightFacePipeline
from face_recognition.infrastructure.sqlite_repository import SqliteRepository

logger = logging.getLogger(__name__)

# 策略注册表
STRATEGY_REGISTRY: dict[str, type[TemplateStrategy]] = {
    "random_one": RandomOneStrategy,
    "mean_all": MeanAllStrategy,
    "manual_three": ManualThreeStrategy,
    "kmeans_k3": KMeansK3Strategy,
    "all_vectors": AllVectorsStrategy,
}


def create_strategy(name: str) -> TemplateStrategy:
    """从名称创建策略实例。"""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略: {name}，可选: {list(STRATEGY_REGISTRY.keys())}")
    strategy_cls = STRATEGY_REGISTRY[name]
    if name in ("random_one", "kmeans_k3"):
        return strategy_cls(seed=42)
    return strategy_cls()


def build_pipeline(config: AppConfig | None = None) -> InsightFacePipeline:
    """装配 InsightFace 管线。"""
    if config is None:
        config = load_config()
    return InsightFacePipeline(
        pack=config.model.pack,
        ctx_id=config.model.ctx_id,
        det_size=config.model.det_size,
    )


def build_repository(config: AppConfig | None = None) -> SqliteRepository:
    """装配 SQLite 向量库。"""
    if config is None:
        config = load_config()
    return SqliteRepository(config.data.sqlite_path)


def build_register_use_case(
    config: AppConfig | None = None,
    pipeline: InsightFacePipeline | None = None,
    repository: SqliteRepository | None = None,
    strategy: TemplateStrategy | None = None,
) -> RegisterFace:
    """装配注册用例。"""
    if config is None:
        config = load_config()
    if pipeline is None:
        pipeline = build_pipeline(config)
    if repository is None:
        repository = build_repository(config)
    if strategy is None:
        strategy = create_strategy(config.recognition.template_strategy)
    return RegisterFace(pipeline, repository, strategy)


def build_recognize_use_case(
    config: AppConfig | None = None,
    pipeline: InsightFacePipeline | None = None,
    repository: SqliteRepository | None = None,
) -> RecognizeFace:
    """装配识别用例。"""
    if config is None:
        config = load_config()
    if pipeline is None:
        pipeline = build_pipeline(config)
    if repository is None:
        repository = build_repository(config)
    recognizer = RecognizeFace(
        pipeline,
        repository,
        threshold=config.recognition.threshold,
    )
    recognizer.refresh_cache()
    return recognizer
