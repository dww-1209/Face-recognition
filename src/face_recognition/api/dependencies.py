"""
依赖装配：所有"具体实现 → 用例"的注入都在这里。
CLI 和 Server 共用同一份装配逻辑。

M4 新增：serve 端 lru_cache 单例 + 实时识别用例工厂。
M1 的 build_* 函数保留给 CLI 用——CLI 是"短命进程"每次新建即可，无需缓存。
"""

import logging
from functools import lru_cache
from pathlib import Path

from face_recognition.application.recognize_face import RecognizeFace
from face_recognition.application.recognize_frame import RecognizeFrame
from face_recognition.application.register_face import RegisterFace
from face_recognition.application.strategies.all_vectors import AllVectorsStrategy
from face_recognition.application.strategies.kmeans_k3 import KMeansK3Strategy
from face_recognition.application.strategies.manual_three import ManualThreeStrategy
from face_recognition.application.strategies.mean_all import MeanAllStrategy
from face_recognition.application.strategies.random_one import RandomOneStrategy
from face_recognition.application.template_matrix import TemplateMatrixService
from face_recognition.domain.interfaces import FacePipeline, PersonRepository, TemplateStrategy
from face_recognition.infrastructure.config_loader import AppConfig, load_config
from face_recognition.infrastructure.insightface_pipeline import InsightFacePipeline
from face_recognition.infrastructure.iou_tracker import IoUTracker
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


# ===== M4：Server 端单例工厂 =====
# 下面用 @lru_cache(maxsize=1) 做"全应用唯一实例"。
# InsightFace 模型加载 ~5s，每个 HTTP 请求重加载会卡死。
# 模型本身无状态，多线程共享读没问题（buffalo_l 已实测）。


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """加载 config.yaml 一次，后续调用直接返回缓存。"""
    return load_config(Path("config.yaml"))


@lru_cache(maxsize=1)
def get_pipeline() -> FacePipeline:
    """InsightFace 模型单例。首次调用 ~5s 加载，后续 O(1) 返回。"""
    cfg = get_config()
    return InsightFacePipeline(
        pack=cfg.model.pack,
        ctx_id=cfg.model.ctx_id,
        det_size=cfg.model.det_size,
    )


@lru_cache(maxsize=1)
def get_repository() -> PersonRepository:
    """SQLite 仓储单例。"""
    cfg = get_config()
    return SqliteRepository(cfg.data.sqlite_path)


@lru_cache(maxsize=1)
def get_recognizer() -> RecognizeFace:
    """RecognizeFace 单例（CLI + Server 共用识别逻辑）。"""
    cfg = get_config()
    recognizer = RecognizeFace(
        get_pipeline(),
        get_repository(),
        threshold=cfg.recognition.threshold,
    )
    recognizer.refresh_cache()
    return recognizer


@lru_cache(maxsize=1)
def get_template_matrix() -> TemplateMatrixService:
    """模板矩阵服务单例。构造时只持有 repo 引用，load 在 lifespan 里调用。"""
    return TemplateMatrixService(repository=get_repository())


def build_recognize_frame_use_case() -> RecognizeFrame:
    """每次 WebSocket 连接创建一个新 RecognizeFrame 实例。

    不能 lru_cache：每个连接需要独立的 IoUTracker（多客户端串号）。
    pipeline + matrix 共享（无状态）。
    """
    cfg = get_config()
    return RecognizeFrame(
        pipeline=get_pipeline(),
        tracker=IoUTracker(
            iou_threshold=cfg.realtime.iou_threshold,
            max_missing=cfg.realtime.track_max_missing_frames,
        ),
        template_matrix=get_template_matrix(),
        threshold=cfg.recognition.threshold,
        recheck_interval=cfg.realtime.recognition_recheck_interval,
    )
