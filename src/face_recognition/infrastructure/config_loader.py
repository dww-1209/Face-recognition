"""配置加载：pydantic 读 config.yaml，启动时字段校验。"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

StrategyName = Literal[
    "random_one", "mean_all", "manual_three", "kmeans_k3", "all_vectors"
]


class ModelConfig(BaseModel):
    pack: str = "buffalo_l"
    ctx_id: int = 0
    det_size: tuple[int, int] = (640, 640)


class RecognitionConfig(BaseModel):
    threshold: float = Field(default=0.45, ge=-1.0, le=1.0)
    template_strategy: StrategyName = "kmeans_k3"


class CameraConfig(BaseModel):
    device_index: int = 0
    resolution: tuple[int, int] = (1280, 720)
    fps: int = Field(default=30, gt=0)


class RealtimeConfig(BaseModel):
    detect_every_n_frames: int = Field(default=1, ge=1)
    recognize_on_new_track: bool = True
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    track_max_missing_frames: int = Field(default=15, ge=0)
    jpeg_quality: int = Field(default=85, ge=1, le=100)


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, gt=0, le=65535)


class DataConfig(BaseModel):
    sqlite_path: Path = Path("data/face.db")
    dataset_root: Path = Path("data/private_dataset")
    lfw_subset: Path = Path("data/lfw_subset")


class EvaluationConfig(BaseModel):
    random_seed: int = 42
    train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    far_targets: list[float] = [0.001, 0.01, 0.1]


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/face.log"


class AppConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    recognition: RecognitionConfig = RecognitionConfig()
    camera: CameraConfig = CameraConfig()
    realtime: RealtimeConfig = RealtimeConfig()
    api: ApiConfig = ApiConfig()
    data: DataConfig = DataConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    logging: LoggingConfig = LoggingConfig()


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """加载配置。优先级：config.local.yaml > config.yaml > 默认值。"""
    config_path = Path(config_path)

    merged: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            merged.update(yaml.safe_load(f) or {})

    local_path = config_path.with_name("config.local.yaml")
    if local_path.exists():
        with open(local_path) as f:
            local = yaml.safe_load(f) or {}
            _deep_merge(merged, local)

    return AppConfig(**merged)


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
