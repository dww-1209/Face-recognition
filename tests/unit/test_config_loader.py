from pathlib import Path

import pytest
import yaml

from face_recognition.infrastructure.config_loader import AppConfig, load_config


@pytest.fixture
def valid_config_path(tmp_path: Path) -> Path:
    cfg = {
        "model": {"pack": "buffalo_l", "ctx_id": 0, "det_size": [640, 640]},
        "recognition": {"threshold": 0.45, "template_strategy": "kmeans_k3"},
        "camera": {"device_index": 0, "resolution": [1280, 720], "fps": 30},
        "realtime": {
            "detect_every_n_frames": 1,
            "recognize_on_new_track": True,
            "iou_threshold": 0.5,
            "track_max_missing_frames": 15,
        },
        "api": {"host": "0.0.0.0", "port": 8000},
        "data": {
            "sqlite_path": "data/face.db",
            "dataset_root": "data/private_dataset",
            "lfw_subset": "data/lfw_subset",
        },
        "evaluation": {
            "random_seed": 42,
            "train_ratio": 0.8,
            "far_targets": [0.001, 0.01, 0.1],
        },
        "logging": {"level": "INFO", "file": "logs/face.log"},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_load_config_returns_app_config(valid_config_path: Path):
    cfg = load_config(valid_config_path)
    assert isinstance(cfg, AppConfig)
    assert cfg.recognition.threshold == 0.45
    assert cfg.recognition.template_strategy == "kmeans_k3"
    assert cfg.evaluation.random_seed == 42
    assert cfg.data.sqlite_path == Path("data/face.db")


def test_invalid_strategy_name_raises(tmp_path: Path):
    cfg = {
        "model": {"pack": "buffalo_l", "ctx_id": 0, "det_size": [640, 640]},
        "recognition": {"threshold": 0.5, "template_strategy": "no_such_strategy"},
        "camera": {"device_index": 0, "resolution": [1280, 720], "fps": 30},
        "realtime": {
            "detect_every_n_frames": 1,
            "recognize_on_new_track": True,
            "iou_threshold": 0.5,
            "track_max_missing_frames": 15,
        },
        "api": {"host": "0.0.0.0", "port": 8000},
        "data": {
            "sqlite_path": "data/face.db",
            "dataset_root": "data/private_dataset",
            "lfw_subset": "data/lfw_subset",
        },
        "evaluation": {
            "random_seed": 42,
            "train_ratio": 0.8,
            "far_targets": [0.001],
        },
        "logging": {"level": "INFO", "file": "logs/face.log"},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(Exception):
        load_config(p)


def test_threshold_out_of_range_raises(tmp_path: Path, valid_config_path: Path):
    raw = yaml.safe_load(valid_config_path.read_text())
    raw["recognition"]["threshold"] = 2.0
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(Exception):
        load_config(p)
