from datetime import datetime
from unittest.mock import MagicMock

import numpy as np

from face_recognition.application.template_matrix import TemplateMatrixService
from face_recognition.domain.entities import FaceEncoding, Person, Template


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_person(pid: str, n_templates: int, seed: int) -> Person:
    """造一个有 n_templates 个模板的 Person。"""
    templates = tuple(
        Template(
            encoding=FaceEncoding(vector=_unit_vec(seed + i), model_version="buffalo_l"),
            source=f"tpl_{i}",
            created_at=datetime(2026, 5, 21),
        )
        for i in range(n_templates)
    )
    return Person(person_id=pid, display_name=pid, templates=templates)


def test_load_builds_matrix_and_pid_list():
    """load 后 matrix.shape = (总模板数, 512)，pid_list 长度一致。"""
    repo = MagicMock()
    repo.list_all.return_value = [
        _make_person("alice", 3, seed=0),    # 3 个模板
        _make_person("bob", 1, seed=10),     # 1 个模板
    ]
    svc = TemplateMatrixService(repository=repo)
    svc.load()
    assert svc.matrix.shape == (4, 512)
    assert svc.pid_list == ["alice", "alice", "alice", "bob"]


def test_query_returns_best_matching_person():
    """query 应返回相似度最高的 person_id 和分数。"""
    repo = MagicMock()
    alice = _make_person("alice", 1, seed=0)
    bob = _make_person("bob", 1, seed=100)
    repo.list_all.return_value = [alice, bob]

    svc = TemplateMatrixService(repository=repo)
    svc.load()

    # query alice 自己的模板向量 → 应该匹配到 alice，相似度 = 1.0
    pid, sim = svc.query(alice.templates[0].encoding.vector)
    assert pid == "alice"
    assert sim > 0.99


def test_reload_picks_up_repository_changes():
    """reload 后矩阵应反映 repository 的新状态。"""
    repo = MagicMock()
    repo.list_all.return_value = [_make_person("alice", 1, seed=0)]

    svc = TemplateMatrixService(repository=repo)
    svc.load()
    assert svc.matrix.shape == (1, 512)

    # 模拟新增 bob
    repo.list_all.return_value = [_make_person("alice", 1, seed=0), _make_person("bob", 2, seed=10)]
    svc.reload()
    assert svc.matrix.shape == (3, 512)
    assert "bob" in svc.pid_list


def test_query_on_empty_matrix_returns_none():
    """库里一个人都没有时 query 应返回 (None, 0.0) 而非崩溃。"""
    repo = MagicMock()
    repo.list_all.return_value = []
    svc = TemplateMatrixService(repository=repo)
    svc.load()
    pid, sim = svc.query(_unit_vec(0))
    assert pid is None
    assert sim == 0.0
