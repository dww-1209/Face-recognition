from collections.abc import Callable
from pathlib import Path

import pytest

from face_recognition.evaluation.data_split import PersonSplit, split_by_person


def _make_person_dir(root: Path, person_id: str, n: int) -> None:
    """造一个 "person_id/" 目录，放 n 张占位 jpg。"""
    d = root / person_id
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"{i:03d}.jpg").write_bytes(b"fake")


def test_split_keeps_persons_disjoint(tmp_path: Path):
    """同一人不能同时出现在 train 和 test。"""
    _make_person_dir(tmp_path, "alice", 10)
    _make_person_dir(tmp_path, "bob", 10)
    splits = split_by_person(tmp_path, train_ratio=0.8, seed=42)

    # 每个 person 都拿到自己的 PersonSplit
    assert {s.person_id for s in splits} == {"alice", "bob"}
    for s in splits:
        # 集合交集为空 = 不重叠（set 操作 & 是交集，| 是并集）
        assert set(s.train_paths).isdisjoint(set(s.test_paths))
        assert len(s.train_paths) + len(s.test_paths) == 10


def test_split_ratio_80_20(tmp_path: Path):
    _make_person_dir(tmp_path, "alice", 10)
    [s] = split_by_person(tmp_path, train_ratio=0.8, seed=42)
    # 10 张 × 80% = 8 张训练
    assert len(s.train_paths) == 8
    assert len(s.test_paths) == 2


def test_split_is_deterministic_with_seed(tmp_path: Path):
    """同 seed → 同切分。"""
    _make_person_dir(tmp_path, "alice", 20)
    a = split_by_person(tmp_path, train_ratio=0.8, seed=42)
    b = split_by_person(tmp_path, train_ratio=0.8, seed=42)
    assert a[0].train_paths == b[0].train_paths


def test_split_skips_persons_with_too_few_images(tmp_path: Path):
    """不足 5 张照片的人直接跳过——80/20 切下来训练或测试可能为 0。"""
    _make_person_dir(tmp_path, "alice", 10)
    _make_person_dir(tmp_path, "tooFew", 3)
    splits = split_by_person(tmp_path, train_ratio=0.8, seed=42, min_images=5)
    assert {s.person_id for s in splits} == {"alice"}


def test_split_ignores_non_directories(tmp_path: Path):
    """根目录里的非文件夹条目（README、隐藏文件）应该被忽略。"""
    _make_person_dir(tmp_path, "alice", 10)
    (tmp_path / "README.md").write_text("notes")
    splits = split_by_person(tmp_path, train_ratio=0.8, seed=42)
    assert len(splits) == 1
