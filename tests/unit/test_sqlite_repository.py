import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from face_recognition.domain.entities import Person, Template
from face_recognition.domain.errors import PersonNotFoundError
from face_recognition.infrastructure.sqlite_repository import SqliteRepository


def test_add_and_get_roundtrip(
    tmp_db_path: Path,
    make_template: Callable[[int, str], Template],
):
    repo = SqliteRepository(tmp_db_path)
    alice = Person(
        person_id="alice",
        display_name="Alice",
        templates=(make_template(1, "centroid_0"), make_template(2, "centroid_1")),
    )
    repo.add(alice)
    fetched = repo.get("alice")
    assert fetched is not None
    assert fetched.person_id == "alice"
    assert fetched.display_name == "Alice"
    assert len(fetched.templates) == 2
    assert np.allclose(
        fetched.templates[0].encoding.vector,
        alice.templates[0].encoding.vector,
    )


def test_get_unknown_returns_none(tmp_db_path: Path):
    repo = SqliteRepository(tmp_db_path)
    assert repo.get("ghost") is None


def test_list_all_orders_by_person_id(
    tmp_db_path: Path,
    make_template: Callable[[int, str], Template],
):
    repo = SqliteRepository(tmp_db_path)
    repo.add(Person("bob", "Bob", (make_template(10, "x"),)))
    repo.add(Person("alice", "Alice", (make_template(20, "x"),)))
    ids = [p.person_id for p in repo.list_all()]
    assert ids == ["alice", "bob"]


def test_add_existing_person_replaces(
    tmp_db_path: Path,
    make_template: Callable[[int, str], Template],
):
    repo = SqliteRepository(tmp_db_path)
    repo.add(Person("alice", "Alice", (make_template(1, "x"),)))
    repo.add(Person("alice", "Alice 2", (make_template(99, "y"), make_template(100, "z"))))
    fetched = repo.get("alice")
    assert fetched is not None
    assert fetched.display_name == "Alice 2"
    assert len(fetched.templates) == 2


def test_remove_nonexistent_raises(tmp_db_path: Path):
    repo = SqliteRepository(tmp_db_path)
    with pytest.raises(PersonNotFoundError):
        repo.remove("ghost")


def test_remove_then_get_returns_none(
    tmp_db_path: Path,
    make_template: Callable[[int, str], Template],
):
    repo = SqliteRepository(tmp_db_path)
    repo.add(Person("alice", "Alice", (make_template(1, "x"),)))
    repo.remove("alice")
    assert repo.get("alice") is None


def test_all_templates_matrix_shape_and_index(
    tmp_db_path: Path,
    make_template: Callable[[int, str], Template],
):
    repo = SqliteRepository(tmp_db_path)
    repo.add(Person("alice", "A", (make_template(1, "x"), make_template(2, "y"))))
    repo.add(Person("bob", "B", (make_template(3, "x"),)))
    matrix, ids = repo.all_templates_matrix()
    assert matrix.shape == (3, 512)
    assert matrix.dtype == np.float32
    assert ids == ["alice", "alice", "bob"]
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_empty_matrix_is_zero_rows(tmp_db_path: Path):
    repo = SqliteRepository(tmp_db_path)
    matrix, ids = repo.all_templates_matrix()
    assert matrix.shape == (0, 512)
    assert ids == []


def test_concurrent_reads_do_not_raise(
    tmp_db_path: Path,
    make_template: Callable[[int, str], Template],
):
    repo = SqliteRepository(tmp_db_path)
    repo.add(Person("alice", "A", (make_template(1, "x"),)))

    errors = []

    def read():
        try:
            repo.get("alice")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=read) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
