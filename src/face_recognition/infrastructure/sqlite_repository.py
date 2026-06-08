"""SQLite 向量库：人员模板的持久化存储。"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from face_recognition.domain.entities import FaceEncoding, Person, Template
from face_recognition.domain.errors import PersonNotFoundError

logger = logging.getLogger(__name__)


class SqliteRepository:
    """基于 SQLite 的人员向量库，实现 PersonRepository 协议。"""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES persons(person_id)
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_templates_person_id "
                "ON templates(person_id)"
            )

    # ---- 写操作 ----

    def add(self, person: Person) -> None:
        """添加人员，已存在则先删旧再插新（幂等）。"""
        with sqlite3.connect(str(self.db_path)) as conn:
            try:
                conn.execute("DELETE FROM templates WHERE person_id = ?", (person.person_id,))
                conn.execute("DELETE FROM persons WHERE person_id = ?", (person.person_id,))
                conn.execute(
                    "INSERT INTO persons (person_id, display_name) VALUES (?, ?)",
                    (person.person_id, person.display_name),
                )
                for tpl in person.templates:
                    conn.execute(
                        "INSERT INTO templates (person_id, vector, source, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            person.person_id,
                            tpl.encoding.vector.tobytes(),
                            tpl.source,
                            tpl.created_at.isoformat(),
                        ),
                    )
            except Exception:
                conn.rollback()
                raise

    def remove(self, person_id: str) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "DELETE FROM persons WHERE person_id = ?", (person_id,)
            )
            if cursor.rowcount == 0:
                raise PersonNotFoundError(f"人员不存在: {person_id}")

    # ---- 读操作 ----

    def get(self, person_id: str) -> Person | None:
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT person_id, display_name FROM persons WHERE person_id = ?",
                (person_id,),
            ).fetchone()
            if row is None:
                return None
            templates = self._load_templates(conn, person_id)
            return Person(
                person_id=row[0],
                display_name=row[1],
                templates=tuple(templates),
            )

    def list_all(self) -> list[Person]:
        persons = []
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT person_id, display_name FROM persons ORDER BY person_id"
            ).fetchall()
            for row in rows:
                templates = self._load_templates(conn, row[0])
                persons.append(
                    Person(
                        person_id=row[0],
                        display_name=row[1],
                        templates=tuple(templates),
                    )
                )
        return persons

    def all_templates_matrix(self) -> tuple[np.ndarray, list[str]]:
        """返回 (M, 512) 矩阵 + 对应的 person_id 列表。"""
        vectors = []
        person_ids = []
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT t.vector, t.person_id FROM templates t "
                "INNER JOIN persons p ON t.person_id = p.person_id "
                "ORDER BY t.id"
            ).fetchall()
            for vector_blob, pid in rows:
                vec = np.frombuffer(vector_blob, dtype=np.float32)
                if vec.shape[0] == 512:
                    vectors.append(vec)
                    person_ids.append(pid)
        if not vectors:
            return np.empty((0, 512), dtype=np.float32), []
        return np.stack(vectors), person_ids

    def _load_templates(self, conn: sqlite3.Connection, person_id: str) -> list[Template]:
        rows = conn.execute(
            "SELECT vector, source, created_at FROM templates WHERE person_id = ?",
            (person_id,),
        ).fetchall()
        templates = []
        for vector_blob, source, created_at_str in rows:
            vec = np.frombuffer(vector_blob, dtype=np.float32)
            created_at = datetime.fromisoformat(created_at_str)
            templates.append(
                Template(
                    encoding=FaceEncoding(vector=vec),
                    source=source,
                    created_at=created_at,
                )
            )
        return templates
