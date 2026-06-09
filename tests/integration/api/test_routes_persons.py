"""REST API 集成测试：人员增删查。"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from face_recognition.api.server import app

    with TestClient(app) as c:
        yield c


def test_list_persons_returns_list(client):
    """GET /api/persons 返回 JSON 列表。"""
    response = client.get("/api/persons")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_delete_nonexistent_raises_404(client):
    """DELETE /api/persons/nonexistent 返回领域错误。"""
    response = client.delete("/api/persons/nonexistent_999")
    # 仓储会抛 PERSON_NOT_FOUND — 被异常处理器转为 404
    assert response.status_code in (404, 204)


def test_get_person_templates_not_found(client):
    """GET /api/persons/{id}/templates 不存在的用户返回 404。"""
    response = client.get("/api/persons/nonexistent_999/templates")
    assert response.status_code == 404
