"""全局异常处理测试。"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from face_recognition.api.server import app

    with TestClient(app) as c:
        yield c


def test_404_on_unknown_route(client):
    """未注册路由 → 404。"""
    response = client.get("/api/nonexistent")
    assert response.status_code == 404
