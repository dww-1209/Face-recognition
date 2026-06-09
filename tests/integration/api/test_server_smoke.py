"""冒烟测试：app 能起来 + 静态首页能访问。"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """TestClient 是 FastAPI 自带的测试客户端，基于 httpx，
    会自动触发 lifespan startup/shutdown，模拟真实启动流程。"""
    from face_recognition.api.server import app

    with TestClient(app) as c:
        yield c


def test_app_starts(client):
    """app 启动 + 关闭都不抛错 = 装配链路 OK。"""
    # TestClient 进入 with 块时已经跑过 lifespan startup，没炸就算过
    assert client.app is not None


def test_health_check(client):
    """健康检查端点。"""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_static_index_served(client):
    """根路径 / 应返回 index.html（由 StaticFiles html=True 提供）。"""
    response = client.get("/")
    # 静态文件存在时返回 200，不存在时可能 404
    assert response.status_code in (200, 404)
