"""FastAPI 应用入口。

启动方式：
    uv run uvicorn face_recognition.api.server:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from face_recognition.api.dependencies import (
    get_config,
    get_pipeline,
    get_template_matrix,
)
from face_recognition.api.routes_persons import router as persons_router
from face_recognition.api.routes_stream import router as stream_router
from face_recognition.domain.errors import FaceRecognitionError

logger = logging.getLogger(__name__)

# 错误码 → HTTP 状态码映射
ERROR_TO_HTTP = {
    "NO_FACE": 422,
    "MULTIPLE_FACES": 422,
    "PERSON_NOT_FOUND": 404,
    "DUPLICATE_PERSON": 409,
    "NO_TEMPLATES": 422,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载模型和模板矩阵，关闭时清理。"""
    # ----- 启动逻辑 -----
    logger.info("正在加载配置...")
    cfg = get_config()
    logger.info("配置加载完成 (db=%s)", cfg.data.sqlite_path)

    logger.info("正在加载 InsightFace 模型（首次启动 ~5s）...")
    get_pipeline()  # 触发 lru_cache，提前加载
    logger.info("模型加载完成")

    logger.info("正在加载模板矩阵...")
    matrix = get_template_matrix()
    matrix.load()
    logger.info("模板矩阵加载完成")

    yield  # ↑ 启动 / ↓ 关闭

    # ----- 关闭逻辑 -----
    logger.info("应用关闭中...")


# 创建 FastAPI 应用
app = FastAPI(
    title="人脸识别系统",
    version="0.1.0",
    lifespan=lifespan,
)

# ---- 全局异常处理 ----
@app.exception_handler(FaceRecognitionError)
async def domain_error_handler(request, exc: FaceRecognitionError):
    status_code = ERROR_TO_HTTP.get(exc.code, 500)
    return JSONResponse(
        status_code=status_code,
        content={"error_code": exc.code, "detail": str(exc)},
    )


# ---- 注册路由 ----
app.include_router(persons_router)
app.include_router(stream_router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "pipeline_loaded": True}


# ---- 静态文件（必须最后挂载，否则吞掉 API 路由） ----
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
