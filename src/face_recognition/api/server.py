"""FastAPI 应用 + WebSocket 实时识别。"""

import asyncio
import logging
import tempfile
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from face_recognition.api.dependencies import (
    build_pipeline,
    build_recognize_use_case,
    build_repository,
    create_strategy,
    load_config,
)
from face_recognition.application.recognize_face import RecognizeFace
from face_recognition.domain.entities import Person
from face_recognition.domain.errors import FaceRecognitionError, MultipleFacesError, NoFaceError

logger = logging.getLogger(__name__)

app = FastAPI(title="人脸识别系统", version="0.1.0")

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

pipeline = None
repository = None
recognizer: RecognizeFace | None = None
_config = None


@app.on_event("startup")
async def startup():
    global pipeline, repository, recognizer, _config
    _config = load_config()
    pipeline = build_pipeline(_config)
    repository = build_repository(_config)
    recognizer = build_recognize_use_case(_config, pipeline=pipeline, repository=repository)
    logger.info("服务启动完成")


# ---- 全局异常处理 ----

ERROR_TO_HTTP = {
    "NO_FACE": 422,
    "MULTIPLE_FACES": 422,
    "PERSON_NOT_FOUND": 404,
    "DUPLICATE_PERSON": 409,
    "NO_TEMPLATES": 422,
}


@app.exception_handler(FaceRecognitionError)
async def domain_error_handler(request, exc: FaceRecognitionError):
    status_code = ERROR_TO_HTTP.get(exc.code, 500)
    return JSONResponse(
        status_code=status_code,
        content={"error_code": exc.code, "detail": str(exc)},
    )


# ---- 页面 ----

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>前端未构建，请访问 /docs</h1>")


# ---- REST API ----

@app.get("/api/persons")
async def list_persons():
    persons = repository.list_all()
    return [
        {
            "person_id": p.person_id,
            "display_name": p.display_name,
            "template_count": p.template_count,
        }
        for p in persons
    ]


@app.post("/api/persons")
async def register_person(
    person_id: str = Form(...),
    display_name: str = Form(...),
    strategy: str = Form("kmeans_k3"),
    images: list[UploadFile] = File(...),
):
    """注册新人：上传多张照片 → 编码 → 生成模板 → 入库。"""
    if not images:
        return JSONResponse(status_code=422, content={"detail": "至少上传 1 张照片"})

    temp_dir = Path(tempfile.mkdtemp(prefix="face_reg_"))

    try:
        # 保存上传文件到临时目录
        image_arrays = []
        for i, upload in enumerate(images):
            contents = await upload.read()
            temp_path = temp_dir / f"{i:04d}_{upload.filename or 'img.jpg'}"
            temp_path.write_bytes(contents)
            img = cv2.imread(str(temp_path))
            if img is not None:
                image_arrays.append(img)

        if not image_arrays:
            return JSONResponse(status_code=422, content={"detail": "无法读取任何有效图片"})

        # 编码
        encodings = []
        for i, img in enumerate(image_arrays):
            try:
                enc = pipeline.encode_single(img)
                encodings.append(enc)
            except (NoFaceError, MultipleFacesError) as e:
                logger.warning(f"注册 {person_id} 跳过图 {i}: {e}")

        if not encodings:
            return JSONResponse(
                status_code=422,
                content={"detail": "所有照片均未检测到单张人脸"},
            )

        # 生成模板
        strat = create_strategy(strategy)
        templates = strat.build(encodings)

        person = Person(
            person_id=person_id,
            display_name=display_name,
            templates=tuple(templates),
        )
        repository.add(person)
        recognizer.refresh_cache()

        return {
            "person_id": person.person_id,
            "display_name": person.display_name,
            "template_count": person.template_count,
        }

    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


@app.delete("/api/persons/{person_id}")
async def delete_person(person_id: str):
    """删除人员。"""
    repository.remove(person_id)
    recognizer.refresh_cache()
    return JSONResponse(status_code=204, content=None)


@app.get("/api/health")
async def health():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}


# ---- WebSocket 实时推流 ----
# 协议：先发二进制 JPEG 帧，再发 JSON 文本元数据（tracks）。
# 前端按 Blob / 文本分流处理，无需显式配对。


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket 连接建立")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await ws.send_json({"error": "CAMERA_LOST"})
        await ws.close()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await ws.send_json({"error": "CAMERA_LOST"})
                break

            # 检测 + 识别
            encodings = pipeline.encode(frame)
            tracks = []
            for i, enc in enumerate(encodings):
                result = recognizer._match(enc)
                tracks.append({
                    "track_id": i,
                    "identity": result.person_id,
                    "similarity": round(result.similarity, 4),
                })

            # 发送二进制 JPEG
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            await ws.send_bytes(jpeg.tobytes())

            # 发送 JSON 元数据
            await ws.send_json({"tracks": tracks})

            await asyncio.sleep(0.033)

    except WebSocketDisconnect:
        logger.info("WebSocket 断开")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
    finally:
        cap.release()
