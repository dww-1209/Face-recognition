"""WebSocket 实时推流：JPEG 帧 + JSON 元数据。"""

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from face_recognition.api.dependencies import (
    build_recognize_frame_use_case,
    get_config,
    get_pipeline,
)
from face_recognition.domain.errors import CameraDisconnectedError
from face_recognition.infrastructure.camera_capture import CameraCapture
from face_recognition.infrastructure.frame_renderer import encode_jpeg, render_tracks

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """实时推流：每帧发送 JPEG 二进制 + 识别结果 JSON。"""
    await ws.accept()
    logger.info("WebSocket 连接建立")

    cfg = get_config()
    # 每个 WebSocket 连接独立 tracker（防止多客户端串号）
    use_case = build_recognize_frame_use_case()

    try:
        cam = CameraCapture(
            device_index=cfg.camera.device_index,
            resolution=cfg.camera.resolution,
        )
    except CameraDisconnectedError as e:
        await ws.send_json({"error": "CAMERA_LOST", "detail": str(e)})
        await ws.close()
        return

    try:
        while True:
            try:
                frame = cam.read()
            except CameraDisconnectedError:
                await ws.send_json({"error": "CAMERA_LOST"})
                break

            # 检测 + 跟踪 + 按需识别
            tracks = use_case.process_frame(frame)

            # 画框 + 写名
            rendered = render_tracks(frame, tracks)

            # 编码 JPEG
            jpeg_bytes = encode_jpeg(rendered, quality=cfg.realtime.jpeg_quality)

            # 发送二进制 JPEG
            await ws.send_bytes(jpeg_bytes)

            # 发送 JSON 元数据
            track_data = []
            for t in tracks:
                track_data.append({
                    "track_id": t.track_id,
                    "bbox": list(t.bbox),
                    "identity": t.person_id,
                    "similarity": round(t.similarity, 4),
                })
            await ws.send_json({"tracks": track_data, "threshold": use_case.threshold})

            await asyncio.sleep(1.0 / cfg.camera.fps)

    except WebSocketDisconnect:
        logger.info("WebSocket 断开")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
    finally:
        cam.release()
