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
    """实时推流：每帧发送 JPEG 二进制 + 识别结果 JSON。

    画面和识别解耦：
    - 每帧都读摄像头 + 渲染 + 发 JPEG（保证画面流畅）
    - 隔 N 帧才跑一次 detect_and_encode（节省算力）
    - 中间帧复用上一次的 tracks 画框
    """
    await ws.accept()
    logger.info("WebSocket 连接建立")

    cfg = get_config()
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

    detect_interval = cfg.realtime.detect_every_n_frames
    last_tracks: list = []
    frame_idx = 0

    try:
        while True:
            try:
                frame = cam.read()
            except CameraDisconnectedError:
                await ws.send_json({"error": "CAMERA_LOST"})
                break

            frame_idx += 1

            # 隔帧检测：只在第 1 帧和每 N 帧跑检测+识别
            if frame_idx == 1 or frame_idx % detect_interval == 0:
                last_tracks = use_case.process_frame(frame)

            # 始终用最新 tracks 画框（无论是否跑了检测）
            rendered = render_tracks(frame, last_tracks)

            # 编码 JPEG
            jpeg_bytes = encode_jpeg(rendered, quality=cfg.realtime.jpeg_quality)

            # 发送二进制 JPEG
            await ws.send_bytes(jpeg_bytes)

            # 发送 JSON 元数据
            track_data = []
            for t in last_tracks:
                track_data.append({
                    "track_id": t.track_id,
                    "bbox": list(t.bbox),
                    "identity": t.person_id,
                    "similarity": round(t.similarity, 4),
                })
            await ws.send_json({"tracks": track_data, "threshold": use_case.threshold})

            # 不 sleep：让摄像头按自己节奏出帧
            await asyncio.sleep(0)

    except WebSocketDisconnect:
        logger.info("WebSocket 断开")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
    finally:
        cam.release()
