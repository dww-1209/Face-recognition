"""WebSocket 实时推流：JPEG 帧 + JSON 元数据。

架构：主线程读帧→发送（流畅画面），后台线程跑检测识别（不卡画面）。
"""

import asyncio
import logging
import threading
import time

import cv2

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
    """实时推流：主线程只读帧+发送，后台线程异步跑检测。"""
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

    # 线程间共享状态
    lock = threading.Lock()
    latest_frame = None
    last_tracks: list = []
    running = True

    def detect_loop():
        """后台线程：每隔 200ms 拿最新帧跑一次检测（5fps 够跟踪用了）。"""
        nonlocal latest_frame, last_tracks
        while running:
            with lock:
                frame = latest_frame
            if frame is not None:
                try:
                    tracks = use_case.process_frame(frame)
                    with lock:
                        last_tracks = tracks
                except Exception as e:
                    logger.warning(f"检测线程出错: {e}")
            time.sleep(0.2)  # 200ms → 每秒 5 次检测

    detect_thread = threading.Thread(target=detect_loop, daemon=True)
    detect_thread.start()

    t_last = time.perf_counter()
    frame_count = 0
    try:
        while True:
            try:
                frame = cam.read()
            except CameraDisconnectedError:
                await ws.send_json({"error": "CAMERA_LOST"})
                break

            t_now = time.perf_counter()
            dt = (t_now - t_last) * 1000
            t_last = t_now
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"帧间隔: {dt:.0f}ms (≈{1000/dt:.0f}fps)")

            # macOS 上 cv2.set 改分辨率无效，手动缩到 640 保证流畅
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, int(h * 640 / w)))

            # 更新最新帧（检测线程会自己来拿）
            with lock:
                latest_frame = frame
                current_tracks = list(last_tracks)

            # 渲染 + 编码 + 发送
            rendered = render_tracks(frame, current_tracks)
            jpeg_bytes = encode_jpeg(rendered, quality=cfg.realtime.jpeg_quality)
            await ws.send_bytes(jpeg_bytes)

            track_data = []
            for t in current_tracks:
                track_data.append({
                    "track_id": t.track_id,
                    "bbox": list(t.bbox),
                    "identity": t.person_id,
                    "similarity": round(t.similarity, 4),
                })
            await ws.send_json({"tracks": track_data, "threshold": use_case.threshold})

    except WebSocketDisconnect:
        logger.info("WebSocket 断开")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
    finally:
        running = False  # 通知检测线程停止（daemon，进程结束自动回收）
        cam.release()    # 检测线程不碰摄像头，直接释放不冲突
