"""OpenCV 摄像头采集封装。"""

import logging
import threading

import cv2
import numpy as np

from face_recognition.domain.errors import CameraDisconnectedError

logger = logging.getLogger(__name__)


class CameraCapture:
    """后台线程持续采集摄像头帧，线程安全的最新帧读取。"""

    def __init__(self, device_index: int = 0, resolution: tuple[int, int] = (1280, 720)):
        self.device_index = device_index
        self.resolution = resolution
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        if not self._cap.isOpened():
            raise CameraDisconnectedError(f"无法打开摄像头 {self.device_index}")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"摄像头启动: {self.resolution}")

    def _capture_loop(self) -> None:
        while self._running and self._cap is not None:
            ret, frame = self._cap.read()
            if not ret:
                logger.error("摄像头读取失败")
                self._running = False
                break
            with self._lock:
                self._frame = frame

    def read(self) -> np.ndarray:
        """获取最新帧（非阻塞）。"""
        with self._lock:
            if self._frame is None:
                raise CameraDisconnectedError("尚无帧可用")
            return self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info("摄像头已释放")
