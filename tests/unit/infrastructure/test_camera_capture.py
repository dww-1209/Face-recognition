from unittest.mock import MagicMock

import numpy as np
import pytest

from face_recognition.domain.errors import CameraDisconnectedError
from face_recognition.infrastructure.camera_capture import CameraCapture


def test_camera_open_failure_raises_domain_error():
    """cv2.VideoCapture 打不开（isOpened()=False）→ 抛 CameraDisconnectedError。"""
    fake_cv2_cap = MagicMock()
    fake_cv2_cap.isOpened.return_value = False
    # cap_factory 是依赖注入点：测试传 mock 工厂，生产传 cv2.VideoCapture
    with pytest.raises(CameraDisconnectedError):
        CameraCapture(device_index=0, resolution=(640, 480), cap_factory=lambda i: fake_cv2_cap)


def test_read_returns_frame_when_ok():
    """read() 返回 (True, frame) 时正常出帧。"""
    fake_cv2_cap = MagicMock()
    fake_cv2_cap.isOpened.return_value = True
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_cv2_cap.read.return_value = (True, fake_frame)

    cam = CameraCapture(device_index=0, resolution=(640, 480), cap_factory=lambda i: fake_cv2_cap)
    frame = cam.read()
    # 形状校验，确认我们没乱处理 frame
    assert frame.shape == (480, 640, 3)


def test_read_returns_false_raises_domain_error():
    """read() 返回 (False, None) 表示采集失败 → 抛 CameraDisconnectedError。"""
    fake_cv2_cap = MagicMock()
    fake_cv2_cap.isOpened.return_value = True
    fake_cv2_cap.read.return_value = (False, None)

    cam = CameraCapture(device_index=0, resolution=(640, 480), cap_factory=lambda i: fake_cv2_cap)
    with pytest.raises(CameraDisconnectedError):
        cam.read()


def test_release_calls_cv2_release():
    """release() 应该转发到 cv2 cap.release，避免摄像头资源泄漏。"""
    fake_cv2_cap = MagicMock()
    fake_cv2_cap.isOpened.return_value = True
    cam = CameraCapture(device_index=0, resolution=(640, 480), cap_factory=lambda i: fake_cv2_cap)
    cam.release()
    fake_cv2_cap.release.assert_called_once()
