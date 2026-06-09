from collections.abc import Callable

import cv2
import numpy as np

from face_recognition.domain.errors import CameraDisconnectedError

# Callable[[int], 任意]：M1 中已用过此类型注解
# 这里 cap_factory 接收 device_index，返回一个有 isOpened/read/release 方法的对象（duck typing）
# 默认指向 cv2.VideoCapture；测试时换成返回 MagicMock 的 lambda
_CapFactory = Callable[[int], "cv2.VideoCapture"]


class CameraCapture:
    """OpenCV VideoCapture 的薄封装，把 cv2 错误码翻译成领域异常。"""

    def __init__(
        self,
        device_index: int,
        resolution: tuple[int, int],
        cap_factory: _CapFactory = cv2.VideoCapture,
    ) -> None:
        self._cap = cap_factory(device_index)
        if not self._cap.isOpened():
            raise CameraDisconnectedError(f"摄像头 {device_index} 无法打开")
        # cv2 的 set 设置不一定生效（取决于驱动），但常见 USB 摄像头都支持。
        # CAP_PROP_FRAME_WIDTH/HEIGHT 是常量整数（在 cv2 命名空间下），和 ffmpeg 的属性一一对应
        w, h = resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def read(self) -> np.ndarray:
        """读取一帧。失败抛 CameraDisconnectedError。"""
        # cv2 的 read 返回 (ret: bool, frame: np.ndarray | None)。
        # 失败原因可能是摄像头被拔、被其他程序占用、驱动崩溃。
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise CameraDisconnectedError("摄像头读取失败")
        # ── 给小白：frame 的颜色顺序是 BGR，不是 RGB（OpenCV 最大的坑） ──
        # cv2.VideoCapture.read() / cv2.imread() 返回的 ndarray 形状 (H, W, 3) uint8，
        # 但**通道顺序是 BGR**（蓝-绿-红），不是新手熟悉的 RGB。这是 OpenCV 1999 年
        # 那版作者遗留的历史习惯，沿用至今。
        # 后果：
        #   - 直接喂 InsightFace `app.get(frame)` ✓ 没事——InsightFace 接 BGR
        #   - 直接喂 cv2.imencode(".jpg", frame) ✓ 没事——cv2 全家都是 BGR
        #   - 直接喂 PIL.Image.fromarray(frame) ✗ 颜色会反（人脸变蓝紫色）
        #   - 直接喂 matplotlib.imshow(frame) ✗ 同上
        # 想给非 OpenCV 工具看，要先转：`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`。
        # 本项目链路全在 OpenCV/InsightFace 内部转，**不需要**手动 cvtColor。
        return frame

    def release(self) -> None:
        """释放摄像头资源。FastAPI 关闭时调用。"""
        self._cap.release()
