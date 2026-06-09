import numpy as np

from face_recognition.infrastructure.frame_renderer import (
    encode_jpeg,
    render_tracks,
)
from face_recognition.infrastructure.iou_tracker import Track


def _blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_render_tracks_draws_on_frame():
    """画框后帧不再全黑——至少画出来的位置有非零像素。"""
    frame = _blank_frame()
    tracks = [Track(track_id=0, bbox=(100, 100, 300, 300), person_id="alice", similarity=0.85)]
    out = render_tracks(frame, tracks)
    # 矩形线条上应该有非零像素（OpenCV 默认线条颜色是绿色 (0, 255, 0)）
    # 我们检查框的左上角 1×1 区域 ——
    assert np.any(out[100:101, 100:101] > 0)
    # 输入不被修改：render_tracks 返回新数组，不污染原 frame
    assert np.all(frame == 0)


def test_render_tracks_handles_unknown_identity():
    """person_id=None（未识别）不报错，应该写"未知"或类似标签。"""
    frame = _blank_frame()
    tracks = [Track(track_id=0, bbox=(50, 50, 200, 200), person_id=None)]
    out = render_tracks(frame, tracks)
    # 不崩溃就是过；具体文字内容不强求（避免 OpenCV 字体测试脆性）
    assert out.shape == frame.shape


def test_encode_jpeg_returns_valid_bytes():
    """encode_jpeg 返回的 bytes 应该以 JPEG magic header 开头。"""
    frame = _blank_frame()
    data = encode_jpeg(frame, quality=80)
    # JPEG 文件头：FF D8 FF（任何编码器都遵循）
    assert data[:3] == b"\xff\xd8\xff"
    assert len(data) > 100  # 全黑图也得有几百字节
