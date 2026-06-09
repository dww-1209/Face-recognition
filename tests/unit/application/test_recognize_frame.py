"""测 RecognizeFrame 用例的编排逻辑。所有外部依赖全 mock。"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from face_recognition.application.recognize_frame import RecognizeFrame
from face_recognition.domain.entities import DetectedFace, FaceEncoding
from face_recognition.infrastructure.iou_tracker import IoUTracker


def _fake_face(bbox, vec_value=0.1):
    """造一个 DetectedFace，模拟 pipeline.detect_and_encode 的返回项。"""
    v = np.full(512, vec_value, dtype=np.float32)
    enc = FaceEncoding(vector=v / np.linalg.norm(v), model_version="buffalo_l")
    return DetectedFace(bbox=bbox, encoding=enc)


def test_first_frame_detects_and_recognizes():
    """第一帧：tracker 是空的 → 所有 detection 都是新 track → 全部触发识别。"""
    # mock 三个依赖
    pipeline = MagicMock()
    pipeline.detect_and_encode.return_value = [
        _fake_face((10, 10, 50, 50), vec_value=0.1),
    ]
    tracker = IoUTracker(iou_threshold=0.5, max_missing=15)
    matrix = MagicMock()
    matrix.query.return_value = ("alice", 0.85)

    use_case = RecognizeFrame(pipeline=pipeline, tracker=tracker,
                              template_matrix=matrix, threshold=0.45,
                              recheck_interval=30)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracks = use_case.process_frame(frame)

    assert len(tracks) == 1
    assert tracks[0].person_id == "alice"
    assert tracks[0].similarity == pytest.approx(0.85)
    assert tracks[0].needs_recognition is False  # 识别完成后置 False
    matrix.query.assert_called_once()


def test_below_threshold_marks_unknown():
    """相似度低于 threshold → person_id 设为 None（未知人）。"""
    pipeline = MagicMock()
    pipeline.detect_and_encode.return_value = [
        _fake_face((10, 10, 50, 50)),
    ]
    tracker = IoUTracker()
    matrix = MagicMock()
    matrix.query.return_value = ("bob", 0.30)  # 远低于阈值 0.45

    use_case = RecognizeFrame(pipeline=pipeline, tracker=tracker,
                              template_matrix=matrix, threshold=0.45,
                              recheck_interval=30)

    tracks = use_case.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))

    assert tracks[0].person_id is None
    assert tracks[0].similarity == pytest.approx(0.30)


def test_existing_track_skips_recognition():
    """已识别过的 track（needs_recognition=False）短期内不再触发 query。"""
    pipeline = MagicMock()
    # 两帧都返回同一位置的脸 → IoU 高 → 同一 track
    pipeline.detect_and_encode.return_value = [_fake_face((10, 10, 50, 50))]
    tracker = IoUTracker(iou_threshold=0.5)
    matrix = MagicMock()
    matrix.query.return_value = ("alice", 0.85)

    # recheck_interval 设大数：保证 2 帧内不会触发重识别
    use_case = RecognizeFrame(pipeline=pipeline, tracker=tracker,
                              template_matrix=matrix, threshold=0.45,
                              recheck_interval=10_000)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    use_case.process_frame(frame)  # 第一帧：识别一次
    use_case.process_frame(frame)  # 第二帧：同一 track，不再识别

    assert matrix.query.call_count == 1


def test_track_re_recognizes_after_recheck_interval():
    """已识别的 track 经过 N 帧后应重新识别一次。"""
    pipeline = MagicMock()
    pipeline.detect_and_encode.return_value = [_fake_face((10, 10, 50, 50))]
    tracker = IoUTracker(iou_threshold=0.5)
    matrix = MagicMock()
    # 第一次识别：低分被判 unknown（person_id=None）
    # 第四次识别（recheck）：高分命中 alice
    matrix.query.side_effect = [("bob", 0.20), ("alice", 0.85)]

    use_case = RecognizeFrame(pipeline=pipeline, tracker=tracker,
                              template_matrix=matrix, threshold=0.45,
                              recheck_interval=3)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # 帧 0：识别 → unknown
    tracks = use_case.process_frame(frame)
    assert tracks[0].person_id is None
    # 帧 1：在 recheck 窗口内，跳过（tracker.update 将 needs_recognition 置 False）
    use_case.process_frame(frame)
    assert matrix.query.call_count == 1
    # 帧 2：frame_count=3, 3%3==0 → 触发重识别 → alice
    tracks = use_case.process_frame(frame)
    assert matrix.query.call_count == 2
    assert tracks[0].person_id == "alice"
    assert tracks[0].similarity == pytest.approx(0.85)


def test_no_faces_returns_empty():
    """画面没人 → 返回空列表，不报错。"""
    pipeline = MagicMock()
    pipeline.detect_and_encode.return_value = []
    tracker = IoUTracker()
    matrix = MagicMock()

    use_case = RecognizeFrame(pipeline=pipeline, tracker=tracker,
                              template_matrix=matrix, threshold=0.45,
                              recheck_interval=30)

    tracks = use_case.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert tracks == []
