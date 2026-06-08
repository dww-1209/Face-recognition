from face_recognition.infrastructure.iou_tracker import IoUTracker


class TestIoUTracker:
    def test_new_box_creates_track(self):
        tracker = IoUTracker(iou_threshold=0.5)
        tracks = tracker.update([(10, 20, 100, 200)])
        assert len(tracks) == 1
        assert tracks[0].needs_recognition is True

    def test_same_box_reuses_track(self):
        tracker = IoUTracker(iou_threshold=0.5)
        t1 = tracker.update([(10, 20, 100, 200)])
        t2 = tracker.update([(12, 22, 102, 202)])  # IoU ≈ 0.92
        assert len(t2) == 1
        assert t2[0].track_id == t1[0].track_id
        assert t2[0].needs_recognition is False

    def test_different_box_creates_new_track(self):
        tracker = IoUTracker(iou_threshold=0.5)
        t1 = tracker.update([(10, 20, 100, 200)])
        t2 = tracker.update([(500, 500, 600, 600)])  # IoU = 0
        assert len(t2) == 2  # 旧 track 还在（missing=1） + 新 track

    def test_track_expires_after_max_missing(self):
        tracker = IoUTracker(iou_threshold=0.5, max_missing=2)
        tracker.update([(10, 20, 100, 200)])
        tracker.update([])  # missing=1
        tracker.update([])  # missing=2
        tracks = tracker.update([])  # missing=3 → 移除
        assert len(tracks) == 0

    def test_iou_one(self):
        assert IoUTracker._iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_iou_zero(self):
        assert IoUTracker._iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
