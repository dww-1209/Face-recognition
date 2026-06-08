import numpy as np

from face_recognition.evaluation.metrics import (
    compute_eer,
    compute_far_frr,
    compute_roc_curve,
    compute_tar_at_far,
    compute_top1_accuracy,
)


class TestROC:
    def test_roc_curve_shape(self):
        genuine = np.array([0.9, 0.8, 0.7, 0.6])
        impostor = np.array([0.3, 0.2, 0.1])
        far, tar, thresholds = compute_roc_curve(genuine, impostor, n_points=100)
        assert len(far) == 100
        assert len(tar) == 100
        assert len(thresholds) == 100
        assert np.all(far >= 0) and np.all(far <= 1)
        assert np.all(tar >= 0) and np.all(tar <= 1)

    def test_perfect_separation_roc(self):
        genuine = np.ones(100) * 0.9
        impostor = np.ones(100) * 0.1
        far, tar, _ = compute_roc_curve(genuine, impostor, n_points=1000)
        # 在 threshold=0.5 附近：FAR≈0, TAR≈1
        assert np.min(far) < 0.01
        assert np.max(tar) > 0.99


class TestEER:
    def test_eer_on_well_separated_data(self):
        genuine = np.random.normal(0.7, 0.1, 1000)
        impostor = np.random.normal(0.3, 0.1, 1000)
        eer, threshold = compute_eer(genuine, impostor)
        assert 0.0 < eer < 0.1
        assert 0.4 < threshold < 0.6

    def test_eer_on_identical_distributions(self):
        data = np.random.normal(0.5, 0.1, 1000)
        eer, _ = compute_eer(data, data.copy())
        assert eer > 0.4  # EER of identical distributions ≈ 0.5


class TestTARatFAR:
    def test_tar_at_far_zero(self):
        genuine = np.array([0.9, 0.8])
        impostor = np.array([0.1, 0.2])
        tar = compute_tar_at_far(genuine, impostor, target_far=0.01)
        assert tar >= 0.0

    def test_tar_at_far_impossible(self):
        genuine = np.array([0.9])
        impostor = np.array([0.9])  # 完全不可区分
        tar = compute_tar_at_far(genuine, impostor, target_far=0.0)
        assert tar >= 0.0


class TestFARFRR:
    def test_low_threshold_high_far(self):
        genuine = np.array([0.8, 0.7])
        impostor = np.array([0.6, 0.5])
        far, frr = compute_far_frr(genuine, impostor, threshold=0.55)
        assert far > 0.0
        assert frr == 0.0

    def test_high_threshold_high_frr(self):
        genuine = np.array([0.8, 0.7])
        impostor = np.array([0.6, 0.5])
        far, frr = compute_far_frr(genuine, impostor, threshold=0.75)
        assert far == 0.0
        assert frr > 0.0


class TestTop1:
    def test_top1_perfect(self):
        q = np.eye(3, 512, dtype=np.float32)  # 3 query, 每行正交
        tpl = q.copy()
        q_ids = ["a", "b", "c"]
        t_ids = ["a", "b", "c"]
        acc = compute_top1_accuracy(q, tpl, q_ids, t_ids)
        assert acc == 1.0
