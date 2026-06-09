import numpy as np
import pytest

from face_recognition.evaluation.metrics import (
    compute_roc,
    compute_eer,
    compute_tar_at_far,
    compute_top1_accuracy,
    compute_top1_with_threshold,
)
from face_recognition.evaluation.types import EvalEncoding, PairResult


def _pair(score: float, is_genuine: bool) -> PairResult:
    return PairResult(score=score, is_genuine=is_genuine, query_person="q", template_person="t")


def test_roc_perfect_separation():
    """完全分开的 case：genuine 全 1.0，impostor 全 0.0 → AUC=1.0，ROC 经过 (0, 1)。"""
    pairs = [_pair(1.0, True), _pair(1.0, True), _pair(0.0, False), _pair(0.0, False)]
    fpr, tpr, thresholds = compute_roc(pairs)
    # ROC 一定从 (0, 0) 起、(1, 1) 终。完美分开时中间会经过 (0, 1)
    # np.isclose 比 == 安全：浮点比较留 1e-7 容差
    assert np.any(np.isclose(fpr, 0.0) & np.isclose(tpr, 1.0))


def test_eer_perfect_classifier_is_zero():
    """完美分类器 EER=0：FAR 和 FRR 在 0% 处相等。"""
    pairs = [_pair(1.0, True), _pair(0.0, False)]
    eer, threshold = compute_eer(pairs)
    # pytest.approx(expected, abs=...) 是浮点比较的标准写法
    assert eer == pytest.approx(0.0, abs=1e-3)


def test_eer_random_classifier_is_around_half():
    """完全随机分类器 EER ≈ 0.5。"""
    rng = np.random.default_rng(42)
    pairs = []
    for _ in range(500):
        pairs.append(_pair(float(rng.random()), True))
        pairs.append(_pair(float(rng.random()), False))
    eer, _ = compute_eer(pairs)
    # 随机分类器理论上 EER=0.5，500 样本下波动允许 ± 0.1
    assert 0.4 < eer < 0.6


def test_tar_at_far_decreases_when_far_threshold_strict():
    """卡更严的 FAR（更小）时 TAR 不应升高（单调性）。"""
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(200):
        # 制造可分但有重叠的分布
        pairs.append(_pair(0.7 + 0.1 * rng.standard_normal(), True))
        pairs.append(_pair(0.3 + 0.1 * rng.standard_normal(), False))
    tar_loose = compute_tar_at_far(pairs, target_far=0.05)   # 5% FAR
    tar_strict = compute_tar_at_far(pairs, target_far=0.001) # 0.1% FAR
    assert tar_loose >= tar_strict


def test_top1_accuracy_picks_correct_template():
    """Top-1：每个测试编码找到余弦最近的 template，命中本人则计数。"""
    # 造 3 个互相正交的"原型向量"
    rng = np.random.default_rng(0)
    a = rng.standard_normal(512); a /= np.linalg.norm(a)
    b = rng.standard_normal(512); b /= np.linalg.norm(b)
    c = rng.standard_normal(512); c /= np.linalg.norm(c)
    test = [
        EvalEncoding(vector=a.astype(np.float32), person_id="A", image_path=""),
        EvalEncoding(vector=b.astype(np.float32), person_id="B", image_path=""),
    ]
    # 每人 1 个模板的简单情形——多模板由 _max_cosine 路径覆盖
    templates = {
        "A": [EvalEncoding(vector=a.astype(np.float32), person_id="A", image_path="")],
        "B": [EvalEncoding(vector=b.astype(np.float32), person_id="B", image_path="")],
        "C": [EvalEncoding(vector=c.astype(np.float32), person_id="C", image_path="")],
    }
    acc = compute_top1_accuracy(test, templates)
    assert acc == pytest.approx(1.0)


def test_top1_with_threshold_rejects_low_scores():
    """带阈值的 Top-1:即便 argmax 选对人,分数 < threshold 也算错(判 unknown)。"""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(512); a /= np.linalg.norm(a)
    b = rng.standard_normal(512); b /= np.linalg.norm(b)
    # 造一个"略偏离 a"的 query——还是和 a 最像,但 cos 不到阈值
    a_blur = a + 0.5 * rng.standard_normal(512)
    a_blur = a_blur / np.linalg.norm(a_blur)

    test = [EvalEncoding(vector=a_blur.astype(np.float32), person_id="A", image_path="")]
    templates = {
        "A": [EvalEncoding(vector=a.astype(np.float32), person_id="A", image_path="")],
        "B": [EvalEncoding(vector=b.astype(np.float32), person_id="B", image_path="")],
    }
    # 无阈值:argmax 选对了 A → 命中
    assert compute_top1_accuracy(test, templates) == pytest.approx(1.0)
    # 阈值 0.99 严格到 a_blur 都过不去 → 拒识 → 算错
    assert compute_top1_with_threshold(test, templates, threshold=0.99) == pytest.approx(0.0)
    # 阈值 0.0 形同虚设 → 退化到无阈值 Top-1
    assert compute_top1_with_threshold(test, templates, threshold=0.0) == pytest.approx(1.0)
