"""
评估指标：ROC、EER、TAR@FAR、FRR/FAR 等。

所有函数输入两组分数：
- genuine_scores: 同人相似度（应高）
- impostor_scores: 异人相似度（应低）
"""

import numpy as np


def compute_roc_curve(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    n_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 ROC 曲线。

    Returns:
        (far_array, tar_array, thresholds)
        其中 TAR = 1 - FRR
    """
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(all_scores.min() - 0.01, all_scores.max() + 0.01, n_points)

    far = np.zeros_like(thresholds)
    tar = np.zeros_like(thresholds)
    n_impostor = len(impostor_scores)
    n_genuine = len(genuine_scores)

    for i, thresh in enumerate(thresholds):
        far[i] = np.sum(impostor_scores >= thresh) / n_impostor
        tar[i] = np.sum(genuine_scores >= thresh) / n_genuine

    return far, tar, thresholds


def compute_eer(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> tuple[float, float]:
    """
    计算 EER（Equal Error Rate）。

    Returns:
        (eer, threshold_at_eer)
    """
    far, tar, thresholds = compute_roc_curve(genuine_scores, impostor_scores)
    frr = 1.0 - tar

    # 找 FAR 和 FRR 最接近的点
    diff = np.abs(far - frr)
    best_idx = int(np.argmin(diff))
    eer = (far[best_idx] + frr[best_idx]) / 2.0

    return float(eer), float(thresholds[best_idx])


def compute_tar_at_far(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    target_far: float = 0.001,
) -> float:
    """
    计算 TAR @ FAR = target_far。

    按 FAR ≤ target_far 找最高 TAR。
    """
    far, tar, thresholds = compute_roc_curve(genuine_scores, impostor_scores)

    # 找所有 FAR ≤ target_far 的索引
    valid = far <= target_far
    if not np.any(valid):
        return 0.0

    # 取其中 TAR 最大的
    best_tar = float(np.max(tar[valid]))
    return best_tar


def compute_far_frr(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    """
    在给定阈值下计算 FAR 和 FRR。

    Returns:
        (far, frr)
    """
    far = float(np.sum(impostor_scores >= threshold) / len(impostor_scores))
    frr = float(np.sum(genuine_scores < threshold) / len(genuine_scores))
    return far, frr


def compute_top1_accuracy(
    query_vectors: np.ndarray,  # (N, 512)
    template_matrix: np.ndarray,  # (M, 512)
    query_person_ids: list[str],
    template_person_ids: list[str],
) -> float:
    """
    计算闭集 Top-1 准确率。

    对每张 query，找相似度最高的模板，判断 person_id 是否匹配。
    """
    if len(query_vectors) == 0:
        return 0.0

    # (N, 512) @ (512, M) → (N, M)
    similarities = query_vectors @ template_matrix.T
    best_indices = np.argmax(similarities, axis=1)

    correct = 0
    for i, best_idx in enumerate(best_indices):
        if query_person_ids[i] == template_person_ids[best_idx]:
            correct += 1

    return correct / len(query_vectors)
