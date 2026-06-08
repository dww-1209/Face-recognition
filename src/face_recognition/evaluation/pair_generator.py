"""
评估配对生成器：生成 Genuine / 库内 Impostor / 库外 Impostor 三组配对。

配对不存储图片，只记录 (query_person_id, template_person_id, image_index)，
由下游负责加载和编码。
"""


import numpy as np


def _pair_count(test_sets: dict[str, list]) -> dict[str, int]:
    """统计每个测试人员有多少张照片。"""
    return {pid: len(imgs) for pid, imgs in test_sets.items()}


def gen_genuine_pairs(
    train_set: dict[str, list[np.ndarray]],
    test_set: dict[str, list[np.ndarray]],
) -> list[tuple[str, str, int]]:
    """
    生成 Genuine 配对：同一人的测试照 vs 自己的模板。

    Returns:
        [(query_person_id, template_person_id, image_index), ...]
        其中 query_person_id == template_person_id
    """
    pairs = []
    for person_id, images in sorted(test_set.items()):
        if person_id not in train_set:
            continue
        for i in range(len(images)):
            pairs.append((person_id, person_id, i))
    return pairs


def gen_closed_impostor_pairs(
    train_set: dict[str, list[np.ndarray]],
    test_set: dict[str, list[np.ndarray]],
) -> list[tuple[str, str, int]]:
    """
    生成库内 Impostor 配对：张三的测试照 vs 李四的模板。

    每张测试照只与第一个不同人的模板配对（避免组合爆炸）。
    """
    train_pids = sorted(train_set.keys())
    pairs = []

    for query_pid, images in sorted(test_set.items()):
        if query_pid not in train_set:
            continue
        # 取第一个与 query 不同的注册人作为 impostor
        for tpl_pid in train_pids:
            if tpl_pid != query_pid:
                for i in range(len(images)):
                    pairs.append((query_pid, tpl_pid, i))
                break  # 每张测试照只配一个 impostor

    return pairs


def gen_open_impostor_pairs(
    open_set: dict[str, list[np.ndarray]],
    train_set: dict[str, list[np.ndarray]],
) -> list[tuple[str, str, int]]:
    """
    生成库外 Impostor 配对：库外陌生人的照片 vs 库内所有模板。

    Returns:
        [(outsider_person_id, template_person_id, image_index), ...]
    """
    train_pids = sorted(train_set.keys())
    pairs = []

    for outsider_pid, images in sorted(open_set.items()):
        # 只取第一个库内人员的模板（所有库内模板都会比较，但配对只记录一组）
        if train_pids:
            for i in range(len(images)):
                pairs.append((outsider_pid, train_pids[0], i))
    return pairs
