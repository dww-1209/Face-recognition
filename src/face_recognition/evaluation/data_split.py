import random
from dataclasses import dataclass
from pathlib import Path

# M1 Task 8 已经定义过的图片扩展名集合，复用同样口径
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class PersonSplit:
    """单个人的切分结果。"""
    person_id: str
    # tuple 而非 list：保持 frozen dataclass 的不可变语义一致性
    train_paths: tuple[Path, ...]
    test_paths: tuple[Path, ...]


def split_by_person(
    dataset_root: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    min_images: int = 5,
) -> list[PersonSplit]:
    """按人切分整个数据集。

    参数：
      dataset_root: 数据集根目录，子目录名 = person_id
      train_ratio:  训练集比例（默认 80%）
      seed:         随机种子（默认 42——和 spec/config.yaml 保持一致）
      min_images:   每人最少需要的图片数；不足的人直接跳过

    返回：
      list[PersonSplit]，按 person_id 字典序排序——保证下游遍历顺序确定
    """
    # 用独立的 Random 实例，不污染全局 random.* 状态（M1 Task 7 random_one 同款理由）
    rng = random.Random(seed)
    splits: list[PersonSplit] = []

    # sorted(...) 让人员遍历顺序在不同文件系统下都一致；
    # 切分本身的随机性由 seed 控制，外层顺序也固定才能完全可复现
    for person_dir in sorted(dataset_root.iterdir()):
        if not person_dir.is_dir():
            continue  # 跳过 README.md、.DS_Store 之类
        # 收集该人所有图片
        images = sorted(
            p for p in person_dir.iterdir()
            if p.suffix.lower() in _IMG_EXTS
        )
        if len(images) < min_images:
            continue

        # rng.sample(seq, k) = 从 seq 里**无放回**抽 k 个，返回新列表（不改原 seq）。
        # 等价于"洗牌后取前 k 个"，但比 shuffle + 切片更直白。
        # 这里的妙处：seq 是排过序的，加上固定 seed → 抽样结果 100% 可复现。
        n_train = int(len(images) * train_ratio)
        train_set = rng.sample(images, n_train)
        # 用 set 差集算"不在训练集里的图片"。set 运算 set(a) - set(b) = a 中减去 b 的元素
        test_set = sorted(set(images) - set(train_set))
        train_set_sorted = sorted(train_set)  # 给 train 也排序，便于断言

        splits.append(PersonSplit(
            person_id=person_dir.name,
            train_paths=tuple(train_set_sorted),
            test_paths=tuple(test_set),
        ))

    return splits
