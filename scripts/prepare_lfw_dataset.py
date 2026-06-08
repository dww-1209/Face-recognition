"""
LFW 数据集准备脚本。

从 sklearn LFW 下载 → 筛选 ≥50 张的类别 → 选 ~30 类 →
按人建文件夹 → 80/20 切分 → 每人类别下建 3 个子文件夹（供 manual_three 策略用）。

子文件夹 images 用符号链接指向根目录原图，避免重复占用磁盘。
"""

import os
import random
import shutil
from pathlib import Path

from sklearn.datasets import fetch_lfw_people

RANDOM_SEED = 42
MIN_IMAGES_PER_PERSON = 30  # LFW ≥50 的仅 12 人，≥30 的有 34 人
TARGET_CLASSES = 34  # 全取（≈ 要求的 30）
TRAIN_RATIO = 0.8

# 输出目录
OUTPUT_DIR = Path("data/lfw_subset")
# sklearn 缓存目录（下载一次后不再重复下载）
CACHE_DIR = Path.home() / "scikit_learn_data"


def download_lfw() -> tuple[list[str], list[list[int]]]:
    """
    下载 LFW funneled 版本。

    关键参数（来自 plan/m2 实测经验）：
    - slice_=None 禁用自动裁切（默认 (70:195, 78:172) 会把 250×250
      裁成 125×94，脸几乎填满整图，SCRFD 检测器需要 padding 才能识别）
    - color=True 保留 RGB 三通道
    - resize=None 保留原始分辨率（否则 SCRFD 检测框坐标会错位）

    Returns:
        (target_names, image_indices): target_names[i] 是第 i 张图的人名，
            image_indices 未使用，从 target 直接推断
    """
    print(f"加载 LFW（min_faces_per_person={MIN_IMAGES_PER_PERSON}）...")
    bunch = fetch_lfw_people(
        min_faces_per_person=MIN_IMAGES_PER_PERSON,
        color=True,
        resize=None,
        slice_=None,  # 禁用自动裁切
        data_home=CACHE_DIR,
        download_if_missing=True,
    )
    print(f"共 {bunch.images.shape[0]} 张图片，{len(bunch.target_names)} 个类别")
    return bunch  # 返回整个 bunch 便于后续处理


def select_classes(bunch, n_classes: int) -> list[int]:
    """
    从 bunch 中选出图片数最多的 n_classes 个类别。

    Returns:
        选中的 target 索引列表
    """
    import numpy as np
    from collections import Counter

    counts = Counter(bunch.target)
    # 按图片数降序排列
    sorted_classes = sorted(counts.items(), key=lambda x: -x[1])
    selected = sorted_classes[:n_classes]
    print(f"选中的 {len(selected)} 个类别：")
    for target_idx, count in selected:
        name = bunch.target_names[target_idx]
        print(f"  {name}: {count} 张")
    return [t for t, _ in selected]


def build_dataset(bunch, selected_targets: list[int], output_dir: Path):
    """
    构建项目数据集目录结构：

    data/lfw_subset/
    ├── train/
    │   └── <person_name>/
    │       ├── <person_name>_0001.jpg      # 原始图片（80% 进 train）
    │       ├── <person_name>_0002.jpg
    │       ├── ...
    │       ├── subset_0/                    # manual_three 子文件夹
    │       │   ├── <person_name>_0001.jpg  # → 符号链接
    │       │   └── ...
    │       ├── subset_1/
    │       │   └── ...
    │       └── subset_2/
    │           └── ...
    └── test/
        └── <person_name>/
            ├── <person_name>_0051.jpg      # 原始图片（20% 进 test）
            └── ...
    """
    import numpy as np
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)

    for target_idx in selected_targets:
        name = bunch.target_names[target_idx]
        # 取该人的所有图片索引
        indices = np.where(bunch.target == target_idx)[0]

        # 按 seed 随机打乱后 80/20 切分
        rng = random.Random(RANDOM_SEED)
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_train = int(len(shuffled) * TRAIN_RATIO)
        train_indices = shuffled[:n_train]
        test_indices = shuffled[n_train:]

        for split, split_indices in [("train", train_indices), ("test", test_indices)]:
            person_dir = output_dir / split / name
            person_dir.mkdir(parents=True, exist_ok=True)

            # 保存原始图片到根目录
            saved_paths = []
            for i, idx in enumerate(split_indices):
                # sklearn 返回的 images 是 float32 [0, 1]（源码 face /= 255.0），
                # 必须先 ×255 再 cast uint8
                img = (bunch.images[idx] * 255).clip(0, 255).astype(np.uint8)
                filename = f"{name}_{i:04d}.jpg"
                filepath = person_dir / filename
                Image.fromarray(img).save(filepath)
                saved_paths.append(filename)

            # 对 train 集，建 3 个子文件夹（均匀分配图片）
            if split == "train":
                # 随机打乱后均匀分 3 份
                rng.shuffle(saved_paths)
                chunk_size = max(1, len(saved_paths) // 3)
                for subset_idx in range(3):
                    subset_dir = person_dir / f"subset_{subset_idx}"
                    subset_dir.mkdir(exist_ok=True)
                    start = subset_idx * chunk_size
                    if subset_idx == 2:
                        end = len(saved_paths)
                    else:
                        end = (subset_idx + 1) * chunk_size
                    for filename in saved_paths[start:end]:
                        # 符号链接指向根目录原图
                        link_path = subset_dir / filename
                        target_path = person_dir / filename
                        if not link_path.exists():
                            link_path.symlink_to(os.path.relpath(target_path, subset_dir))

            print(f"  [{split}] {name}: {len(split_indices)} 张图片")


def main():
    random.seed(RANDOM_SEED)

    # 如果已存在则跳过下载
    if (OUTPUT_DIR / "train").exists() and list((OUTPUT_DIR / "train").iterdir()):
        print(f"数据集已存在于 {OUTPUT_DIR.resolve()}，跳过下载。")
        print(f"如需重新生成请删除后重跑：rm -rf {OUTPUT_DIR}")
        return

    bunch = download_lfw()
    selected = select_classes(bunch, TARGET_CLASSES)
    build_dataset(bunch, selected, OUTPUT_DIR)

    # 输出统计
    train_people = list((OUTPUT_DIR / "train").iterdir())
    test_people = list((OUTPUT_DIR / "test").iterdir())
    print(f"\n完成！train: {len(train_people)} 人, test: {len(test_people)} 人")

    # 验证 subfolder 结构
    for person_dir in sorted(train_people)[:3]:
        n_subsets = len(list(person_dir.glob("subset_*")))
        n_links = sum(1 for _ in (person_dir / "subset_0").iterdir())
        n_images = len(list(person_dir.glob("*.jpg")))
        print(f"  验证 {person_dir.name}: {n_images} 张图片, "
              f"{n_subsets} 个子文件夹, subset_0 有 {n_links} 个链接")


if __name__ == "__main__":
    main()
