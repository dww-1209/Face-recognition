"""
数据切分：读取已预分割的 LFW 数据集。

数据集由 scripts/prepare_lfw_dataset.py 生成，已按 80/20 按人切分到 train/ test/。
本模块提供加载接口，供 evaluation 和 register 复用。

约定目录结构：
  data/lfw_subset/
  ├── train/<person_name>/*.jpg          # 80% 注册集
  │             /subset_0/*.jpg          # → manual_three 子文件夹
  │             /subset_1/*.jpg
  │             /subset_2/*.jpg
  └── test/<person_name>/*.jpg           # 20% 测试集（无子文件夹）
"""

from pathlib import Path

import cv2
import numpy as np


def load_images(directory: Path) -> list[np.ndarray]:
    """加载目录下所有图片（仅 jpg/png 文件，跳过子目录）。"""
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img_path in sorted(directory.glob(ext)):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
    return images


def load_train_set(root_dir: str | Path) -> dict[str, list[np.ndarray]]:
    """
    加载训练集所有人员图片。

    Returns:
        {person_id: [images]} 字典
    """
    root = Path(root_dir) / "train"
    if not root.is_dir():
        raise FileNotFoundError(f"训练集目录不存在: {root}")

    result = {}
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        images = load_images(person_dir)
        if images:
            result[person_dir.name] = images
    return result


def load_test_set(root_dir: str | Path) -> dict[str, list[np.ndarray]]:
    """
    加载测试集所有人员图片。

    Returns:
        {person_id: [images]} 字典
    """
    root = Path(root_dir) / "test"
    if not root.is_dir():
        raise FileNotFoundError(f"测试集目录不存在: {root}")

    result = {}
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        images = load_images(person_dir)
        if images:
            result[person_dir.name] = images
    return result


def load_train_subset_images(
    root_dir: str | Path, person_id: str
) -> list[list[np.ndarray]]:
    """
    加载某人的 manual_three 子文件夹图片（3 组）。

    Returns:
        [subset_0_images, subset_1_images, subset_2_images]
    """
    person_dir = Path(root_dir) / "train" / person_id
    groups = []
    for subset_name in ("subset_0", "subset_1", "subset_2"):
        subset_dir = person_dir / subset_name
        if subset_dir.is_dir():
            groups.append(load_images(subset_dir))
        else:
            groups.append([])
    return groups
