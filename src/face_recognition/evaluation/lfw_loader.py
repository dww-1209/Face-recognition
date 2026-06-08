"""
LFW 库外陌生人加载器。

用于生成 open impostor 配对：从 LFW 中取与库内人员不重叠的类别。
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class LFWOpenSetLoader:
    """
    从 LFW sklearn 数据集中加载库外陌生人。

    注意：库内人员来自 prepare_lfw_dataset.py 选出的 ~30 类。
    库外陌生人从 LFW 中取其他类别（与库内不重叠）。
    """

    def __init__(self, min_faces: int = 10, random_seed: int = 42):
        self.min_faces = min_faces
        self.random_seed = random_seed

    def load_outsiders(
        self, in_library_names: set[str], n_outsiders: int = 50
    ) -> dict[str, list[np.ndarray]]:
        """
        从 LFW 中加载库外陌生人。

        Args:
            in_library_names: 库内人员名集合，将与 LFW 类别取差集。
            n_outsiders: 需要的库外人数。

        Returns:
            {person_name: [images]} — 每个库外人取 1 张图片。
        """
        from sklearn.datasets import fetch_lfw_people

        # 下载完整的 LFW（不限类别数）
        bunch = fetch_lfw_people(
            min_faces_per_person=self.min_faces,
            color=True,
            resize=None,
            slice_=None,
            download_if_missing=True,
        )

        import random

        rng = random.Random(self.random_seed)

        outsiders = {}
        for target_idx, name in enumerate(bunch.target_names):
            if name in in_library_names:
                continue

            indices = np.where(bunch.target == target_idx)[0]
            if len(indices) == 0:
                continue

            # 从该人的图片中随机选 1 张
            chosen = rng.choice(indices)
            img = (bunch.images[chosen] * 255).clip(0, 255).astype(np.uint8)
            outsiders[name] = [img]

            if len(outsiders) >= n_outsiders:
                break

        logger.info(f"加载 {len(outsiders)} 个库外陌生人")
        return outsiders
