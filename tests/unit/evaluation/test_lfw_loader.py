from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from face_recognition.evaluation.lfw_loader import (
    LfwImage,
    load_lfw_subset,
)


# 这是个慢测试（要联网下载几十 MB），用 -m slow 显式才跑
@pytest.mark.slow
def test_load_lfw_subset_returns_n_images_real():
    """真下载 LFW 拿 50 张图，验证形状和数量。需联网 + sklearn cache。"""
    images = load_lfw_subset(n_persons=50, seed=42)
    assert len(images) == 50
    # LFW 图默认 250×250 RGB；fetch_lfw_people 默认 resize=0.5 → 62×47×3，
    # 但我们要求按 None resize 拿原图给 InsightFace 用。具体在 Step 3 解释参数
    img = images[0]
    assert isinstance(img, LfwImage)
    assert img.image.ndim == 3      # H × W × C
    assert img.image.dtype == np.uint8
    assert img.person_name != ""


def test_load_lfw_subset_uses_seed_for_determinism():
    """同 seed 选同样 50 个人——避免每次跑实验抽不同人导致不可复现。"""
    # 用 patch 替换 fetch_lfw_people 返回假数据，避免真下载
    fake_bunch = MagicMock()
    # sklearn 的 Bunch 对象暴露 .images / .target / .target_names
    # 假装库里有 200 个人，每人 1 张图
    fake_bunch.images = np.random.RandomState(0).rand(200, 100, 100, 3) * 255
    fake_bunch.target = np.arange(200)
    fake_bunch.target_names = np.array([f"P{i}" for i in range(200)])

    # patch 装饰器：在测试期间把目标对象替换成 mock，结束后还原。
    # ── 给小白：为什么 patch 路径写 lfw_loader.fetch_lfw_people 而不是 sklearn.datasets.fetch_lfw_people ──
    # 这是 mock.patch **最容易踩的坑**："patch 的目标是被使用的位置，不是定义位置。"
    #   1) `lfw_loader.py` 顶部写了 `from sklearn.datasets import fetch_lfw_people`，
    #      这一刻 Python 把函数对象绑定到 lfw_loader 模块下的同名变量上——`lfw_loader.
    #      fetch_lfw_people` 现在指向那个函数。
    #   2) 后续 `load_lfw_subset` 内部调用的是 `lfw_loader.fetch_lfw_people` 这个**模块属性**。
    #   3) patch 要替换的就是这个属性。如果误写 `patch("sklearn.datasets.fetch_lfw_people")`，
    #      只改了 sklearn 那边的引用——lfw_loader 里早已绑定的旧引用一动不动，mock 失效，
    #      测试还是会真的去下载 LFW 数据集（500MB，且联网失败就报错）。
    # 一句话记法：**patch 谁 import 它的那个文件**，不是 patch 它定义的地方。
    with patch("face_recognition.evaluation.lfw_loader.fetch_lfw_people", return_value=fake_bunch):
        a = load_lfw_subset(n_persons=10, seed=42)
        b = load_lfw_subset(n_persons=10, seed=42)
    assert [x.person_name for x in a] == [x.person_name for x in b]
