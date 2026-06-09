import random
from dataclasses import dataclass

import numpy as np
# sklearn.datasets.fetch_lfw_people = 一行 LFW 下载器。
#   - 第一次调用：从 vis-www.cs.umass.edu 下载 233MB tarball，解压到 ~/scikit_learn_data/lfw_home/
#   - 之后调用：直接读缓存，几乎零成本
#   - 选它而非"用户手动放图到 data/lfw_subset/"：可复现性更强（任何机器跑都拿到同一份）
from sklearn.datasets import fetch_lfw_people


@dataclass(frozen=True)
class LfwImage:
    """LFW 一张图。和 EvalEncoding 不同——LFW 阶段还没过 ArcFace，没有向量。"""
    image: np.ndarray   # H × W × 3 uint8（注意 sklearn 默认是 RGB，cv2 是 BGR——下游做转换）
    person_name: str    # LFW 里的姓名，作为伪 person_id 使用


def load_lfw_subset(n_persons: int = 50, seed: int = 42) -> list[LfwImage]:
    """从 LFW 抽 n_persons 个不同的人，每人取 1 张图。

    设计选择：
      - **每人只抽 1 张**：评估只需要"不同陌生人"的多样性，同人多张反而引入相关性
      - **从图库抽人而非按 LFW 的 official splits**：我们不做闭集分类，
        不需要 sklearn 的 train/test 划分，自己抽样更直白
    """
    # 三个关键参数（少一个都跑不通,已踩坑）：
    # 1. color=True       拿 RGB 三通道（默认是灰度）。InsightFace 需要彩色。
    # 2. resize=None      不缩放,保持 250×250。（默认 resize=0.5 → 125×125,小脸检测吃力）
    # 3. slice_=None      ⚠️ 必须显式禁用默认的紧裁切。
    #     fetch_lfw_people 默认 slice_=(slice(70,195), slice(78,172)) → 125×94,
    #     脸几乎填满整图,SCRFD 检测器需要 padding 才能识别 → **100% 检测失败**。
    #     设 slice_=None 拿原始 250×250,脸占大约一半,SCRFD 才有空间画 anchor。
    # min_faces_per_person=1：默认是 50,会把 LFW 13000+ 人砍到 158 人。
    #     我们只要"任意陌生人",门槛设到 1 拿全 5749 个人备选。
    bunch = fetch_lfw_people(
        color=True, resize=None, slice_=None, min_faces_per_person=1
    )

    # bunch.target 是 (N,) 的 person 索引数组，bunch.target_names 是名字列表
    # numpy.unique(arr) = 返回去重后排序的数组——拿到所有不同的 person 索引
    unique_persons = np.unique(bunch.target)

    # 用独立 Random 实例做抽样
    rng = random.Random(seed)
    chosen_indices = rng.sample(list(unique_persons), n_persons)

    images: list[LfwImage] = []
    for pid in chosen_indices:
        # np.where(condition) 返回满足条件的下标元组。bunch.target == pid 是布尔数组，
        # where 给出每个 True 的位置。[0] 取第 0 维（一维数组只有这维）
        candidate_idx = np.where(bunch.target == pid)[0]
        # 该人多张图里取第一张就行（评估侧不关心选哪张，只关心是陌生人）
        first_idx = int(candidate_idx[0])
        # ⚠️ **bunch.images 是 float32 范围 [0.0, 1.0]**,不是 [0, 255]。
        # 这一点 sklearn 文档不显眼,源码 _lfw.py 里 `face /= 255.0` 才看得到——
        # 直接 .astype(np.uint8) 会得到**全零数组**(所有像素 < 1.0 → 截断为 0),
        # 下游 SCRFD 检测全军覆没。必须先 ×255 还原到 [0, 255] 再 cast。
        # .clip(0, 255) 防止极端浮点抖动越界;.astype(uint8) 截断小数。
        img_uint8 = (bunch.images[first_idx] * 255.0).clip(0, 255).astype(np.uint8)
        images.append(LfwImage(
            image=img_uint8,
            person_name=str(bunch.target_names[pid]),
        ))

    return images
