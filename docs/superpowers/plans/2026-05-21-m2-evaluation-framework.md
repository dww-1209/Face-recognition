# M2 评估框架实施计划（5 策略消融 + ROC/EER + 阈值定标）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `evaluation/` 模块的 6 个文件 + 跑通 5 策略消融实验，产出 ROC/EER/TAR\@FAR=1e-3 三组指标，最终把最优阈值写回 `config.yaml`。

**Architecture:** 5 步流水线 = `data_split`（按人 80/20 切）→ `lfw_loader`（拉库外陌生人）→ `pair_generator`（造 Genuine/库内 Impostor/库外 Impostor 三组对）→ `metrics`（ROC/EER/TAR\@FAR）→ `reports`（CSV/PNG/Markdown 落盘）→ `run_ablation`（编排所有策略跑一遍）。所有计算在内存里完成，**不污染**生产 SQLite。

**Tech Stack:** scikit-learn（LFW 下载 + 暴力余弦）、numpy（向量批处理）、matplotlib（ROC 图 + 直方图）、pandas（CSV 写盘 + 表格）、pytest（合成数据单测）。

---

## 任务清单（10 个）

> 教材风格：首次出现的 API/装饰器/方法详细解释，第二次起从简。M1 已经讲过的（np.linalg.norm、np.stack、np.random.default_rng、@dataclass(frozen=True)、Protocol、pytest fixture、cv2.imread 等）一律不再重复。

| # | Task | 类型 |
| --- | --- | --- |
| 1 | 评估实体（EvalEncoding、PairResult、StrategyMetrics 等数据类） | TDD |
| 2 | data_split.py：按人 80/20 切分 | TDD |
| 3 | lfw_loader.py：sklearn 拉 LFW 抽 50 人 | TDD（mock + slow 集成） |
| 4 | pair_generator.py：造三组配对（Genuine / 库内 / 库外） | TDD |
| 5 | metrics.py：ROC、EER、TAR\@FAR、Top-1 | TDD（合成数据） |
| 6 | reports.py：CSV、ROC 叠图、直方图、Markdown 报告 | TDD（落盘断言） |
| 7 | embedder：把图片目录批量转成 EvalEncoding 集合（共用辅助） | TDD |
| 8 | run_ablation.py：主入口编排 5 策略 × 3 评估组 | 集成 |
| 9 | **跑真实实验**（M3 任务）：在私有数据集上跑 ablation，看结果 | 手动 |
| 10 | 把最优阈值/策略写进 config.yaml + summary.md 写结论 | 手动 |

---

### Task 1: 评估专用数据类

**Files:**
- Create: `src/face_recognition/evaluation/types.py`
- Test: `tests/unit/evaluation/test_types.py`

为什么单独建一个 `types.py`？评估流水线有自己的语义概念（`EvalEncoding` 含 `person_id` 和 `image_path`，与 domain 的 `FaceEncoding` 区别——后者是"纯向量+模型版本"，前者带"它来自谁"），混进 domain 会让 domain 知道太多评估细节。保持评估自洽。

- [ ] **Step 1: 写失败的测试 `tests/unit/evaluation/test_types.py`**

```python
# 评估实体的最小冒烟测试：能造、字段齐、不可变
import numpy as np
import pytest

# 评估专用数据类：和 domain.entities 的区别在 Step 3 详细解释
from face_recognition.evaluation.types import (
    EvalEncoding,
    PairResult,
    StrategyMetrics,
)


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_eval_encoding_is_frozen():
    enc = EvalEncoding(
        vector=_unit_vec(0),
        person_id="alice",
        image_path="data/private_dataset/alice/000.jpg",
    )
    # frozen=True 在 M1 Task 1 已经解释过：试图改字段会抛 FrozenInstanceError
    # （dataclasses 模块自带的异常类型，标记"我是 frozen 但你想改我"）
    with pytest.raises(Exception):
        enc.person_id = "bob"  # type: ignore[misc]


def test_pair_result_carries_score_and_label():
    # PairResult 表示"一次比对的产出"：相似度分数 + 是否同人（ground truth）
    pair = PairResult(score=0.83, is_genuine=True, query_person="alice", template_person="alice")
    assert pair.score == 0.83
    assert pair.is_genuine is True


def test_strategy_metrics_holds_all_indicators():
    # StrategyMetrics 是给 reports 模块的统一交付物——5 策略每个一份
    m = StrategyMetrics(
        strategy_name="kmeans_k3",
        eer=0.04,
        eer_threshold=0.62,
        tar_at_far_1e3=0.91,
        top1_accuracy=0.96,
        roc_fpr=np.array([0.0, 0.5, 1.0]),
        roc_tpr=np.array([0.0, 0.95, 1.0]),
        n_genuine=200,
        n_impostor=4500,
    )
    assert m.strategy_name == "kmeans_k3"
    assert 0.0 < m.eer < 1.0
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/evaluation/test_types.py -v
```

预期：`ModuleNotFoundError: face_recognition.evaluation.types`

- [ ] **Step 3: 实现 `src/face_recognition/evaluation/types.py`**

```python
# 评估专用值对象。全部 frozen：评估管线"产出 → 写报告"是单向流，没有"修改"语义。
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class EvalEncoding:
    """评估侧的"一张图的向量"。

    与 domain.entities.FaceEncoding 的区别：
      - FaceEncoding 不知道这张图属于谁、来自哪——它就是"模型输出向量+版本"
      - EvalEncoding 必须带 person_id（同/异人配对靠它）和 image_path（出错时回溯）
    把"评估的元数据"和"领域的纯向量"分离开，domain 层永远不知道 image_path。
    """
    # numpy ndarray 不是 hashable 类型，在 frozen dataclass 里要小心；
    # 但 @dataclass(frozen=True) 默认 eq=True 会让实例可哈希——只要你不 hash 它就没事。
    # 评估管线确实不会 hash EvalEncoding，所以 OK。
    vector: np.ndarray
    person_id: str
    image_path: str


@dataclass(frozen=True)
class PairResult:
    """一次"查询 vs 模板"比对的结果。

    is_genuine = True：query 和 template 是同一个人（应该高分通过）
    is_genuine = False：不同人（应该低分拒绝）
    评估指标的所有计算最终归结为一堆 PairResult 的统计。
    """
    score: float
    is_genuine: bool
    query_person: str
    template_person: str


@dataclass(frozen=True)
class StrategyMetrics:
    """单个策略跑完后的全部指标。reports.py 接收 list[StrategyMetrics] 写表/画图。"""
    strategy_name: str
    eer: float                 # Equal Error Rate
    eer_threshold: float       # 达到 EER 的相似度阈值
    tar_at_far_1e3: float      # 卡 FAR=0.1% 时的真员工接受率
    top1_accuracy: float       # 闭集 Top-1（同人 query 找最相似 → 是不是本人）
    # field(default_factory=...) 是 dataclass 的"工厂默认值"语法。
    #   - 不能写 `roc_fpr: np.ndarray = np.array([])`——所有实例会**共享同一个数组**
    #     （和 list/dict 默认参数同样的坑）
    #   - default_factory 接受一个无参函数，每次构造时调用生成新实例
    roc_fpr: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    roc_tpr: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    n_genuine: int = 0
    n_impostor: int = 0
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/evaluation/test_types.py -v
```

预期：3 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/evaluation/types.py tests/unit/evaluation/test_types.py
git commit -m "feat(evaluation): 加 EvalEncoding/PairResult/StrategyMetrics 数据类"
```

---

### Task 2: 数据切分（按人 80/20）

**Files:**
- Create: `src/face_recognition/evaluation/data_split.py`
- Test: `tests/unit/evaluation/test_data_split.py`

按"每个人"切：每人 N 张照片 → 80%×N 进注册集、20%×N 进测试集。**不是**按总样本切——后者会让同一个人同时出现在两边，污染评估。

- [ ] **Step 1: 写失败的测试**

```python
from collections.abc import Callable
from pathlib import Path

import pytest

from face_recognition.evaluation.data_split import PersonSplit, split_by_person


def _make_person_dir(root: Path, person_id: str, n: int) -> None:
    """造一个 "person_id/" 目录，放 n 张占位 jpg。"""
    d = root / person_id
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"{i:03d}.jpg").write_bytes(b"fake")


def test_split_keeps_persons_disjoint(tmp_path: Path):
    """同一人不能同时出现在 train 和 test。"""
    _make_person_dir(tmp_path, "alice", 10)
    _make_person_dir(tmp_path, "bob", 10)
    splits = split_by_person(tmp_path, train_ratio=0.8, seed=42)

    # 每个 person 都拿到自己的 PersonSplit
    assert {s.person_id for s in splits} == {"alice", "bob"}
    for s in splits:
        # 集合交集为空 = 不重叠（set 操作 & 是交集，| 是并集）
        assert set(s.train_paths).isdisjoint(set(s.test_paths))
        assert len(s.train_paths) + len(s.test_paths) == 10


def test_split_ratio_80_20(tmp_path: Path):
    _make_person_dir(tmp_path, "alice", 10)
    [s] = split_by_person(tmp_path, train_ratio=0.8, seed=42)
    # 10 张 × 80% = 8 张训练
    assert len(s.train_paths) == 8
    assert len(s.test_paths) == 2


def test_split_is_deterministic_with_seed(tmp_path: Path):
    """同 seed → 同切分。"""
    _make_person_dir(tmp_path, "alice", 20)
    a = split_by_person(tmp_path, train_ratio=0.8, seed=42)
    b = split_by_person(tmp_path, train_ratio=0.8, seed=42)
    assert a[0].train_paths == b[0].train_paths


def test_split_skips_persons_with_too_few_images(tmp_path: Path):
    """不足 5 张照片的人直接跳过——80/20 切下来训练或测试可能为 0。"""
    _make_person_dir(tmp_path, "alice", 10)
    _make_person_dir(tmp_path, "tooFew", 3)
    splits = split_by_person(tmp_path, train_ratio=0.8, seed=42, min_images=5)
    assert {s.person_id for s in splits} == {"alice"}


def test_split_ignores_non_directories(tmp_path: Path):
    """根目录里的非文件夹条目（README、隐藏文件）应该被忽略。"""
    _make_person_dir(tmp_path, "alice", 10)
    (tmp_path / "README.md").write_text("notes")
    splits = split_by_person(tmp_path, train_ratio=0.8, seed=42)
    assert len(splits) == 1
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/evaluation/test_data_split.py -v
```

- [ ] **Step 3: 实现 `src/face_recognition/evaluation/data_split.py`**

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/evaluation/test_data_split.py -v
```

预期：5 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/evaluation/data_split.py tests/unit/evaluation/test_data_split.py
git commit -m "feat(evaluation): 加按人 80/20 切分（PersonSplit + split_by_person）"
```

---

### Task 3: LFW 库外陌生人加载器

**Files:**
- Create: `src/face_recognition/evaluation/lfw_loader.py`
- Test: `tests/unit/evaluation/test_lfw_loader.py`

库外 Impostor 评估"陌生人冒充员工"。最方便的来源是 **scikit-learn** 内置的 `fetch_lfw_people` 函数——它会自动下载 LFW 人脸库到本地缓存（`~/scikit_learn_data/`），无需手动管理数据。

- [ ] **Step 1: 写失败的测试**

```python
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

    # patch 装饰器：在测试期间把目标对象替换成 mock，结束后还原
    # 字符串路径要写"被使用的位置"——lfw_loader.py 里 import 之后的 fetch_lfw_people 引用
    with patch("face_recognition.evaluation.lfw_loader.fetch_lfw_people", return_value=fake_bunch):
        a = load_lfw_subset(n_persons=10, seed=42)
        b = load_lfw_subset(n_persons=10, seed=42)
    assert [x.person_name for x in a] == [x.person_name for x in b]
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/evaluation/test_lfw_loader.py -v
```

- [ ] **Step 3: 实现 `src/face_recognition/evaluation/lfw_loader.py`**

```python
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
    # color=True：拿 RGB 三通道（默认是灰度）。InsightFace 需要彩色。
    # resize=None：不缩放，拿原始 250×250。（fetch_lfw_people 默认 resize=0.5→125×125，
    #              对小脸检测不够友好；InsightFace 检测器内部会再缩到 det_size）
    # min_faces_per_person=1：默认是 50，会把 LFW 13000+ 人砍到 158 人。
    #              我们只要"任意陌生人"，门槛设到 1 拿全 5749 个人备选。
    bunch = fetch_lfw_people(color=True, resize=None, min_faces_per_person=1)

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
        # bunch.images 是 (N, H, W, 3) float64 范围 0~255；要转 uint8 给 InsightFace
        img_uint8 = bunch.images[first_idx].astype(np.uint8)
        images.append(LfwImage(
            image=img_uint8,
            person_name=str(bunch.target_names[pid]),
        ))

    return images
```

- [ ] **Step 4: 跑单元测试（不联网部分）确认通过**

```bash
uv run pytest tests/unit/evaluation/test_lfw_loader.py -v -m "not slow"
```

预期：1 passed, 1 deselected

- [ ] **Step 5: 跑慢测试一次（首次会下载 LFW）**

```bash
uv run pytest tests/unit/evaluation/test_lfw_loader.py -v -m slow
```

预期：1 passed（首跑约 1~3 分钟下载，再跑秒级）

- [ ] **Step 6: 把 slow 标签注册到 pyproject.toml（如未注册）**

检查 `pyproject.toml` 的 `[tool.pytest.ini_options].markers`，确保有 `"slow: ..."`。M1 时已加，无需重复。

- [ ] **Step 7: commit**

```bash
git add src/face_recognition/evaluation/lfw_loader.py tests/unit/evaluation/test_lfw_loader.py
git commit -m "feat(evaluation): 加 LFW 加载器（sklearn fetch_lfw_people）"
```

---


### Task 4: 配对生成器（Genuine / 库内 Impostor / 库外 Impostor）

**Files:**
- Create: `src/face_recognition/evaluation/pair_generator.py`
- Test: `tests/unit/evaluation/test_pair_generator.py`

评估的核心是"配对"——把两个 EvalEncoding 拼起来算余弦分，再贴上"是否同人"的真值标签。三组配对各有用途：

- **Genuine**：同一人的 query 和 template 配对——理想情况分数应该高（≥ 阈值）
- **库内 Impostor**：库里 A 的 query 配 B 的 template——应该低（< 阈值）
- **库外 Impostor**：LFW 陌生人 query 配库内任何人的 template——更应该低

> spec 里**省去库内 Impostor**（用户决策："冗余"）。但代码我们仍然实现，给后续想加回来留口子；run_ablation 默认只跑 Genuine + 库外。

- [ ] **Step 1: 写失败的测试**

```python
import numpy as np
import pytest

from face_recognition.evaluation.pair_generator import (
    generate_genuine_pairs,
    generate_closed_impostor_pairs,
    generate_open_impostor_pairs,
)
from face_recognition.evaluation.types import EvalEncoding, PairResult


def _enc(person_id: str, seed: int, path: str = "x.jpg") -> EvalEncoding:
    """造一个单位向量 EvalEncoding，方便测试。"""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return EvalEncoding(vector=v / np.linalg.norm(v), person_id=person_id, image_path=path)


def test_genuine_pairs_only_same_person():
    """Genuine 应该全部 is_genuine=True 且 query/template 同人。"""
    queries = [_enc("alice", 0), _enc("alice", 1), _enc("bob", 2)]
    templates = {"alice": _enc("alice", 10), "bob": _enc("bob", 11)}
    pairs = generate_genuine_pairs(queries, templates)
    # 3 个 query 各自和"自己人"的 template 配对 → 3 个 PairResult
    assert len(pairs) == 3
    assert all(p.is_genuine for p in pairs)
    assert all(p.query_person == p.template_person for p in pairs)


def test_genuine_pairs_skip_query_without_template():
    """如果某人在 templates 里没有条目，他的 query 应该被跳过（不报错）。"""
    queries = [_enc("alice", 0), _enc("ghost", 1)]
    templates = {"alice": _enc("alice", 10)}
    pairs = generate_genuine_pairs(queries, templates)
    assert len(pairs) == 1
    assert pairs[0].query_person == "alice"


def test_closed_impostor_only_different_person():
    """库内 Impostor 必须 query_person != template_person。"""
    queries = [_enc("alice", 0)]
    templates = {"alice": _enc("alice", 10), "bob": _enc("bob", 11), "carol": _enc("carol", 12)}
    pairs = generate_closed_impostor_pairs(queries, templates)
    # alice 的 query 配 bob、carol 的 template → 2 对
    assert len(pairs) == 2
    assert all(not p.is_genuine for p in pairs)
    assert all(p.query_person != p.template_person for p in pairs)


def test_open_impostor_pairs_all_negative():
    """库外陌生人 vs 库内任何 template → 全部 is_genuine=False。"""
    lfw_queries = [_enc("LFW_Stranger_1", 0, "lfw/x.jpg")]
    templates = {"alice": _enc("alice", 10), "bob": _enc("bob", 11)}
    pairs = generate_open_impostor_pairs(lfw_queries, templates)
    assert len(pairs) == 2
    assert all(not p.is_genuine for p in pairs)


def test_score_is_cosine_similarity():
    """分数 = 单位向量点积（余弦相似度，范围 [-1, 1]，同向接近 1）。"""
    # 自己跟自己点积 = 1
    same = _enc("alice", 0)
    pairs = generate_genuine_pairs([same], {"alice": same})
    # 浮点比较留余地（np.float32 累加误差通常 < 1e-6）
    assert pairs[0].score == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/evaluation/test_pair_generator.py -v
```

- [ ] **Step 3: 实现 `src/face_recognition/evaluation/pair_generator.py`**

```python
# Genuine / 库内 Impostor / 库外 Impostor 三种配对的生成器。
# 共同模式：双重循环 + 算余弦 + 包成 PairResult。
import numpy as np

from face_recognition.evaluation.types import EvalEncoding, PairResult


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """两个**已 L2 归一化**的向量的余弦相似度 = 点积。
    M1 已经多次出现，这里只一行：np.dot(a, b) 标量返回。
    """
    # float() 把 np.float32 转成 Python float，避免下游 dataclass 字段类型不一致
    return float(np.dot(a, b))


def generate_genuine_pairs(
    queries: list[EvalEncoding],
    templates: dict[str, EvalEncoding],
) -> list[PairResult]:
    """同人配对：每个 query 找到自己 person_id 对应的 template。

    queries: 测试集（每张测试图一个 EvalEncoding）
    templates: 注册集产物（多模板策略压缩后每人 1 个 EvalEncoding——
               策略给出多个时这里取代表向量；具体策略侧的事 Task 7 再细谈）
    """
    pairs: list[PairResult] = []
    for q in queries:
        # dict.get 在 key 不存在时返回 None，不抛 KeyError——
        # 这里允许"测试集出现库里没有的人"，体现开放集场景的健壮性
        tpl = templates.get(q.person_id)
        if tpl is None:
            continue
        pairs.append(PairResult(
            score=_cosine(q.vector, tpl.vector),
            is_genuine=True,
            query_person=q.person_id,
            template_person=tpl.person_id,
        ))
    return pairs


def generate_closed_impostor_pairs(
    queries: list[EvalEncoding],
    templates: dict[str, EvalEncoding],
) -> list[PairResult]:
    """库内异人配对：query 和库里**其他人**的 template 配。

    数量级：N 个测试 query × (M-1) 个其他模板 → 增长很快，
    35 人每人 10 张测试 → 350 × 34 = 11900 对。spec 已决定**不上线**这组评估，
    但函数留着以备后续切回来。
    """
    pairs: list[PairResult] = []
    for q in queries:
        # dict.items() 同时拿 (key, value)——比 for k in d 再 d[k] 高效一点
        for tpl_pid, tpl in templates.items():
            if tpl_pid == q.person_id:
                continue  # 跳过同人，那是 genuine 的活
            pairs.append(PairResult(
                score=_cosine(q.vector, tpl.vector),
                is_genuine=False,
                query_person=q.person_id,
                template_person=tpl_pid,
            ))
    return pairs


def generate_open_impostor_pairs(
    lfw_queries: list[EvalEncoding],
    templates: dict[str, EvalEncoding],
) -> list[PairResult]:
    """库外陌生人配对：LFW 里的人作为 query，配库里所有 template。

    所有产出 is_genuine=False（陌生人不可能"是"库里的任何人）。
    数量级：50 个 LFW × 35 个库内 → 1750 对，比库内 impostor 小一个量级。
    """
    pairs: list[PairResult] = []
    for q in lfw_queries:
        for tpl_pid, tpl in templates.items():
            pairs.append(PairResult(
                score=_cosine(q.vector, tpl.vector),
                is_genuine=False,
                query_person=q.person_id,   # "LFW_xxx" 风格的伪 ID
                template_person=tpl_pid,
            ))
    return pairs
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/evaluation/test_pair_generator.py -v
```

预期：5 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/evaluation/pair_generator.py tests/unit/evaluation/test_pair_generator.py
git commit -m "feat(evaluation): 加三组配对生成器（Genuine / 库内 / 库外 Impostor）"
```

---

### Task 5: 评估指标（ROC / EER / TAR\@FAR / Top-1）

**Files:**
- Create: `src/face_recognition/evaluation/metrics.py`
- Test: `tests/unit/evaluation/test_metrics.py`

四个指标的数学定义：

- **ROC（Receiver Operating Characteristic）**：横轴 FAR（False Accept Rate，把 Impostor 当成员工的比例），纵轴 TAR/TPR（True Accept Rate，正确识别员工的比例）。给定一组阈值 t，每个 t 算一个 (FAR, TAR) 点，连成曲线。
- **EER（Equal Error Rate）**：FAR = FRR（False Reject Rate = 1 - TAR）那个点的错误率。直观理解："漏识别"和"错识别"被罚得一样重时的折中点。
- **TAR\@FAR=1e-3**：业界标准——把"陌生人冒充率"压到千分之一时，真员工还能被认出来的比例。门禁更怕"放进坏人"，所以卡 FAR 比卡 EER 更贴近真实场景。
- **Top-1 Accuracy**：闭集分类口径——同人 query 找最相似的 template，是不是本人。和 ROC 互补：ROC 看二分类（是/不是），Top-1 看排序（最像谁）。

实现策略：sklearn 已有 `roc_curve`，我们只在它输出之上算 EER 和 TAR\@FAR。

- [ ] **Step 1: 写失败的测试**

```python
import numpy as np
import pytest

from face_recognition.evaluation.metrics import (
    compute_roc,
    compute_eer,
    compute_tar_at_far,
    compute_top1_accuracy,
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
    templates = {
        "A": EvalEncoding(vector=a.astype(np.float32), person_id="A", image_path=""),
        "B": EvalEncoding(vector=b.astype(np.float32), person_id="B", image_path=""),
        "C": EvalEncoding(vector=c.astype(np.float32), person_id="C", image_path=""),
    }
    acc = compute_top1_accuracy(test, templates)
    assert acc == pytest.approx(1.0)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/evaluation/test_metrics.py -v
```

- [ ] **Step 3: 实现 `src/face_recognition/evaluation/metrics.py`**

```python
import numpy as np
# sklearn.metrics.roc_curve(y_true, y_score) → (fpr, tpr, thresholds)
#   - y_true: 0/1 标签数组（True=正例，这里"同人"）
#   - y_score: 模型输出的"是正例的置信度"——我们用余弦分
#   - 返回值都是 (T,) 一维数组，T 是不同阈值数（去重后）
#   - thresholds 从大到小排，对应曲线从 (0, 0) 走到 (1, 1)
from sklearn.metrics import roc_curve

from face_recognition.evaluation.types import EvalEncoding, PairResult


def _pairs_to_arrays(pairs: list[PairResult]) -> tuple[np.ndarray, np.ndarray]:
    """把 PairResult 列表拆成 (scores, labels) 数组，喂给 sklearn。"""
    # 列表推导式两次扫一遍——50K 量级配对下可忽略；如要优化可改用 np.fromiter
    scores = np.array([p.score for p in pairs], dtype=np.float64)
    labels = np.array([1 if p.is_genuine else 0 for p in pairs], dtype=np.int64)
    return scores, labels


def compute_roc(pairs: list[PairResult]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 ROC 曲线，返回 (FPR, TPR, thresholds)。"""
    scores, labels = _pairs_to_arrays(pairs)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return fpr, tpr, thresholds


def compute_eer(pairs: list[PairResult]) -> tuple[float, float]:
    """计算 EER 和对应阈值。

    EER 定义：FAR = FRR 那个点。
      FRR = 1 - TPR
      条件 FAR = FRR → fpr = 1 - tpr → fpr + tpr = 1
    我们沿曲线找"|fpr - (1 - tpr)| 最小"的点，返回该点 fpr 当 EER。
    """
    fpr, tpr, thresholds = compute_roc(pairs)
    # 全数组运算 fpr - (1 - tpr) = fpr + tpr - 1
    # np.argmin(arr) 返回最小元素的索引——绝对值最小 = 最接近"FAR=FRR"的那一格
    diffs = np.abs(fpr - (1.0 - tpr))
    idx = int(np.argmin(diffs))
    eer = float((fpr[idx] + (1.0 - tpr[idx])) / 2.0)  # 取两个误差的平均更稳
    return eer, float(thresholds[idx])


def compute_tar_at_far(pairs: list[PairResult], target_far: float = 1e-3) -> float:
    """卡定 FAR 时的 TAR。

    target_far 通常取 1e-3（千分之一），门禁场景的事实标准。
    实现：在 ROC 曲线上找 fpr ≤ target_far 的最右一格，返回该格的 tpr。
    """
    fpr, tpr, _ = compute_roc(pairs)
    # 布尔索引：fpr <= target_far 是 (T,) bool 数组
    # np.where(...)[0] 返回所有 True 位置的下标
    valid = np.where(fpr <= target_far)[0]
    if len(valid) == 0:
        # 极端情况：所有 fpr 都 > target_far（数据太烂或样本太少）
        return 0.0
    # 取最右那个——FAR 最接近上限的位置 TPR 最高
    return float(tpr[valid[-1]])


def compute_top1_accuracy(
    test_set: list[EvalEncoding],
    templates: dict[str, EvalEncoding],
) -> float:
    """闭集 Top-1：每个 test 找余弦最相似的 template，命中本人则算对。

    只统计 template 里有的那些人；test 里"陌生人"应已被调用方过滤。
    """
    # 把 templates 的 person_id 顺序固定下来，便于 argmax 后回查
    template_ids = list(templates.keys())
    if not template_ids:
        return 0.0
    # np.stack 把多个一维向量堆成 (M, 512) 矩阵；axis=0 沿"人"维堆叠（默认）
    template_matrix = np.stack([templates[pid].vector for pid in template_ids])

    correct = 0
    total = 0
    for q in test_set:
        if q.person_id not in templates:
            continue  # 不在库里的人不计入闭集 Top-1
        # 矩阵 × 向量 = (M, 512) @ (512,) = (M,) 得分向量
        # @ 运算符就是 numpy.matmul 的语法糖
        scores = template_matrix @ q.vector
        pred = template_ids[int(np.argmax(scores))]
        if pred == q.person_id:
            correct += 1
        total += 1
    return correct / total if total else 0.0
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/evaluation/test_metrics.py -v
```

预期：5 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/evaluation/metrics.py tests/unit/evaluation/test_metrics.py
git commit -m "feat(evaluation): 加 ROC/EER/TAR@FAR/Top-1 指标计算"
```

---

### Task 6: 报告产出（CSV / ROC 叠图 / 直方图 / Markdown）

**Files:**
- Create: `src/face_recognition/evaluation/reports.py`
- Test: `tests/unit/evaluation/test_reports.py`

报告模块负责"把 `list[StrategyMetrics]` 落地成人能看的东西"。四种产物：

1. **CSV 表**：`reports/ablation_summary.csv`，pandas 一行写盘
2. **ROC 叠图**：5 策略的 ROC 画在同一张 `reports/roc_curves.png` 上方便对比
3. **分数直方图**：每个策略一张 `reports/hist_<strategy>.png`（Genuine vs Impostor 分布）
4. **Markdown 报告**：`reports/summary.md`，把表格+图片引用拼一起，答辩报告可直接抄

- [ ] **Step 1: 写失败的测试**

```python
from pathlib import Path

import numpy as np
import pytest

from face_recognition.evaluation.reports import (
    write_csv,
    plot_roc_curves,
    plot_score_histogram,
    write_markdown,
)
from face_recognition.evaluation.types import PairResult, StrategyMetrics


def _make_metrics(name: str, eer: float = 0.05) -> StrategyMetrics:
    return StrategyMetrics(
        strategy_name=name,
        eer=eer,
        eer_threshold=0.62,
        tar_at_far_1e3=0.91,
        top1_accuracy=0.96,
        roc_fpr=np.array([0.0, 0.05, 1.0]),
        roc_tpr=np.array([0.0, 0.95, 1.0]),
        n_genuine=200,
        n_impostor=4500,
    )


def test_write_csv_creates_file_with_correct_columns(tmp_path: Path):
    metrics = [_make_metrics("kmeans_k3"), _make_metrics("mean_all", eer=0.07)]
    csv_path = tmp_path / "summary.csv"
    write_csv(metrics, csv_path)
    assert csv_path.exists()
    # 用 pandas 读回来核对——别手撕 CSV 字符串
    import pandas as pd
    df = pd.read_csv(csv_path)
    assert set(df.columns) >= {"strategy_name", "eer", "tar_at_far_1e3", "top1_accuracy"}
    assert len(df) == 2


def test_plot_roc_curves_creates_png(tmp_path: Path):
    metrics = [_make_metrics("a"), _make_metrics("b")]
    out = tmp_path / "roc.png"
    plot_roc_curves(metrics, out)
    assert out.exists()
    # PNG 文件至少应该有几 KB——空图也得有 magic header
    assert out.stat().st_size > 1000


def test_plot_score_histogram_separates_genuine_and_impostor(tmp_path: Path):
    pairs = [
        PairResult(0.9, True, "a", "a"),
        PairResult(0.85, True, "b", "b"),
        PairResult(0.3, False, "a", "b"),
        PairResult(0.2, False, "b", "a"),
    ]
    out = tmp_path / "hist.png"
    plot_score_histogram(pairs, "kmeans_k3", out)
    assert out.exists()


def test_write_markdown_includes_metrics_and_image_refs(tmp_path: Path):
    metrics = [_make_metrics("kmeans_k3", eer=0.04)]
    md = tmp_path / "summary.md"
    # 报告需要引用之前生成的 png——传相对路径
    write_markdown(
        metrics=metrics,
        output_path=md,
        roc_image="roc_curves.png",
        hist_images={"kmeans_k3": "hist_kmeans_k3.png"},
    )
    text = md.read_text()
    assert "kmeans_k3" in text
    assert "0.04" in text or "4.00%" in text  # eer 数字格式由实现决定
    assert "roc_curves.png" in text
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/evaluation/test_reports.py -v
```

- [ ] **Step 3: 实现 `src/face_recognition/evaluation/reports.py`**

```python
from pathlib import Path

# matplotlib 在测试环境里默认会试图打开窗口，no-display 服务器上会报错。
# 'Agg' 后端 = "anti-grain geometry"，纯写 PNG 不开 GUI。必须在 import pyplot 前设。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from face_recognition.evaluation.types import PairResult, StrategyMetrics


def write_csv(metrics: list[StrategyMetrics], output_path: Path) -> None:
    """把所有策略的标量指标写成 CSV。

    只写"标量列"（eer/tar/top1/n_*），不写 roc_fpr/roc_tpr——
    那俩是数组，不适合塞 CSV，PNG 图里已经画过了。
    """
    rows = [
        {
            "strategy_name": m.strategy_name,
            "eer": m.eer,
            "eer_threshold": m.eer_threshold,
            "tar_at_far_1e3": m.tar_at_far_1e3,
            "top1_accuracy": m.top1_accuracy,
            "n_genuine": m.n_genuine,
            "n_impostor": m.n_impostor,
        }
        for m in metrics
    ]
    # pd.DataFrame(records).to_csv 一行写盘；index=False 不写行号列
    pd.DataFrame(rows).to_csv(output_path, index=False)


def plot_roc_curves(metrics: list[StrategyMetrics], output_path: Path) -> None:
    """把所有策略的 ROC 画在同一张图上，便于答辩讲故事。"""
    # plt.figure(figsize=(w, h)) 单位是英寸；(8, 6) 是论文图的常见尺寸
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in metrics:
        # plot(x, y, label=...) 画线；label 用于后面 legend()
        ax.plot(m.roc_fpr, m.roc_tpr, label=f"{m.strategy_name} (EER={m.eer:.3f})")
    # 对角线 = 随机分类器，作为 baseline 视觉参照
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="random")
    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("True Accept Rate (TAR)")
    ax.set_title("ROC Curves — 5 Strategy Ablation")
    ax.legend(loc="lower right")  # 图例放右下不挡曲线
    ax.grid(alpha=0.3)
    # tight_layout 自动调边距避免 label 被裁
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)  # 显式关闭释放内存——大量画图时关键


def plot_score_histogram(
    pairs: list[PairResult],
    strategy_name: str,
    output_path: Path,
) -> None:
    """把同一策略下 Genuine / Impostor 的分数分布画成叠加直方图。

    理想情况：两组分布分得很开，中间几乎不重叠。看图能直接判断这个策略好不好。
    """
    genuine = [p.score for p in pairs if p.is_genuine]
    impostor = [p.score for p in pairs if not p.is_genuine]

    fig, ax = plt.subplots(figsize=(8, 5))
    # bins=50 把 [-1, 1] 切 50 格；alpha=0.5 半透明让两组重叠区可见
    # density=True 画频率（积分=1）而非计数——genuine/impostor 数量差悬殊时必须
    ax.hist(genuine, bins=50, alpha=0.5, label=f"Genuine (n={len(genuine)})", density=True)
    ax.hist(impostor, bins=50, alpha=0.5, label=f"Impostor (n={len(impostor)})", density=True)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution — {strategy_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def write_markdown(
    metrics: list[StrategyMetrics],
    output_path: Path,
    roc_image: str,
    hist_images: dict[str, str],
) -> None:
    """生成 Markdown 报告，把表格+图片引用拼起来。

    roc_image / hist_images 用**相对路径**——markdown 渲染器从 .md 文件所在目录解析。
    output_path 与图片位于同目录时直接传文件名即可。
    """
    lines: list[str] = []
    lines.append("# 5 策略消融评估报告\n")
    lines.append(f"覆盖 {len(metrics)} 个策略，每策略 Genuine/Impostor 配对见下表。\n")
    lines.append("## 总览\n")
    # Markdown 表格：表头 + 分隔行 + 数据行
    lines.append("| Strategy | EER | TAR\\@FAR=1e-3 | Top-1 | EER thresh | n(Gen) | n(Imp) |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in metrics:
        lines.append(
            f"| {m.strategy_name} "
            f"| {m.eer:.4f} "
            f"| {m.tar_at_far_1e3:.4f} "
            f"| {m.top1_accuracy:.4f} "
            f"| {m.eer_threshold:.4f} "
            f"| {m.n_genuine} | {m.n_impostor} |"
        )
    lines.append("\n## ROC 叠图\n")
    lines.append(f"![ROC]({roc_image})\n")
    lines.append("## 各策略分数分布\n")
    for name, img in hist_images.items():
        lines.append(f"### {name}\n")
        lines.append(f"![hist-{name}]({img})\n")
    output_path.write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/evaluation/test_reports.py -v
```

预期：4 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/evaluation/reports.py tests/unit/evaluation/test_reports.py
git commit -m "feat(evaluation): 加报告产出（CSV / ROC 叠图 / 直方图 / Markdown）"
```

---

### Task 7: 批量编码辅助（图片目录 → EvalEncoding 集合）

**Files:**
- Create: `src/face_recognition/evaluation/embedder.py`
- Test: `tests/unit/evaluation/test_embedder.py`

run_ablation 要把"一堆图片路径 / LFW ndarray"喂进 ArcFace 拿向量。这块逻辑是**评估侧专用胶水**——和 M1 的 `register_face` 不一样：
- register 走 SQLite 落盘
- evaluation 全在内存里跑，不污染生产 DB

所以单独抽个 `embedder.py`，复用 M1 的 `FacePipeline`（`buffalo_l` 一站式接口）但不接仓储。

- [ ] **Step 1: 写失败的测试**

```python
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from face_recognition.domain.entities import FaceEncoding
from face_recognition.domain.errors import NoFaceError
from face_recognition.evaluation.embedder import (
    encode_image_paths,
    encode_lfw_images,
)
from face_recognition.evaluation.lfw_loader import LfwImage
from face_recognition.evaluation.types import EvalEncoding


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_encode_image_paths_returns_one_eval_encoding_per_path(tmp_path: Path):
    """encode_image_paths 应该跳过无脸/读取失败，返回有效编码列表。"""
    # 造 3 个假 jpg 文件
    paths = []
    for i in range(3):
        p = tmp_path / f"alice_{i}.jpg"
        p.write_bytes(b"fake")
        paths.append(p)

    # mock pipeline：前两张正常返回向量，第三张抛 NoFaceError 模拟"无脸"
    # —— M1 约定 FacePipeline.encode_single 检测不到脸时**抛异常**而非返回 None
    pipeline = MagicMock()
    fe1 = FaceEncoding(vector=_unit_vec(0), model_version="buffalo_l")
    fe2 = FaceEncoding(vector=_unit_vec(1), model_version="buffalo_l")
    # side_effect 接列表时按调用顺序逐项产出；元素若是 Exception 实例则被 raise
    # 这是 MagicMock 的特殊语义（M1 测试已用过同款套路）
    pipeline.encode_single.side_effect = [fe1, fe2, NoFaceError("无脸")]

    result = encode_image_paths(pipeline, paths, person_id="alice")
    assert len(result) == 2  # 第三张被跳过
    assert all(isinstance(r, EvalEncoding) for r in result)
    assert all(r.person_id == "alice" for r in result)


def test_encode_lfw_images_uses_person_name_as_id():
    """LFW 路径下 person_id 来自 person_name，image_path 留 'lfw://<name>'。"""
    pipeline = MagicMock()
    pipeline.encode_single.return_value = FaceEncoding(
        vector=_unit_vec(0), model_version="buffalo_l"
    )
    lfw_imgs = [
        LfwImage(image=np.zeros((250, 250, 3), dtype=np.uint8), person_name="George_W_Bush"),
    ]
    result = encode_lfw_images(pipeline, lfw_imgs)
    assert len(result) == 1
    assert result[0].person_id == "George_W_Bush"
    assert result[0].image_path.startswith("lfw://")


def test_encode_lfw_images_skips_no_face():
    """无脸 LFW 图（罕见但要防御）：抛 NoFaceError 时跳过。"""
    pipeline = MagicMock()
    pipeline.encode_single.side_effect = NoFaceError("无脸")
    lfw_imgs = [LfwImage(image=np.zeros((250, 250, 3), dtype=np.uint8), person_name="Ghost")]
    assert encode_lfw_images(pipeline, lfw_imgs) == []
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/evaluation/test_embedder.py -v
```

- [ ] **Step 3: 实现 `src/face_recognition/evaluation/embedder.py`**

```python
import logging
from pathlib import Path

import cv2
import numpy as np

# 复用 M1 的 FacePipeline Protocol——评估只是另一个调用方
from face_recognition.domain.errors import NoFaceError
from face_recognition.domain.interfaces import FacePipeline
from face_recognition.evaluation.lfw_loader import LfwImage
from face_recognition.evaluation.types import EvalEncoding

logger = logging.getLogger(__name__)


def encode_image_paths(
    pipeline: FacePipeline,
    paths: list[Path],
    person_id: str,
) -> list[EvalEncoding]:
    """把一组图片路径批量编码成 EvalEncoding 列表，跳过无脸/读取失败的。

    所有路径默认是同一个 person_id——评估侧通常按"人/目录"分批调用。
    """
    out: list[EvalEncoding] = []
    for p in paths:
        # cv2.imread 失败时返回 None（不抛异常，注意！）
        img = cv2.imread(str(p))
        if img is None:
            logger.warning("无法读取图片，跳过：%s", p)
            continue
        # M1 约定：FacePipeline.encode_single 在检测不到脸时**抛 NoFaceError**，
        # 不是返回 None。评估侧捕获该异常并跳过——单张失败不应中断整个流水线。
        # try/except 缩到最小范围（只包 encode_single 这一行）：
        # 把 EvalEncoding 构造放外面，避免把它的错误也吞掉
        try:
            face = pipeline.encode_single(img)
        except NoFaceError:
            logger.warning("未检测到人脸，跳过：%s", p)
            continue
        out.append(EvalEncoding(
            vector=face.vector,
            person_id=person_id,
            image_path=str(p),
        ))
    return out


def encode_lfw_images(
    pipeline: FacePipeline,
    images: list[LfwImage],
) -> list[EvalEncoding]:
    """把 LFW 内存图片批量编码。

    注意：sklearn fetch_lfw_people 给的是 RGB（H, W, 3）uint8，
    InsightFace.encode_single 期望的是 BGR（OpenCV 默认）——必须转通道。
    """
    out: list[EvalEncoding] = []
    for lfw in images:
        # cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) = numpy 切片 [..., ::-1] 的等价语义
        # 选 cvtColor 更显式，读代码的人一眼看出"在做色彩空间转换"
        bgr = cv2.cvtColor(lfw.image, cv2.COLOR_RGB2BGR)
        try:
            face = pipeline.encode_single(bgr)
        except NoFaceError:
            logger.warning("LFW 图未检测到脸，跳过：%s", lfw.person_name)
            continue
        out.append(EvalEncoding(
            vector=face.vector,
            person_id=lfw.person_name,
            # image_path 用伪 URI 标记来源——回溯时一眼看出是 LFW
            image_path=f"lfw://{lfw.person_name}",
        ))
    return out
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/evaluation/test_embedder.py -v
```

预期：3 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/evaluation/embedder.py tests/unit/evaluation/test_embedder.py
git commit -m "feat(evaluation): 加批量编码辅助（encode_image_paths / encode_lfw_images）"
```

---

### Task 8: run_ablation 主入口（5 策略 × 评估流水线）

**Files:**
- Create: `src/face_recognition/evaluation/run_ablation.py`
- Test: `tests/integration/test_run_ablation.py`（集成，仅冒烟）

主入口编排所有前面的部件。流程：

```
load 数据 → split → 对每个 strategy:
                       train_set → 策略压缩 → templates
                       test_set  → encode    → genuine_pairs
                       lfw_set   → encode    → open_impostor_pairs
                       compute metrics → 累加到 list
→ 汇总 metrics → 写 CSV / 画 ROC / 画 Histogram / 写 Markdown
```

策略实现复用 M1 `application/strategies/`（5 个 `TemplateStrategy`）——评估侧只调用，不重写。

- [ ] **Step 1: 写集成冒烟测试 `tests/integration/test_run_ablation.py`**

> 这个测试用 mock pipeline 跑通整条流水线，验证文件落盘正常。**不依赖真模型**，几秒级。

```python
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from face_recognition.evaluation.run_ablation import run_ablation


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_dataset(root: Path, n_persons: int, n_imgs_per_person: int) -> None:
    """造合成数据集：每人若干占位 jpg。embedder 会被 mock，不需要真图片。"""
    for i in range(n_persons):
        d = root / f"person_{i:02d}"
        d.mkdir()
        for j in range(n_imgs_per_person):
            (d / f"{j:03d}.jpg").write_bytes(b"fake")


@pytest.mark.slow
def test_run_ablation_produces_all_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """跑通整条流水线，确认 CSV / ROC.png / 多张 hist.png / summary.md 都落盘。"""
    dataset = tmp_path / "private"
    dataset.mkdir()
    _make_dataset(dataset, n_persons=5, n_imgs_per_person=10)
    out_dir = tmp_path / "reports"
    out_dir.mkdir()

    # mock pipeline：每张图返回一个稳定的向量（同人接近、异人发散）
    from face_recognition.domain.entities import FaceEncoding

    fake_pipeline = MagicMock()

    def fake_encode(img):
        # 用 id(img) 当随机种子——不同 ndarray 对象拿不同向量
        return FaceEncoding(vector=_unit_vec(id(img) % 100000), model_version="buffalo_l")

    fake_pipeline.encode_single.side_effect = fake_encode

    # mock LFW loader 返回 3 张陌生人，省得真下载
    from face_recognition.evaluation import lfw_loader
    fake_lfw = [
        lfw_loader.LfwImage(image=np.zeros((250, 250, 3), dtype=np.uint8), person_name=f"LFW_{i}")
        for i in range(3)
    ]
    monkeypatch.setattr(lfw_loader, "load_lfw_subset", lambda **kw: fake_lfw)

    run_ablation(
        dataset_root=dataset,
        output_dir=out_dir,
        pipeline=fake_pipeline,
        n_lfw=3,
        seed=42,
    )

    # 6 件落盘：summary.csv / roc_curves.png / 5×hist_*.png / summary.md
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "roc_curves.png").exists()
    assert (out_dir / "summary.md").exists()
    # 5 个策略各自一张直方图
    hists = list(out_dir.glob("hist_*.png"))
    assert len(hists) == 5
```

- [ ] **Step 2: 跑测试确认失败（模块不存在）**

```bash
uv run pytest tests/integration/test_run_ablation.py -v -m slow
```

- [ ] **Step 3: 实现 `src/face_recognition/evaluation/run_ablation.py`**

```python
import logging
from pathlib import Path

import numpy as np

# 5 个策略来自 M1 application 层，**各自一个子模块**——M1 没写
# `application/strategies/__init__.py` 的 re-export，所以这里必须按子模块路径分别 import。
# 如果以后想批量 import，可在 M1 里给 strategies/__init__.py 加 `from .random_one import ...`
from face_recognition.application.strategies.all_vectors import AllVectorsStrategy
from face_recognition.application.strategies.kmeans_k3 import KMeansK3Strategy
from face_recognition.application.strategies.manual_three import ManualThreeStrategy
from face_recognition.application.strategies.mean_all import MeanAllStrategy
from face_recognition.application.strategies.random_one import RandomOneStrategy
from face_recognition.domain.entities import FaceEncoding
from face_recognition.domain.interfaces import FacePipeline, TemplateStrategy
from face_recognition.evaluation import lfw_loader, reports
from face_recognition.evaluation.data_split import PersonSplit, split_by_person
from face_recognition.evaluation.embedder import encode_image_paths, encode_lfw_images
from face_recognition.evaluation.metrics import (
    compute_eer,
    compute_roc,
    compute_tar_at_far,
    compute_top1_accuracy,
)
from face_recognition.evaluation.pair_generator import (
    generate_genuine_pairs,
    generate_open_impostor_pairs,
)
from face_recognition.evaluation.types import EvalEncoding, PairResult, StrategyMetrics

logger = logging.getLogger(__name__)


# 5 个策略以"名字 → 实例"映射列出，方便循环消融。
# RandomOneStrategy / KMeansK3Strategy 构造时要传 seed（M1 dependencies.build_strategy 同款做法）；
# 其余三个无参。
def _all_strategies(seed: int) -> dict[str, TemplateStrategy]:
    return {
        "random_one": RandomOneStrategy(seed=seed),
        "mean_all": MeanAllStrategy(),
        "manual_three": ManualThreeStrategy(),
        "kmeans_k3": KMeansK3Strategy(seed=seed),
        "all_vectors": AllVectorsStrategy(),
    }


def _select_one_template_per_person(
    person_id: str,
    train_encodings: list[EvalEncoding],
    strategy: TemplateStrategy,
) -> EvalEncoding | None:
    """跑一个策略，把训练集压缩成"代表向量"。

    策略可能返回 1 个（mean_all）或多个（kmeans_k3=3、all_vectors=N）Template。
    评估指标按"每人 1 个代表向量"算 ROC——多模板的策略取平均向量再归一化作为代表。
    这是评估侧的简化口径，**不**影响生产侧的多模板检索。
    """
    if not train_encodings:
        return None

    # M1 定义的策略接口签名（domain/interfaces.py）：
    #   def build(self, encodings: list[FaceEncoding]) -> list[Template]
    # 入参是 FaceEncoding（带 model_version），出参是 Template（带 encoding/source/created_at）。
    # 评估侧手头是 EvalEncoding，需要先映射到 FaceEncoding 再传给策略。
    fe_list = [
        FaceEncoding(vector=e.vector, model_version="buffalo_l")
        for e in train_encodings
    ]
    templates = strategy.build(fe_list)
    if not templates:
        return None

    # Template 里的向量在 .encoding.vector，先抽出来再聚合
    template_vectors = [t.encoding.vector for t in templates]
    # 多个模板向量求平均 + L2 归一化 → 单个代表向量
    # 这是评估口径的近似；生产识别时 5 策略各自的检索逻辑由 application 层负责
    avg = np.mean(template_vectors, axis=0)
    avg = avg / np.linalg.norm(avg)
    # 取第一张图的 image_path 当代表，仅用于 debug 回溯
    return EvalEncoding(
        vector=avg.astype(np.float32),
        person_id=person_id,
        image_path=train_encodings[0].image_path,
    )


def run_ablation(
    dataset_root: Path,
    output_dir: Path,
    pipeline: FacePipeline,
    n_lfw: int = 50,
    seed: int = 42,
    target_far: float = 1e-3,
) -> list[StrategyMetrics]:
    """跑完整 5 策略消融实验，所有产物写入 output_dir。

    返回 list[StrategyMetrics] 给调用方（CLI 命令、Notebook、测试）做断言或追加分析。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[1/5] 切分数据集 (seed=%d)…", seed)
    splits: list[PersonSplit] = split_by_person(dataset_root, seed=seed)
    logger.info("  得到 %d 人", len(splits))

    logger.info("[2/5] 编码训练集 + 测试集…")
    # 提前把每个人的 train/test 图片都编码好——5 策略复用相同的向量集，避免重复推理
    train_encodings_per_person: dict[str, list[EvalEncoding]] = {}
    test_encodings: list[EvalEncoding] = []
    for sp in splits:
        train_encodings_per_person[sp.person_id] = encode_image_paths(
            pipeline, list(sp.train_paths), sp.person_id
        )
        test_encodings.extend(
            encode_image_paths(pipeline, list(sp.test_paths), sp.person_id)
        )

    logger.info("[3/5] 加载 LFW 库外陌生人 (n=%d)…", n_lfw)
    lfw_imgs = lfw_loader.load_lfw_subset(n_persons=n_lfw, seed=seed)
    lfw_encodings = encode_lfw_images(pipeline, lfw_imgs)

    logger.info("[4/5] 跑 5 策略并计算指标…")
    all_metrics: list[StrategyMetrics] = []
    # 在每策略循环里我们要把 pairs 也存下来给直方图用
    pairs_per_strategy: dict[str, list[PairResult]] = {}

    for name, strategy in _all_strategies(seed).items():
        logger.info("  策略：%s", name)
        # 4.1 用策略压缩每个人 train_set → 1 个代表向量
        templates: dict[str, EvalEncoding] = {}
        for pid, encs in train_encodings_per_person.items():
            tpl = _select_one_template_per_person(pid, encs, strategy)
            if tpl is not None:
                templates[pid] = tpl

        # 4.2 造配对：Genuine + 库外 Impostor（spec 已决定省去库内 Impostor）
        genuine = generate_genuine_pairs(test_encodings, templates)
        open_imp = generate_open_impostor_pairs(lfw_encodings, templates)
        all_pairs = genuine + open_imp
        pairs_per_strategy[name] = all_pairs

        # 4.3 计算指标
        eer, eer_thresh = compute_eer(all_pairs)
        tar = compute_tar_at_far(all_pairs, target_far=target_far)
        top1 = compute_top1_accuracy(test_encodings, templates)
        fpr, tpr, _ = compute_roc(all_pairs)

        all_metrics.append(StrategyMetrics(
            strategy_name=name,
            eer=eer,
            eer_threshold=eer_thresh,
            tar_at_far_1e3=tar,
            top1_accuracy=top1,
            roc_fpr=fpr,
            roc_tpr=tpr,
            n_genuine=len(genuine),
            n_impostor=len(open_imp),
        ))

    logger.info("[5/5] 写报告到 %s …", output_dir)
    reports.write_csv(all_metrics, output_dir / "summary.csv")
    reports.plot_roc_curves(all_metrics, output_dir / "roc_curves.png")
    hist_images: dict[str, str] = {}
    for name, pairs in pairs_per_strategy.items():
        rel = f"hist_{name}.png"
        reports.plot_score_histogram(pairs, name, output_dir / rel)
        hist_images[name] = rel
    reports.write_markdown(
        all_metrics,
        output_dir / "summary.md",
        roc_image="roc_curves.png",
        hist_images=hist_images,
    )

    logger.info("完成。最优策略（按 EER）：%s", min(all_metrics, key=lambda m: m.eer).strategy_name)
    return all_metrics
```

- [ ] **Step 4: 跑集成测试确认通过**

```bash
uv run pytest tests/integration/test_run_ablation.py -v -m slow
```

预期：1 passed

- [ ] **Step 5: 加 CLI 入口**

修改 `src/face_recognition/api/cli.py`，添加新命令 `evaluate`。

> 注意：M1 的 `_setup` callback 把 `AppConfig` 实例放到 `ctx.obj`（不是装配好的容器）。
> 其他子命令（`register` / `recognize`）的套路都是 `cfg = ctx.obj` → 调 `build_xxx_use_case(cfg)`
> 现场装配。`evaluate` 沿用同款套路：拿 cfg → 调 `build_pipeline(cfg)` 现造 pipeline。

```python
# 在 cli.py 已有的 import 之后追加
from pathlib import Path

import typer

from face_recognition.api.dependencies import build_pipeline
from face_recognition.evaluation.run_ablation import run_ablation


# 在已有的 app = typer.Typer(...) 之后追加命令
@app.command()
def evaluate(
    ctx: typer.Context,
    dataset: Path = typer.Option(..., help="私有数据集根目录"),
    output: Path = typer.Option(Path("reports"), help="报告输出目录"),
    n_lfw: int = typer.Option(50, help="LFW 抽样人数"),
) -> None:
    """跑 5 策略消融评估，产出 ROC/EER/TAR\\@FAR 报告。"""
    # ctx.obj 是 M1 _setup 塞进去的 AppConfig，跟 register/recognize 子命令同款
    cfg = ctx.obj
    pipeline = build_pipeline(cfg)
    run_ablation(
        dataset_root=dataset,
        output_dir=output,
        pipeline=pipeline,
        n_lfw=n_lfw,
        seed=cfg.evaluation.random_seed,
    )
    typer.echo(f"报告已写到 {output}/summary.md")
```

- [ ] **Step 6: commit**

```bash
git add src/face_recognition/evaluation/run_ablation.py \
        src/face_recognition/api/cli.py \
        tests/integration/test_run_ablation.py
git commit -m "feat(evaluation): 加 run_ablation 主入口 + CLI evaluate 命令"
```

---

### Task 9（M3）：在私有数据集上跑真实消融实验

**Files:**
- Run: 已写好的 `face_recognition.api.cli evaluate` 命令
- Output: `reports/summary.md` 等

> 这是**手动任务**——不写代码，只跑命令、看结果、做判断。前 8 个 task 完成后才能开始。

- [ ] **Step 1: 准备数据集**

把私有数据集放到 `data/private_dataset/`，按"每人一个文件夹"组织：

```
data/private_dataset/
├── alice/
│   ├── 000.jpg
│   ├── 001.jpg
│   └── ... (≥ 20 张，理想 60 张)
├── bob/
└── ...
```

**确认（再看一眼以免事故）**：
- `.gitignore` 已包含 `data/`（spec 强制要求私人照片不入仓）
- 每人至少 5 张，否则 split 会跳过该人

- [ ] **Step 2: 第一次跑（少量数据快速冒烟）**

```bash
uv run python -m face_recognition.api.cli evaluate \
    --dataset data/private_dataset \
    --output reports \
    --n-lfw 20
```

预期：约 2~5 分钟（取决于 GPU），生成 `reports/summary.md` 等 8 个文件。

- [ ] **Step 3: 跑正式实验**

把 `--n-lfw 20` 改成 `--n-lfw 50`（spec 决策值），重跑：

```bash
uv run python -m face_recognition.api.cli evaluate \
    --dataset data/private_dataset \
    --output reports \
    --n-lfw 50
```

- [ ] **Step 4: 看结果**

```bash
open reports/summary.md         # macOS：在 VS Code/浏览器里看
open reports/roc_curves.png
```

**重点检查**：

1. **5 个策略 EER 都给出来了吗**？（任何一行 NaN/缺失都说明 split 太小或数据有问题）
2. **EER 数字是否合理**？ArcFace + 干净人脸数据通常 EER < 0.05；如果 > 0.15 大概率是数据集太脏（很多模糊/侧脸）
3. **5 策略的相对排序**？预期：`mean_all` ≈ `kmeans_k3` > `manual_three` > `random_one`，`all_vectors` 视测试集大小可能略有波动
4. **TAR\@FAR=1e-3 是否 ≥ 0.85**？低于这个值的话生产门槛 0.45 会拒掉太多真员工

**如果结果有异常**：先看 `reports/hist_*.png` 直方图——
- Genuine 分布拖长尾、低于 0.5：说明很多正例分错（脸太花/年龄差大）
- Impostor 分布往右拖：说明负例分错（光照相似度被错认）

---

### Task 10（M3）：把最优阈值 / 策略写回 config.yaml + summary 写结论

**Files:**
- Modify: `config.yaml`（recognition.threshold、registration.default_strategy）
- Modify: `reports/summary.md`（追加结论段，进答辩报告）

- [ ] **Step 1: 决定最优策略**

按以下优先级从 `summary.csv` 选：

1. **首选 EER 最低**（理论最优）
2. **若多个策略 EER 接近（差 < 0.005）**：选 TAR\@FAR=1e-3 最高的（更贴近门禁场景）
3. **若仍并列**：选 `kmeans_k3`（spec 默认推荐，注册成本低于 all_vectors）

- [ ] **Step 2: 决定生产阈值**

阈值候选有两个：

- **EER 阈值**（`eer_threshold` 列）：FAR/FRR 平衡点
- **FAR=1e-3 对应阈值**：更严格，门禁推荐

门禁场景**强烈推荐用 FAR=1e-3 阈值**（不在 csv 里直接给，但能从 ROC 反算——也可临时加个 `--show-thresholds` flag，但 spec 已定 0.45 起点，先用这个值起步即可）。

实际操作：把 `summary.csv` 里最优策略那行的 `eer_threshold` 拿出来，**乘 1.05~1.10** 当生产阈值（让 FAR 比 EER 点更严一点）。

- [ ] **Step 3: 改 config.yaml**

```yaml
# 找到 recognition 段
recognition:
  threshold: 0.<最优策略 eer_threshold × 1.05>  # 由消融实验定标，2026-05-XX 跑

# 找到 registration 段
registration:
  default_strategy: kmeans_k3  # 改成实验选出来的最优策略名
```

- [ ] **Step 4: 在 reports/summary.md 末尾追加结论段**

模板（直接 copy 到 markdown 末尾，把 `<>` 替换成实测数据）：

```markdown
## 结论

经 5 策略消融实验，最优策略为 **<strategy_name>**：
- EER = <0.0XX>，对应阈值 <0.X>
- TAR\@FAR=1e-3 = <0.XX>
- Top-1 闭集准确率 = <0.XX>

生产配置（已写入 `config.yaml`）：
- `recognition.threshold = <0.X>`（在 EER 阈值基础上 ×1.05 让 FAR 更严）
- `registration.default_strategy = <strategy_name>`

**为什么不选 all_vectors**：相比 mean_all/kmeans_k3，all_vectors 检索时要遍历每人 N 个向量，
35 人 × 50 张 = 1750 次余弦——远超 200 向量的暴力检索假设；指标提升不到 0.005，不值得。

**为什么不选 random_one**：单张照片受光照/角度影响大，EER 比 mean_all 高 <X> 个百分点，
答辩时也不好讲故事。
```

- [ ] **Step 5: commit**

```bash
git add config.yaml reports/summary.md
git commit -m "chore: 用 M2 消融结果定标生产阈值与默认策略"
```

- [ ] **Step 6: 打 M2 tag**

```bash
git tag -a v0.2-evaluation -m "M2 + M3 评估框架完成；阈值已用消融数据定标"
git push origin main --tags
```

---

## 完成标准（Definition of Done）

- [ ] `evaluation/` 下 6 个文件齐：types / data_split / lfw_loader / pair_generator / metrics / reports + run_ablation + embedder
- [ ] 单元测试 ≥ 18 个，全部 pass
- [ ] 集成测试 1 个（mock pipeline 跑通流水线），pass
- [ ] CLI `evaluate` 命令在私有数据集上跑通，产出 `reports/summary.md`
- [ ] 5 策略指标全部出齐（无 NaN）
- [ ] `config.yaml` 用消融数据定标完毕
- [ ] M2 tag 已打
