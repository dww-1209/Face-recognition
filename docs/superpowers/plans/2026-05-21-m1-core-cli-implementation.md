# M1 核心 CLI 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 spec §1 中 M1 阶段的全部内容——可用 CLI 完成"按文件夹批量注册人员 → 识别一张照片 → 列出/删除人员"，含 5 个模板生成策略与 SQLite 持久化。

**Architecture:** 按 spec §3 的清洁架构四层（domain / application / infrastructure / api）逐层从内向外实现。先写 domain 实体与 Protocol，再写 infrastructure 层 SQLite 与 InsightFace pipeline，再写 application 层用例与 5 个策略，最后写 api 层 Typer CLI 与依赖装配。

**Tech Stack:** Python 3.12 / uv / InsightFace `buffalo_l` / ONNX Runtime (macOS CPU + CoreML) / numpy / scikit-learn / SQLite (stdlib) / Typer / pydantic-settings / pytest / ruff / mypy。

**Reference Documents:**
- 设计文档：`docs/superpowers/specs/2026-05-21-face-recognition-design.md`
- 项目指令：`CLAUDE.md`（本地，不在仓库）
- 配置模板：`config.yaml`

---

## 范围与不在范围

### M1 包含

- `domain/` 全部三个文件（实体、Protocol、异常）
- `infrastructure/insightface_pipeline.py`：buffalo_l 一站式封装
- `infrastructure/sqlite_repository.py`：向量库（含 BLOB 序列化、(M, 512) 矩阵导出）
- `infrastructure/config_loader.py`：pydantic-settings 加载 config.yaml
- `application/strategies/`：base + 5 个策略（random_one / mean_all / manual_three / kmeans_k3 / all_vectors）
- `application/register_face.py`：注册用例
- `application/recognize_face.py`：识别用例
- `api/dependencies.py`：依赖装配
- `api/cli.py`：Typer CLI（register / recognize / list / remove）
- 单元测试覆盖：5 策略、SQLite 仓库、注册/识别用例
- 集成测试：用 InsightFace 自带 sample 图跑通"注册→识别"全链路

### M1 不含（留给后续）

- `infrastructure/camera_capture.py`、`iou_tracker.py`（→ M4）
- `api/server.py`、`api/static/index.html`（→ M4/M5）
- `evaluation/` 全部内容（→ M2）

---

## 文件结构与责任

下面这张表锁定每个文件的产出与依赖，避免分歧。所有文件路径相对仓库根目录。

| 文件 | 责任 | 依赖 |
| --- | --- | --- |
| `src/face_recognition/domain/entities.py` | `FaceEncoding`、`Template`、`Person`、`RecognitionResult` 四个 frozen dataclass | numpy |
| `src/face_recognition/domain/interfaces.py` | `FacePipeline`、`PersonRepository`、`TemplateStrategy` 三个 Protocol | numpy, entities |
| `src/face_recognition/domain/errors.py` | 领域异常层次 | 仅 stdlib |
| `src/face_recognition/infrastructure/config_loader.py` | `AppConfig` (pydantic BaseSettings) + `load_config()` | pydantic-settings, yaml |
| `src/face_recognition/infrastructure/sqlite_repository.py` | `SqliteRepository` 实现 `PersonRepository` | sqlite3, numpy, domain |
| `src/face_recognition/infrastructure/insightface_pipeline.py` | `InsightFacePipeline` 实现 `FacePipeline` | insightface, opencv, numpy, domain |
| `src/face_recognition/application/strategies/base.py` | （空）只占位，5 个策略各自直接实现 `TemplateStrategy` Protocol | — |
| `src/face_recognition/application/strategies/random_one.py` | `RandomOneStrategy` | random, domain |
| `src/face_recognition/application/strategies/mean_all.py` | `MeanAllStrategy` | numpy, domain |
| `src/face_recognition/application/strategies/manual_three.py` | `ManualThreeStrategy`（前 3 张取，简化版） | domain |
| `src/face_recognition/application/strategies/kmeans_k3.py` | `KMeansK3Strategy` | scikit-learn, numpy, domain |
| `src/face_recognition/application/strategies/all_vectors.py` | `AllVectorsStrategy` | domain |
| `src/face_recognition/application/register_face.py` | `RegisterFace` 用例 | domain, strategies, infrastructure (注入) |
| `src/face_recognition/application/recognize_face.py` | `RecognizeFace` 用例 | numpy, domain |
| `src/face_recognition/api/dependencies.py` | `build_pipeline()`, `build_repository()`, `build_strategy()`, `build_register_use_case()`, `build_recognize_use_case()` | infrastructure, application, config |
| `src/face_recognition/api/cli.py` | Typer app: `register`, `recognize`, `list`, `remove` | typer, dependencies |

测试文件结构对应：

```
tests/
├── unit/
│   ├── test_entities.py
│   ├── test_errors.py
│   ├── test_config_loader.py
│   ├── test_sqlite_repository.py
│   ├── test_strategies.py
│   ├── test_register_face.py
│   └── test_recognize_face.py
├── integration/
│   └── test_register_recognize_e2e.py
└── conftest.py                          # 共享 fixtures：临时 db、合成 encoding 等
```

---

## 任务清单（13 个）

执行顺序遵循依赖：domain → infrastructure → application → api → 集成测试。每个任务都是一个完整的 TDD 循环（红→绿→commit）。

---

### Task 1: 领域实体（FaceEncoding、Template、Person、RecognitionResult）

**Files:**
- Create: `src/face_recognition/domain/entities.py`
- Test: `tests/unit/test_entities.py`

- [ ] **Step 1: 写失败的测试 `tests/unit/test_entities.py`**

```python
# numpy 是 Python 科学计算的事实标准库，约定别名 np
# 我们用它造测试向量、检查范数等
import numpy as np

# pytest 是 Python 最流行的测试框架。这里 import 它本身是为了用 pytest.raises / pytest.approx
import pytest

# datetime 用来给 Template 加"创建时间"字段
from datetime import datetime

# 从我们要实现的模块里导入 4 个实体类
# 注意：这一行会失败（因为 entities.py 还是空的）——这正是 TDD"先写失败测试"的重点
from face_recognition.domain.entities import (
    FaceEncoding,
    Template,
    Person,
    RecognitionResult,
)


def _unit_vector(seed: int = 0) -> np.ndarray:
    """生成一个 L2 归一化的随机 512 维向量，用于测试时模拟 ArcFace 输出。

    "L2 归一化" = 向量除以它自己的长度，使最终长度为 1。
    几何意义：把向量"压"到单位球面上，方向不变、长度统一为 1。
    ArcFace 真实输出也是单位球面上的向量，所以测试用同样形态来模拟最逼真。

    函数名前的下划线 `_` 是 Python 约定的"模块私有"标志——
    告诉读者"这是测试内部辅助函数，不要从外面 import 它"。
    """

    # np.random.default_rng(seed) 是 NumPy 1.17+ 推荐的随机数生成器构造方式
    #   - 旧 API 是 np.random.seed(...)（修改全局状态，多线程下相互干扰）
    #   - 新 API 返回一个独立的 Generator 对象，互不影响
    #   - seed 相同时生成的"随机"数完全一样——保证测试可复现
    rng = np.random.default_rng(seed)

    # rng.standard_normal(N) = 从标准正态分布 N(0, 1) 抽 N 个独立样本
    #   - 标准正态：均值 0、方差 1 的钟形分布
    #   - 为什么不用 rng.random()（[0, 1) 均匀分布）？
    #     正态分布在每个维度上对称地分散，归一化后向量在球面上分布更均匀；
    #     均匀分布会让向量倾向某些角落，对模拟"特征向量"不真实
    # .astype(np.float32) = 把数组类型转成 32 位浮点
    #   - NumPy 默认是 float64（双精度）；ArcFace 实际输出是 float32
    #   - 强制对齐 float32 让测试与生产环境的数值精度一致
    v = rng.standard_normal(512).astype(np.float32)

    # np.linalg.norm(v) 默认计算 L2 范数 = sqrt(v[0]² + v[1]² + ... + v[511]²)
    #   - "linalg" = "linear algebra"（线性代数）模块缩写
    #   - 想算别的范数可传 ord=1（绝对值之和）或 ord=np.inf（最大绝对值）
    #   - 对 512 维向量结果是它的"长度"，恒为正数
    # v / norm 用 NumPy 的"广播（broadcasting）"：标量除向量等于逐元素除
    #   - 等价于 [v[0]/norm, v[1]/norm, ..., v[511]/norm]
    #   - 除完后向量的 L2 范数严格等于 1（除浮点误差）
    return v / np.linalg.norm(v)


# 测试函数命名约定：以 test_ 开头 pytest 才会自动发现并执行
def test_face_encoding_requires_512_dim():
    """构造 FaceEncoding 时如果维度不是 512，应抛 ValueError。"""
    # pytest.raises(ExpectedException) 是 pytest 提供的"断言异常"上下文管理器
    #   - 如果 with 块内代码抛了 ValueError，测试通过
    #   - 如果没抛任何异常 / 抛了别的异常，测试失败
    # np.zeros(128, dtype=np.float32) 创建一个全 0 的 128 维 float32 向量
    #   - 维度 128 != 512，应触发我们的校验逻辑
    with pytest.raises(ValueError):
        FaceEncoding(vector=np.zeros(128, dtype=np.float32), model_version="buffalo_l")


def test_face_encoding_requires_l2_normalized():
    """构造 FaceEncoding 时如果向量没有 L2 归一化，应抛 ValueError。"""
    # np.ones(512) 是全 1 向量，它的 L2 范数 = sqrt(512) ≈ 22.63，远不为 1
    # 这种"未归一化"的向量进入领域层是 bug，必须拦下
    with pytest.raises(ValueError):
        FaceEncoding(vector=np.ones(512, dtype=np.float32), model_version="buffalo_l")


def test_face_encoding_cosine_similarity_self_is_one():
    """同一个 FaceEncoding 与自己的余弦相似度应该是 1.0。"""
    enc = FaceEncoding(vector=_unit_vector(0), model_version="buffalo_l")
    # pytest.approx 用于浮点近似比较——直接 == 1.0 在浮点世界几乎永远不成立
    #   - abs=1e-6 表示允许 ±0.000001 的误差
    #   - 也可以传 rel=1e-6 用相对误差
    assert enc.cosine_similarity(enc) == pytest.approx(1.0, abs=1e-6)


def test_face_encoding_cosine_similarity_orthogonal_is_zero():
    """两个互相正交（垂直）的单位向量，余弦相似度应该是 0。"""
    # 构造两个特殊的单位向量：v1 沿 x 轴方向，v2 沿 y 轴方向，互相正交
    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0  # 第 0 维 = 1，其他全 0；它的 L2 范数 = 1，是合法 unit vector
    v2 = np.zeros(512, dtype=np.float32)
    v2[1] = 1.0
    e1 = FaceEncoding(vector=v1, model_version="buffalo_l")
    e2 = FaceEncoding(vector=v2, model_version="buffalo_l")
    # 正交向量点积 = 0，因此余弦相似度 = 0
    # 这测试除了功能正确，还顺便验证 cosine_similarity 用的是点积公式
    assert e1.cosine_similarity(e2) == pytest.approx(0.0, abs=1e-6)


def test_face_encoding_is_frozen():
    """frozen dataclass 的字段不可修改，赋值会抛异常。"""
    enc = FaceEncoding(vector=_unit_vector(0), model_version="buffalo_l")
    # 注释 # FrozenInstanceError 解释：dataclasses 模块定义的专用异常
    # 它继承自 AttributeError；这里用宽泛的 Exception 是因为不想让测试和具体异常类型耦合
    with pytest.raises(Exception):  # FrozenInstanceError
        enc.model_version = "other"


def test_person_templates_must_be_tuple():
    """Person.templates 必须是 tuple（不可变），不是 list。"""
    enc = FaceEncoding(vector=_unit_vector(0), model_version="buffalo_l")
    # Template 三个字段都给齐：encoding / source / created_at
    # datetime.now() 拿当前时间；测试里用什么时间无所谓
    tpl = Template(encoding=enc, source="test", created_at=datetime.now())
    # (tpl,) 是单元素元组；写 (tpl) 会被解析成普通括号表达式而非元组
    # 这是 Python 容易踩的坑：单元素元组必须有逗号
    person = Person(person_id="alice", display_name="Alice", templates=(tpl,))
    # isinstance(x, T) 判断 x 是否为 T 类型（包括子类）
    assert isinstance(person.templates, tuple)


def test_recognition_result_unknown_has_none_person_id():
    """识别结果"未知"时 person_id 应是 None，相似度低于阈值。"""
    r = RecognitionResult(person_id=None, similarity=0.3, threshold=0.45)
    # is None 比 == None 更好：is 比较"同一对象"，是 None 检测的官方推荐写法
    assert r.person_id is None
    assert r.similarity < r.threshold
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/test_entities.py -v
```

预期：`ModuleNotFoundError: No module named 'face_recognition.domain.entities'`（文件还没有内容）

- [ ] **Step 3: 实现 `src/face_recognition/domain/entities.py`**

```python
# dataclasses 是 Python 3.7+ 标准库，让定义"只装数据的类"少写一堆样板代码
# 没有它你要手写 def __init__(self, x, y): self.x = x; self.y = y; ... 一长串
from dataclasses import dataclass

# datetime 类型用于 Template.created_at 字段
from datetime import datetime

import numpy as np

# ===== 模块级常量 =====
# 下划线开头是 Python 约定的"私有"标志，不会被 from xxx import * 导出
_EMBED_DIM = 512                # ArcFace ResNet100 输出向量维度，buffalo_l 模型固定为 512
_NORM_TOLERANCE = 1e-3          # L2 范数容忍度：理论 ||v||=1，浮点实际可能是 0.9999~1.0001 都接受


# ===== FaceEncoding：领域层最核心的实体 =====
# @dataclass 是装饰器，作用是给 class 自动生成几个常用方法：
#   - __init__   构造函数（按字段顺序接参）
#   - __repr__   打印表示（FaceEncoding(vector=..., model_version='buffalo_l')）
#   - __eq__     相等比较
# 不用它你要手写每一个，几十行的样板代码
#
# frozen=True 让实例创建后所有字段不可修改：
#   - 给字段赋值会抛 dataclasses.FrozenInstanceError
#   - 这正是"领域实体不可变"原则的代码体现——避免被外部代码偷偷改坏导致难追的 bug
@dataclass(frozen=True)
class FaceEncoding:
    """ArcFace 对一张人脸输出的 512 维特征向量（已 L2 归一化）。

    设计要点：
    - vector 必须 (512,) float32 且 ||v|| ≈ 1
    - 归一化后两个 encoding 的余弦相似度 = 它们的点积，范围 [-1, 1]
    - 这是整个识别系统的"通用货币"——所有相似度计算都基于它

    "model_version" 字段记录这个向量是哪个模型出的；将来切换模型时
    可以拒绝跨模型比对（比如 buffalo_l 的向量不能和 buffalo_s 的混着比）
    """

    # 用类型注解声明字段，dataclass 据此生成 __init__
    # np.ndarray 是 NumPy 数组类型；shape/dtype 在 __post_init__ 中校验
    vector: np.ndarray
    model_version: str

    # __post_init__ 是 dataclass 提供的特殊方法：自动生成的 __init__ 跑完后会调用它
    # 用途：在构造结束后做字段校验。如果字段不合法，抛异常拦截构造
    # 这是"在系统边界做校验"的核心实践——非法实体根本无法存在
    def __post_init__(self) -> None:
        # .shape 是 NumPy 数组的形状属性，返回元组
        # 一维 512 长度向量的 shape 是 (512,)，注意逗号——Python 单元素元组写法
        if self.vector.shape != (_EMBED_DIM,):
            # f-string 是 Python 3.6+ 的字符串格式化语法
            # f"...{expr}..." 中 {expr} 会被求值后插入字符串
            raise ValueError(
                f"FaceEncoding 必须是 ({_EMBED_DIM},) 维，收到 {self.vector.shape}"
            )
        # np.linalg.norm 默认计算 L2 范数（向量长度）
        # float(...) 把 NumPy 标量转 Python 原生 float
        #   - 避免后续 JSON 序列化、日志打印等场景遇到 numpy 类型的小麻烦
        norm = float(np.linalg.norm(self.vector))
        if abs(norm - 1.0) > _NORM_TOLERANCE:
            # {norm:.4f} 是格式化指令——保留 4 位小数
            raise ValueError(f"FaceEncoding 必须 L2 归一化（||v||=1），当前 ||v||={norm:.4f}")

    # other 参数类型注解写成 "FaceEncoding"（带引号）的字符串
    # 因为定义到这里时 FaceEncoding 这个名字还没"完全建好"，直接用会报错
    # 用引号叫"前向引用（forward reference）"，Python 会延迟解析
    def cosine_similarity(self, other: "FaceEncoding") -> float:
        """计算与另一个 FaceEncoding 的余弦相似度。

        因为两个向量都已 L2 归一化，余弦相似度等价于它们的点积——
        无需再除以范数（都是 1），省一步且更稳。
        """
        # np.dot 在两个一维数组上 = 它们的内积/点积 = sum(a[i] * b[i])
        # 内积公式 vs 余弦相似度：cos = (a·b) / (||a||·||b||)
        # 因为 ||a||=||b||=1，所以 cos = a·b（点积）
        return float(np.dot(self.vector, other.vector))


# ===== Template：单条模板向量 =====
# 一个 Person 可能有 1~50 条 Template，取决于使用的策略：
#   - random_one / mean_all 策略：1 条
#   - manual_three / kmeans_k3：3 条
#   - all_vectors：N 条（注册集所有照片）
@dataclass(frozen=True)
class Template:
    """单条模板向量。识别时把库内所有 Template 凑成一个矩阵做暴力检索。"""

    encoding: FaceEncoding
    source: str                 # 来源标识，如 "kmeans_centroid_0" / "raw_photo_3.jpg" / "mean"
    created_at: datetime


# ===== Person：领域聚合根 =====
# 在领域驱动设计（DDD）中，"聚合根"是访问一组相关实体的唯一入口
# 这里 Person 是访问其下所有 Template 的入口——你不能在系统中独立地存一个无主 Template
@dataclass(frozen=True)
class Person:
    """库内人员 = 唯一 ID + 名字 + 多个模板向量。"""

    person_id: str              # 唯一 ID，本项目中直接用文件夹名（如 "alice"）
    display_name: str           # 展示名，前端用
    # tuple[Template, ...] 类型注解：含义是"任意长度的 Template 元组"
    #   - 末尾的 ... 是 Ellipsis 字面量，表示"可变长"
    #   - 用 tuple 而非 list：与 frozen=True 配合，让 Person 真正不可变
    #     list 是可变容器（即使外层 frozen，list 内容仍可被偷偷改）
    #     tuple 是不可变容器，外层加内层都不可变，消除一类 bug
    templates: tuple[Template, ...]


# ===== RecognitionResult：识别用例的输出 =====
@dataclass(frozen=True)
class RecognitionResult:
    """把识别决定 + 置信度 + 当时阈值打包返回。

    为什么把 threshold 也放进结果里？
    - 便于审计与日志：将来回看历史识别记录，能立即看出"当时用的阈值"
    - 阈值会随着评估实验更新（spec §6 提到的 config.yaml 调整）
    """

    # str | None 是 PEP 604 联合类型语法（Python 3.10+ 支持）
    #   - 等价于旧写法 Optional[str] 或 Union[str, None]
    #   - None 表示未识别（库为空 / 最高相似度 < 阈值）
    person_id: str | None
    similarity: float           # 与最匹配模板的相似度，[-1, 1]
    threshold: float            # 当时使用的阈值
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/test_entities.py -v
```

预期：7 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/domain/entities.py tests/unit/test_entities.py
git commit -m "feat(domain): 加 FaceEncoding/Template/Person/RecognitionResult 实体"
```

---

### Task 2: 领域异常层次

**Files:**
- Create: `src/face_recognition/domain/errors.py`
- Test: `tests/unit/test_errors.py`

- [ ] **Step 1: 写失败的测试 `tests/unit/test_errors.py`**

```python
import pytest

# 一次 import 多个名字时，把它们写在括号里换行，可读性更好
from face_recognition.domain.errors import (
    FaceRecognitionError,
    NoFaceError,
    MultipleFacesError,
    PersonNotFoundError,
    DuplicatePersonError,
    LowConfidenceError,
    PersonHasNoTemplatesError,
    CameraDisconnectedError,
)


def test_all_errors_inherit_base():
    """所有具体异常都应继承自 FaceRecognitionError 基类。

    为什么需要基类？API 层可以一次 try ... except FaceRecognitionError，
    捕获我们项目所有领域异常，统一映射成 HTTP 4xx/5xx；
    不会因为忘了某个具体异常而漏处理。
    """
    # 把所有具体异常类放到一个元组里，循环检查
    # 元组比 list 更适合"固定不变的几个东西"——也是 Python 习惯
    for cls in (
        NoFaceError, MultipleFacesError, PersonNotFoundError,
        DuplicatePersonError, LowConfidenceError,
        PersonHasNoTemplatesError, CameraDisconnectedError,
    ):
        # issubclass(child, parent) 检查 child 是不是 parent 的子类（包括孙子辈等）
        # 注意：issubclass 接受类对象作参数；isinstance 接受实例对象
        assert issubclass(cls, FaceRecognitionError)


def test_each_error_has_stable_code():
    """每个异常类都应有稳定的 code 字符串属性。

    为什么需要 code？将来前端要根据错误类型展示中文提示
    （比如 NO_FACE → "未检测到人脸，请正对镜头"）。
    类名可能改名重构，但 code 是稳定 API——前后端约定的"暗号"。
    """
    # 类属性可以直接用 ClassName.attr 访问，不需要先创建实例
    assert NoFaceError.code == "NO_FACE"
    assert MultipleFacesError.code == "MULTIPLE_FACES"
    assert PersonNotFoundError.code == "PERSON_NOT_FOUND"
    assert DuplicatePersonError.code == "DUPLICATE_PERSON"
    assert LowConfidenceError.code == "LOW_CONFIDENCE"
    assert PersonHasNoTemplatesError.code == "NO_TEMPLATES"
    assert CameraDisconnectedError.code == "CAMERA_LOST"


def test_multiple_faces_carries_count():
    """MultipleFacesError 应携带具体检出的人脸数量。

    为什么需要 count？错误信息能更精确："检出 3 张脸（要求 1 张）"
    比"检出多张脸"更有用——日志里能直接看出问题严重程度。
    """
    err = MultipleFacesError(count=3)
    # 实例属性用 instance.attr 访问
    # 这里 count 是 MultipleFacesError 的实例属性（每个实例独立存储），
    # 不是类属性（所有实例共享）
    assert err.count == 3


def test_can_be_raised_and_caught_by_base():
    """子类异常应能被基类的 except 块捕获——这是基类设计的核心目的。"""
    # raise 关键字抛异常；NoFaceError(...) 创建异常实例
    # except 子句捕获时按"是不是基类的实例"判断，所以子类抛出能被基类捕获
    with pytest.raises(FaceRecognitionError):
        raise NoFaceError("没检测到人脸")
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/test_errors.py -v
```

- [ ] **Step 3: 实现 `src/face_recognition/domain/errors.py`**

```python
# Python 的所有异常都继承自内置 Exception 类
# 自定义异常的标准做法：写一个 class，继承 Exception 即可
# 这是整个项目领域异常的"祖先"——任何业务错误都应通过它的子类抛出
class FaceRecognitionError(Exception):
    """所有领域异常的基类。

    用法：
        try:
            ...
        except FaceRecognitionError as e:
            print(e.code)  # 给前端的稳定错误码
    """

    # 类属性（class attribute）：所有实例共享同一个 code
    # 子类可以覆盖它（写 code = "NO_FACE" 等）
    # 类型注解 `code: str = "..."` 表示"code 是 str 类型，默认值 ..."
    code: str = "FACE_RECOGNITION_ERROR"


# class Child(Parent): 表示 Child 继承自 Parent
# 子类自动获得父类的所有属性和方法（除非显式覆盖）
class NoFaceError(FaceRecognitionError):
    """图中未检测到人脸时抛出。"""

    # 这里覆盖父类的 code 属性
    # Python 的属性查找：先在子类找，找不到再去父类找——所以子类 code 会胜出
    code = "NO_FACE"


class MultipleFacesError(FaceRecognitionError):
    """图中检测到多张脸（注册/识别要求单脸）时抛出。"""

    code = "MULTIPLE_FACES"

    # 显式定义 __init__ 接受额外参数 count
    # 注意：父类 Exception 的 __init__ 接受 message 参数；这里我们想保留这能力 + 加 count
    # message: str | None = None 是默认参数：调用时不传则为 None
    def __init__(self, count: int, message: str | None = None) -> None:
        # super().__init__(...) 调用父类（Exception）的构造函数
        # 'super()' 是 Python 引用父类的标准方式
        # 传给它的 message 会成为异常的"消息"，用 str(err) 能打印出来
        # 'message or f"..."' 是 Python 习惯：message 真值时用 message，否则用 fallback
        #   - None / 空字符串 / 0 / False / 空列表都算"假值"
        super().__init__(message or f"检出 {count} 张脸（要求 1 张）")
        # self.count 是实例属性，每个 MultipleFacesError 实例自己存一份
        # 区别于上面的类属性 code（所有实例共享）
        self.count = count


# 后续异常都是简单子类，只覆盖 code 即可——不需要额外字段或自定义 __init__
class PersonNotFoundError(FaceRecognitionError):
    """要查询/删除的 person_id 不在数据库时抛出。"""
    code = "PERSON_NOT_FOUND"


class DuplicatePersonError(FaceRecognitionError):
    """注册时 person_id 已存在（如要求"严格唯一"模式）时抛出。"""
    code = "DUPLICATE_PERSON"


class LowConfidenceError(FaceRecognitionError):
    """识别相似度低于阈值时使用——通常不抛而是返回 RecognitionResult(person_id=None)。
    保留这个异常类型给将来如有"必须强匹配"的场景用。
    """
    code = "LOW_CONFIDENCE"


class PersonHasNoTemplatesError(FaceRecognitionError):
    """注册某人时所有照片都识别失败、一个模板都没生成的情况。"""
    code = "NO_TEMPLATES"


class CameraDisconnectedError(FaceRecognitionError):
    """实时识别中摄像头中途断开。M4 阶段才会真正抛出，这里先定义好。"""
    code = "CAMERA_LOST"
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/test_errors.py -v
```

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/domain/errors.py tests/unit/test_errors.py
git commit -m "feat(domain): 加领域异常层次（NoFace/MultipleFaces/PersonNotFound 等）"
```

---

### Task 3: 领域 Protocol 接口

**Files:**
- Create: `src/face_recognition/domain/interfaces.py`

无需测试——Protocol 只是类型标记，没有运行时逻辑。后续具体实现的测试会自动验证它们符合 Protocol。

- [ ] **Step 1: 实现 `src/face_recognition/domain/interfaces.py`**

```python
from typing import Protocol

import numpy as np

from face_recognition.domain.entities import (
    FaceEncoding,
    Person,
    Template,
)


class FacePipeline(Protocol):
    def encode(self, image: np.ndarray) -> list[FaceEncoding]: ...

    def encode_single(self, image: np.ndarray) -> FaceEncoding: ...


class PersonRepository(Protocol):
    def add(self, person: Person) -> None: ...

    def get(self, person_id: str) -> Person | None: ...

    def remove(self, person_id: str) -> None: ...

    def list_all(self) -> list[Person]: ...

    def all_templates_matrix(self) -> tuple[np.ndarray, list[str]]:
        """返回 (M, 512) 模板矩阵 + 长度 M 的 person_id 列表（每行对应一个模板）"""
        ...


class TemplateStrategy(Protocol):
    name: str

    def build(self, encodings: list[FaceEncoding]) -> list[Template]: ...
```

- [ ] **Step 2: 跑 mypy 验证类型协议正确**

```bash
uv run mypy src/face_recognition/domain/
```

预期：Success: no issues found

- [ ] **Step 3: commit**

```bash
git add src/face_recognition/domain/interfaces.py
git commit -m "feat(domain): 加 FacePipeline/PersonRepository/TemplateStrategy Protocol"
```

---

### Task 4: 共享测试 fixtures（conftest.py）

**Files:**
- Create: `tests/conftest.py`

后续多个测试会需要"造合成 FaceEncoding"和"临时 SQLite 文件"。集中到 conftest 避免重复。

- [ ] **Step 1: 实现 `tests/conftest.py`**

```python
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from face_recognition.domain.entities import FaceEncoding, Template


@pytest.fixture
def make_encoding() -> Callable[[int], FaceEncoding]:
    """生成确定性、L2 归一化的 FaceEncoding。同一 seed → 同一向量。"""

    def _make(seed: int) -> FaceEncoding:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(512).astype(np.float32)
        v /= np.linalg.norm(v)
        return FaceEncoding(vector=v, model_version="test")

    return _make


@pytest.fixture
def make_template(make_encoding: Callable[[int], FaceEncoding]) -> Callable[[int, str], Template]:
    def _make(seed: int, source: str = "test") -> Template:
        return Template(
            encoding=make_encoding(seed),
            source=source,
            created_at=datetime(2026, 1, 1),
        )

    return _make


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_face.db"
```

- [ ] **Step 2: 跑测试套件确认 fixture 不破坏现有测试**

```bash
uv run pytest tests/unit/ -v
```

预期：之前 11 个测试仍然 pass。

- [ ] **Step 3: commit**

```bash
git add tests/conftest.py
git commit -m "test: 加共享 fixtures（make_encoding/make_template/tmp_db_path）"
```

---

### Task 5: 配置加载器（pydantic-settings + config.yaml）

**Files:**
- Create: `src/face_recognition/infrastructure/config_loader.py`
- Test: `tests/unit/test_config_loader.py`

- [ ] **Step 1: 写失败的测试 `tests/unit/test_config_loader.py`**

```python
from pathlib import Path

import pytest
import yaml

from face_recognition.infrastructure.config_loader import AppConfig, load_config


@pytest.fixture
def valid_config_path(tmp_path: Path) -> Path:
    cfg = {
        "model": {"pack": "buffalo_l", "ctx_id": 0, "det_size": [640, 640]},
        "recognition": {"threshold": 0.45, "template_strategy": "kmeans_k3"},
        "camera": {"device_index": 0, "resolution": [1280, 720], "fps": 30},
        "realtime": {
            "detect_every_n_frames": 1,
            "recognize_on_new_track": True,
            "iou_threshold": 0.5,
            "track_max_missing_frames": 15,
        },
        "api": {"host": "0.0.0.0", "port": 8000},
        "data": {
            "sqlite_path": "data/face.db",
            "dataset_root": "data/private_dataset",
            "lfw_subset": "data/lfw_subset",
        },
        "evaluation": {"random_seed": 42, "train_ratio": 0.8, "far_targets": [0.001, 0.01, 0.1]},
        "logging": {"level": "INFO", "file": "logs/face.log"},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_load_config_returns_app_config(valid_config_path: Path):
    cfg = load_config(valid_config_path)
    assert isinstance(cfg, AppConfig)
    assert cfg.recognition.threshold == 0.45
    assert cfg.recognition.template_strategy == "kmeans_k3"
    assert cfg.evaluation.random_seed == 42
    assert cfg.data.sqlite_path == Path("data/face.db")


def test_invalid_strategy_name_raises(tmp_path: Path):
    cfg = {
        "model": {"pack": "buffalo_l", "ctx_id": 0, "det_size": [640, 640]},
        "recognition": {"threshold": 0.5, "template_strategy": "no_such_strategy"},
        "camera": {"device_index": 0, "resolution": [1280, 720], "fps": 30},
        "realtime": {
            "detect_every_n_frames": 1,
            "recognize_on_new_track": True,
            "iou_threshold": 0.5,
            "track_max_missing_frames": 15,
        },
        "api": {"host": "0.0.0.0", "port": 8000},
        "data": {
            "sqlite_path": "data/face.db",
            "dataset_root": "data/private_dataset",
            "lfw_subset": "data/lfw_subset",
        },
        "evaluation": {"random_seed": 42, "train_ratio": 0.8, "far_targets": [0.001]},
        "logging": {"level": "INFO", "file": "logs/face.log"},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(Exception):  # pydantic ValidationError
        load_config(p)


def test_threshold_out_of_range_raises(tmp_path: Path, valid_config_path: Path):
    raw = yaml.safe_load(valid_config_path.read_text())
    raw["recognition"]["threshold"] = 2.0
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(Exception):
        load_config(p)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/test_config_loader.py -v
```

- [ ] **Step 3: 实现 `src/face_recognition/infrastructure/config_loader.py`**

```python
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

StrategyName = Literal[
    "random_one", "mean_all", "manual_three", "kmeans_k3", "all_vectors"
]


class ModelConfig(BaseModel):
    pack: str
    ctx_id: int
    det_size: tuple[int, int]


class RecognitionConfig(BaseModel):
    threshold: float = Field(ge=-1.0, le=1.0)
    template_strategy: StrategyName


class CameraConfig(BaseModel):
    device_index: int
    resolution: tuple[int, int]
    fps: int = Field(gt=0)


class RealtimeConfig(BaseModel):
    detect_every_n_frames: int = Field(ge=1)
    recognize_on_new_track: bool
    iou_threshold: float = Field(ge=0.0, le=1.0)
    track_max_missing_frames: int = Field(ge=0)


class ApiConfig(BaseModel):
    host: str
    port: int = Field(gt=0, le=65535)


class DataConfig(BaseModel):
    sqlite_path: Path
    dataset_root: Path
    lfw_subset: Path


class EvaluationConfig(BaseModel):
    random_seed: int
    train_ratio: float = Field(gt=0.0, lt=1.0)
    far_targets: list[float]

    @field_validator("far_targets")
    @classmethod
    def _all_in_range(cls, v: list[float]) -> list[float]:
        for x in v:
            if not 0.0 < x < 1.0:
                raise ValueError(f"FAR 目标必须在 (0, 1) 内，收到 {x}")
        return v


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    file: str


class AppConfig(BaseModel):
    model: ModelConfig
    recognition: RecognitionConfig
    camera: CameraConfig
    realtime: RealtimeConfig
    api: ApiConfig
    data: DataConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig


def load_config(path: Path | str = "config.yaml") -> AppConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(raw)
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/test_config_loader.py -v
```

预期：3 passed

- [ ] **Step 5: 验证生产 config.yaml 可加载**

```bash
uv run python -c "from face_recognition.infrastructure.config_loader import load_config; print(load_config('config.yaml'))"
```

预期：打印出 AppConfig 实例（一长串字段值）

- [ ] **Step 6: commit**

```bash
git add src/face_recognition/infrastructure/config_loader.py tests/unit/test_config_loader.py
git commit -m "feat(infrastructure): 加 pydantic config_loader（YAML + 字段校验）"
```

---

### Task 6: SQLite 仓库（PersonRepository 实现）

**Files:**
- Create: `src/face_recognition/infrastructure/sqlite_repository.py`
- Test: `tests/unit/test_sqlite_repository.py`

- [ ] **Step 1: 写失败的测试 `tests/unit/test_sqlite_repository.py`**

```python
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from face_recognition.domain.entities import Person, Template
from face_recognition.domain.errors import PersonNotFoundError
from face_recognition.infrastructure.sqlite_repository import SqliteRepository


def test_add_and_get_roundtrip(
    tmp_db_path: Path,
    make_template: Callable[[int, str], Template],
):
    repo = SqliteRepository(tmp_db_path)
    alice = Person(
        person_id="alice",
        display_name="Alice",
        templates=(make_template(1, "centroid_0"), make_template(2, "centroid_1")),
    )
    repo.add(alice)

    fetched = repo.get("alice")
    assert fetched is not None
    assert fetched.person_id == "alice"
    assert fetched.display_name == "Alice"
    assert len(fetched.templates) == 2
    assert np.allclose(
        fetched.templates[0].encoding.vector,
        alice.templates[0].encoding.vector,
    )


def test_get_unknown_returns_none(tmp_db_path: Path):
    repo = SqliteRepository(tmp_db_path)
    assert repo.get("ghost") is None


def test_list_all_orders_by_person_id(
    tmp_db_path: Path,
    make_template: Callable[[int, str], Template],
):
    repo = SqliteRepository(tmp_db_path)
    repo.add(Person("bob", "Bob", (make_template(10, "x"),)))
    repo.add(Person("alice", "Alice", (make_template(20, "x"),)))
    ids = [p.person_id for p in repo.list_all()]
    assert ids == ["alice", "bob"]


def test_add_existing_person_replaces(
    tmp_db_path: Path,
    make_template: Callable[[int, str], Template],
):
    repo = SqliteRepository(tmp_db_path)
    repo.add(Person("alice", "Alice", (make_template(1, "x"),)))
    # 重新注册：模板内容变化
    repo.add(Person("alice", "Alice 2", (make_template(99, "y"), make_template(100, "z"))))
    fetched = repo.get("alice")
    assert fetched is not None
    assert fetched.display_name == "Alice 2"
    assert len(fetched.templates) == 2


def test_remove_nonexistent_raises(tmp_db_path: Path):
    repo = SqliteRepository(tmp_db_path)
    with pytest.raises(PersonNotFoundError):
        repo.remove("ghost")


def test_remove_then_get_returns_none(
    tmp_db_path: Path,
    make_template: Callable[[int, str], Template],
):
    repo = SqliteRepository(tmp_db_path)
    repo.add(Person("alice", "Alice", (make_template(1, "x"),)))
    repo.remove("alice")
    assert repo.get("alice") is None


def test_all_templates_matrix_shape_and_index(
    tmp_db_path: Path,
    make_template: Callable[[int, str], Template],
):
    repo = SqliteRepository(tmp_db_path)
    repo.add(Person("alice", "A", (make_template(1, "x"), make_template(2, "y"))))
    repo.add(Person("bob", "B", (make_template(3, "x"),)))
    matrix, ids = repo.all_templates_matrix()
    assert matrix.shape == (3, 512)
    assert matrix.dtype == np.float32
    assert ids == ["alice", "alice", "bob"]
    # 每行仍归一化
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_empty_matrix_is_zero_rows(tmp_db_path: Path):
    repo = SqliteRepository(tmp_db_path)
    matrix, ids = repo.all_templates_matrix()
    assert matrix.shape == (0, 512)
    assert ids == []
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/test_sqlite_repository.py -v
```

- [ ] **Step 3: 实现 `src/face_recognition/infrastructure/sqlite_repository.py`**

```python
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from face_recognition.domain.entities import FaceEncoding, Person, Template
from face_recognition.domain.errors import PersonNotFoundError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    person_id     TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS templates (
    person_id     TEXT NOT NULL,
    template_idx  INTEGER NOT NULL,
    vector        BLOB NOT NULL,
    source        TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (person_id, template_idx),
    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
);
"""


class SqliteRepository:
    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)

    def add(self, person: Person) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM persons WHERE person_id = ?", (person.person_id,))
            self._conn.execute(
                "INSERT INTO persons (person_id, display_name) VALUES (?, ?)",
                (person.person_id, person.display_name),
            )
            for idx, tpl in enumerate(person.templates):
                self._conn.execute(
                    "INSERT INTO templates (person_id, template_idx, vector, source, "
                    "model_version, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        person.person_id,
                        idx,
                        tpl.encoding.vector.astype(np.float32).tobytes(),
                        tpl.source,
                        tpl.encoding.model_version,
                        tpl.created_at.isoformat(),
                    ),
                )

    def get(self, person_id: str) -> Person | None:
        row = self._conn.execute(
            "SELECT display_name FROM persons WHERE person_id = ?", (person_id,)
        ).fetchone()
        if row is None:
            return None
        display_name = row[0]
        tpl_rows = self._conn.execute(
            "SELECT vector, source, model_version, created_at FROM templates "
            "WHERE person_id = ? ORDER BY template_idx",
            (person_id,),
        ).fetchall()
        templates = tuple(
            Template(
                encoding=FaceEncoding(
                    vector=np.frombuffer(blob, dtype=np.float32),
                    model_version=mv,
                ),
                source=src,
                created_at=datetime.fromisoformat(ts),
            )
            for blob, src, mv, ts in tpl_rows
        )
        return Person(person_id=person_id, display_name=display_name, templates=templates)

    def remove(self, person_id: str) -> None:
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM persons WHERE person_id = ?", (person_id,)
            )
            if cur.rowcount == 0:
                raise PersonNotFoundError(person_id)

    def list_all(self) -> list[Person]:
        ids = [
            row[0]
            for row in self._conn.execute(
                "SELECT person_id FROM persons ORDER BY person_id"
            )
        ]
        return [p for p in (self.get(pid) for pid in ids) if p is not None]

    def all_templates_matrix(self) -> tuple[np.ndarray, list[str]]:
        rows = self._conn.execute(
            "SELECT person_id, vector FROM templates "
            "ORDER BY person_id, template_idx"
        ).fetchall()
        if not rows:
            return np.zeros((0, 512), dtype=np.float32), []
        ids = [r[0] for r in rows]
        matrix = np.stack(
            [np.frombuffer(r[1], dtype=np.float32) for r in rows]
        )
        return matrix, ids
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/test_sqlite_repository.py -v
```

预期：8 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/infrastructure/sqlite_repository.py tests/unit/test_sqlite_repository.py
git commit -m "feat(infrastructure): 加 SqliteRepository（向量 BLOB + 矩阵导出）"
```

---

### Task 7: 5 个模板生成策略

**Files:**
- Create:
  - `src/face_recognition/application/strategies/random_one.py`
  - `src/face_recognition/application/strategies/mean_all.py`
  - `src/face_recognition/application/strategies/manual_three.py`
  - `src/face_recognition/application/strategies/kmeans_k3.py`
  - `src/face_recognition/application/strategies/all_vectors.py`
- Test: `tests/unit/test_strategies.py`

- [ ] **Step 1: 写失败的测试 `tests/unit/test_strategies.py`**

```python
from collections.abc import Callable

import numpy as np
import pytest

from face_recognition.application.strategies.all_vectors import AllVectorsStrategy
from face_recognition.application.strategies.kmeans_k3 import KMeansK3Strategy
from face_recognition.application.strategies.manual_three import ManualThreeStrategy
from face_recognition.application.strategies.mean_all import MeanAllStrategy
from face_recognition.application.strategies.random_one import RandomOneStrategy
from face_recognition.domain.entities import FaceEncoding


def _all_unit_norm(templates):
    return all(
        abs(float(np.linalg.norm(t.encoding.vector)) - 1.0) < 1e-3 for t in templates
    )


@pytest.fixture
def encs(make_encoding: Callable[[int], FaceEncoding]) -> list[FaceEncoding]:
    return [make_encoding(i) for i in range(40)]


def test_random_one_returns_one_template(encs: list[FaceEncoding]):
    out = RandomOneStrategy(seed=42).build(encs)
    assert len(out) == 1
    assert _all_unit_norm(out)


def test_random_one_deterministic_with_seed(encs: list[FaceEncoding]):
    a = RandomOneStrategy(seed=42).build(encs)
    b = RandomOneStrategy(seed=42).build(encs)
    assert np.array_equal(a[0].encoding.vector, b[0].encoding.vector)


def test_mean_all_returns_one_normalized_centroid(encs: list[FaceEncoding]):
    out = MeanAllStrategy().build(encs)
    assert len(out) == 1
    assert _all_unit_norm(out)


def test_manual_three_takes_first_three(encs: list[FaceEncoding]):
    out = ManualThreeStrategy().build(encs)
    assert len(out) == 3
    assert _all_unit_norm(out)
    # 应取前三张
    for i, tpl in enumerate(out):
        assert np.allclose(tpl.encoding.vector, encs[i].vector)


def test_manual_three_with_fewer_takes_all(make_encoding):
    out = ManualThreeStrategy().build([make_encoding(0), make_encoding(1)])
    assert len(out) == 2


def test_kmeans_k3_returns_three_normalized_centroids(encs):
    out = KMeansK3Strategy(seed=42).build(encs)
    assert len(out) == 3
    assert _all_unit_norm(out)


def test_kmeans_k3_with_fewer_than_3_falls_back(make_encoding):
    out = KMeansK3Strategy(seed=42).build([make_encoding(0), make_encoding(1)])
    # 不足 3 张时降级为返回所有原始向量
    assert len(out) == 2


def test_all_vectors_returns_all_inputs(encs):
    out = AllVectorsStrategy().build(encs)
    assert len(out) == len(encs)


def test_empty_input_raises(encs):
    for strategy in (
        RandomOneStrategy(seed=0),
        MeanAllStrategy(),
        ManualThreeStrategy(),
        KMeansK3Strategy(seed=0),
        AllVectorsStrategy(),
    ):
        with pytest.raises(ValueError):
            strategy.build([])
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/test_strategies.py -v
```

- [ ] **Step 3: 实现 `random_one.py`**

```python
# src/face_recognition/application/strategies/random_one.py
import random
from datetime import datetime

from face_recognition.domain.entities import FaceEncoding, Template


class RandomOneStrategy:
    name = "random_one"

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def build(self, encodings: list[FaceEncoding]) -> list[Template]:
        if not encodings:
            raise ValueError("RandomOneStrategy 至少需要 1 个 encoding")
        chosen = self._rng.choice(encodings)
        return [
            Template(
                encoding=chosen,
                source="random_one",
                created_at=datetime.utcnow(),
            )
        ]
```

- [ ] **Step 4: 实现 `mean_all.py`**

```python
# src/face_recognition/application/strategies/mean_all.py
from datetime import datetime

import numpy as np

from face_recognition.domain.entities import FaceEncoding, Template


class MeanAllStrategy:
    name = "mean_all"

    def build(self, encodings: list[FaceEncoding]) -> list[Template]:
        if not encodings:
            raise ValueError("MeanAllStrategy 至少需要 1 个 encoding")
        stacked = np.stack([e.vector for e in encodings])
        centroid = stacked.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        model_version = encodings[0].model_version
        return [
            Template(
                encoding=FaceEncoding(
                    vector=centroid.astype(np.float32),
                    model_version=model_version,
                ),
                source="mean_all",
                created_at=datetime.utcnow(),
            )
        ]
```

- [ ] **Step 5: 实现 `manual_three.py`**

```python
# src/face_recognition/application/strategies/manual_three.py
from datetime import datetime

from face_recognition.domain.entities import FaceEncoding, Template


class ManualThreeStrategy:
    """简化版：取前 3 张。生产中可换成根据照片标签（正光/侧光/逆光）挑选。"""

    name = "manual_three"

    def build(self, encodings: list[FaceEncoding]) -> list[Template]:
        if not encodings:
            raise ValueError("ManualThreeStrategy 至少需要 1 个 encoding")
        chosen = encodings[:3]
        return [
            Template(encoding=e, source=f"manual_{i}", created_at=datetime.utcnow())
            for i, e in enumerate(chosen)
        ]
```

- [ ] **Step 6: 实现 `kmeans_k3.py`**

```python
# src/face_recognition/application/strategies/kmeans_k3.py
from datetime import datetime

import numpy as np
from sklearn.cluster import KMeans

from face_recognition.domain.entities import FaceEncoding, Template


class KMeansK3Strategy:
    name = "kmeans_k3"

    def __init__(self, k: int = 3, seed: int = 42) -> None:
        self._k = k
        self._seed = seed

    def build(self, encodings: list[FaceEncoding]) -> list[Template]:
        if not encodings:
            raise ValueError("KMeansK3Strategy 至少需要 1 个 encoding")
        if len(encodings) < self._k:
            return [
                Template(
                    encoding=e,
                    source=f"kmeans_fallback_{i}",
                    created_at=datetime.utcnow(),
                )
                for i, e in enumerate(encodings)
            ]
        stacked = np.stack([e.vector for e in encodings])
        km = KMeans(n_clusters=self._k, random_state=self._seed, n_init=10).fit(stacked)
        model_version = encodings[0].model_version
        templates = []
        for i, c in enumerate(km.cluster_centers_):
            normed = c / np.linalg.norm(c)
            templates.append(
                Template(
                    encoding=FaceEncoding(
                        vector=normed.astype(np.float32),
                        model_version=model_version,
                    ),
                    source=f"kmeans_centroid_{i}",
                    created_at=datetime.utcnow(),
                )
            )
        return templates
```

- [ ] **Step 7: 实现 `all_vectors.py`**

```python
# src/face_recognition/application/strategies/all_vectors.py
from datetime import datetime

from face_recognition.domain.entities import FaceEncoding, Template


class AllVectorsStrategy:
    name = "all_vectors"

    def build(self, encodings: list[FaceEncoding]) -> list[Template]:
        if not encodings:
            raise ValueError("AllVectorsStrategy 至少需要 1 个 encoding")
        return [
            Template(encoding=e, source=f"all_{i}", created_at=datetime.utcnow())
            for i, e in enumerate(encodings)
        ]
```

- [ ] **Step 8: 跑测试确认通过**

```bash
uv run pytest tests/unit/test_strategies.py -v
```

预期：9 passed

- [ ] **Step 9: commit**

```bash
git add src/face_recognition/application/strategies/ tests/unit/test_strategies.py
git commit -m "feat(strategies): 加 5 个模板生成策略（random/mean/manual/kmeans/all）"
```

---

### Task 8: 注册用例（RegisterFace）

**Files:**
- Create: `src/face_recognition/application/register_face.py`
- Test: `tests/unit/test_register_face.py`

`RegisterFace` 接收一个人的所有照片路径 → 调 `FacePipeline.encode_single` → 调 `TemplateStrategy.build` → 写入 `PersonRepository`。无脸照片记 warning 跳过；该人全部失败抛 `PersonHasNoTemplatesError`。

- [ ] **Step 1: 写失败的测试 `tests/unit/test_register_face.py`**

```python
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from face_recognition.application.register_face import RegisterFace
from face_recognition.application.strategies.kmeans_k3 import KMeansK3Strategy
from face_recognition.domain.entities import FaceEncoding, Person, Template
from face_recognition.domain.errors import (
    NoFaceError,
    PersonHasNoTemplatesError,
)


@pytest.fixture
def stub_pipeline(make_encoding: Callable[[int], FaceEncoding]) -> MagicMock:
    """encode_single 按调用顺序返回 seed=0,1,2,... 的合成 encoding。"""
    counter = {"i": 0}

    def _encode_single(_image: np.ndarray) -> FaceEncoding:
        i = counter["i"]
        counter["i"] += 1
        return make_encoding(i)

    pipeline = MagicMock()
    pipeline.encode_single.side_effect = _encode_single
    return pipeline


@pytest.fixture
def stub_repo() -> MagicMock:
    repo = MagicMock()
    return repo


@pytest.fixture
def fake_image_loader() -> Callable[[Path], np.ndarray]:
    return lambda path: np.zeros((112, 112, 3), dtype=np.uint8)


def _make_person_dir(tmp_path: Path, person_id: str, n_imgs: int) -> Path:
    d = tmp_path / person_id
    d.mkdir()
    for i in range(n_imgs):
        (d / f"{i:03d}.jpg").write_bytes(b"fake")
    return d


def test_register_one_person_with_kmeans_k3(
    tmp_path: Path,
    stub_pipeline: MagicMock,
    stub_repo: MagicMock,
    fake_image_loader,
):
    person_dir = _make_person_dir(tmp_path, "alice", 10)
    use_case = RegisterFace(
        pipeline=stub_pipeline,
        repository=stub_repo,
        strategy=KMeansK3Strategy(seed=42),
        image_loader=fake_image_loader,
    )
    use_case.execute_for_person(person_dir)

    stub_repo.add.assert_called_once()
    person: Person = stub_repo.add.call_args.args[0]
    assert person.person_id == "alice"
    assert len(person.templates) == 3  # KMeans K=3 出 3 模板


def test_skips_images_without_face(
    tmp_path: Path,
    make_encoding,
    stub_repo: MagicMock,
    fake_image_loader,
):
    person_dir = _make_person_dir(tmp_path, "alice", 5)
    pipeline = MagicMock()
    # 第 0,2 张 NoFaceError，剩 3 张正常
    seq = [NoFaceError("无脸"), make_encoding(0), NoFaceError("无脸"), make_encoding(1), make_encoding(2)]
    pipeline.encode_single.side_effect = seq

    use_case = RegisterFace(
        pipeline=pipeline,
        repository=stub_repo,
        strategy=KMeansK3Strategy(seed=42),
        image_loader=fake_image_loader,
    )
    use_case.execute_for_person(person_dir)

    person: Person = stub_repo.add.call_args.args[0]
    assert len(person.templates) == 3  # 3 张成功 → KMeans 3 簇


def test_all_images_fail_raises(
    tmp_path: Path,
    stub_repo: MagicMock,
    fake_image_loader,
):
    person_dir = _make_person_dir(tmp_path, "alice", 3)
    pipeline = MagicMock()
    pipeline.encode_single.side_effect = NoFaceError("无脸")

    use_case = RegisterFace(
        pipeline=pipeline,
        repository=stub_repo,
        strategy=KMeansK3Strategy(seed=42),
        image_loader=fake_image_loader,
    )
    with pytest.raises(PersonHasNoTemplatesError):
        use_case.execute_for_person(person_dir)
    stub_repo.add.assert_not_called()


def test_execute_dir_skips_failed_people(
    tmp_path: Path,
    stub_repo: MagicMock,
    fake_image_loader,
    make_encoding,
):
    _make_person_dir(tmp_path, "alice", 3)
    _make_person_dir(tmp_path, "bob", 3)
    pipeline = MagicMock()
    # alice 3 张全失败；bob 3 张全成功
    pipeline.encode_single.side_effect = [
        NoFaceError("无脸"), NoFaceError("无脸"), NoFaceError("无脸"),
        make_encoding(0), make_encoding(1), make_encoding(2),
    ]

    use_case = RegisterFace(
        pipeline=pipeline,
        repository=stub_repo,
        strategy=KMeansK3Strategy(seed=42),
        image_loader=fake_image_loader,
    )
    summary = use_case.execute(tmp_path)
    assert summary.persons_succeeded == 1
    assert summary.persons_failed == 1
    assert stub_repo.add.call_count == 1
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/test_register_face.py -v
```

- [ ] **Step 3: 实现 `src/face_recognition/application/register_face.py`**

```python
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from face_recognition.domain.entities import FaceEncoding, Person
from face_recognition.domain.errors import (
    FaceRecognitionError,
    PersonHasNoTemplatesError,
)
from face_recognition.domain.interfaces import (
    FacePipeline,
    PersonRepository,
    TemplateStrategy,
)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisterSummary:
    persons_succeeded: int
    persons_failed: int
    images_processed: int
    images_skipped: int


class RegisterFace:
    def __init__(
        self,
        pipeline: FacePipeline,
        repository: PersonRepository,
        strategy: TemplateStrategy,
        image_loader: Callable[[Path], np.ndarray],
    ) -> None:
        self._pipeline = pipeline
        self._repo = repository
        self._strategy = strategy
        self._load_image = image_loader

    def execute_for_person(self, person_dir: Path) -> int:
        """注册单个人。返回成功提取的图片数。失败抛 PersonHasNoTemplatesError。"""
        encodings: list[FaceEncoding] = []
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue
            try:
                img = self._load_image(img_path)
                enc = self._pipeline.encode_single(img)
                encodings.append(enc)
            except FaceRecognitionError as e:
                logger.warning("跳过 %s: %s", img_path, e)

        if not encodings:
            raise PersonHasNoTemplatesError(
                f"{person_dir.name}: 全部照片无法提取人脸"
            )

        templates = self._strategy.build(encodings)
        person = Person(
            person_id=person_dir.name,
            display_name=person_dir.name,
            templates=tuple(templates),
        )
        self._repo.add(person)
        return len(encodings)

    def execute(self, dataset_dir: Path) -> RegisterSummary:
        """批量注册整个数据集。"""
        succeeded = failed = images_processed = images_skipped = 0
        for person_dir in sorted(dataset_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            try:
                n = self.execute_for_person(person_dir)
                succeeded += 1
                images_processed += n
            except PersonHasNoTemplatesError as e:
                logger.error("跳过此人: %s", e)
                failed += 1
        return RegisterSummary(
            persons_succeeded=succeeded,
            persons_failed=failed,
            images_processed=images_processed,
            images_skipped=images_skipped,
        )
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/test_register_face.py -v
```

预期：4 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/application/register_face.py tests/unit/test_register_face.py
git commit -m "feat(application): 加 RegisterFace 用例（容忍单张失败、整人失败抛错）"
```

---

### Task 9: 识别用例（RecognizeFace）

**Files:**
- Create: `src/face_recognition/application/recognize_face.py`
- Test: `tests/unit/test_recognize_face.py`

`RecognizeFace` 输入一张图 → `FacePipeline.encode_single` → 用 `repository.all_templates_matrix()` 一次矩阵乘法找最相似 → 过阈值返回，否则 `person_id=None`。

- [ ] **Step 1: 写失败的测试 `tests/unit/test_recognize_face.py`**

```python
from unittest.mock import MagicMock

import numpy as np
import pytest

from face_recognition.application.recognize_face import RecognizeFace
from face_recognition.domain.entities import FaceEncoding


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_recognize_returns_best_match_above_threshold():
    pipeline = MagicMock()
    repo = MagicMock()

    query_vec = _unit_vec(0)
    pipeline.encode_single.return_value = FaceEncoding(query_vec, "test")

    # 3 个模板：第 1 个就是 query 自己（相似度=1），其他差远
    other = _unit_vec(99)
    matrix = np.stack([_unit_vec(11), query_vec, other])
    repo.all_templates_matrix.return_value = (matrix, ["bob", "alice", "carol"])

    use_case = RecognizeFace(pipeline=pipeline, repository=repo, threshold=0.5)
    result = use_case.execute(np.zeros((112, 112, 3), dtype=np.uint8))

    assert result.person_id == "alice"
    assert result.similarity == pytest.approx(1.0, abs=1e-5)
    assert result.threshold == 0.5


def test_recognize_returns_none_when_below_threshold():
    pipeline = MagicMock()
    repo = MagicMock()
    pipeline.encode_single.return_value = FaceEncoding(_unit_vec(0), "test")
    matrix = np.stack([_unit_vec(50), _unit_vec(60)])
    repo.all_templates_matrix.return_value = (matrix, ["bob", "carol"])

    use_case = RecognizeFace(pipeline=pipeline, repository=repo, threshold=0.99)
    result = use_case.execute(np.zeros((112, 112, 3), dtype=np.uint8))

    assert result.person_id is None
    assert result.similarity < 0.99


def test_recognize_with_empty_repo_returns_none():
    pipeline = MagicMock()
    repo = MagicMock()
    pipeline.encode_single.return_value = FaceEncoding(_unit_vec(0), "test")
    repo.all_templates_matrix.return_value = (np.zeros((0, 512), dtype=np.float32), [])

    use_case = RecognizeFace(pipeline=pipeline, repository=repo, threshold=0.5)
    result = use_case.execute(np.zeros((112, 112, 3), dtype=np.uint8))
    assert result.person_id is None
    assert result.similarity == 0.0
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/test_recognize_face.py -v
```

- [ ] **Step 3: 实现 `src/face_recognition/application/recognize_face.py`**

```python
import numpy as np

from face_recognition.domain.entities import RecognitionResult
from face_recognition.domain.interfaces import FacePipeline, PersonRepository


class RecognizeFace:
    def __init__(
        self,
        pipeline: FacePipeline,
        repository: PersonRepository,
        threshold: float,
    ) -> None:
        self._pipeline = pipeline
        self._repo = repository
        self._threshold = threshold

    def execute(self, image: np.ndarray) -> RecognitionResult:
        enc = self._pipeline.encode_single(image)
        matrix, person_ids = self._repo.all_templates_matrix()
        if matrix.shape[0] == 0:
            return RecognitionResult(
                person_id=None, similarity=0.0, threshold=self._threshold
            )
        sims = matrix @ enc.vector
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= self._threshold:
            return RecognitionResult(
                person_id=person_ids[best_idx],
                similarity=best_sim,
                threshold=self._threshold,
            )
        return RecognitionResult(
            person_id=None, similarity=best_sim, threshold=self._threshold
        )
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/test_recognize_face.py -v
```

预期：3 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/application/recognize_face.py tests/unit/test_recognize_face.py
git commit -m "feat(application): 加 RecognizeFace 用例（一次矩阵乘法 + 阈值判别）"
```

---

### Task 10: InsightFace pipeline 实现

**Files:**
- Create: `src/face_recognition/infrastructure/insightface_pipeline.py`
- Test: `tests/integration/test_insightface_pipeline.py`（标记 `@pytest.mark.gpu`）

这是唯一调用 `insightface` 库的地方。集成测试用 `insightface.app.FaceAnalysis()` 自带的示例图。**测试默认跳过**（需要下载模型 + 较慢），用 `pytest -m gpu` 显式跑。

- [ ] **Step 1: 写集成测试 `tests/integration/test_insightface_pipeline.py`**

```python
import numpy as np
import pytest

from face_recognition.domain.errors import NoFaceError
from face_recognition.infrastructure.insightface_pipeline import InsightFacePipeline


pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def pipeline() -> InsightFacePipeline:
    return InsightFacePipeline(model_pack="buffalo_l", ctx_id=-1, det_size=(640, 640))


def test_encode_single_on_real_face(pipeline: InsightFacePipeline):
    # 用 InsightFace 自带的示例图（包目录下 sample-images/）
    import insightface
    from pathlib import Path
    sample = Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"
    if not sample.exists():
        pytest.skip(f"InsightFace 自带示例图未找到: {sample}")
    import cv2
    img = cv2.imread(str(sample))
    enc = pipeline.encode_single(img)
    assert enc.vector.shape == (512,)
    assert abs(float(np.linalg.norm(enc.vector)) - 1.0) < 1e-3


def test_encode_no_face_raises(pipeline: InsightFacePipeline):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(NoFaceError):
        pipeline.encode_single(blank)
```

- [ ] **Step 2: 实现 `src/face_recognition/infrastructure/insightface_pipeline.py`**

```python
import numpy as np
from insightface.app import FaceAnalysis

from face_recognition.domain.entities import FaceEncoding
from face_recognition.domain.errors import MultipleFacesError, NoFaceError


class InsightFacePipeline:
    def __init__(
        self,
        model_pack: str = "buffalo_l",
        ctx_id: int = 0,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        self._model_pack = model_pack
        self._app = FaceAnalysis(name=model_pack)
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

    def encode(self, image: np.ndarray) -> list[FaceEncoding]:
        faces = self._app.get(image)
        return [
            FaceEncoding(
                vector=self._normalize(f.embedding.astype(np.float32)),
                model_version=self._model_pack,
            )
            for f in faces
        ]

    def encode_single(self, image: np.ndarray) -> FaceEncoding:
        faces = self._app.get(image)
        if not faces:
            raise NoFaceError("图中未检出人脸")
        if len(faces) > 1:
            raise MultipleFacesError(count=len(faces))
        emb = faces[0].embedding.astype(np.float32)
        return FaceEncoding(
            vector=self._normalize(emb), model_version=self._model_pack
        )

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < 1e-12:
            raise ValueError("InsightFace 输出零向量，模型异常")
        return v / n
```

- [ ] **Step 3: 跑集成测试（首次会下载 buffalo_l 模型，约 300MB）**

```bash
uv run pytest tests/integration/test_insightface_pipeline.py -v -m gpu
```

预期：2 passed（首次跑约 1~3 分钟，含模型下载）

- [ ] **Step 4: 默认套件不跑这些（快）**

```bash
uv run pytest -v -m "not gpu"
```

预期：之前所有单测 pass，集成被跳过

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/infrastructure/insightface_pipeline.py tests/integration/test_insightface_pipeline.py
git commit -m "feat(infrastructure): 加 InsightFacePipeline（buffalo_l 一站式封装）"
```

---

### Task 11: 依赖装配（dependencies.py）

**Files:**
- Create: `src/face_recognition/api/dependencies.py`

无单元测试——依赖装配本质就是工厂函数串联，集成测试 (Task 13) 会自动验证它能跑起来。

- [ ] **Step 1: 实现 `src/face_recognition/api/dependencies.py`**

```python
from pathlib import Path

import cv2
import numpy as np

from face_recognition.application.recognize_face import RecognizeFace
from face_recognition.application.register_face import RegisterFace
from face_recognition.application.strategies.all_vectors import AllVectorsStrategy
from face_recognition.application.strategies.kmeans_k3 import KMeansK3Strategy
from face_recognition.application.strategies.manual_three import ManualThreeStrategy
from face_recognition.application.strategies.mean_all import MeanAllStrategy
from face_recognition.application.strategies.random_one import RandomOneStrategy
from face_recognition.domain.interfaces import (
    FacePipeline,
    PersonRepository,
    TemplateStrategy,
)
from face_recognition.infrastructure.config_loader import AppConfig, load_config
from face_recognition.infrastructure.insightface_pipeline import InsightFacePipeline
from face_recognition.infrastructure.sqlite_repository import SqliteRepository


def build_config(path: Path | str = "config.yaml") -> AppConfig:
    return load_config(path)


def build_pipeline(cfg: AppConfig) -> FacePipeline:
    return InsightFacePipeline(
        model_pack=cfg.model.pack,
        ctx_id=cfg.model.ctx_id,
        det_size=cfg.model.det_size,
    )


def build_repository(cfg: AppConfig) -> PersonRepository:
    return SqliteRepository(cfg.data.sqlite_path)


def build_strategy(name: str, seed: int) -> TemplateStrategy:
    table: dict[str, TemplateStrategy] = {
        "random_one": RandomOneStrategy(seed=seed),
        "mean_all": MeanAllStrategy(),
        "manual_three": ManualThreeStrategy(),
        "kmeans_k3": KMeansK3Strategy(seed=seed),
        "all_vectors": AllVectorsStrategy(),
    }
    if name not in table:
        raise ValueError(f"未知策略: {name}")
    return table[name]


def _load_bgr_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"无法读图: {path}")
    return img


def build_register_use_case(cfg: AppConfig, strategy_name: str) -> RegisterFace:
    return RegisterFace(
        pipeline=build_pipeline(cfg),
        repository=build_repository(cfg),
        strategy=build_strategy(strategy_name, seed=cfg.evaluation.random_seed),
        image_loader=_load_bgr_image,
    )


def build_recognize_use_case(cfg: AppConfig) -> RecognizeFace:
    return RecognizeFace(
        pipeline=build_pipeline(cfg),
        repository=build_repository(cfg),
        threshold=cfg.recognition.threshold,
    )
```

- [ ] **Step 2: 跑 mypy 校验类型**

```bash
uv run mypy src/face_recognition/api/dependencies.py
```

预期：Success: no issues found（如有错先修）

- [ ] **Step 3: commit**

```bash
git add src/face_recognition/api/dependencies.py
git commit -m "feat(api): 加 dependencies 装配工厂（CLI 与 server 共用）"
```

---

### Task 12: Typer CLI

**Files:**
- Create: `src/face_recognition/api/cli.py`

CLI 暴露 4 个命令：

- `register --strategy <name> --dataset <dir>`：批量注册
- `recognize <image_path>`：识别单张图
- `list`：列出库内人员
- `remove <person_id>`：删除人员

无单元测试——CLI 是薄壳，每条命令都是一行调用用例。集成测试覆盖。

- [ ] **Step 1: 实现 `src/face_recognition/api/cli.py`**

```python
import logging
from pathlib import Path

import cv2
import typer

from face_recognition.api.dependencies import (
    build_config,
    build_recognize_use_case,
    build_register_use_case,
    build_repository,
)
from face_recognition.domain.errors import FaceRecognitionError

app = typer.Typer(
    name="face-recognition",
    no_args_is_help=True,
    help="基于 ArcFace 的开放集人脸识别系统 CLI",
)


@app.callback()
def _setup(
    config: Path = typer.Option("config.yaml", "--config", "-c", help="配置文件路径"),
    ctx: typer.Context = None,  # type: ignore[assignment]
) -> None:
    cfg = build_config(config)
    logging.basicConfig(
        level=cfg.logging.level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ctx.obj = cfg


@app.command()
def register(
    ctx: typer.Context,
    dataset: Path = typer.Option(..., "--dataset", help="数据集根目录（按文件夹分人）"),
    strategy: str = typer.Option(
        ...,
        "--strategy",
        help="模板策略：random_one / mean_all / manual_three / kmeans_k3 / all_vectors",
    ),
) -> None:
    """批量注册整个数据集。"""
    cfg = ctx.obj
    use_case = build_register_use_case(cfg, strategy)
    summary = use_case.execute(dataset)
    typer.echo(
        f"完成: 成功 {summary.persons_succeeded} 人 / 失败 {summary.persons_failed} 人; "
        f"处理图片 {summary.images_processed} 张"
    )


@app.command()
def recognize(ctx: typer.Context, image: Path) -> None:
    """识别单张图片中的人脸。"""
    cfg = ctx.obj
    use_case = build_recognize_use_case(cfg)
    img = cv2.imread(str(image))
    if img is None:
        typer.echo(f"无法读取图片: {image}", err=True)
        raise typer.Exit(code=1)
    try:
        result = use_case.execute(img)
    except FaceRecognitionError as e:
        typer.echo(f"识别失败 [{e.code}]: {e}", err=True)
        raise typer.Exit(code=2)
    if result.person_id is None:
        typer.echo(f"未知人员（最高相似度 {result.similarity:.4f} < 阈值 {result.threshold}）")
    else:
        typer.echo(f"识别为: {result.person_id}（相似度 {result.similarity:.4f}）")


@app.command(name="list")
def list_persons(ctx: typer.Context) -> None:
    """列出库内所有人员。"""
    cfg = ctx.obj
    repo = build_repository(cfg)
    persons = repo.list_all()
    if not persons:
        typer.echo("（库为空）")
        return
    typer.echo(f"共 {len(persons)} 人：")
    for p in persons:
        typer.echo(f"  {p.person_id}  ({len(p.templates)} 模板)  - {p.display_name}")


@app.command()
def remove(ctx: typer.Context, person_id: str) -> None:
    """从库中删除人员。"""
    cfg = ctx.obj
    repo = build_repository(cfg)
    try:
        repo.remove(person_id)
        typer.echo(f"已删除: {person_id}")
    except FaceRecognitionError as e:
        typer.echo(f"删除失败 [{e.code}]: {e}", err=True)
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: 验证 CLI 自带 help 工作**

```bash
uv run python -m face_recognition.api.cli --help
```

预期：看到四条子命令的 help（register / recognize / list / remove）

- [ ] **Step 3: commit**

```bash
git add src/face_recognition/api/cli.py
git commit -m "feat(api): 加 Typer CLI（register/recognize/list/remove）"
```

---

### Task 13: 端到端集成测试

**Files:**
- Create: `tests/integration/test_register_recognize_e2e.py`

**最关键的一步**：用真实 InsightFace 模型 + 真实 SQLite，模拟一个完整的"注册 → 识别"业务流程。这一步通过，意味着 M1 实质完成。

- [ ] **Step 1: 写集成测试**

```python
from pathlib import Path

import cv2
import insightface
import numpy as np
import pytest

from face_recognition.application.recognize_face import RecognizeFace
from face_recognition.application.register_face import RegisterFace
from face_recognition.application.strategies.kmeans_k3 import KMeansK3Strategy
from face_recognition.infrastructure.insightface_pipeline import InsightFacePipeline
from face_recognition.infrastructure.sqlite_repository import SqliteRepository


pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def pipeline() -> InsightFacePipeline:
    return InsightFacePipeline(model_pack="buffalo_l", ctx_id=-1, det_size=(640, 640))


def _sample_image_path() -> Path:
    p = Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"
    if not p.exists():
        pytest.skip(f"InsightFace 自带示例图未找到: {p}")
    return p


def _populate_person_dir(target_dir: Path, sample: Path, n: int) -> None:
    target_dir.mkdir()
    src = cv2.imread(str(sample))
    for i in range(n):
        # 微小扰动避免 KMeans 因完全相同向量而退化
        noise = (np.random.RandomState(i).rand(*src.shape) * 5).astype(np.uint8)
        cv2.imwrite(str(target_dir / f"{i:03d}.jpg"), cv2.add(src, noise))


def test_register_and_recognize_same_person(
    tmp_path: Path,
    pipeline: InsightFacePipeline,
):
    sample = _sample_image_path()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _populate_person_dir(dataset_dir / "alice", sample, n=10)

    repo = SqliteRepository(tmp_path / "test.db")
    register = RegisterFace(
        pipeline=pipeline,
        repository=repo,
        strategy=KMeansK3Strategy(seed=42),
        image_loader=lambda p: cv2.imread(str(p)),
    )
    register.execute(dataset_dir)
    assert len(repo.list_all()) == 1

    # 用同一张原图查询
    query_img = cv2.imread(str(sample))
    recognize = RecognizeFace(pipeline=pipeline, repository=repo, threshold=0.5)
    result = recognize.execute(query_img)
    assert result.person_id == "alice"
    assert result.similarity > 0.5


def test_recognize_unknown_returns_none(
    tmp_path: Path,
    pipeline: InsightFacePipeline,
):
    """库为空时识别返回未知。"""
    sample = _sample_image_path()
    repo = SqliteRepository(tmp_path / "test.db")
    recognize = RecognizeFace(pipeline=pipeline, repository=repo, threshold=0.5)
    result = recognize.execute(cv2.imread(str(sample)))
    assert result.person_id is None
```

- [ ] **Step 2: 跑集成测试**

```bash
uv run pytest tests/integration/test_register_recognize_e2e.py -v -m gpu
```

预期：2 passed（约 30 秒，复用之前 task 10 已下载的模型）

- [ ] **Step 3: 跑全部测试做一次 sanity check**

```bash
uv run pytest -v
uv run pytest -v -m gpu
```

预期：单元测试全部 pass；集成测试全部 pass

- [ ] **Step 4: 跑 ruff & mypy 整体检查**

```bash
uv run ruff check src tests
uv run mypy src/face_recognition
```

预期：无 lint 错误，无类型错误（如有需要修则修）

- [ ] **Step 5: commit + 打 M1 tag**

```bash
git add tests/integration/test_register_recognize_e2e.py
git commit -m "test(integration): 加 register→recognize 端到端测试"
git tag -a m1-complete -m "M1: 核心 CLI 完成（注册/识别/SQLite/5 策略）"
```

- [ ] **Step 6: 手动验证 CLI 真的能用**

把一两个真实测试照片放到 `data/private_dataset/yourself/`，跑：

```bash
uv run python -m face_recognition.api.cli register \
    --dataset data/private_dataset \
    --strategy kmeans_k3
uv run python -m face_recognition.api.cli list
uv run python -m face_recognition.api.cli recognize data/private_dataset/yourself/000.jpg
uv run python -m face_recognition.api.cli remove yourself
```

预期：每条命令都打印合理结果。如果有 bug，加新测试覆盖再修。

---

## 完成标准

执行完 13 个任务，M1 算完整交付，必须满足：

- [ ] `uv run pytest -m "not gpu"` 全部 pass（单元测试）
- [ ] `uv run pytest -m gpu` 全部 pass（集成测试，含端到端）
- [ ] `uv run ruff check src tests` 0 错误
- [ ] `uv run mypy src/face_recognition` 0 错误
- [ ] CLI 四条命令（register / recognize / list / remove）手动跑通
- [ ] git 历史里至少 13 个 commit + 1 个 `m1-complete` tag
- [ ] `git push origin main --tags` 推送到 GitHub

完成后即可进入 **M2 评估框架** 的实施计划。
