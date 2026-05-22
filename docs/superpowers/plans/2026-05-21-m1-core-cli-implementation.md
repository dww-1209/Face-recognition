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
from datetime import datetime, timezone

# 从我们要实现的模块里导入 4 个实体类
# 注意：这一行会失败（因为 entities.py 还是空的）——这正是 TDD"先写失败测试"的重点。
#
# ── 给小白解释这条 import 路径 ──
# 为什么是 `face_recognition.domain.entities` 而不是 `src.face_recognition.domain.entities`
# 也不是相对路径 `from .entities import ...`？
#   1) 项目用的是 **src layout**：源码全部放在 `src/face_recognition/` 下，根目录只放
#      pyproject.toml / tests / docs。pyproject.toml 里 `[tool.hatch.build.targets.wheel]
#      packages = ["src/face_recognition"]` 告诉打包工具"src/ 下面的 face_recognition
#      是真正的包名"——`src/` 这一层在 import 路径里**不出现**。
#   2) `uv sync` 会以 editable 模式把项目装进虚拟环境，所以 `face_recognition` 就和
#      安装的 numpy 一样可以用绝对路径 import；测试不在包内，用相对 import 反而出错。
#   3) src layout 的好处：在项目根跑 `python -c "import face_recognition"` 一定走的是
#      安装版而非误走当前目录里的同名 py 文件——避免"测试通过但发布到别的环境就崩"。
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
from datetime import datetime, timezone

import numpy as np

# ===== 模块级常量 =====
# 下划线开头是 Python 约定的"私有"标志，不会被 from xxx import * 导出
_EMBED_DIM = 512                # ArcFace ResNet100 输出向量维度，buffalo_l 模型固定为 512
_NORM_TOLERANCE = 1e-3          # L2 范数容忍度：理论 ||v||=1，浮点实际可能是 0.9999~1.0001 都接受
# ── 给小白解释这个数 1e-3 ──
#   Q1: 为什么是 1e-3 (= 0.001)？
#       float32 的相对精度大约 1e-7；一次 L2 归一化 + SQLite BLOB 序列化往返 + 余弦点积，
#       误差累积大约在 1e-4 量级。1e-3 比累积误差大一个数量级——给足"安全边距"，
#       不会因正常浮点抖动而误判向量"没归一化"。
#   Q2: 为什么不用 1e-6 或 1e-9？
#       太严了。1e-6 比累积误差还小，会出现"理论上该过、实际偶尔抛 InvalidEncodingError"
#       的玄学失败，调试起来非常痛苦。
#   Q3: 为什么不用 1e-1？
#       太松。L2 范数 0.9 还能过容忍——但 0.9 不是单位球面，余弦点积就不再等价于余弦
#       相似度，下游所有阈值都失效。1e-3 是"够松到不抖、够严到不掩盖真 bug"的折中。


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


class EncodingError(FaceRecognitionError):
    """编码异常:模型输出零向量、L2 归一化分母为 0 等"模型本身坏了"的情况。

    与 NoFaceError(图里就没脸)区分:NoFaceError 是输入数据问题,EncodingError 是模型问题。
    """
    code = "ENCODING_ERROR"
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
# typing.Protocol = "结构化子类型"协议，Python 3.8+ 引入。
#   - 传统继承：class Dog(Animal)，必须显式声明继承关系才算 Animal
#   - Protocol：只要一个类**长得像** Protocol（同名方法、同名签名），就算实现了它
#   - 类比 Go 的 interface、Rust 的 trait（隐式实现）
#   - 好处：domain 层不需要 import infrastructure（避免反向依赖），
#          只要 SqliteRepository 定义了 add/get/... 方法签名匹配，类型检查就过
#   - 对比 abc.ABC：ABC 强制继承，会让 domain 反向依赖到具体实现位置
#
# ── 给小白：3 行最小对比例子 ──
#   # 传统 ABC（必须显式继承才算实现）：
#   class Animal(ABC):
#       @abstractmethod
#       def speak(self) -> str: ...
#   class Dog(Animal):           # ← 必须写 (Animal)
#       def speak(self) -> str: return "汪"
#
#   # Protocol（只要方法名 + 签名一致就算）：
#   class Speaker(Protocol):
#       def speak(self) -> str: ...
#   class Dog:                    # ← 不写继承
#       def speak(self) -> str: return "汪"
#   def call(s: Speaker) -> str: return s.speak()
#   call(Dog())                   # ← 类型检查通过：Dog 长得像 Speaker
#
# 关键差异：用 Protocol 时 Dog 自己**完全不知道** Speaker 存在；用 ABC 时 Dog
# 必须 import Animal 才能继承。本项目 SqliteRepository 不需要 import domain 的
# Protocol——它只要把方法名/签名实现对就行——这正是清洁架构"依赖反转"的体现。
from typing import Protocol

import numpy as np

# 从同包 entities 导入领域实体。注意这里的 import 路径用绝对导入
# （from face_recognition.domain... 而不是 from .entities），
# 是因为我们用 src layout 装成包，绝对导入更明确、IDE 跳转更稳。
from face_recognition.domain.entities import (
    FaceEncoding,
    Person,
    Template,
)


class FacePipeline(Protocol):
    # 模型版本字符串(如 "buffalo_l")。下游评估管线把它作为 FaceEncoding.model_version 写入,
    # 避免在评估代码里硬编码字符串——换模型时只在 InsightFacePipeline 构造处改一处。
    # 用属性而非方法:配置一次,运行期不变;Protocol 里写成裸属性即可。
    model_version: str

    # 方法体只写 ... （三个点，叫 Ellipsis 字面量），表示"什么都不做的占位"。
    #   - 在 Protocol 里，方法体不会被执行，只用来声明签名
    #   - 也可以写 pass，但 ... 是 Protocol/Stub 的社区惯例
    # 这个方法约定："输入一张 BGR 图（OpenCV 读出来的 ndarray），返回所有人脸的 FaceEncoding 列表"
    def encode(self, image: np.ndarray) -> list[FaceEncoding]: ...

    # encode_single：注册流程用，要求图里**正好 1 张脸**，多于或少于都报错。
    # 拆成两个方法：encode 给"识别 / 多人"用，encode_single 给"注册 / 单人"用，
    # 调用方语义清晰，不用每次手动检查 len(faces)。
    def encode_single(self, image: np.ndarray) -> FaceEncoding: ...


class PersonRepository(Protocol):
    # 把"人脸库"抽象成增删改查 + 一个矩阵导出方法。
    # 谁来实现？SqliteRepository（Task 6）。
    # application/ 层只依赖这个 Protocol，永远不知道底层是 SQLite 还是别的。

    def add(self, person: Person) -> None: ...

    # 返回类型 `Person | None` 是 PEP 604 联合类型语法（Python 3.10+）。
    #   - 旧写法：Optional[Person]，需要 from typing import Optional
    #   - 新写法：直接 |，更像 TypeScript / Rust
    # None 语义："没找到这个人"，由调用方决定怎么处理（不抛异常，因为"找不到"是预期路径）。
    def get(self, person_id: str) -> Person | None: ...

    # 区别：remove 找不到要**抛异常**（PersonNotFoundError），不返回 None。
    # 为什么？删除是**有副作用**的命令，调用方期待"删了"或"明确失败"，
    # 静默 no-op 容易掩盖 bug（比如拼错 person_id）。
    def remove(self, person_id: str) -> None: ...

    def list_all(self) -> list[Person]: ...

    def all_templates_matrix(self) -> tuple[np.ndarray, list[str]]:
        """返回 (M, 512) 模板矩阵 + 长度 M 的 person_id 列表（每行对应一个模板）

        识别时关键性能优化：把所有模板拼成一个 (M, 512) 大矩阵，
        和查询向量做一次矩阵乘法 (M, 512) @ (512,) → (M,)，得到 M 个相似度。
        比"逐人遍历 + 逐模板算余弦"快 10x 以上（NumPy 底层用 BLAS）。
        """
        ...


class TemplateStrategy(Protocol):
    # Protocol 也可以声明**类属性**（不是方法）。这里要求实现类必须有 `name: str` 属性。
    # 用途：CLI 拿 strategy.name 打日志、记录到数据库 source 字段。
    name: str

    # build：核心方法。给一堆 FaceEncoding（一个人的所有照片），
    # 返回若干 Template（库里实际存的模板向量）。
    # - random_one  → 1 个 Template（随机选 1 张）
    # - mean_all    → 1 个 Template（全部平均）
    # - manual_three → 3 个
    # - kmeans_k3   → 3 个（聚类质心）
    # - all_vectors → N 个（全保留）
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
# conftest.py = pytest 的"魔法文件"。
#   - pytest 自动发现并加载，不需要 import
#   - 同目录及子目录的所有测试都能直接用里面定义的 fixture
#   - 放在 tests/ 根目录 → 整个测试套件共享
# 命名是 pytest 硬约定，**只能叫这个名字**。

# collections.abc 里的 Callable 是"可调用类型"的标准类型注解。
#   - Callable[[ArgType], ReturnType] 表示"接收 ArgType、返回 ReturnType 的函数"
#   - Python 3.9 之前要从 typing 导，3.9+ 推荐 collections.abc（与运行时同源）
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from face_recognition.domain.entities import FaceEncoding, Template


# @pytest.fixture 装饰器：把一个函数注册成"测试可注入的依赖"。
#   - 测试函数声明形参 make_encoding，pytest 自动调用这个 fixture，把返回值塞进去
#   - 类似依赖注入容器：测试不用自己 new，写在参数列表里就有
@pytest.fixture
def make_encoding() -> Callable[[int], FaceEncoding]:
    """生成确定性、L2 归一化的 FaceEncoding。同一 seed → 同一向量。

    注意：这个 fixture 返回的不是 FaceEncoding 本身，而是一个**生成函数**。
    叫"factory fixture"模式——测试需要多个不同 seed 的 encoding 时，
    直接 make_encoding(0)、make_encoding(1) 即可，避免给每个变体写一个 fixture。
    """

    # _make 前缀下划线 = Python 约定的"私有函数"标记，
    # 提示读者这是 fixture 内部的工厂，不是给外面用的（虽然 Python 没有真正的访问控制）。
    def _make(seed: int) -> FaceEncoding:
        # 与 Task 1 的 _unit_vector 同样的逻辑：default_rng + standard_normal + L2 归一。
        # 已经在 Task 1 详细解释过，这里不重复。
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(512).astype(np.float32)
        # `v /= np.linalg.norm(v)` 是 `v = v / norm` 的"原地修改"简写。
        #   - 等价但更省一次内存分配
        #   - 仅当 v 是可变数组（numpy ndarray）才能用，tuple/frozenset 不行
        v /= np.linalg.norm(v)
        return FaceEncoding(vector=v, model_version="test")

    # 把内部函数返回出去，让测试持有这个工厂。
    return _make


# fixture 之间可以互相依赖：make_template 在参数列表里写 make_encoding，
# pytest 会先调 make_encoding 拿到工厂，再传给 make_template。
@pytest.fixture
def make_template(make_encoding: Callable[[int], FaceEncoding]) -> Callable[[int, str], Template]:
    def _make(seed: int, source: str = "test") -> Template:
        return Template(
            encoding=make_encoding(seed),
            source=source,
            # datetime(2026, 1, 1) = 固定一个时间戳，避免测试因"现在几点"而不可复现
            created_at=datetime(2026, 1, 1),
        )

    return _make


# tmp_path 是 pytest **内置** fixture，每个测试函数自动拿到一个**独立的临时目录**。
#   - 路径形如 /private/var/folders/.../pytest-of-user/test_xxx0/
#   - 测试结束后 pytest 默认保留最近 3 次（方便排查失败），更早的自动清理
#   - 多个测试并行也不会冲突
#
# ── 给小白：fixture 注入是怎么发生的 ──
# pytest 看到测试函数（或下面这个 fixture）的形参里出现了 `tmp_path`，就**按形参名
# 查找** 同名的 fixture，把它的返回值塞进来。所以：
#   - 形参名必须是 `tmp_path` 一字不差。写成 `tmp_pth` / `tmppath` / `tmp_dir`
#     都不会注入——pytest 找不到匹配的 fixture，最终拿到的是普通的 None 或者直接抛
#     "fixture not found" 错。
#   - 不要在函数体里手动 `tmp_path = something`——那会盖掉注入的值。
#   - 类型注解 `: Path` 是给 IDE 和 mypy 看的，对 pytest 注入机制没影响（pytest 只
#     看名字）。但写上类型对自己最有好处：自动补全 `tmp_path.read_text()` 之类的方法。
@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    # 在临时目录里凑一个 .db 路径返回。注意这里只是**路径**，
    # 文件本身由 SqliteRepository 第一次连接时创建。
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
# pyyaml 库：把 YAML 字符串 ↔ Python dict 互转。
#   - yaml.safe_load(s)  → 把 YAML 文本解析成 dict
#   - yaml.safe_dump(d)  → 把 dict 序列化成 YAML 文本
#   - "safe" 版本不会执行 YAML 里的可执行标签（!!python/object 之类），
#     避免读到攻击者构造的 YAML 时被代码注入
import yaml

from face_recognition.infrastructure.config_loader import AppConfig, load_config


@pytest.fixture
def valid_config_path(tmp_path: Path) -> Path:
    """构造一份合法的 YAML 配置文件，返回路径。"""
    # cfg 是个嵌套 dict，结构和 config.yaml 一一对应。
    # 测试不直接读项目根目录的 config.yaml，因为：
    #   1. 测试要可复现，不依赖外部文件
    #   2. 测试要能验证不同字段值（合法/非法），原文件改不动
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
    # tmp_path / "config.yaml" 用 / 重载——pathlib.Path 把 / 实现成路径拼接。
    # 比 os.path.join(str(tmp_path), "config.yaml") 简洁、跨平台（Windows 也对）。
    p = tmp_path / "config.yaml"
    # Path.write_text(s) = 一行写入文本文件，自动 open / write / close
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_load_config_returns_app_config(valid_config_path: Path):
    cfg = load_config(valid_config_path)
    # isinstance(obj, Cls) 检查 obj 是否是 Cls 的实例（含子类）。
    # 这里验证 load_config 返回的是真正的 AppConfig 类，不是某个 dict / namespace。
    assert isinstance(cfg, AppConfig)
    # 链式属性访问：cfg.recognition.threshold——Pydantic 会把嵌套 dict 自动转成嵌套对象
    assert cfg.recognition.threshold == 0.45
    assert cfg.recognition.template_strategy == "kmeans_k3"
    assert cfg.evaluation.random_seed == 42
    # 注意：YAML 里写的是字符串 "data/face.db"，但被 Pydantic 自动转成了 Path 对象
    # （因为 DataConfig.sqlite_path 字段类型注解为 Path）。这就是 pydantic 的"类型驱动转换"。
    assert cfg.data.sqlite_path == Path("data/face.db")


def test_invalid_strategy_name_raises(tmp_path: Path):
    """合法 YAML 但策略名超出枚举范围 → 必须报错。"""
    cfg = {
        "model": {"pack": "buffalo_l", "ctx_id": 0, "det_size": [640, 640]},
        # 故意填一个不在白名单里的策略名
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
    # 注：捕获 Exception 是兜底；pydantic 实际会抛 ValidationError，
    # 但这里我们关心"会不会抛"，不关心具体类型。
    with pytest.raises(Exception):
        load_config(p)


def test_threshold_out_of_range_raises(tmp_path: Path, valid_config_path: Path):
    """字段值越界（threshold 必须在 [-1, 1]）→ 必须报错。"""
    # 复用 valid_config_path 的内容，只改一个字段，避免重复写整套 dict。
    raw = yaml.safe_load(valid_config_path.read_text())
    raw["recognition"]["threshold"] = 2.0  # 超出余弦相似度的合法范围
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
# typing.Literal["a", "b", "c"] = "字符串只能是这几个值之一"的类型。
#   - 不是 enum，但效果类似——传别的字符串 mypy / pydantic 都会报错
#   - 比 Enum 轻量：不用定义 class，直接在类型注解里写枚举值
from typing import Literal

import yaml
# pydantic = Python 最流行的"数据校验 + 类型转换"库，FastAPI 也基于它。
#   - BaseModel：所有 schema 类的基类，给类加上"字段校验、JSON 序列化、类型转换"能力
#   - Field：声明字段的额外约束（最大值、最小值、长度、正则…），返回一个特殊"字段描述符"
#   - field_validator：自定义校验函数装饰器，用于内置约束表达不了的逻辑
from pydantic import BaseModel, Field, field_validator

# 用 Literal 给"模板策略名"圈定白名单。
# 后面 RecognitionConfig.template_strategy: StrategyName 时，
# pydantic 看到这是 Literal，传非白名单值会立刻 ValidationError。
StrategyName = Literal[
    "random_one", "mean_all", "manual_three", "kmeans_k3", "all_vectors"
]


# 每个嵌套配置块对应 config.yaml 里的一节。分块的好处：
#   1. 校验粒度细——单字段错误能精准定位到 ModelConfig 还是 RecognitionConfig
#   2. 在代码里访问时层次清晰：cfg.model.pack 比 cfg["model"]["pack"] 安全
class ModelConfig(BaseModel):
    pack: str
    ctx_id: int
    # tuple[int, int] = "正好两个 int 的元组"。
    # YAML 里写 [640, 640] 是 list，pydantic 会自动转成 tuple（类型驱动转换）。
    # 用 tuple 而不是 list：tuple 不可变，对"分辨率 / 尺寸"这种语义上"两个数捆绑"的值更合适。
    det_size: tuple[int, int]


class RecognitionConfig(BaseModel):
    # Field(ge=-1.0, le=1.0) = greater-or-equal / less-or-equal 约束。
    # ge=大于等于、gt=大于、le=小于等于、lt=小于。
    # 余弦相似度数学上必在 [-1, 1]，越界配置直接拒。
    threshold: float = Field(ge=-1.0, le=1.0)
    # 用上面定义的 Literal 类型，限制取值
    template_strategy: StrategyName


class CameraConfig(BaseModel):
    device_index: int
    resolution: tuple[int, int]
    fps: int = Field(gt=0)  # 必须 > 0


class RealtimeConfig(BaseModel):
    detect_every_n_frames: int = Field(ge=1)  # 至少 1 帧检一次（0 没意义）
    recognize_on_new_track: bool
    iou_threshold: float = Field(ge=0.0, le=1.0)  # IoU 数学上 [0, 1]
    track_max_missing_frames: int = Field(ge=0)
    # JPEG 编码质量,1=最差/最小,100=最佳/最大。85 是肉眼无损临界值,
    # 兼顾画质和 WebSocket 带宽(640×480 约 30~60KB/帧)。
    jpeg_quality: int = Field(85, ge=1, le=100)


class ApiConfig(BaseModel):
    host: str
    port: int = Field(gt=0, le=65535)  # TCP 端口范围


class DataConfig(BaseModel):
    # 字段类型为 Path → pydantic 自动把 YAML 里的字符串转成 pathlib.Path。
    # 用 Path 而非 str：后续代码 mkdir / read_text / 拼路径都更安全，不用反复 Path(...)。
    sqlite_path: Path
    dataset_root: Path
    lfw_subset: Path


class EvaluationConfig(BaseModel):
    random_seed: int
    train_ratio: float = Field(gt=0.0, lt=1.0)
    far_targets: list[float]

    # @field_validator("far_targets") = "我要给 far_targets 字段额外加一段校验逻辑"。
    # 第一参数是字段名字符串（pydantic v2 写法）。
    @field_validator("far_targets")
    # @classmethod 必须紧跟在 @field_validator 下面（pydantic v2 要求）。
    #   - classmethod = 类方法，第一个参数是类本身（cls）而不是实例（self）
    #   - 校验在"还没有实例"时就要跑，所以拿不到 self
    @classmethod
    def _all_in_range(cls, v: list[float]) -> list[float]:
        for x in v:
            # 0.0 < x < 1.0 是 Python 的"链式比较"，等价于 (0.0 < x) and (x < 1.0)
            # 比 C/Java 的 (x > 0 && x < 1) 更直观
            if not 0.0 < x < 1.0:
                # f-string：f"...{表达式}..." 是 Python 3.6+ 的格式化字符串，
                # 把表达式的值嵌进去，比 .format() 简洁
                raise ValueError(f"FAR 目标必须在 (0, 1) 内，收到 {x}")
        # 校验函数必须**返回值**——pydantic 用返回值作为最终字段值（允许在校验时顺便做转换）
        return v


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    file: str


class AppConfig(BaseModel):
    """整个 config.yaml 的顶层 schema。把上面 8 个子模型组合起来。"""
    model: ModelConfig
    recognition: RecognitionConfig
    camera: CameraConfig
    realtime: RealtimeConfig
    api: ApiConfig
    data: DataConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig


def load_config(path: Path | str = "config.yaml") -> AppConfig:
    """读 YAML 文件，返回校验过的 AppConfig 对象。配置错误立即抛 ValidationError。"""
    # 接受 Path 或 str，统一转成 Path 处理
    path = Path(path)
    # encoding="utf-8" 显式指定编码，避免 Windows 默认 GBK 读中文注释翻车
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    # AppConfig.model_validate(d) 是 pydantic v2 的入口：
    #   - 接收 dict / list / 任意可序列化结构
    #   - 按 schema 校验 + 类型转换
    #   - 失败抛 pydantic.ValidationError（带具体字段路径，比如 "recognition.threshold"）
    # （v1 是 AppConfig.parse_obj(d)，已废弃；新代码必须用 model_validate）
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
    """添加 → 取出 → 数据应完全一致（最经典的 roundtrip 测试）。"""
    repo = SqliteRepository(tmp_db_path)
    alice = Person(
        person_id="alice",
        display_name="Alice",
        templates=(make_template(1, "centroid_0"), make_template(2, "centroid_1")),
    )
    repo.add(alice)

    fetched = repo.get("alice")
    # `is not None` 而非 `!= None`：在 Task 1 已解释（None 是单例，用 is 比较身份）
    assert fetched is not None
    assert fetched.person_id == "alice"
    assert fetched.display_name == "Alice"
    assert len(fetched.templates) == 2
    # np.allclose(a, b) = "两个浮点数组在容差范围内相等吗"。
    #   - 浮点数 a == b 危险：1.0 / 3 * 3 != 1.0（IEEE 754 精度损失）
    #   - allclose 默认 rtol=1e-5、atol=1e-8，按 |a-b| ≤ atol + rtol·|b| 判定
    #   - 写向量比较时**永远用 allclose**，不要直接 ==
    assert np.allclose(
        fetched.templates[0].encoding.vector,
        alice.templates[0].encoding.vector,
    )


def test_get_unknown_returns_none(tmp_db_path: Path):
    """查不存在的人 → 返回 None（不是抛异常）。"""
    repo = SqliteRepository(tmp_db_path)
    assert repo.get("ghost") is None


def test_list_all_orders_by_person_id(
    tmp_db_path: Path,
    make_template: Callable[[int, str], Template],
):
    """list_all 必须按 person_id 字典序排序——保证调用端拿到稳定顺序。"""
    repo = SqliteRepository(tmp_db_path)
    repo.add(Person("bob", "Bob", (make_template(10, "x"),)))  # 先插 bob
    repo.add(Person("alice", "Alice", (make_template(20, "x"),)))  # 后插 alice
    # 列表推导式：[expr for x in iterable] = "对 iterable 里每个元素求 expr 收集成列表"。
    # 等价于 list(map(lambda p: p.person_id, repo.list_all()))，但更 Pythonic。
    ids = [p.person_id for p in repo.list_all()]
    # 即使插入顺序是 bob → alice，输出也必须是 alice → bob（按 ID 排序）
    assert ids == ["alice", "bob"]


def test_add_existing_person_replaces(
    tmp_db_path: Path,
    make_template: Callable[[int, str], Template],
):
    """对同一个 person_id 二次 add → 应该**整体替换**老数据，而非追加。"""
    repo = SqliteRepository(tmp_db_path)
    repo.add(Person("alice", "Alice", (make_template(1, "x"),)))
    # 用同一个 id 再插，display_name 和模板都变
    repo.add(Person("alice", "Alice 2", (make_template(99, "y"), make_template(100, "z"))))
    fetched = repo.get("alice")
    assert fetched is not None
    assert fetched.display_name == "Alice 2"  # 取新名字
    assert len(fetched.templates) == 2  # 不是 1+2=3，是覆盖后的 2


def test_remove_nonexistent_raises(tmp_db_path: Path):
    """删不存在的人 → 抛 PersonNotFoundError（不是静默 no-op）。"""
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
    """识别用矩阵导出：行数 = 总模板数，列数 = 512，索引列表对齐每行的 person_id。"""
    repo = SqliteRepository(tmp_db_path)
    repo.add(Person("alice", "A", (make_template(1, "x"), make_template(2, "y"))))  # 2 模板
    repo.add(Person("bob", "B", (make_template(3, "x"),)))  # 1 模板
    matrix, ids = repo.all_templates_matrix()
    # ndarray.shape 是个 tuple，描述每个维度的长度。(3, 512) = 3 行 512 列。
    assert matrix.shape == (3, 512)
    # ndarray.dtype 是数据类型；比较时和 np.float32 这个类型对象比，不是字符串
    assert matrix.dtype == np.float32
    # alice 有 2 模板 → ids 列表前两个是 "alice"，第三个是 "bob"
    assert ids == ["alice", "alice", "bob"]

    # 每行依然是单位向量？
    # np.linalg.norm(matrix, axis=1) = 沿 axis=1（列方向）算范数。
    #   - axis=0 → 对每列求 → 输出 shape (512,)
    #   - axis=1 → 对每行求 → 输出 shape (3,)
    #   - 不传 axis → 对整个数组求一个标量
    # 这里要每行的范数，所以 axis=1。
    norms = np.linalg.norm(matrix, axis=1)
    # atol=1e-3 = absolute tolerance，允许 |a-b| ≤ 0.001 的误差。
    # 为什么放宽？float32 精度有限，存进 SQLite BLOB 再读出来累计误差，1e-3 足够区分"还是单位向量"。
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_empty_matrix_is_zero_rows(tmp_db_path: Path):
    """库为空时，矩阵应该是 (0, 512) 而不是 None 或抛错——下游矩阵乘法不需要特判。"""
    repo = SqliteRepository(tmp_db_path)
    matrix, ids = repo.all_templates_matrix()
    assert matrix.shape == (0, 512)
    assert ids == []


def test_concurrent_reads_do_not_raise(
    tmp_db_path: Path,
    make_template: Callable[..., Template],
):
    """多线程并发调用同一个 repo 不应崩溃。

    M4 的 FastAPI 在线程池里跑同步代码——如果 SqliteRepository 没设
    check_same_thread=False 或没加锁,这里会抛 sqlite3.ProgrammingError。
    """
    import threading  # 测试内部 import,避免污染模块顶部

    repo = SqliteRepository(tmp_db_path)
    repo.add(Person(person_id="alice", display_name="Alice", templates=(make_template(),)))

    errors: list[BaseException] = []

    def worker():
        try:
            for _ in range(50):
                repo.get("alice")
                repo.all_templates_matrix()
        except BaseException as e:  # noqa: BLE001 - 测试要捕所有异常
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"并发访问出错: {errors}"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/test_sqlite_repository.py -v
```

- [ ] **Step 3: 实现 `src/face_recognition/infrastructure/sqlite_repository.py`**

```python
# sqlite3 是 Python **标准库**自带的 SQLite 客户端——不需要 pip install。
#   - SQLite = 单文件嵌入式数据库，整个库就是一个 .db 文件
#   - 35 人 × 几个模板 = 几百行数据，SQLite 完全够用
#   - 选 SQLite 而非 PostgreSQL/MySQL：零配置、无服务器、和项目同生命周期
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from face_recognition.domain.entities import FaceEncoding, Person, Template
from face_recognition.domain.errors import PersonNotFoundError

# 用模块级常量 _SCHEMA 存建表 SQL。下划线前缀 = 模块私有。
# 三引号字符串：跨多行的 SQL 直接写进来，比 Python 列表逐条好读。
_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    person_id     TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL
);

-- templates 表：一个人对应多行（每行一个模板向量）
-- BLOB = Binary Large Object，存任意二进制。SQLite 没有"数组"原生类型，
-- 把 numpy 数组的字节序列存进 BLOB 是简单又高效的方案。
CREATE TABLE IF NOT EXISTS templates (
    person_id     TEXT NOT NULL,
    template_idx  INTEGER NOT NULL,
    vector        BLOB NOT NULL,
    source        TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    -- 复合主键：(person_id, template_idx) 联合唯一
    PRIMARY KEY (person_id, template_idx),
    -- 外键：指向 persons.person_id；ON DELETE CASCADE = 删人时自动级联删掉所有模板
    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
);
"""


class SqliteRepository:
    """PersonRepository Protocol 的具体实现（domain/interfaces.py 里定义的接口）。

    注意：这个类**没有显式继承** PersonRepository。只要方法签名匹配，
    Protocol 就认为它"算"实现了——结构化子类型，前面 Task 3 已解释。
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        # parents=True 自动创建中间目录（mkdir -p 等价物）
        # exist_ok=True 目录已存在不报错（默认存在会抛 FileExistsError）
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # sqlite3.connect(path)：打开（或创建）一个 SQLite 数据库连接。
        # 文件不存在会自动创建。返回 Connection 对象，后续所有操作走它。
        #
        # check_same_thread=False 是关键 —— Python 的 sqlite3 模块默认要求
        # "创建 connection 的线程才能用它"。M4 的 FastAPI / WebSocket 会在线
        # 程池里跑同步代码（CLI 用例被 `run_in_threadpool` 调度），如果不关掉
        # 这个检查就会抛 "SQLite objects created in a thread can only be used
        # in that same thread"。
        #
        # 关掉检查后必须自己加锁——SQLite 的 connection 本身**不是**线程安全
        # 的。下面的 self._lock 保证同一时刻只有一个线程在操作 connection。
        # 35 人量级的项目里这把锁完全不构成性能瓶颈。
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._lock = threading.Lock()
        # PRAGMA foreign_keys = ON：SQLite 默认**不**强制外键，必须显式打开。
        # 不开的话上面定义的 FOREIGN KEY 形同虚设，删人时不会级联删模板。
        self._conn.execute("PRAGMA foreign_keys = ON")
        # executescript 可以一次跑多条 SQL（用 ; 分隔）。execute 一次只能一条。
        self._conn.executescript(_SCHEMA)
        # ── 给小白：整套并发设计的几个常见疑问 ──
        # Q1: 为什么不每次操作都 `with sqlite3.connect(...)` 临时新建一个 connection？
        #     一个 connect 大约要 1~3ms（解析 db 文件头 + 建立 page cache）。M4 实时
        #     识别每帧都要 `query()` 查模板矩阵，每秒 30 帧 × 3ms = 90ms 纯开销，CPU
        #     白白吃掉 9%。维持长连接 + Lock 串行化，连接开销摊到 0。
        # Q2: 为什么用普通 Lock 而不是 RLock（可重入锁）？
        #     普通 Lock 不可重入：同一线程二次 `with self._lock:` 会自我死锁。但本类
        #     的方法之间**互不调用**——`add` 不会调 `get`、`get` 不会调 `list_all`，所以
        #     不会出现"嵌套加锁"的情况。普通 Lock 比 RLock 快一点（少一次重入计数）。
        #     例外：`list_all` 内部刻意把"取数据"和"还原对象"拆成两段，前段在锁内拿
        #     到 rows 后**释放锁**，后段不持锁地构造 Person——见下方 list_all 注释。
        # Q3: 为什么不用每个请求一个 connection（FastAPI 依赖注入风格）？
        #     可以但更复杂：要写 contextmanager + 依赖注入，且 SQLite WAL 模式才支持
        #     多 connection 高并发。本项目并发量小（35 人门禁场景峰值 5 QPS），单
        #     connection + 锁是更直白、更易讲清楚的方案。
        # Q4: check_same_thread=False 不加锁直接用会怎样？
        #     SQLite 的 connection 内部有 prepared statement 缓存，多线程同时操作会
        #     破坏这个缓存，出现段错误（C 层崩溃）或脏数据。Python 层可能不报错只
        #     返回错乱结果——这种 bug 极难复现，所以"关检查"和"加锁"必须配套。

    def add(self, person: Person) -> None:
        # `with self._lock:` 串行化所有 connection 操作；`with self._conn:`
        # 在锁内开启 SQLite 事务（无异常 COMMIT，有异常 ROLLBACK）。
        # 保证"删旧 + 插新 + 插模板"要么全成功，要么全回滚（原子性）。
        with self._lock, self._conn:
            # 先 DELETE 再 INSERT，实现"重复 add 等价于覆盖"。
            # 配合外键级联，老模板会自动被删，不留垃圾数据。
            # 参数化查询：?  占位符 + 元组传值。**永远不要**用 f-string 拼 SQL，
            # 否则一旦 person_id 含 ' 字符就 SQL 注入。
            self._conn.execute("DELETE FROM persons WHERE person_id = ?", (person.person_id,))
            self._conn.execute(
                "INSERT INTO persons (person_id, display_name) VALUES (?, ?)",
                (person.person_id, person.display_name),
            )
            # enumerate(iterable) = 同时遍历"索引 + 元素"。等价于：
            #   for idx in range(len(person.templates)):
            #       tpl = person.templates[idx]
            # 但更 Pythonic、对任何可迭代对象都行（不仅是 list）。
            for idx, tpl in enumerate(person.templates):
                self._conn.execute(
                    # SQL 字符串拼接：相邻的字符串字面量自动拼起来（C 语言也有这个特性）
                    "INSERT INTO templates (person_id, template_idx, vector, source, "
                    "model_version, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        person.person_id,
                        idx,
                        # ndarray.astype(np.float32) 确保类型一致（防止意外是 float64）。
                        # ndarray.tobytes() 把内存里的字节序列原样导出 → 存 BLOB。
                        # 512 个 float32 = 512 × 4 = 2048 字节 / 模板。
                        tpl.encoding.vector.astype(np.float32).tobytes(),
                        tpl.source,
                        tpl.encoding.model_version,
                        # datetime.isoformat() = "2026-01-01T00:00:00"。
                        # 用 ISO 8601 字符串存而非 Unix 时间戳：直接读得懂、SQL 排序也对。
                        tpl.created_at.isoformat(),
                    ),
                )

    def get(self, person_id: str) -> Person | None:
        with self._lock:
            # execute() 返回 Cursor；.fetchone() 取下一行（没有则 None）；
            # .fetchall() 取所有剩余行；不调直接迭代 cursor 是流式取。
            row = self._conn.execute(
                "SELECT display_name FROM persons WHERE person_id = ?", (person_id,)
            ).fetchone()
            if row is None:
                return None
            # 行结果是 tuple，按 SELECT 字段顺序索引
            display_name = row[0]
            tpl_rows = self._conn.execute(
                "SELECT vector, source, model_version, created_at FROM templates "
                "WHERE person_id = ? ORDER BY template_idx",  # 按 idx 排序还原插入顺序
                (person_id,),
            ).fetchall()
        # 生成器表达式 + tuple() 包裹 = 直接造 tuple，不建中间 list。
        # 写法 (expr for x in iter) 用圆括号；和列表推导式 [expr for x in iter] 区别仅此。
        templates = tuple(
            Template(
                encoding=FaceEncoding(
                    # np.frombuffer(bytes, dtype) = 把字节序列**零拷贝**解读成 ndarray。
                    # 比 np.array(list(bytes)) 快几十倍——直接共享内存视图。
                    # ── dtype 必须严格匹配存进去时用的类型 ──
                    # 存的时候 `vector.astype(np.float32).tobytes()` 写入 512×4=2048 字节；
                    # 读的时候若误写 dtype=np.float64（每数 8 字节），frombuffer 会把
                    # 2 个 float32 当作 1 个 float64 解释，得到 256 个垃圾 float——余弦
                    # 点积出来的结果完全无意义、但程序**不会报错**。这是新手最容易踩的
                    # 静默大坑，写错 dtype 就只能靠"识别准确率诡异低"反推。
                    vector=np.frombuffer(blob, dtype=np.float32),
                    model_version=mv,
                ),
                source=src,
                # datetime.fromisoformat(s) = isoformat() 的反操作
                created_at=datetime.fromisoformat(ts),
            )
            # 元组解包：每个 tpl_rows 元素是 (vector, source, mv, ts) 四元组，
            # 直接 for blob, src, mv, ts in ... 可以同时拿到 4 个变量
            for blob, src, mv, ts in tpl_rows
        )
        return Person(person_id=person_id, display_name=display_name, templates=templates)

    def remove(self, person_id: str) -> None:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM persons WHERE person_id = ?", (person_id,)
            )
            # cursor.rowcount = 上一条 DML 影响的行数。0 行 = 这个 person_id 本来就不存在。
            # 我们约定"删不存在的人是异常"，所以抛 PersonNotFoundError。
            if cur.rowcount == 0:
                raise PersonNotFoundError(person_id)

    def list_all(self) -> list[Person]:
        # 先在锁内取所有 id（短事务），再逐个 self.get(pid) —— 注意 get() 内部
        # 自己也会拿锁，所以这里**不要**把外层 with self._lock 包住整个方法，
        # 否则同一线程重入就会死锁（threading.Lock 不可重入）。
        with self._lock:
            ids = [
                row[0]
                for row in self._conn.execute(
                    "SELECT person_id FROM persons ORDER BY person_id"
                )
            ]
        # 列表推导带 if 过滤：[expr for x in iter if cond]
        # 这里其实 self.get(pid) 不会返回 None（因为 pid 来自 persons 表本身），
        # 但加一道 None 过滤让类型检查器闭嘴（mypy 不知道这个不变量）。
        return [p for p in (self.get(pid) for pid in ids) if p is not None]

    def all_templates_matrix(self) -> tuple[np.ndarray, list[str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT person_id, vector FROM templates "
                "ORDER BY person_id, template_idx"  # 双重排序保证矩阵行顺序确定
            ).fetchall()
        # 空库的特判：返回形状 (0, 512) 的"空矩阵"。
        # 为什么不直接返回 None？让调用端少写 if：
        #   matrix @ query 在 (0, 512) 上是合法的，得到长度为 0 的相似度数组，argmax 自然报错——
        #   调用端只要检查 matrix.shape[0] == 0 即可，比"matrix is None"更对称。
        if not rows:
            return np.zeros((0, 512), dtype=np.float32), []
        ids = [r[0] for r in rows]
        # np.stack(list_of_1d_arrays) = "把若干一维数组堆成二维"。
        #   - 每个 (512,) 一维向量 → 堆完是 (M, 512)
        #   - axis=0（默认）：在新轴 0 堆叠
        #   - 替代品：np.array([...]) 也行，但 stack 语义更明确"我要堆叠成新维度"
        matrix = np.stack(
            [np.frombuffer(r[1], dtype=np.float32) for r in rows]
        )
        return matrix, ids
```

> **为什么不每次方法新开一条 connection？**
> 那也是合法做法，但每次 `sqlite3.connect()` 都要重新解析文件、重放 WAL、
> 加载 schema 缓存——M4 实时识别每帧都要查矩阵，这开销不可忽略。本项目数据
> 量小、QPS 低，**一条共享连接 + Lock** 是最简且性能最好的折中。生产服务
> 才需要 connection pool。

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/test_sqlite_repository.py -v
```

预期：9 passed

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
    """辅助函数：验证所有模板向量都是单位向量（L2 范数 ≈ 1）。"""
    # all(iterable) = "iterable 里所有值都为真吗"。任何一个 False 立刻短路返回 False。
    # 配合生成器表达式（不建中间 list），写"全员校验"的标准句式。
    return all(
        abs(float(np.linalg.norm(t.encoding.vector)) - 1.0) < 1e-3 for t in templates
    )


# 共用 fixture：40 个不同的 FaceEncoding，模拟一个人有 40 张照片
@pytest.fixture
def encs(make_encoding: Callable[[int], FaceEncoding]) -> list[FaceEncoding]:
    # range(40) = 生成 0..39 的整数序列（懒求值）
    return [make_encoding(i) for i in range(40)]


def test_random_one_returns_one_template(encs: list[FaceEncoding]):
    out = RandomOneStrategy(seed=42).build(encs)
    assert len(out) == 1
    assert _all_unit_norm(out)


def test_random_one_deterministic_with_seed(encs: list[FaceEncoding]):
    """同一 seed 必须给同样的随机选择——可复现是评估实验的基本要求。"""
    a = RandomOneStrategy(seed=42).build(encs)
    b = RandomOneStrategy(seed=42).build(encs)
    # np.array_equal(a, b) = "两个数组完全相同吗（包括 shape 和每个元素）"。
    # 与 np.allclose 区别：array_equal 是**精确相等**（适合"完全一样的随机选择"场景），
    # allclose 是"在容差内相等"（适合涉及浮点运算的比较，比如均值、归一化）。
    assert np.array_equal(a[0].encoding.vector, b[0].encoding.vector)


def test_mean_all_returns_one_normalized_centroid(encs: list[FaceEncoding]):
    out = MeanAllStrategy().build(encs)
    assert len(out) == 1
    assert _all_unit_norm(out)


def test_manual_three_takes_first_three(encs: list[FaceEncoding]):
    out = ManualThreeStrategy().build(encs)
    assert len(out) == 3
    assert _all_unit_norm(out)
    # 验证简化版"取前三"语义：第 i 个输出 = 输入第 i 个原向量
    for i, tpl in enumerate(out):
        assert np.allclose(tpl.encoding.vector, encs[i].vector)


def test_manual_three_with_fewer_takes_all(make_encoding):
    """不到 3 张时不要崩溃——尽量返回有多少给多少。"""
    out = ManualThreeStrategy().build([make_encoding(0), make_encoding(1)])
    assert len(out) == 2


def test_kmeans_k3_returns_three_normalized_centroids(encs):
    out = KMeansK3Strategy(seed=42).build(encs)
    assert len(out) == 3
    assert _all_unit_norm(out)


def test_kmeans_k3_with_fewer_than_3_falls_back(make_encoding):
    """不到 3 张时降级，避免 sklearn 因 n_samples < n_clusters 报错。"""
    out = KMeansK3Strategy(seed=42).build([make_encoding(0), make_encoding(1)])
    assert len(out) == 2


def test_manual_three_with_fewer_than_3_returns_what_it_has(make_encoding):
    """manual_three 用 encodings[:3] 宽容切片——只有 2 张就返回 2 张，不报错。
    这是评估流水线里"某人照片不够"的兜底：不让 5 策略对照实验里 manual_three 一家独崩。"""
    out = ManualThreeStrategy().build([make_encoding(0), make_encoding(1)])
    assert len(out) == 2
    out_one = ManualThreeStrategy().build([make_encoding(0)])
    assert len(out_one) == 1


def test_all_vectors_returns_all_inputs(encs):
    out = AllVectorsStrategy().build(encs)
    assert len(out) == len(encs)


def test_empty_input_raises(encs):
    """所有策略在空输入下都应抛 ValueError（防止后续计算被零向量污染）。"""
    # 用 tuple 装多个策略实例 → for 循环里逐个验证。
    # 比"5 个独立测试"更紧凑，但失败时 pytest 只报第一个崩的，定位略弱。
    # 这里 trade-off 取紧凑——这种"统一行为"的测试通常用循环。
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

# Python 标准库 random 模块——纯 Python 的随机数生成器，与 numpy.random 是两套独立实现。
#   - random.choice(seq) = 从序列里等概率挑一个
#   - 这里**不用 np.random** 是因为：选 FaceEncoding 对象不是数值运算，标准库更轻量
import random
from datetime import datetime, timezone

from face_recognition.domain.entities import FaceEncoding, Template


class RandomOneStrategy:
    # 类属性 `name`：实现 TemplateStrategy Protocol 要求的 name 字段。
    # 写在 class 顶层（不在 __init__ 里）→ 所有实例共享同一个值，节省内存。
    name = "random_one"

    def __init__(self, seed: int = 42) -> None:
        # random.Random(seed) = 创建一个**独立的随机数生成器实例**，
        # 不污染全局 random.* 的状态。
        # 类比：numpy 用 default_rng(seed)，标准库用 Random(seed)，效果对应。
        self._rng = random.Random(seed)

    def build(self, encodings: list[FaceEncoding]) -> list[Template]:
        # `if not encodings` 利用 Python 的"真值测试"：空列表为 False、非空为 True。
        # 等价于 `if len(encodings) == 0`，但更 Pythonic。
        if not encodings:
            raise ValueError("RandomOneStrategy 至少需要 1 个 encoding")
        # rng.choice(seq) 等价于 random.choice，但用我们这个独立生成器
        chosen = self._rng.choice(encodings)
        return [
            Template(
                encoding=chosen,
                source="random_one",
                # 当前 UTC 时间。datetime.utcnow() 在 Python 3.12+ 已 deprecate;
                # 现代写法是 datetime.now(timezone.utc) → 拿到 timezone-aware datetime,
                # 再 .replace(tzinfo=None) 抹掉时区信息变 naive(SQLite 存储更省事)。
                # 后续所有策略沿用本写法,保持时间字段一致。
                # ── 给小白：为什么要 `.replace(tzinfo=None)` 这一步骚操作 ──
                # Python 的 datetime 分两种：
                #   - **naive**（无时区）：datetime(2026,1,1,12,0)，"几点几分"但不知道哪个时区
                #   - **aware**（带时区）：datetime(2026,1,1,12,0, tzinfo=timezone.utc)
                # 这两种**不能直接比较**，混着用 `dt1 < dt2` 会抛 TypeError。
                # SQLite 的 TEXT 字段存 isoformat 字符串：aware 会写成
                # "2026-01-01T12:00:00+00:00"，naive 写成 "2026-01-01T12:00:00"——
                # 用 fromisoformat 读回来类型就跟着分裂。整套系统全用 naive UTC（先 now(utc)
                # 拿到正确的 UTC 时刻，再 replace 抹掉 tzinfo）= 所有 datetime 都可比较，
                # 也不会因为本地时区不同导致测试在不同机器上结果不一样。
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        ]
```

- [ ] **Step 4: 实现 `mean_all.py`**

```python
# src/face_recognition/application/strategies/mean_all.py
from datetime import datetime, timezone

import numpy as np

from face_recognition.domain.entities import FaceEncoding, Template


class MeanAllStrategy:
    """对一个人的所有照片向量取平均，再归一化为 1 个模板。

    数学直觉：每张照片是单位球面上的一个点，平均就是这堆点的"重心"。
    重心一般不在球面上（长度 < 1），所以最后必须再 L2 归一化拉回球面。
    优点：只存 1 个向量，识别最快。缺点：极端姿态（侧脸、闭眼）会被"平滑掉"。
    """

    name = "mean_all"

    def build(self, encodings: list[FaceEncoding]) -> list[Template]:
        if not encodings:
            raise ValueError("MeanAllStrategy 至少需要 1 个 encoding")
        # np.stack([v1, v2, ...]) 已在 Task 6 解释：把若干一维数组堆成二维 (N, 512)
        stacked = np.stack([e.vector for e in encodings])
        # ndarray.mean(axis=0) = 沿行方向（"竖着"）求平均。
        #   - axis=0：对每列做平均 → 输出 shape (512,)，得到"重心向量"
        #   - axis=1：对每行做平均 → 输出 shape (N,)，每张图变成一个标量（没意义）
        #   - 不传 axis：对所有元素求一个标量
        # 记忆口诀："axis=k 把第 k 维消掉"。这里消掉行（N），保留列（512）。
        centroid = stacked.mean(axis=0)
        # 归一化：除以自己的 L2 范数 → 得到单位向量
        centroid = centroid / np.linalg.norm(centroid)
        # 取第 0 个 encoding 的 model_version 作为代表。
        # 假设：同一批 encoding 都来自同一模型版本（注册时一次性生成）。
        # 若假设破裂（混合不同模型）这里会丢信息——但实际上不可能发生。
        model_version = encodings[0].model_version
        return [
            Template(
                encoding=FaceEncoding(
                    vector=centroid.astype(np.float32),
                    model_version=model_version,
                ),
                source="mean_all",
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        ]
```

- [ ] **Step 5: 实现 `manual_three.py`**

```python
# src/face_recognition/application/strategies/manual_three.py
from datetime import datetime, timezone

from face_recognition.domain.entities import FaceEncoding, Template


class ManualThreeStrategy:
    """简化版：取前 3 张。生产中可换成根据照片标签（正光/侧光/逆光）挑选。

    为什么本项目用"前 3 张"作为简化？真正的"人工挑 3 张"需要用户给每张照片打 tag
    （正脸/侧脸/逆光），数据管道复杂。期末项目阶段不上这套——用列表前 3 张占位，
    评估实验里如果它表现不如 KMeans，结论就是"自动聚类比朴素人工挑选好"。
    """

    name = "manual_three"

    def build(self, encodings: list[FaceEncoding]) -> list[Template]:
        if not encodings:
            raise ValueError("ManualThreeStrategy 至少需要 1 个 encoding")
        # encodings[:3] = Python 切片：取前 3 个元素。
        #   - 不会越界：少于 3 个时返回所有，不抛错
        #   - 这是 Python "宽容切片"的特性，与 list[index] 不同（后者越界会抛 IndexError）
        chosen = encodings[:3]
        # 列表推导式：把每个原始 encoding 包成 Template，记录 source="manual_0/1/2"
        return [
            Template(encoding=e, source=f"manual_{i}", created_at=datetime.now(timezone.utc).replace(tzinfo=None))
            for i, e in enumerate(chosen)
        ]
```

- [ ] **Step 6: 实现 `kmeans_k3.py`**

```python
# src/face_recognition/application/strategies/kmeans_k3.py
from datetime import datetime, timezone

import numpy as np
# sklearn (scikit-learn) = Python 最经典的传统机器学习库（聚类、回归、SVM、随机森林等）。
# KMeans 是无监督聚类算法的代表——不需要标签，自己把数据分成 K 簇。
from sklearn.cluster import KMeans

from face_recognition.domain.entities import FaceEncoding, Template


class KMeansK3Strategy:
    """对一个人的 N 张照片做 K=3 聚类，取 3 个簇心做模板。

    为什么这是"理论上最好的多模板策略"？
    - 一个人的所有照片在 512 维空间形成一个"散点云"；
    - 散点云通常有几个亚结构：正脸、侧脸、不同光照；
    - K-means 自动找出 3 个最具代表性的"中心"，比"任选 3 张"更稳；
    - 比 mean_all 多 2 个模板，覆盖姿态/光照变化；比 all_vectors（50 个）省存储。
    """

    name = "kmeans_k3"

    def __init__(self, k: int = 3, seed: int = 42) -> None:
        # 类内私有字段用 _ 前缀（约定，不强制）
        self._k = k
        self._seed = seed

    def build(self, encodings: list[FaceEncoding]) -> list[Template]:
        if not encodings:
            raise ValueError("KMeansK3Strategy 至少需要 1 个 encoding")
        # 降级路径：照片数 < K 时 KMeans 会报错（聚类数不能多于样本数），
        # 直接退化为"all_vectors"行为，不挑了
        if len(encodings) < self._k:
            return [
                Template(
                    encoding=e,
                    source=f"kmeans_fallback_{i}",
                    created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                )
                for i, e in enumerate(encodings)
            ]
        # 把 list[FaceEncoding] 堆成 (N, 512) 矩阵——sklearn 全部输入都要 2D 数组
        stacked = np.stack([e.vector for e in encodings])
        # KMeans 三个关键参数：
        #   - n_clusters=k：聚类数（这里 K=3）
        #   - random_state=seed：内部随机初始化（K-means++ 选起点）的种子，固定 → 可复现
        #   - n_init=10：用 10 组不同的随机起点跑 10 次，取最好的那次。
        #     不传 n_init 在 sklearn 1.4+ 会发警告（默认值变化），显式写最稳。
        # .fit(X) = 在 X 上运行算法；返回训练好的 KMeans 对象（链式调用）
        km = KMeans(n_clusters=self._k, random_state=self._seed, n_init=10).fit(stacked)
        model_version = encodings[0].model_version
        # 用空 list + append 是 Python 经典模式。
        # 也可写成列表推导，但里面要做归一化等多步操作，append 更易读。
        templates = []
        # km.cluster_centers_ = shape (K, 512) 数组，每行一个簇心
        # （sklearn 约定：训练后的属性以 _ 结尾，区分于"传入的超参"）
        for i, c in enumerate(km.cluster_centers_):
            # 簇心一般不在单位球面上，必须重新归一化（与 mean_all 同理）
            normed = c / np.linalg.norm(c)
            templates.append(
                Template(
                    encoding=FaceEncoding(
                        vector=normed.astype(np.float32),
                        model_version=model_version,
                    ),
                    source=f"kmeans_centroid_{i}",
                    created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )
        return templates
```

- [ ] **Step 7: 实现 `all_vectors.py`**

```python
# src/face_recognition/application/strategies/all_vectors.py
from datetime import datetime, timezone

from face_recognition.domain.entities import FaceEncoding, Template


class AllVectorsStrategy:
    """全保留：每张照片都进库当模板。

    这是"召回率最高"的方案（任何一张原图都能被自己 100% 匹配上），
    但模板数 N 倍增长，识别时矩阵乘法成本随之增加。
    评估实验里它通常是 TAR 最高、但与 mean/kmeans 差距很小——
    用消融对比"加 50× 存储换微弱提升"是否值得。
    """

    name = "all_vectors"

    def build(self, encodings: list[FaceEncoding]) -> list[Template]:
        if not encodings:
            raise ValueError("AllVectorsStrategy 至少需要 1 个 encoding")
        return [
            Template(encoding=e, source=f"all_{i}", created_at=datetime.now(timezone.utc).replace(tzinfo=None))
            for i, e in enumerate(encodings)
        ]
```

- [ ] **Step 8: 跑测试确认通过**

```bash
uv run pytest tests/unit/test_strategies.py -v
```

预期：10 passed

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
# unittest.mock = Python 标准库的"假对象"工具集，用于隔离单元测试。
#   - MagicMock = 万能假对象：访问任何属性 / 调用任何方法都返回新的 MagicMock，不报错
#   - 单元测试不能真的去调 InsightFace（慢 + 要 GPU），用 mock 占位
from unittest.mock import MagicMock

import numpy as np
import pytest

from face_recognition.application.register_face import RegisterFace
from face_recognition.application.strategies.all_vectors import AllVectorsStrategy
from face_recognition.application.strategies.kmeans_k3 import KMeansK3Strategy
from face_recognition.domain.entities import FaceEncoding, Person, Template
from face_recognition.domain.errors import (
    NoFaceError,
    PersonHasNoTemplatesError,
)


@pytest.fixture
def stub_pipeline(make_encoding: Callable[[int], FaceEncoding]) -> MagicMock:
    """encode_single 按调用顺序返回 seed=0,1,2,... 的合成 encoding。"""
    # 闭包 + 可变 dict 实现"调用计数器"。
    # 为什么不用 `nonlocal i`？因为 Python 的闭包对**简单变量**的写需要 nonlocal 声明，
    # 而对**容器内字段**的修改不需要——dict["i"] += 1 只是读 dict 引用、改它内部值。
    # 这是 Python 闭包的常见绕路写法。
    counter = {"i": 0}

    def _encode_single(_image: np.ndarray) -> FaceEncoding:
        # 形参以 _ 开头：约定"我知道有这个参数但故意不用它"，
        # 让 lint 工具不警告"unused argument"
        i = counter["i"]
        counter["i"] += 1
        return make_encoding(i)

    # MagicMock() 创建一个万能假对象
    pipeline = MagicMock()
    # mock.method.side_effect = func：调用 mock.method(...) 时**实际执行 func**，
    # 把 func 的返回值当作 mock 的返回值。
    # 比 .return_value（每次都返回同一个值）灵活——这里要"每次返回不同向量"。
    pipeline.encode_single.side_effect = _encode_single
    return pipeline


@pytest.fixture
def stub_repo() -> MagicMock:
    """空白 mock 仓库——测试只关心 RegisterFace 是否调用 .add()，不关心数据真存哪。"""
    repo = MagicMock()
    return repo


@pytest.fixture
def fake_image_loader() -> Callable[[Path], np.ndarray]:
    # lambda path: ... = 匿名函数，等价于：
    #   def _loader(path): return np.zeros((112, 112, 3), dtype=np.uint8)
    # 单元测试不需要真读图，返回个全 0 数组占位即可——pipeline 是 mock，不会真用这张图。
    # 形状 (112, 112, 3) 模拟一张 112×112 BGR 图（ArcFace 标准对齐尺寸）。
    return lambda path: np.zeros((112, 112, 3), dtype=np.uint8)


def _make_person_dir(tmp_path: Path, person_id: str, n_imgs: int) -> Path:
    """辅助函数：在 tmp_path 下造一个 "person_id/" 目录，里面放 n 张假图。"""
    d = tmp_path / person_id
    d.mkdir()
    for i in range(n_imgs):
        # f"{i:03d}" = 把 i 格式化成"宽度 3、左侧补 0 的十进制"。
        #   - i=0 → "000"，i=12 → "012"
        # 文件名补零是为了字符串排序和数字顺序一致（"010" < "002" 在字典序下是错的，"002" < "010" 才对）
        # write_bytes(b"...") = b 前缀 = bytes 字面量，不是 str。这里只是占位，不真读
        (d / f"{i:03d}.jpg").write_bytes(b"fake")
    return d


def test_register_one_person_with_kmeans_k3(
    tmp_path: Path,
    stub_pipeline: MagicMock,
    stub_repo: MagicMock,
    fake_image_loader,
):
    """正常流程：10 张照片用 kmeans_k3 → 仓库收到 1 个 person、3 个模板。"""
    person_dir = _make_person_dir(tmp_path, "alice", 10)
    use_case = RegisterFace(
        pipeline=stub_pipeline,
        repository=stub_repo,
        strategy=KMeansK3Strategy(seed=42),
        image_loader=fake_image_loader,
    )
    use_case.execute_for_person(person_dir)

    # MagicMock 自带的断言方法：检查 .add() 是否被调用了**正好 1 次**
    stub_repo.add.assert_called_once()
    # mock.call_args 记录最后一次调用的参数。.args 是位置参数 tuple，[0] 是第一个。
    # `person: Person =` 是变量类型注解（不影响运行，给 IDE 和类型检查器看的）
    person: Person = stub_repo.add.call_args.args[0]
    assert person.person_id == "alice"
    assert len(person.templates) == 3  # KMeans K=3 出 3 模板


def test_skips_images_without_face(
    tmp_path: Path,
    make_encoding,
    stub_repo: MagicMock,
    fake_image_loader,
):
    """部分照片无脸 → warning 跳过，剩下的能成功就成功；返回的 skipped 计数准确。"""
    person_dir = _make_person_dir(tmp_path, "alice", 5)
    pipeline = MagicMock()
    # side_effect 也可以传**列表**：第 i 次调用返回（或抛出）列表第 i 项。
    # 列表元素是异常对象 → 抛该异常；其它对象 → 作为返回值。
    seq = [NoFaceError("无脸"), make_encoding(0), NoFaceError("无脸"), make_encoding(1), make_encoding(2)]
    pipeline.encode_single.side_effect = seq

    use_case = RegisterFace(
        pipeline=pipeline,
        repository=stub_repo,
        strategy=KMeansK3Strategy(seed=42),
        image_loader=fake_image_loader,
    )
    n_ok, n_skip = use_case.execute_for_person(person_dir)
    assert n_ok == 3
    assert n_skip == 2

    person: Person = stub_repo.add.call_args.args[0]
    assert len(person.templates) == 3  # 3 张成功 → KMeans 3 簇


def test_all_images_fail_raises(
    tmp_path: Path,
    stub_repo: MagicMock,
    fake_image_loader,
):
    """全部照片无脸 → 抛 PersonHasNoTemplatesError，仓库不被调用。"""
    person_dir = _make_person_dir(tmp_path, "alice", 3)
    pipeline = MagicMock()
    # 直接传单个异常实例 → 每次调用都抛这个（不是只抛一次）
    pipeline.encode_single.side_effect = NoFaceError("无脸")

    use_case = RegisterFace(
        pipeline=pipeline,
        repository=stub_repo,
        strategy=KMeansK3Strategy(seed=42),
        image_loader=fake_image_loader,
    )
    with pytest.raises(PersonHasNoTemplatesError):
        use_case.execute_for_person(person_dir)
    # assert_not_called = 检查 mock 从未被调用——验证"失败时不写库"
    stub_repo.add.assert_not_called()


def test_execute_dir_skips_failed_people(
    tmp_path: Path,
    stub_repo: MagicMock,
    fake_image_loader,
    make_encoding,
):
    """批量注册：1 人失败、1 人成功 → 不应整体崩，统计应分别记录。"""
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
    # mock.call_count = 调用总次数。Bob 1 次成功 → 仓库.add 调 1 次
    assert stub_repo.add.call_count == 1
    # bob 3 张全成功 → images_processed = 3, images_skipped = 0;
    # alice 整人失败抛 PersonHasNoTemplatesError,这条路径 skipped 不累加
    # (设计取舍:整人失败已用 persons_failed 单独计数,无需在 images_skipped 双重表达)
    assert summary.images_processed == 3
    assert summary.images_skipped == 0


def test_partial_failure_accumulates_skipped(
    tmp_path: Path,
    stub_repo: MagicMock,
    fake_image_loader,
    make_encoding,
):
    """部分失败的人:成功一张 + 失败两张 → images_skipped=2 累加进 summary。"""
    _make_person_dir(tmp_path, "alice", 3)
    pipeline = MagicMock()
    pipeline.encode_single.side_effect = [
        make_encoding(0),  # 成功
        NoFaceError("无脸"),
        NoFaceError("无脸"),
    ]

    use_case = RegisterFace(
        pipeline=pipeline,
        repository=stub_repo,
        # all_vectors 不要求多张,1 张也能注册——这样 alice 不会整人失败,skipped 才会被累加
        strategy=AllVectorsStrategy(),
        image_loader=fake_image_loader,
    )
    summary = use_case.execute(tmp_path)
    assert summary.persons_succeeded == 1
    assert summary.persons_failed == 0
    assert summary.images_processed == 1
    assert summary.images_skipped == 2
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/test_register_face.py -v
```

- [ ] **Step 3: 实现 `src/face_recognition/application/register_face.py`**

```python
# Python 标准库的 logging。CLAUDE.md 第 2 节明确"用 basicConfig 即可，不上 loguru/structlog"。
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

# set 字面量：{".jpg", ".jpeg", ...}（注意大括号 + 元素，不是 dict）。
# 用 set 而非 list：判断 `ext in _IMG_EXTS` 是 O(1) 哈希查找，list 是 O(n) 线性扫描。
# 模块级私有常量，所有方法共享。
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# logging.getLogger(__name__) 标准用法：
#   - __name__ = 当前模块的全限定名（如 "face_recognition.application.register_face"）
#   - 同一模块多次 getLogger 返回**同一个**实例（logger 是单例工厂）
#   - 这样不同模块的日志会带不同前缀，便于过滤和定位
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisterSummary:
    """注册批次结果统计。frozen=True 与 Task 1 的解释相同（不可变值对象）。"""
    persons_succeeded: int
    persons_failed: int
    images_processed: int
    images_skipped: int


class RegisterFace:
    """用例（use case）：表达一个完整的业务动作 = "注册一批人脸到库里"。

    依赖通过构造函数注入（DI），不在内部 new。这就是为什么测试可以传 MagicMock：
    用例不知道 pipeline 是真模型还是假对象，只要符合 Protocol 即可。
    """

    def __init__(
        self,
        pipeline: FacePipeline,
        repository: PersonRepository,
        strategy: TemplateStrategy,
        # image_loader 是函数而非对象——把"读图"做成可注入的依赖，
        # 测试时换成 lambda 返回假图，生产时传 cv2.imread 包装
        image_loader: Callable[[Path], np.ndarray],
    ) -> None:
        self._pipeline = pipeline
        self._repo = repository
        self._strategy = strategy
        self._load_image = image_loader

    def execute_for_person(self, person_dir: Path) -> tuple[int, int]:
        """注册单个人。返回 (成功图片数, 跳过图片数)。失败抛 PersonHasNoTemplatesError。

        跳过包含两类:扩展名不是图片(.txt/.DS_Store 等)、解码或检测失败(NoFaceError 等)。
        调用方拿到这俩数字往 RegisterSummary 里累加,用户在 CLI 末尾能看到准确统计。
        """
        # 显式标注 list[FaceEncoding]——空 list 的元素类型 mypy 推不出，主动告知
        encodings: list[FaceEncoding] = []
        skipped = 0
        # Path.iterdir() = 列出该目录下所有条目（不递归），返回 Path 生成器。
        # sorted(...) 强制确定顺序——文件系统遍历顺序在不同平台不一致，
        # 排序保证测试和实际运行结果可复现。
        for img_path in sorted(person_dir.iterdir()):
            # Path.suffix = 文件扩展名（含点）。.lower() 防大写 .JPG 漏过滤。
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue  # 跳过非图片（如 .DS_Store、.txt 标注文件）—不计入 skipped,扩展名过滤是预筛
            try:
                img = self._load_image(img_path)
                enc = self._pipeline.encode_single(img)
                encodings.append(enc)
            # except FaceRecognitionError 捕获我们自定义异常的**整个家族**
            # （NoFaceError、MultipleFacesError 等都是它的子类，见 Task 2）。
            # 不捕获 Exception——避免吞掉真正的程序 bug（KeyError 之类）。
            except FaceRecognitionError as e:
                # logging 的 % 占位符语法：logger.warning(fmt, *args)
                # 优势 vs f-string：仅在 WARNING 级别真的输出时才做字符串拼接，
                # 调用 .debug() 时如果当前级别是 INFO，省下拼接成本（虽然这里不重要）
                logger.warning("跳过 %s: %s", img_path, e)
                skipped += 1

        # 全部失败 → 这个人彻底注册不上，向上抛
        if not encodings:
            raise PersonHasNoTemplatesError(
                f"{person_dir.name}: 全部照片无法提取人脸"
            )

        # 走到这里：至少有一张照片成功了。交给策略生成模板。
        templates = self._strategy.build(encodings)
        person = Person(
            # Path.name = 路径最后一段（不含父目录）。"data/alice/" → "alice"
            person_id=person_dir.name,
            display_name=person_dir.name,  # 简化：用文件夹名做显示名
            # tuple(list) 把 list 转 tuple，因为 Person.templates 字段类型是 tuple
            templates=tuple(templates),
        )
        self._repo.add(person)
        return len(encodings), skipped

    def execute(self, dataset_dir: Path) -> RegisterSummary:
        """批量注册整个数据集。"""
        # 一行多重赋值：a = b = c = 0 → 所有变量都指向同一个 0（int 是不可变所以没问题）
        succeeded = failed = images_processed = images_skipped = 0
        for person_dir in sorted(dataset_dir.iterdir()):
            # Path.is_dir() = 是不是目录（不是文件、不是符号链接到不存在）
            if not person_dir.is_dir():
                continue
            try:
                n_ok, n_skip = self.execute_for_person(person_dir)
                succeeded += 1
                images_processed += n_ok
                images_skipped += n_skip
            # 单人失败时**不向上传播**——只记 error 然后继续下一个人。
            # 这是"批处理容错"的标准做法：1 个人挂了不连累整批。
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

预期：5 passed

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
    """造一个固定 seed 的单位向量。和 conftest 的 make_encoding 类似，但只返 ndarray。"""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_recognize_returns_best_match_above_threshold():
    """库里包含与查询完全相同的模板 → 应识别为对应人，相似度 ≈ 1.0。"""
    pipeline = MagicMock()
    repo = MagicMock()

    query_vec = _unit_vec(0)
    # mock.method.return_value = X：调用 mock.method(...) 时直接返回 X（每次都一样）
    pipeline.encode_single.return_value = FaceEncoding(query_vec, "test")

    # 3 个模板：第 1 个就是 query 自己（相似度=1），其他差远
    other = _unit_vec(99)
    matrix = np.stack([_unit_vec(11), query_vec, other])
    # 注意：返回的是 tuple（matrix, ids），与 Protocol 签名一致
    repo.all_templates_matrix.return_value = (matrix, ["bob", "alice", "carol"])

    use_case = RecognizeFace(pipeline=pipeline, repository=repo, threshold=0.5)
    # np.zeros((112, 112, 3), dtype=np.uint8) = 一张全黑的 112×112 BGR 图（占位）
    # uint8 = 0~255 整数，OpenCV 默认图像 dtype
    result = use_case.execute(np.zeros((112, 112, 3), dtype=np.uint8))

    assert result.person_id == "alice"
    # pytest.approx(value, abs=tolerance) = "近似等于"断言。
    # 浮点比较防精度坑：query · query = 1.0 理论上精确，但浮点舍入可能给 0.99999999
    # abs=1e-5 表示绝对误差容忍 0.00001
    assert result.similarity == pytest.approx(1.0, abs=1e-5)
    assert result.threshold == 0.5


def test_recognize_returns_none_when_below_threshold():
    """所有相似度都 < 阈值 → 返回 person_id=None（开放集拒识）。"""
    pipeline = MagicMock()
    repo = MagicMock()
    pipeline.encode_single.return_value = FaceEncoding(_unit_vec(0), "test")
    matrix = np.stack([_unit_vec(50), _unit_vec(60)])
    repo.all_templates_matrix.return_value = (matrix, ["bob", "carol"])

    # 阈值故意调高到 0.99——随机向量之间相似度通常很小，肯定过不了
    use_case = RecognizeFace(pipeline=pipeline, repository=repo, threshold=0.99)
    result = use_case.execute(np.zeros((112, 112, 3), dtype=np.uint8))

    assert result.person_id is None
    assert result.similarity < 0.99


def test_recognize_with_empty_repo_returns_none():
    """库为空 → 安全返回 None（不要崩在 argmax）。"""
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
    """识别用例：1 张图 → 提取向量 → 与库内所有模板比 → 阈值判别。

    核心算法：余弦相似度 = 单位向量点积。因为我们保证所有向量都 L2 归一化，
    所以"余弦相似度" ≡ "点积"。识别整库 = 一次矩阵乘法。
    """

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
        # 第 1 步：把图喂给 pipeline，提取 1 个 FaceEncoding（含 512 维向量）
        enc = self._pipeline.encode_single(image)
        # 第 2 步：从仓库一次性拿出所有模板矩阵。M = 总模板数。
        matrix, person_ids = self._repo.all_templates_matrix()
        # ndarray.shape[0] = 第 0 维大小，即"行数"=总模板数。
        # 库为空时 (0, 512)，避免后面 argmax 在空数组上炸（会抛 ValueError）
        if matrix.shape[0] == 0:
            return RecognitionResult(
                person_id=None, similarity=0.0, threshold=self._threshold
            )
        # 第 3 步：核心计算——一次矩阵乘法搞定所有相似度。
        # `matrix @ enc.vector` 是 Python 3.5+ 的"矩阵乘法运算符"（@），
        # 等价于 np.matmul(matrix, enc.vector) 或 matrix.dot(enc.vector)。
        # 形状变化：(M, 512) @ (512,) = (M,)
        # 每个元素 sims[i] = matrix[i] · enc.vector = 第 i 个模板与查询的余弦相似度
        # （因为两者都已归一化，所以点积 = cos θ）。
        sims = matrix @ enc.vector
        # np.argmax(arr) = 返回最大值的索引（int 类型），等价于 arr.argmax()
        # 显式 int(...) 转换：argmax 返回 numpy.int64，转回 Python int 让类型注解干净
        best_idx = int(np.argmax(sims))
        # numpy 标量转 Python float，理由同上
        best_sim = float(sims[best_idx])
        # 第 4 步：阈值判别——超过阈值才认领，否则返回未知（开放集核心特性）
        if best_sim >= self._threshold:
            return RecognitionResult(
                person_id=person_ids[best_idx],
                similarity=best_sim,
                threshold=self._threshold,
            )
        # 仍然返回 best_sim（不是 0），方便上层日志展示"虽然没认出但最高也只有 0.42"
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
- Test: `tests/integration/test_insightface_pipeline.py`（标记 `@pytest.mark.integration`）

> **测试 marker 命名约定**：所有"真模型 + 真 SQLite"的集成测试一律打 `@pytest.mark.integration`。
> 默认 `pytest` 只跑单元测试（毫秒级），`pytest -m integration` 才显式跑这一批。
> 之所以不叫 `gpu`：我们用 `ctx_id=-1` 强制走 CPU，名字会误导贡献者以为必须有显卡。

这是唯一调用 `insightface` 库的地方。集成测试用 `insightface.app.FaceAnalysis()` 自带的示例图。**测试默认跳过**（需要下载模型 + 较慢），用 `pytest -m integration` 显式跑。

> **pyproject.toml 必须先注册 marker**（M0 已加 `gpu`/`slow`，本任务里改成 `integration`）：
> ```toml
> [tool.pytest.ini_options]
> markers = [
>     "integration: 集成测试（真模型 + 真 SQLite，需要本地下载的 buffalo_l 权重）",
>     "slow: 耗时 > 1 秒的测试",
> ]
> addopts = "-ra --strict-markers"
> ```
> `--strict-markers` 让未注册的 marker 直接报错，能在 PR 阶段抓到名字写错的情况。

- [ ] **Step 1: 写集成测试 `tests/integration/test_insightface_pipeline.py`**

```python
import numpy as np
import pytest

from face_recognition.domain.errors import NoFaceError
from face_recognition.infrastructure.insightface_pipeline import InsightFacePipeline


# pytestmark = pytest.mark.xxx 是模块级标记：本文件所有测试都打上 xxx 标签。
# 这里给所有测试打 @pytest.mark.integration，等价于在每个 test_ 函数前加 @pytest.mark.integration。
# 配合 pyproject.toml 里 markers = ["integration: ..."]，运行 pytest -m integration 才跑这些。
pytestmark = pytest.mark.integration


# scope="module" = fixture 的"生存期"。
#   - "function"（默认）：每个测试函数独立创建一份
#   - "module"：本文件内所有测试共享一份，跑完才销毁
#   - "session"：整个 pytest 进程共享一份
# InsightFace 模型加载要 1~3 秒 + 占显存，每个测试都重建太浪费 → module 级共享
@pytest.fixture(scope="module")
def pipeline() -> InsightFacePipeline:
    # ctx_id=-1 = 强制走 CPU。集成测试不一定有 GPU 可用，CPU 能跑就行（慢点）
    return InsightFacePipeline(model_pack="buffalo_l", ctx_id=-1, det_size=(640, 640))


def test_encode_single_on_real_face(pipeline: InsightFacePipeline):
    """用 InsightFace 包内置示例图测试一张正常人脸的端到端流程。"""
    # 函数内 import：测试套件层面只在真跑时才 import insightface，
    # 避免模块加载阶段就 import 重型库——配合 -m "not integration" 跳过的场景
    import insightface
    from pathlib import Path
    # insightface.__file__ = insightface 包的入口文件路径。
    # .parent = 包目录。拼上 "data/images/t1.jpg" 找到包内自带示例。
    sample = Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"
    if not sample.exists():
        # pytest.skip(reason) = 主动跳过这个测试，不算失败也不算成功
        pytest.skip(f"InsightFace 自带示例图未找到: {sample}")
    import cv2
    # cv2.imread(path) = OpenCV 读图，返回 BGR 顺序的 ndarray（**不是 RGB**！）。
    # 注：cv2 不接受 Path 对象，必须 str(path)
    img = cv2.imread(str(sample))
    enc = pipeline.encode_single(img)
    # 验证产出形状和归一化（不验证具体数值——那是测 InsightFace 库本身了，不是测我们的封装）
    assert enc.vector.shape == (512,)
    assert abs(float(np.linalg.norm(enc.vector)) - 1.0) < 1e-3


def test_encode_no_face_raises(pipeline: InsightFacePipeline):
    """全黑图无脸 → 抛 NoFaceError（约定的语义异常）。"""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(NoFaceError):
        pipeline.encode_single(blank)
```

- [ ] **Step 2: 实现 `src/face_recognition/infrastructure/insightface_pipeline.py`**

```python
import numpy as np
# insightface.app.FaceAnalysis 是 InsightFace 官方提供的"一站式管线"：
#   detect 检测 → 5 点关键点 → 仿射对齐 → ArcFace ResNet100 出 512 维向量。
# 我们整个项目唯一允许 import insightface 的文件就是这里——见 CLAUDE.md 第 3 节。
from insightface.app import FaceAnalysis

from face_recognition.domain.entities import FaceEncoding
from face_recognition.domain.errors import EncodingError, MultipleFacesError, NoFaceError


class InsightFacePipeline:
    """实现 domain/interfaces.py 的 FacePipeline Protocol。

    封装 buffalo_l 模型的"加载 + 推理"两个动作。所有 InsightFace 接口
    细节都关进这个类，外部只看 encode/encode_single 两个方法。
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        ctx_id: int = 0,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        self._model_pack = model_pack
        # 暴露给 FacePipeline Protocol 的 model_version 属性。
        # 评估侧用它构造 FaceEncoding,避免硬编码 "buffalo_l"。
        self.model_version = model_pack
        # FaceAnalysis(name="buffalo_l") = 选择模型组合包。
        # 首次调用会自动从云端下载到 ~/.insightface/models/ 缓存（约 300MB）。
        self._app = FaceAnalysis(name=model_pack)
        # prepare = "把模型加载到设备上准备推理"。
        # ctx_id：0/1/2... 选 GPU 卡号；-1 走 CPU。
        # det_size：检测器输入分辨率，越大检测越准但越慢。
        # 这一步耗时（CPU 上 1~3s），所以只在 __init__ 调一次。
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

    def encode(self, image: np.ndarray) -> list[FaceEncoding]:
        """识别多张脸（实时识别用）。给一张可能含多人的图，返回每张脸的 encoding。"""
        # FaceAnalysis.get(img) = 一次性跑完整个流水线，返回 Face 对象列表。
        # 每个 Face 对象有 .bbox（边框）/.kps（关键点）/.embedding（向量）/.det_score（置信度）等属性
        faces = self._app.get(image)
        return [
            FaceEncoding(
                # f.embedding 是原始 numpy 数组，转 float32 + L2 归一化
                vector=self._normalize(f.embedding.astype(np.float32)),
                model_version=self._model_pack,
            )
            for f in faces
        ]

    def encode_single(self, image: np.ndarray) -> FaceEncoding:
        """注册时用：图里**必须正好 1 张脸**，多于或少于都是异常情况。"""
        faces = self._app.get(image)
        if not faces:
            raise NoFaceError("图中未检出人脸")
        if len(faces) > 1:
            raise MultipleFacesError(count=len(faces))
        emb = faces[0].embedding.astype(np.float32)
        return FaceEncoding(
            vector=self._normalize(emb), model_version=self._model_pack
        )

    # @staticmethod = 静态方法，不接收 self / cls，调用时不需要实例。
    # 这里 _normalize 不依赖任何实例字段，纯函数 → 标记为 static 让意图更清晰。
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        # 1e-12 是非常严格的"接近 0"判定。InsightFace 正常输出永远不会是零向量，
        # 出现 → 模型加载失败或输入有严重问题，应立即报错而非除以 0 出 NaN。
        if n < 1e-12:
            # 用领域异常而非裸 ValueError,与 errors.py 异常层次一致——
            # 上层全局 handler/CLI 可以按 FaceRecognitionError 基类统一捕获。
            raise EncodingError("InsightFace 输出零向量,模型异常")
        return v / n
```

- [ ] **Step 3: 跑集成测试（首次会下载 buffalo_l 模型，约 300MB）**

```bash
uv run pytest tests/integration/test_insightface_pipeline.py -v -m integration
```

预期：2 passed（首次跑约 1~3 分钟，含模型下载）

- [ ] **Step 4: 默认套件不跑这些（快）**

```bash
uv run pytest -v -m "not integration"
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

# cv2 = OpenCV 的 Python 绑定。imread/imwrite/imshow 等图像 IO 函数都在这里。
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


# 这个文件叫"组合根（Composition Root）"——所有"具体实现 → 用例"的装配只发生在这里。
# CLI 和 FastAPI server 都从这里要"造好的对象"，自己不知道 SqliteRepository、InsightFacePipeline 长什么样。
# 改实现（比如把 SQLite 换成 JSON 文件）只需要改这一个文件。

def build_config(path: Path | str = "config.yaml") -> AppConfig:
    """读配置——只是 load_config 的薄壳，统一入口。"""
    return load_config(path)


def build_pipeline(cfg: AppConfig) -> FacePipeline:
    """造一个真正的 InsightFacePipeline。返回类型故意标 FacePipeline 而非 InsightFacePipeline——
    调用方只要按 Protocol 用就行，看不到具体类型。"""
    return InsightFacePipeline(
        model_pack=cfg.model.pack,
        ctx_id=cfg.model.ctx_id,
        det_size=cfg.model.det_size,
    )


def build_repository(cfg: AppConfig) -> PersonRepository:
    return SqliteRepository(cfg.data.sqlite_path)


def build_strategy(name: str, seed: int) -> TemplateStrategy:
    """根据策略名（字符串）造对应的 TemplateStrategy 实例——典型"工厂方法"。"""
    # dict 字面量映射：name → 实例。注意这里**直接 new 了 5 个**实例，
    # 即便最后只用其中 1 个——5 个策略都很轻（无模型加载），浪费可忽略。
    # 如果将来某个策略变重，再改成 lazy 字典：{"name": lambda: HeavyStrategy()}
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
    """生产用的"图片加载器"——RegisterFace 通过依赖注入接收。"""
    # cv2.imread 失败时**返回 None 而非抛异常**（OpenCV 的设计奇葩点之一），
    # 必须显式检查
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"无法读图: {path}")
    return img


def build_register_use_case(cfg: AppConfig, strategy_name: str) -> RegisterFace:
    """组装 RegisterFace 所需的全部依赖。"""
    return RegisterFace(
        pipeline=build_pipeline(cfg),
        repository=build_repository(cfg),
        strategy=build_strategy(strategy_name, seed=cfg.evaluation.random_seed),
        # 把刚定义的私有函数作为参数传进去——Python 函数是"一等公民"，可以像普通对象一样传递
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
# Typer = 基于 Click 的现代 CLI 框架，由 FastAPI 同一作者出品。
#   - 用 Python 类型注解自动生成命令参数解析（不像 argparse 要手写 add_argument）
#   - 自动生成 --help、shell 自动补全等
#   - 选 Typer 而非 argparse：函数签名即 CLI 接口，DRY
import typer

from face_recognition.api.dependencies import (
    build_config,
    build_recognize_use_case,
    build_register_use_case,
    build_repository,
)
from face_recognition.domain.errors import FaceRecognitionError

# typer.Typer() = 创建一个 CLI 应用对象。可以理解为 FastAPI 里的 app=FastAPI()。
#   - name：在 --help 顶部显示的程序名
#   - no_args_is_help=True：不带任何参数运行时直接显示 help（默认行为是报错）
#   - help：程序顶层描述
app = typer.Typer(
    name="face-recognition",
    no_args_is_help=True,
    help="基于 ArcFace 的开放集人脸识别系统 CLI",
)


# @app.callback() 装饰器注册的函数会在**任何子命令执行前**先跑。
# 用途：装载配置、初始化日志，把结果通过 typer.Context 传给子命令。
@app.callback()
def _setup(
    # typer.Context 必须是**第一个**形参,且不给默认值——Typer 会在调用时自动注入真实 Context;
    # 给 None 默认值是老版本写法,会触发 mypy 警告还需 `# type: ignore`,反而掩盖真正的类型问题。
    ctx: typer.Context,
    # typer.Option(default, "--name", "-n", help="...") = 声明一个选项。
    #   - 第一个位置参数 = 默认值("..." Ellipsis 表示"必填",普通值表示"可选")
    #   - 后续字符串 = CLI 上的长短选项名
    config: Path = typer.Option("config.yaml", "--config", "-c", help="配置文件路径"),
) -> None:
    cfg = build_config(config)
    # logging.basicConfig 配置根 logger：级别 + 格式。
    # 项目刻意用最朴素的 logging（CLAUDE.md 第 2 节：禁用 loguru/structlog）。
    # format 占位符：%(asctime)s 时间、%(levelname)s 级别、%(name)s logger 名、%(message)s 内容
    logging.basicConfig(
        level=cfg.logging.level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # ctx.obj 是 Click/Typer 约定的"共享对象"位，把 cfg 塞进去给子命令用。
    # 子命令通过 `ctx: typer.Context` 形参拿到 ctx，再 ctx.obj 取
    ctx.obj = cfg


# @app.command() 注册一个子命令。函数名 → 子命令名（register、recognize、remove）。
@app.command()
def register(
    ctx: typer.Context,
    # `...`（Ellipsis）作为 typer.Option 的默认值表示"必填"——不传 CLI 会报错
    dataset: Path = typer.Option(..., "--dataset", help="数据集根目录（按文件夹分人）"),
    strategy: str = typer.Option(
        ...,
        "--strategy",
        help="模板策略：random_one / mean_all / manual_three / kmeans_k3 / all_vectors",
    ),
) -> None:
    """批量注册整个数据集。"""  # docstring → 自动作为子命令的 --help 描述
    cfg = ctx.obj
    use_case = build_register_use_case(cfg, strategy)
    summary = use_case.execute(dataset)
    # typer.echo(s) ≈ print(s)，但是 Click 推荐的"CLI 安全输出"——
    # 自动处理编码、stdout/stderr 路由、不输出 None 等
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
        # err=True → 输出到 stderr 而非 stdout（便于脚本 2> 重定向错误流）
        typer.echo(f"无法读取图片: {image}", err=True)
        # typer.Exit(code=N) = 优雅退出并返回 shell 退出码 N。
        # 不要用 sys.exit() 或 return——Typer 内部要做清理工作。
        raise typer.Exit(code=1)
    try:
        result = use_case.execute(img)
    except FaceRecognitionError as e:
        # 暴露错误 code 字段（Task 2 我们给每个领域异常类设了 code）
        typer.echo(f"识别失败 [{e.code}]: {e}", err=True)
        raise typer.Exit(code=2)
    if result.person_id is None:
        # f-string 的 :.4f 格式说明符 = "保留 4 位小数的浮点"。Task 1 已解释。
        typer.echo(f"未知人员（最高相似度 {result.similarity:.4f} < 阈值 {result.threshold}）")
    else:
        typer.echo(f"识别为: {result.person_id}（相似度 {result.similarity:.4f}）")


# 显式 name="list" 因为 list 是 Python 内置类型——不能直接拿 list 做函数名（会遮蔽内置）。
# 函数实际叫 list_persons，但 CLI 上是 list 子命令。
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
        # 字符串里多个空格只是为了对齐输出，不是格式化必需
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


# 标准入口守卫：只有"直接 python xxx.py 运行"才执行 app()，被 import 时不执行。
# __name__ 在直接执行时是 "__main__"，被 import 时是模块全名。
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


# 整文件标记为集成测试——默认 pytest 跑不到，要 pytest -m integration 显式触发
pytestmark = pytest.mark.integration


# 模块级 pipeline fixture，复用 Task 10 同款写法（避免每个测试都重新加载模型）
@pytest.fixture(scope="module")
def pipeline() -> InsightFacePipeline:
    return InsightFacePipeline(model_pack="buffalo_l", ctx_id=-1, det_size=(640, 640))


def _sample_image_path() -> Path:
    """定位 InsightFace 自带的样图。"""
    p = Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"
    if not p.exists():
        pytest.skip(f"InsightFace 自带示例图未找到: {p}")
    return p


def _populate_person_dir(target_dir: Path, sample: Path, n: int) -> None:
    """把同一张样图复制 n 份并加微小扰动，模拟"一个人的 n 张照片"。"""
    target_dir.mkdir()
    src = cv2.imread(str(sample))
    for i in range(n):
        # 给图加 0~5 的高斯白噪声，避免 KMeans 在"全部完全相同"上退化（数学上协方差 0 会让算法不稳）。
        # np.random.RandomState(i) = 旧 API 的随机数生成器，等价于 default_rng（第二次出现，不重复解释）
        # rand(*src.shape) = 解包 shape 元组作为参数，等于 rand(H, W, 3)
        # *0~5 后 .astype(np.uint8) → 噪声范围 [0, 5]
        noise = (np.random.RandomState(i).rand(*src.shape) * 5).astype(np.uint8)
        # cv2.add(a, b) = 饱和加法（像素超过 255 截断为 255，不像 + 会溢出回绕）
        # cv2.imwrite(path, img) = 写图（按扩展名自动选编码格式，.jpg 走 JPEG）
        cv2.imwrite(str(target_dir / f"{i:03d}.jpg"), cv2.add(src, noise))


def test_register_and_recognize_same_person(
    tmp_path: Path,
    pipeline: InsightFacePipeline,
):
    """完整流程：注册 alice 的 10 张照片 → 用其中一张查询 → 应识别为 alice。"""
    sample = _sample_image_path()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _populate_person_dir(dataset_dir / "alice", sample, n=10)

    repo = SqliteRepository(tmp_path / "test.db")
    register = RegisterFace(
        pipeline=pipeline,
        repository=repo,
        strategy=KMeansK3Strategy(seed=42),
        # lambda 一行匿名函数包 cv2.imread——避免依赖 dependencies.py 的 _load_bgr_image
        image_loader=lambda p: cv2.imread(str(p)),
    )
    register.execute(dataset_dir)
    assert len(repo.list_all()) == 1

    # 用同一张原图查询
    query_img = cv2.imread(str(sample))
    recognize = RecognizeFace(pipeline=pipeline, repository=repo, threshold=0.5)
    result = recognize.execute(query_img)
    assert result.person_id == "alice"
    # 同一张图（虽然加过噪声）查回去，相似度应远高于阈值——
    # 用 0.5 是粗略下限；真实场景同人不同照通常 0.6~0.9
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
uv run pytest tests/integration/test_register_recognize_e2e.py -v -m integration
```

预期：2 passed（约 30 秒，复用之前 task 10 已下载的模型）

- [ ] **Step 3: 跑全部测试做一次 sanity check**

```bash
uv run pytest -v
uv run pytest -v -m integration
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

- [ ] `uv run pytest -m "not integration"` 全部 pass（单元测试）
- [ ] `uv run pytest -m integration` 全部 pass（集成测试，含端到端；M0 时若 marker 名为 `gpu` 一并改成 `integration`）
- [ ] `uv run ruff check src tests` 0 错误
- [ ] `uv run mypy src/face_recognition` 0 错误
- [ ] CLI 四条命令（register / recognize / list / remove）手动跑通
- [ ] git 历史里至少 13 个 commit + 1 个 `m1-complete` tag
- [ ] `git push origin main --tags` 推送到 GitHub

完成后即可进入 **M2 评估框架** 的实施计划。
