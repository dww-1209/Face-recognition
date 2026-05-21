# 基于 ArcFace 的人脸识别系统 — 设计文档

> 期末项目设计 spec
> 日期：2026-05-21
> 范围：从需求到模块边界、数据流、错误处理、测试策略与交付物的端到端设计

---

## 0. 目录

1. [项目目标与范围](#1-项目目标与范围)
2. [关键决策快照](#2-关键决策快照)
3. [整体架构与目录结构](#3-整体架构与目录结构)
4. [核心抽象接口与实体](#4-核心抽象接口与实体)
5. [数据流（注册 / 识别 / 评估）](#5-数据流注册--识别--评估)
6. [错误处理与配置管理](#6-错误处理与配置管理)
7. [测试策略与项目交付物](#7-测试策略与项目交付物)
8. [方法论附录（用于答辩报告引用）](#8-方法论附录用于答辩报告引用)

---

## 1. 项目目标与范围

### 1.1 项目目标

构建一个**支持动态增删人员的开放集人脸识别系统**，演示形态为 **CLI + FastAPI + 后端摄像头实时识别**。

- **库内规模**：约 35 人（一个班级量级），每人至少 20 张照片
- **应用场景**：小型门禁 / 公司考勤 / 个人相册管理
- **核心能力**：注册（5 种模板生成策略消融对比）、识别（开放集，含未知拒识）、动态增删

### 1.2 项目非目标（明确不做）

- **不进行任何模型训练或微调**：完全使用 InsightFace 官方预训练模型 `buffalo_l`
- **不引入工业级向量数据库**（FAISS / Milvus / Qdrant）
- **不做小程序、PWA、Gradio**等额外前端形态
- **不做分布式 / 高可用 / 重试 / 降级**等过度工程
- **不做模型推理性能优化**（量化、TensorRT、ONNX Runtime）

### 1.3 成功标准

1. 系统能在浏览器中实时显示摄像头画面并对画面中所有人脸进行识别
2. 5 个模板生成策略的评估实验产出完整 ROC 曲线、EER、TAR\@FAR=1e-3 三组指标
3. 选定最优策略后，库内识别 TAR\@FAR=1e-3 ≥ 90%（保守目标）
4. 答辩演示时能现场新注册一名同学并立即被识别（动态增删能力）

---

## 2. 关键决策快照

| 维度 | 决定 |
| --- | --- |
| **交付形态** | CLI + FastAPI + 后端摄像头实时识别（OpenCV 直读 + WebSocket 推流） |
| **架构** | 清洁架构四层（domain / application / infrastructure / api） |
| **模型** | InsightFace `buffalo_l` 一站式（SCRFD 检测 + 关键点对齐 + ArcFace ResNet100 识别），权重冻结 |
| **硬件** | NVIDIA GPU 推理 |
| **向量数据库** | SQLite + numpy 暴力检索，向量以 BLOB 存储 |
| **数据来源** | 私有数据集（按文件夹分人，无标注） |
| **数据规模参考** | ~35 人 × 每人 ≥ 20 张照片 |
| **数据切分** | 按人 80/20，注册集 80% 进库、测试集 20% 评估，不打乱跨人 |
| **库外集** | 从 LFW 抽 50 人作"陌生人"，每人 1 张 |
| **多模板策略** | **5 策略消融实验**：① 随机1张  ② 全部平均  ③ 人工挑3张  ④ KMeans K=3  ⑤ 全量存 |
| **评估三元组** | Genuine + 库内 Impostor + 库外 Impostor |
| **核心指标** | ROC、EER、TAR\@FAR=1e-3、FAR/FRR；附 Top-1 闭集准确率 |
| **实时识别策略** | 每帧检测 + IoU 跟踪，识别按需触发（新框/未识别框） |
| **前端** | 单文件 HTML + JS + Tailwind/Pico CDN，FastAPI StaticFiles 托管 |

---

## 3. 整体架构与目录结构

### 3.1 架构原则

采用**清洁架构（Clean Architecture）四层**，依赖方向**只能从外向内**：

- `api` 依赖 `application`
- `application` 依赖 `domain`
- `infrastructure` 实现 `domain` 定义的接口

这保证 ArcFace、SQLite、InsightFace 这些"外部细节"哪天想换都不会影响核心业务逻辑。

### 3.2 目录树

采用 **src layout**（PyPA 推荐的现代标准）：所有源码集中在唯一顶层包 `face_recognition` 下，避免装包路径与源码路径分歧、避免顶层名字污染。
import 路径统一形如 `from face_recognition.domain.entities import FaceEncoding`。

```
face-recognition/                         # 仓库根
├── src/
│   └── face_recognition/                 # 唯一顶层包
│       ├── __init__.py
│       ├── domain/                       # 第1层：核心业务，零外部依赖
│       │   ├── entities.py               # FaceEncoding, Person, Template
│       │   ├── interfaces.py             # 抽象接口（Pipeline/Repo/Strategy）
│       │   └── errors.py                 # 领域异常层次
│       │
│       ├── application/                  # 第2层：用例编排
│       │   ├── register_face.py          # 注册流程用例
│       │   ├── recognize_face.py         # 识别流程用例
│       │   └── strategies/               # 5 个模板生成策略
│       │       ├── base.py               # TemplateStrategy 协议
│       │       ├── random_one.py         # 策略1
│       │       ├── mean_all.py           # 策略2
│       │       ├── manual_three.py       # 策略3
│       │       ├── kmeans_k3.py          # 策略4
│       │       └── all_vectors.py        # 策略5
│       │
│       ├── infrastructure/               # 第3层：具体实现
│       │   ├── insightface_pipeline.py   # buffalo_l 一站式（检测+对齐+特征）
│       │   ├── sqlite_repository.py      # 向量数据库
│       │   ├── camera_capture.py         # OpenCV VideoCapture 封装
│       │   ├── iou_tracker.py            # 实时识别帧间跟踪
│       │   └── config_loader.py          # 加载 config.yaml（pydantic-settings）
│       │
│       ├── api/                          # 第4层：入口
│       │   ├── cli.py                    # Typer CLI（register / recognize / eval）
│       │   ├── server.py                 # FastAPI 应用 + WebSocket
│       │   ├── dependencies.py           # 依赖注入装配
│       │   └── static/index.html         # 单文件前端
│       │
│       └── evaluation/                   # 评估实验（独立于四层）
│           ├── data_split.py             # 80/20 按人切分
│           ├── lfw_loader.py             # LFW 库外集加载
│           ├── pair_generator.py         # Genuine / 库内 / 库外 Impostor 配对
│           ├── metrics.py                # ROC, EER, TAR@FAR, FAR, FRR
│           └── run_ablation.py           # 主入口：5 策略 × 3 评估组
│
├── tests/                                # 测试（不打包）
│   ├── unit/                             # 各模块单测，用 mock
│   └── integration/                      # 端到端：注册 → 识别
│
├── data/                                 # 数据（.gitignore）
│   ├── private_dataset/                  # 私有数据，按文件夹分人
│   ├── lfw_subset/                       # LFW 抽取的 50 张陌生人
│   └── face.db                           # SQLite 文件
│
├── reports/                              # 评估输出（.gitignore，仅入仓 .gitkeep）
├── logs/                                 # 日志（.gitignore）
├── scripts/                              # 演示脚本
│
├── docs/superpowers/{specs,plans}/       # 设计文档与实施计划
│
├── config.yaml                           # 全部可调参数
├── pyproject.toml                        # 含 [tool.hatch.build.targets.wheel]
├── uv.lock                               # 依赖锁文件
├── CLAUDE.md                             # AI 协作指令
└── README.md
```

### 3.3 几个关键设计说明

1. **`domain/interfaces.py` 是架构的"心脏"**：所有抽象接口集中在这里，让 application 层完全不知道下层用的是 InsightFace 还是别的库。

2. **`infrastructure/insightface_pipeline.py` 把检测、对齐、特征提取打包成一个对象**：因为 buffalo_l 的 `app.get(img)` 一行就出全部结果，没必要拆成三个独立类。这个类同时实现 `FaceDetector` 和 `FaceEncoder` 行为。

3. **`evaluation/` 独立于四层**：评估实验是离线脚本，不是产品功能。它复用 infrastructure 层的 SQLite 仓库和管线，但有自己的数据切分、配对、指标代码。

4. **`api/dependencies.py` 是依赖装配点**：所有"哪个具体实现注入哪个用例"的逻辑都在这里。CLI 和 FastAPI 共用同一份装配。

5. **没有 `services/` / `utils/` / `helpers/` 万能桶**：清洁架构里这些一般是"职责不清"的信号。

---

## 4. 核心抽象接口与实体

### 4.1 实体（`domain/entities.py`）

```python
from dataclasses import dataclass
from datetime import datetime
import numpy as np

@dataclass(frozen=True)
class FaceEncoding:
    """ArcFace 输出的 512 维向量，已 L2 归一化。"""
    vector: np.ndarray              # shape=(512,), dtype=float32, ||v||=1
    model_version: str              # "buffalo_l/2024-xx" 用于将来换模型时识别

    def cosine_similarity(self, other: "FaceEncoding") -> float:
        # 因为已归一化，余弦相似度 = 点积，范围 [-1, 1]
        return float(np.dot(self.vector, other.vector))

@dataclass(frozen=True)
class Template:
    """单条模板向量（一个 Person 可能有多条 Template）。"""
    encoding: FaceEncoding
    source: str                     # "kmeans_centroid_0" / "raw_photo_3.jpg" / "mean"
    created_at: datetime

@dataclass(frozen=True)
class Person:
    """库内人员（领域聚合根）。"""
    person_id: str                  # 文件夹名直接当 ID
    display_name: str               # 展示名，前端用
    templates: tuple[Template, ...] # 不可变；策略不同模板数不同（1/3/40 等）

@dataclass(frozen=True)
class RecognitionResult:
    """识别用例的输出。"""
    person_id: str | None           # None 表示未识别
    similarity: float               # 与最匹配模板的相似度
    threshold: float                # 当时使用的阈值
```

### 4.2 接口（`domain/interfaces.py`）

用 `typing.Protocol` 而非 `abc.ABC`——更轻量，不需要继承。

```python
from typing import Protocol
import numpy as np

class FacePipeline(Protocol):
    """一站式：图 → 检测 → 对齐 → 编码。对应 buffalo_l 的 app.get()"""
    def encode(self, image: np.ndarray) -> list[FaceEncoding]:
        """返回图中所有人脸的编码（可能 0 个或多个）"""

    def encode_single(self, image: np.ndarray) -> FaceEncoding:
        """要求图中恰好 1 张脸；0 张或多张抛 NoFaceError / MultipleFacesError"""

class PersonRepository(Protocol):
    """向量库的抽象。SQLite 实现位于 infrastructure/。"""
    def add(self, person: Person) -> None: ...
    def get(self, person_id: str) -> Person | None: ...
    def remove(self, person_id: str) -> None: ...
    def list_all(self) -> list[Person]: ...
    def all_templates_matrix(self) -> tuple[np.ndarray, list[str]]:
        """返回 (M, 512) 矩阵 + 长度 M 的 person_id 列表"""

class TemplateStrategy(Protocol):
    """5 种策略的统一接口。"""
    name: str
    def build(self, encodings: list[FaceEncoding]) -> list[Template]: ...
```

### 4.3 关键设计决策

1. **`FaceEncoding.vector` 始终 L2 归一化**：写进类的不变量。归一化后余弦相似度 = 点积，整个系统所有相似度计算都是 `np.dot`，永远不需要除范数。

2. **`Person` 用 `tuple[Template, ...]` 而非 `list`**：dataclass `frozen=True` 配不可变容器，让实体真正不可变。

3. **`PersonRepository.all_templates_matrix()` 返回矩阵而非逐人迭代**：识别用例需要的是一次性矩阵乘法 `M @ q`，一次得到 N 个相似度。

4. **`TemplateStrategy` 是 Protocol 而非基类**：5 个策略各写一个简单类，不需要继承体系。

5. **错误用领域异常表达**：见 [§6.2](#62-领域异常层次-domainerrorspy)。

6. **明确不做的抽象**：
   - 没有 `Detector` 和 `Encoder` 两个独立接口（buffalo_l 一站式）
   - 没有 `SimilarityCalculator` 接口（余弦相似度就是 numpy 一行）
   - 没有 `Photo` 实体（图像就是 `np.ndarray`）

---

## 5. 数据流（注册 / 识别 / 评估）

### 5.1 流程一：注册（一次性，离线批量）

```
用户调用: python -m face_recognition.api.cli register --strategy kmeans_k3 --dataset data/private_dataset/
                                          ↓
api.cli.register_command(strategy_name, dataset_dir)
                                          ↓
dependencies.py 装配 RegisterFace 用例（注入 InsightFacePipeline + SqliteRepo + 选定 Strategy）
                                          ↓
RegisterFace.execute(dataset_dir):
   for person_dir in os.listdir(dataset_dir):
      person_id = person_dir.name
      encodings = []
      for img_path in person_dir.glob("*.jpg"):
         img = cv2.imread(img_path)
         try:
             enc = pipeline.encode_single(img)
             encodings.append(enc)
         except (NoFaceError, MultipleFacesError) as e:
             logger.warning(f"{img_path}: {e}, 跳过")

      if len(encodings) == 0:
          logger.error(f"{person_id} 全部照片无法识别人脸，跳过此人")
          continue

      templates = strategy.build(encodings)              # 5 选 1
      person = Person(person_id, display_name=person_id, templates=tuple(templates))
      repo.add(person)
```

**关键设计点：**
- 注册**幂等**：`repo.add` 内部如果 person_id 已存在，先删旧再插新
- **单张失败不影响整体**：人脸检测失败、双脸、模糊只记 warning 跳过
- 5 个策略**只在这一步切换**：换策略只换 `--strategy` 参数

### 5.2 流程二：实时识别（持续运行）

```
启动: uvicorn api.server:app
浏览器打开: http://localhost:8000

后端启动时：
  - 加载 buffalo_l 模型到 GPU
  - 加载 SQLite 所有模板到内存矩阵 (M, 512)
  - 启动 OpenCV 摄像头采集
  - 启动 WebSocket 端点 /ws/stream

每帧（30 FPS）：
  采集线程: cap.read() → frame
  检测线程（每帧都跑）:
       boxes, kps = pipeline.detect(frame)
  IoUTracker.update(boxes):
       新框：分配 track_id, 标记需要识别
       沿用框：track_id 不变，复用上次身份
  仅对"需要识别"的框：
       enc = pipeline.encode_aligned(...)
       similarities = M @ enc.vector              # 一次矩阵乘法
       best_idx = argmax(similarities)
       if similarities[best_idx] >= threshold:
           track.identity = repo_index[best_idx]
       else:
           track.identity = None
  渲染：在 frame 上画框 + 写名字
  WebSocket 推 JPEG 给所有连接的浏览器
```

**教学性说明：**
1. **"识别按需"省算力**：1 个人在镜头前 5 秒，30 FPS × 5 = 150 帧。每帧都识别要 150 次。用 IoU 跟踪后只第 1 帧识别，后 149 帧复用——从 150 降到 1。
2. **threshold 来源**：评估实验最优工作点写入 `config.yaml`，不硬编码。
3. **(M, 512) 矩阵内存**：策略5全量约 1400 × 512 × 4 = 2.7MB，毫无压力。

### 5.3 流程三：5 策略消融评估（离线一次跑完）

```
python -m face_recognition.evaluation.run_ablation --dataset data/private_dataset/ --lfw data/lfw_subset/

run_ablation():
  # Step 1: 数据切分（只切一次，5 策略复用）
  train_set, test_set = data_split.split_by_person(dataset, ratio=0.8)

  # Step 2: 配对一次（5 策略复用）
  genuine_pairs       = pair_generator.gen_genuine(test_set, train_set)
  closed_impostor     = pair_generator.gen_closed_impostor(test_set, train_set)
  open_impostor       = pair_generator.gen_open_impostor(lfw_set, train_set)

  results = {}
  for strategy in [RandomOne, MeanAll, ManualThree, KMeansK3, AllVectors]:
      # Step 3: 该策略重建模板库（in-memory）
      templates_db = build_templates(train_set, strategy)

      # Step 4: 三组分数
      genuine_scores = [max_similarity(q, templates_db[p]) for q, p in genuine_pairs]
      closed_scores  = [max_similarity(q, templates_db[p]) for q, p in closed_impostor]
      open_scores    = [max_similarity(q, templates_db[p]) for q, p in open_impostor]

      # Step 5: 指标
      results[strategy.name] = {
          "EER": compute_eer(genuine_scores, closed_scores + open_scores),
          "TAR@FAR=1e-3": compute_tar_at_far(...),
          "ROC_curve": compute_roc(...),
          "Top-1": compute_top1(...),
      }

  reports.save_csv(results)         # 表格
  reports.save_roc_plot(results)    # 5 曲线叠图
  reports.save_markdown(results)    # 含决策建议
```

**关键设计点：**
- `RANDOM_SEED = 42` 写死，可复现
- 配对只生成一次，5 策略对比公平
- `max_similarity(query, templates)` 是统一抽象（多模板取最大值就是"多模板"的精髓）
- 评估**不污染生产数据库**

---

## 6. 错误处理与配置管理

### 6.1 三层错误处理哲学

| 层级 | 哲学 | 例子 |
| --- | --- | --- |
| **domain** | 抛领域异常，不做任何处理 | `NoFaceError`、`PersonNotFoundError` |
| **application** | 决定哪些错误吞下、哪些传出 | 注册时单张照片无脸→吞；用例输入参数错→传 |
| **api** | 把领域异常映射成用户能看懂的形式 | CLI 中文 + 退出码；HTTP 4xx/5xx + JSON |

### 6.2 领域异常层次（`domain/errors.py`）

```python
class FaceRecognitionError(Exception):
    """所有领域异常的基类。"""
    code: str

class NoFaceError(FaceRecognitionError):              code = "NO_FACE"
class MultipleFacesError(FaceRecognitionError):       code = "MULTIPLE_FACES"
class PersonNotFoundError(FaceRecognitionError):      code = "PERSON_NOT_FOUND"
class DuplicatePersonError(FaceRecognitionError):     code = "DUPLICATE_PERSON"
class LowConfidenceError(FaceRecognitionError):       code = "LOW_CONFIDENCE"
class PersonHasNoTemplatesError(FaceRecognitionError):code = "NO_TEMPLATES"
class CameraDisconnectedError(FaceRecognitionError):  code = "CAMERA_LOST"
```

### 6.3 三类典型错误的处理

**1. 注册批量遇到坏照片** —— 容忍单张失败，整人失败才中断当前人。最终 CLI 输出汇总：`成功 33 人 / 跳过 2 人；共处理 1875 张照片，跳过 47 张`。

**2. 实时识别摄像头断开** —— 让上层决定。采集线程抛 `CameraDisconnectedError`，FastAPI 捕获后给 WebSocket 推 `{"type": "error", "code": "CAMERA_LOST"}`，前端弹提示。

**3. HTTP 接口统一异常处理**：

```python
@app.exception_handler(FaceRecognitionError)
async def domain_error_handler(request, exc: FaceRecognitionError):
    status_code = ERROR_TO_HTTP.get(exc.code, 500)
    return JSONResponse(status_code=status_code,
        content={"error_code": exc.code, "message": str(exc)})

ERROR_TO_HTTP = {
    "NO_FACE": 422, "MULTIPLE_FACES": 422,
    "PERSON_NOT_FOUND": 404, "DUPLICATE_PERSON": 409,
    "LOW_CONFIDENCE": 200,
}
```

### 6.4 配置管理（`config.yaml`）

```yaml
model:
  pack: "buffalo_l"
  ctx_id: 0                       # GPU; -1=CPU
  det_size: [640, 640]

recognition:
  threshold: 0.45                 # 占位；评估后写入
  template_strategy: "kmeans_k3"

camera:
  device_index: 0
  resolution: [1280, 720]
  fps: 30

realtime:
  detect_every_n_frames: 1
  recognize_on_new_track: true
  iou_threshold: 0.5
  track_max_missing_frames: 15

api:
  host: "0.0.0.0"
  port: 8000

data:
  sqlite_path: "data/face.db"
  dataset_root: "data/private_dataset"
  lfw_subset: "data/lfw_subset"

evaluation:
  random_seed: 42
  train_ratio: 0.8
  far_targets: [0.001, 0.01, 0.1]

logging:
  level: "INFO"
  file: "logs/face.log"
```

加载用 `pydantic-settings`：字段类型校验、字段说明都有，配置错误启动就报错。

### 6.5 明确不做的事

- **没有重试逻辑**：本地系统、单机部署，YAGNI
- **没有"降级模式"**：模型加载失败就启动失败，不要造祸根
- **`threshold` 由评估决定**：写进 config 之前是占位值
- **日志不结构化**：`logging.basicConfig` + 文件输出足够

---

## 7. 测试策略与项目交付物

### 7.1 测试金字塔

```
            ┌─────────────────┐
            │  端到端 (3~5)    │   注册→识别→评估全链路冒烟
            ├─────────────────┤
            │  集成 (10+)     │   真模型 + 真 SQLite，不联网
            ├─────────────────┤
            │  单元 (30+)     │   纯逻辑，全 mock，毫秒级
            └─────────────────┘
```

### 7.2 单元测试重点：5 策略 + ROC/EER

```python
def test_kmeans_k3_returns_three_normalized_templates():
    encs = make_random_encodings(40)
    templates = KMeansK3Strategy().build(encs)
    assert len(templates) == 3
    assert all(np.isclose(np.linalg.norm(t.encoding.vector), 1.0) for t in templates)

def test_mean_all_returns_normalized_centroid():
    encs = [random_unit_vector() for _ in range(50)]
    [tpl] = MeanAllStrategy().build(encs)
    assert np.isclose(np.linalg.norm(tpl.encoding.vector), 1.0)

def test_random_one_is_deterministic_with_seed():
    encs = make_random_encodings(50)
    s1 = RandomOneStrategy(seed=42).build(encs)
    s2 = RandomOneStrategy(seed=42).build(encs)
    assert s1[0].encoding.vector.tobytes() == s2[0].encoding.vector.tobytes()

def test_eer_on_synthetic_data():
    genuine  = np.random.normal(0.7, 0.1, 1000)
    impostor = np.random.normal(0.3, 0.1, 1000)
    eer, threshold = compute_eer(genuine, impostor)
    assert 0.45 < threshold < 0.55
    assert eer < 0.05
```

**不写**"`encode_single` 在某图返回 512 维"这种测试——那是测 InsightFace 库本身。

### 7.3 集成测试

```python
def test_register_then_recognize_same_person(tmp_path, real_pipeline):
    repo = SqliteRepository(tmp_path / "test.db")
    register = RegisterFace(real_pipeline, repo, KMeansK3Strategy())
    register.execute_for_person("alice", alice_imgs)

    recognize = RecognizeFace(real_pipeline, repo, threshold=0.45)
    result = recognize.execute(alice_test_img)
    assert result.person_id == "alice"
```

集成测试覆盖：注册→识别同人、注册→查未知库外、删除→查不到、阈值边界附近行为。

### 7.4 端到端冒烟测试（3 个）

1. `test_cli_register_then_recognize.py`：subprocess 调 CLI，全链路
2. `test_evaluation_pipeline_runs.py`：mini 数据集跑完 5 策略，断言生成 ROC 图、CSV
3. `test_fastapi_health.py`：起服务，请求 `GET /api/persons`，断言 200

WebSocket 实时识别端到端不写自动化测试，列入手工验证清单。

### 7.5 手工验证清单（README）

| 场景 | 预期 |
| --- | --- |
| 浏览器打开首页 | 摄像头画面，连接绿色 |
| 库内人员入镜 | 框上显示姓名，相似度 > 阈值 |
| 库外陌生人入镜 | 框上显示"未知" |
| 多人同时入镜 | 各自识别，不串号 |
| 切出视野再回来 | 同一 track，身份延续 |
| API：POST /persons 注册 | 数据库 + 内存矩阵都更新 |
| API：DELETE /persons/{id} | 该人立刻无法识别 |

### 7.6 项目交付物清单

1. **代码仓库**：完整 `face-recognition/`
2. **数据**：私有数据集（`.gitignore`）+ LFW 子集获取脚本
3. **预训练模型**：`insightface` 包自动下载 `buffalo_l`
4. **配置文件**：`config.yaml`（含最终阈值）
5. **报告材料**：
   - `docs/superpowers/specs/2026-05-21-face-recognition-design.md` 本设计
   - `evaluation/reports/ablation_results.csv` 5 策略 × 各指标
   - `evaluation/reports/roc_curves.png` 5 条 ROC 曲线叠图
   - `evaluation/reports/threshold_analysis.png` Genuine/Impostor 分布直方图
   - `evaluation/reports/summary.md` 评估结论与策略推荐
6. **演示脚本**：`scripts/demo.sh`
7. **README**：装环境、跑评估、起 demo 的命令

### 7.7 项目里程碑

| 阶段 | 内容 | 估时 |
| --- | --- | --- |
| **M1** | 搭骨架：domain + infrastructure + 基础 CLI 注册识别 | 2~3 天 |
| **M2** | 评估代码：数据切分 + 配对 + 5 策略 + 指标 + 报告 | 3 天 |
| **M3** | 跑评估实验，确定最优策略与阈值，**写进 config** | 0.5 天 |
| **M4** | FastAPI + 实时识别 + IoU 跟踪 + WebSocket | 2 天 |
| **M5** | 单文件前端 HTML + 联调 | 1 天 |
| **M6** | 文档 + 演示脚本 + 答辩 PPT 数据图表 | 1~2 天 |

**M3 是关键节点**：评估实验决定最优策略与阈值，之后实时识别才有数可用。

---

## 8. 方法论附录（用于答辩报告引用）

> 本章节专为期末报告写作准备，把实验设计的"为什么"用学术语言写清楚。

### 8.1 ArcFace 在本系统中扮演的角色

ArcFace 是 2019 年提出的**深度人脸识别损失函数**（Additive Angular Margin Loss），其核心思想是在分类训练时对决策边界施加角度间隔，强制模型把同一人的特征向量在 512 维超球面上聚拢、不同人的特征向量推远。

**关键认识**：训练完成后，ArcFace 模型推理时**只取主干网络的输出（512 维特征向量）**，分类头被丢弃。这意味着：

1. **零训练**：本项目使用 InsightFace 官方在 Glint360K（约 36 万人脸 ID、1700 万张照片）上训练好的 `buffalo_l`，权重完全冻结。
2. **零分类头限制**：注册一个新人不需要"加一个输出节点重新训练"，只需把他的照片喂给模型拿 512 维向量存入数据库即可。这正是系统**支持动态增删**的根本原因。
3. **相似度 = 余弦相似度**：两张脸的相似度就是它们 512 维向量的点积（向量已 L2 归一化）。

### 8.2 多模板策略消融实验设计

#### 8.2.1 为什么要对比多种策略

每人有数十张照片可用于注册，"如何把这些照片转化为存入数据库的模板向量"有多种合理选择。不同选择在**识别准确率、检索效率、存储开销、对噪声照片鲁棒性**上各有取舍。学术规范的做法是用消融实验逐一对比，用数据决定最优。

#### 8.2.2 5 个策略

| 策略 | 注册方式 | 模板数/人 |
| --- | --- | --- |
| 1 | 随机选 1 张照片 | 1 |
| 2 | 全部照片取平均（再 L2 归一化） | 1 |
| 3 | 人工挑 3 张（正光/侧光/逆光） | 3 |
| 4 | KMeans 聚类 K=3，取质心 | 3 |
| 5 | 全部照片直接存（多向量检索） | 全部 |

#### 8.2.3 多维度对比表

| 维度 | 策略1 | 策略2 | 策略3 | 策略4 | 策略5 |
| --- | --- | --- | --- | --- | --- |
| 识别准确率 | 低 | 中 | 中-高 | 高 | 最高 |
| 检索耗时 | 最快 | 最快 | 快 | 快 | 慢 ~50× |
| 数据库大小 | 1× | 1× | 3× | 3× | 全部× |
| 对噪声照片鲁棒性 | 差 | 中 | 好 | 好 | 差 |
| 自动化程度 | 高 | 高 | 低（需人工） | 高 | 高 |
| **未知人员拒识率（FAR）** | 偏高 | 中 | 中 | 低 | 偏高 |

**为什么策略 5（全量存）拒识率反而偏高**：库里照片越多，库外陌生人只要凑巧和某张噪声照片相似就会被误识为对应员工——这是"全量存"的隐藏短板，能体现思考深度。

### 8.3 开放集人脸识别评估方法论

#### 8.3.1 闭集 vs 开放集

- **闭集识别**：测试样本必属于库内 N 人之一，问题是"是 N 人中哪个"。指标：Top-1 准确率。
- **开放集识别**：测试样本可能根本不在库内（如门禁场景的陌生人），问题是"是 N 人中某个，还是谁都不是"。指标：FAR/FRR/EER/ROC。

本项目定位为开放集识别，因此**仅报告 Top-1 准确率不充分**——必须用 ROC 体系评估。

#### 8.3.2 数据切分（按 80/20 比例）

每人的照片数量不一定相同（私有数据集本身分布不均），统一按 80/20 比例切分：

```
某人 N 张照片（N ≥ 20）
├── 注册集 (80% × N) ──→ 用于生成模板向量（5 策略各自的输入）
└── 测试集 (20% × N) ──→ 用于评估，绝不进入数据库
```

切分使用固定 `RANDOM_SEED = 42`，保证实验可复现。

测试集**不进入数据库**是关键原则，否则就是"考前给答案"，准确率会虚高。

#### 8.3.3 三组配对的物理含义

**1. Genuine 对（应被识别为同一人）**
- 同一人的测试照 vs 自己的模板
- 比对数 = 库内人员数 × 每人测试照数（量级 10²~10³）
- 得到一组"同人相似度分数"，理论上应高（接近 1.0）

**2. 库内 Impostor 对（应被识别为不同人）**
- 张三的测试照 vs 李四/王五……的模板
- 比对数 = 库内人员数 × 每人测试照数 ×（库内人员数 − 1）（量级 10⁴）
- 得到一组"异人相似度分数"，理论上应低
- **核心评估目标**：系统会不会把张三认成李四（**身份混淆**）

**3. 库外 Impostor 对（应被全部拒识）**
- LFW 抽取的 50 个库外陌生人 vs 库内所有模板
- 比对数 = 50 × 库内人员数（量级 10³）
- **核心评估目标**：陌生人冒充员工的能力（**冒充攻击**）

库内和库外 Impostor 评估的是**两种不同的失败模式**，缺一不可。

#### 8.3.4 阈值与指标

阈值 τ 是夹在 Impostor 和 Genuine 分数分布之间的判定线：

```
    Impostor 分数分布          Genuine 分数分布
   ┌──────────┐              ┌──────────┐
   │   低     │              │    高    │
───┴──────────┴──────┬───────┴──────────┴───→  相似度
                     τ
```

- **τ 设太低** → 陌生人被误认成员工（FAR 高，假接受）
- **τ 设太高** → 真员工被拒识（FRR 高，假拒绝）

**核心指标**：
- **EER（Equal Error Rate）**：FAR == FRR 的工作点，常用平衡指标
- **TAR\@FAR=1e-3**：把 FAR 卡到 0.1%，看真员工接受率，工业界标准报法
- **ROC 曲线**：FAR 为横轴、TAR (=1−FRR) 为纵轴，曲线越靠左上越好

5 个策略各画一条 ROC 叠加在同一张图上 → 报告核心图表。

### 8.4 实时识别工程优化思想

朴素方案"每帧都跑完整识别管线"在实时场景下会出现**身份抖动**——同一张脸因光照微变可能这帧识别为张三、下一帧"未知"。

工程改进**IoU 跟踪 + 识别按需**：
1. 每帧只跑检测（快）
2. 用 IoU 匹配前后帧的人脸框 → 同一人的连续观测保持同一 `track_id`
3. 只在 track 首次出现或长时间未识别时跑一次识别管线（慢）
4. 同一 track 内沿用上次识别结果

这样：30 FPS × 5 秒站镜头 = 150 帧 → 朴素方案识别 150 次，优化方案识别 1 次。算力降至 1/150，画面身份不抖动。

### 8.5 系统架构选型

采用**清洁架构（Clean Architecture）四层**：

- 依赖方向只能从外向内：api → application → domain，infrastructure 实现 domain 接口
- domain 层零外部依赖（不 import insightface、不 import sqlite）
- 切换 InsightFace 为其他模型、切换 SQLite 为 FAISS，只需改 infrastructure 层一处文件

**对答辩的意义**：当被问"如果以后规模扩大要换 FAISS 怎么办"，可以直接回答"换 `infrastructure/sqlite_repository.py` 一个文件即可，application 层和 api 层不需任何修改"——这是清洁架构的直接价值。

---

## 文档版本

| 日期 | 变更 |
| --- | --- |
| 2026-05-21 | 初版 |
