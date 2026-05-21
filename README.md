# 基于 ArcFace 的人脸识别系统

> 本科期末项目 · 开放集人脸识别 · 支持动态增删人员

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 简介

一个**开放集人脸识别系统**，基于 InsightFace 官方预训练模型 `buffalo_l`（ResNet100 + ArcFace），无需任何模型训练。系统支持：

- **动态增删人员**：注册新人只需提取特征向量入库，无需重训
- **5 种模板策略消融对比**：随机选取 / 全部平均 / 人工挑选 / KMeans 聚类 / 全量存储
- **开放集识别**：能区分库内人员与库外陌生人（基于阈值的拒识）
- **实时识别**：FastAPI + OpenCV + WebSocket，浏览器实时查看摄像头识别结果
- **完整学术评估**：ROC 曲线、EER、TAR\@FAR=1e-3、FAR/FRR 等指标

## 技术栈

| 组件 | 技术选型 |
| --- | --- |
| 人脸检测 + 对齐 + 特征 | [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l`（SCRFD + ArcFace ResNet100） |
| 向量数据库 | SQLite + numpy |
| Web 服务 | FastAPI + WebSocket |
| 摄像头 | OpenCV `VideoCapture` |
| 前端 | 单文件 HTML + Pico.css |
| 配置管理 | YAML + pydantic-settings |
| 包管理 | [uv](https://github.com/astral-sh/uv) |
| Python | 3.12 |

## 架构

采用 **src layout**，所有源码归集在 `src/face_recognition/` 顶层包下；
内部按清洁架构（Clean Architecture）分四层，依赖方向只能从外向内：

```
api ──→ application ──→ domain
              ↑              ↑
              └──── infrastructure（实现 domain 中的 Protocol）
```

```
face-recognition/
├── src/face_recognition/
│   ├── domain/         # 第1层：核心业务，零外部依赖（实体 + 接口 + 异常）
│   ├── application/    # 第2层：用例编排（注册 / 识别）+ 5 种模板策略
│   ├── infrastructure/ # 第3层：InsightFace / SQLite / 摄像头 / IoU 跟踪
│   ├── api/            # 第4层：Typer CLI + FastAPI / WebSocket
│   └── evaluation/     # 离线评估实验（独立于四层）
├── tests/{unit,integration}/
├── data/               # 数据集（gitignore）
├── reports/            # 评估输出（gitignore）
├── docs/superpowers/{specs,plans}/
├── config.yaml         # 全部可调参数
└── pyproject.toml
```

## 为什么这样组织 —— 架构决策复习手册

> 这一节解释每个结构选择背后的"为什么"，便于以后复习与答辩讲解。
> 任何工程结构都不是品味问题——它是为解决某些具体问题而存在的。

### 1. 为什么用 `src/` layout（而不是把代码摊在根目录）

**问题**：如果 Python 包就放在项目根目录，运行 `python` 或 `pytest` 时根目录会**自动加进 `sys.path`**，于是 `import face_recognition` 找的永远是源码本身——你从来没有真正测试过"装好的包"能不能跑。

**后果**：等你换台机器、换台同学、换个 CI 环境，`pip install` 装出来的包可能因为打包配置漏配（比如忘了把 `data/*.json` 列进 `package_data`）而炸——但你本地永远不会发现。

**src layout 的解法**：项目根目录下没有 `face_recognition/`，只有 `src/face_recognition/`。`sys.path` 自动机制不会找到它，你必须 `pip install -e .`（或 `uv sync` 自动做这件事）才能 import 成功。这强制让"开发环境"和"装包后环境"走同一条路径，打包配置错了你**自己**就会先发现。

**这是 PyPA（Python 官方包装组织）推荐的现代标准**。FastAPI、Pydantic、Black、Anthropic SDK 等主流项目都用 src layout。

### 2. 为什么 src 下又包一层 `face_recognition/`

如果直接 `src/{domain,application,api,...}` 这种五个并列顶层包的写法，会有三个问题：

- **命名空间污染**：`domain` / `api` 这种通用名很容易和别的库重名，import 时随机命中
- **装包路径混乱**：`pip install` 把这五个名字全部塞进 `site-packages`，让你的项目"占领"了五个全局名字
- **跨项目冲突**：未来你做第二个项目也想叫 `domain`，两个项目装在同一环境里就互相覆盖

**单一顶层包 `face_recognition`**让你的代码全部归属于一个唯一的命名空间，干净、隔离、不与他人冲突。这是工业项目的默认做法。

### 3. 为什么用清洁架构（Clean Architecture）四层

朴素写法是把所有代码扔在一个 `main.py` 或几个 `*.py` 里，"能跑就行"。这对小项目可以，但带来三个长期问题：

- **想换技术栈就要改满代码**：比如想从 SQLite 换成 FAISS，朴素写法里"调用 SQLite"的代码可能散落在 20 个文件里
- **测试困难**：业务逻辑和外部依赖（模型、数据库、文件系统）耦合在一起，单元测试需要把所有外部依赖一起启动
- **新人理解成本高**：没有清晰的边界，新人接手要把整个系统读完才能改一行代码

**清洁架构通过依赖方向规则解决这些问题**：

```
api ──→ application ──→ domain
              ↑              ↑
              └──── infrastructure（实现 domain 中的 Protocol）
```

- **依赖只能从外向内**：`api` 可以调 `application`，`application` 可以调 `domain`，但反过来不行
- **`domain/` 层零外部依赖**：里面没有 `import insightface`、没有 `import sqlite3`、没有 `import fastapi`——纯粹的业务概念（一个 `FaceEncoding` 是什么、一个 `Person` 由什么构成）
- **`infrastructure/` 实现 `domain/` 中定义的接口（Protocol）**：把"具体技术"隔离在最外层

**这样换技术栈只需改 infrastructure 层一处**——例如未来想从 SQLite 换成 FAISS，只改 `sqlite_repository.py` 一个文件，`application` 和 `domain` 层一行不动。这是清洁架构最直接的回报。

### 4. 各层的具体职责

| 层 | 它知道什么 | 它不知道什么 | 例子 |
| --- | --- | --- | --- |
| **domain** | 业务概念（人脸编码、人员、模板） | 怎么提取特征、怎么存数据库、怎么响应 HTTP | `FaceEncoding`、`Person`、`PersonRepository`（Protocol） |
| **application** | 用例流程（注册一个人 = 提特征→生成模板→存库） | 用什么模型提特征、用什么数据库存 | `RegisterFace.execute()`、5 个 `TemplateStrategy` |
| **infrastructure** | 具体技术细节（怎么调 InsightFace、怎么写 SQLite、怎么读摄像头） | 业务流程怎么编排 | `InsightFacePipeline`、`SqliteRepository` |
| **api** | 怎么暴露给用户（命令行参数、HTTP 端点、WebSocket 帧格式） | 业务规则、数据存储 | `cli.py`、`server.py` |

**判断一段代码该放哪一层**：问自己"如果我把 InsightFace 换成 FaceNet，这段代码要不要改？"——要改的属于 infrastructure；不用改的属于 application 或 domain。

### 5. 为什么 `evaluation/` 独立于四层

评估实验是**离线的科研行为**，不是产品功能。它有自己的数据切分、配对、指标计算逻辑，不在用户使用流程里。

把它放进 `application/` 会污染产品代码（用户运行 CLI 注册时不应加载评估代码）；放进 `infrastructure/` 在概念上不对（它不是"具体技术实现"）。所以单独一个目录，复用 infrastructure 的仓库和 pipeline，但有自己独立的脚本入口。

**这种"主线产品 + 离线实验"的并行结构**在工业 ML 项目里非常常见：训练 / 评估 / 数据生成等 pipeline 通常都不属于线上服务的四层。

### 6. 为什么 `config.yaml` 独立

把可调参数（识别阈值、GPU 卡号、跟踪窗口、数据路径）从代码里抽出来，集中到一个 YAML 文件：

- **改阈值不改代码**：评估完发现最优阈值是 0.42，改 yaml 一行就行，`git diff` 干净
- **不同环境不同配置**：开发用 CPU、部署用 GPU，复制一份 `config.local.yaml` 即可
- **可复现性**：`random_seed: 42` 写在 yaml 里，"我用这个种子跑出来的结果"是论文级承诺
- **启动校验**：用 `pydantic-settings` 加载，类型不对启动就崩溃，不留运行时雷

**核心原则**：代码里**绝不出现**像 `0.45`、`"data/face.db"`、`(640, 640)` 这种魔数 / 硬编码。

### 7. 为什么 `tests/` 在项目根而不是包内

测试代码不应该被 `pip install` 一起装到用户机器上。放在仓库根目录但**不在 `src/` 里**，自然就不会被打包进发行版。这是社区惯例。

### 8. 为什么 `data/` 和 `reports/` 在根目录而不在包里

它们是**运行时文件**：数据集会变、报告会重生成。不应当作"源码"对待，所以不进 `src/`。同时 `.gitignore` 屏蔽内容（只入仓 `.gitkeep` 占位），避免把私人照片或临时报告 push 上去。

### 9. 一句话总结这套架构

> **干净的边界 + 明确的依赖方向 + 配置外置**——让代码可读、可测、可换、可复现。
> 任何看起来"多余"的层级或文件，背后都对应一个具体可以避免的工程灾难。

---

## 快速开始

### 环境要求

- Python 3.12
- NVIDIA GPU（CUDA），CPU 也能跑但实时识别会卡
- macOS / Linux / Windows

### 安装

```bash
# 安装 uv（如果还没装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆仓库
git clone https://github.com/<your-username>/face-recognition.git
cd face-recognition

# 创建虚拟环境并装依赖（uv 会自动选 Python 3.12）
uv sync
```

### 准备数据

```
data/
├── private_dataset/       # 自己的数据集，按文件夹分人
│   ├── alice/
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   └── bob/
│       └── ...
└── lfw_subset/            # LFW 抽样作为库外陌生人（脚本待加）
```

### 注册人员

```bash
uv run python -m face_recognition.api.cli register \
    --strategy kmeans_k3 \
    --dataset data/private_dataset/
```

### 跑评估实验（5 策略消融）

```bash
uv run python -m face_recognition.evaluation.run_ablation \
    --dataset data/private_dataset/ \
    --lfw data/lfw_subset/
```

输出会写入 `reports/`：
- `ablation_results.csv`：5 策略 × 各指标的对比表
- `roc_curves.png`：5 条 ROC 曲线叠图
- `summary.md`：评估结论与策略推荐

### 启动 Web 实时识别

```bash
uv run uvicorn face_recognition.api.server:app --host 0.0.0.0 --port 8000
```

浏览器打开 http://localhost:8000 即可看到摄像头实时识别画面。

## 测试

```bash
uv run pytest                            # 全部
uv run pytest tests/unit/                # 单元（毫秒级）
uv run pytest tests/integration/         # 集成（需 GPU + 模型）
```

## 项目里程碑

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| M1 | 核心 CLI（注册 / 识别 / SQLite） | ⏳ |
| M2 | 评估框架（5 策略 + ROC + EER） | ⏳ |
| M3 | 跑评估实验，确定最优策略与阈值 | ⏳ |
| M4 | FastAPI + 实时识别 + IoU 跟踪 + WebSocket | ⏳ |
| M5 | 单文件前端 HTML | ⏳ |
| M6 | 文档 + 演示脚本 + 答辩材料 | ⏳ |

## 致谢

- [InsightFace](https://github.com/deepinsight/insightface) — 提供预训练模型 `buffalo_l`
- [ArcFace 论文](https://arxiv.org/abs/1801.07698)（Deng et al., 2019）— 损失函数设计
- [LFW Dataset](http://vis-www.cs.umass.edu/lfw/) — 库外评估数据来源

## 许可

[MIT](LICENSE)
