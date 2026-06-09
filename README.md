# 基于 ArcFace 的人脸识别系统

> 本科期末项目 · 开放集人脸识别 · 支持动态增删人员

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-96%20passed-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 简介

一个**开放集人脸识别系统**，基于 InsightFace 官方预训练模型 `buffalo_l`（ResNet100 + ArcFace），无需任何模型训练。系统支持：

- **动态增删人员**：注册新人只需提取特征向量入库，无需重训
- **5 种模板策略消融对比**：随机选取 / 全部平均 / 人工挑选 / KMeans 聚类 / 全量存储
- **开放集识别**：能区分库内人员与库外陌生人（基于阈值的拒识）
- **实时识别**：FastAPI + OpenCV + WebSocket，浏览器实时查看摄像头识别结果
- **完整学术评估**：ROC 曲线、EER、TAR\@FAR=1e-3、Top-1 等指标

## 技术栈

| 组件 | 技术选型 |
| --- | --- |
| 人脸检测 + 对齐 + 特征 | [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l`（SCRFD + ArcFace ResNet100） |
| 向量数据库 | SQLite + numpy 暴力检索 |
| Web 服务 | FastAPI + WebSocket |
| 摄像头 | OpenCV `VideoCapture` |
| 前端 | 单文件 HTML + Pico.css CDN |
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
│   ├── domain/         # 第1层：实体 + 接口（Protocol）+ 异常
│   │   ├── entities.py       # FaceEncoding, Template, Person, DetectedFace 等
│   │   ├── interfaces.py     # FacePipeline, PersonRepository, TemplateStrategy
│   │   └── errors.py         # 领域异常层次
│   ├── application/    # 第2层：用例编排
│   │   ├── register_face.py       # 注册用例（支持文件夹 + 内存帧）
│   │   ├── recognize_face.py      # 单图识别用例（矩阵乘法检索）
│   │   ├── recognize_frame.py     # 实时帧识别（检测→跟踪→按需识别）
│   │   ├── template_matrix.py     # 内存模板矩阵（热重载）
│   │   └── strategies/            # 5 种模板生成策略
│   ├── infrastructure/ # 第3层：InsightFace / SQLite / 摄像头 / IoU 跟踪
│   │   ├── insightface_pipeline.py # buffalo_l 一站式管线
│   │   ├── sqlite_repository.py    # SQLite 向量库（BLOB 存储）
│   │   ├── camera_capture.py       # OpenCV 摄像头封装
│   │   ├── iou_tracker.py          # IoU 多目标跟踪
│   │   ├── frame_renderer.py       # 画框/写名/JPEG 编码
│   │   └── config_loader.py        # pydantic 配置加载
│   ├── api/            # 第4层：CLI + REST + WebSocket
│   │   ├── cli.py              # Typer CLI（6 个命令）
│   │   ├── server.py           # FastAPI 应用骨架（lifespan + 异常处理）
│   │   ├── routes_persons.py   # REST: 人员 CRUD
│   │   ├── routes_stream.py    # WebSocket: 实时推流
│   │   ├── dependencies.py     # 依赖装配（CLI 工厂 + Server 单例）
│   │   └── static/index.html   # 单文件前端（262 行）
│   └── evaluation/     # 离线评估（独立于四层）
│       ├── types.py            # EvalEncoding, PairResult, StrategyMetrics
│       ├── data_split.py       # 按人 80/20 切分
│       ├── lfw_loader.py       # sklearn LFW 下载器
│       ├── embedder.py         # 批量编码辅助
│       ├── pair_generator.py   # Genuine/库内/库外 三组配对
│       ├── metrics.py          # ROC, EER, TAR@FAR, Top-1
│       ├── reports.py          # CSV + ROC 叠图 + 直方图 + Markdown
│       └── run_ablation.py     # 5 策略消融主入口
├── tests/
│   ├── unit/                    # 单元测试（毫秒级，32 模块）
│   │   ├── application/         # template_matrix, recognize_frame
│   │   ├── evaluation/          # types, data_split, lfw_loader, embedder,
│   │   │                          pair_generator, metrics, reports
│   │   └── infrastructure/      # iou_tracker, camera_capture, frame_renderer
│   └── integration/             # 集成测试
│       └── api/                 # server smoke, routes_persons, error_handling
├── data/               # 数据集（gitignore）
├── reports/            # 评估输出（gitignore）
├── scripts/            # 数据准备脚本
├── docs/superpowers/{specs,plans}/  # 设计文档 + 教材级实施计划
├── config.yaml         # 全部可调参数
└── pyproject.toml
```

## 快速开始

### 环境要求

- Python 3.12+
- macOS / Linux / Windows
- GPU 可选（CPU 也能跑，实时识别稍慢）

### 安装

```bash
# 安装 uv（如果还没装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆仓库
git clone https://github.com/dww-1209/Face-recognition.git
cd Face-recognition

# 创建虚拟环境并装依赖（首次会自动下载 InsightFace 模型 ~250MB）
uv sync
```

### 准备数据

```bash
# 一键下载 LFW 并生成 80/20 按人切分数据集
uv run python scripts/prepare_lfw_dataset.py
```

输出结构：
```
data/lfw_subset/
├── train/<person_name>/*.jpg          # 80% 注册集
│             /subset_0/*.jpg          # → manual_three 子文件夹
│             /subset_1/*.jpg
│             /subset_2/*.jpg
└── test/<person_name>/*.jpg           # 20% 测试集
```

用自己的数据集时，按同样目录结构组织即可。

### CLI 命令

```bash
# 注册人员
uv run python -m face_recognition.api.cli register \
    --dataset data/lfw_subset/train --strategy kmeans_k3

# 单图识别
uv run python -m face_recognition.api.cli recognize data/test/alice/001.jpg

# 列出 / 删除
uv run python -m face_recognition.api.cli list
uv run python -m face_recognition.api.cli remove alice

# 5 策略消融评估
uv run python -m face_recognition.api.cli evaluate \
    --dataset data/lfw_subset --n-lfw 50 --output reports

# 启动 Web 服务
uv run python -m face_recognition.api.cli serve
```

### 评估输出

评估完成后 `reports/` 下生成：
- `summary.csv`：5 策略 × 各指标对比表
- `roc_curves.png`：5 条 ROC 曲线叠图
- `hist_<strategy>.png`：各策略 Genuine vs Impostor 分数分布
- `summary.md`：Markdown 报告（可直接抄进答辩 PPT）

### 启动 Web 实时识别

```bash
uv run uvicorn face_recognition.api.server:app --host 0.0.0.0 --port 8000
```

浏览器打开 http://localhost:8000 ——人员管理 + 实时识别两个 Tab。

> **macOS 注意**：终端需要摄像头权限（系统偏好设置 → 隐私与安全性 → 相机）。

## 测试

```bash
uv run pytest                            # 全部（96 个测试，~18s）
uv run pytest tests/unit/                # 单元（毫秒级）
uv run pytest tests/integration/         # 集成（需模型）
uv run pytest -m slow                    # 慢测试（含 LFW 下载）
```

## 项目里程碑

| 阶段 | 内容 | 测试 | 状态 |
| --- | --- | --- | --- |
| M1 | 核心 CLI（注册 / 识别 / SQLite）+ 5 策略 | 32 | ✅ |
| M2 | 评估框架（ROC / EER / TAR\@FAR / Top-1） | 28 | ✅ |
| M3 | 跑评估实验，确定最优策略与阈值 | — | ✅ |
| M4 | FastAPI + 实时识别 + IoU 跟踪 + WebSocket | 22 | ✅ |
| M5 | 单文件前端 HTML（Pico.css + WebSocket Canvas） | — | ✅ |
| M6 | 文档 + 演示脚本 + 答辩材料 | — | ⏳ |

## 给 AI 协作者的指引

本项目有完整的 `CLAUDE.md`（项目级指令）和 `docs/superpowers/plans/`（教材级实施计划）。
每个 plan 都经过逐字验证——plan 中的代码块与实际源文件 **byte-for-byte 一致**，
可从零开始跟着 TDD 步骤复现整个项目。

## 致谢

- [InsightFace](https://github.com/deepinsight/insightface) — 提供预训练模型 `buffalo_l`
- [ArcFace 论文](https://arxiv.org/abs/1801.07698)（Deng et al., 2019）— 损失函数设计
- [LFW Dataset](http://vis-www.cs.umass.edu/lfw/) — 库外评估数据来源

## 许可

[MIT](LICENSE)
