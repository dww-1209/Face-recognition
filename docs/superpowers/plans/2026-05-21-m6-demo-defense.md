# M6 演示与答辩准备实施计划

> **For agentic workers:** 本计划与 M1~M5 不同，**不是 TDD 任务列表**，而是"答辩前的工程整理 + 报告撰写 + 演示脚本"清单。每条 TODO 是一项独立可勾选的工作。

**Goal:** 把 M1~M5 的成果打包成可答辩的工程：跑完整套消融实验出图、写答辩报告、录演示视频、清理仓库。

**预计工作量:** 半天到一天（取决于报告章节深度）。

---

## 工作清单

### A. 实验数据落地

- [ ] **A1. 准备数据集**

  在 `data/private_dataset/` 下按以下结构组织：
  ```
  data/private_dataset/
  ├── alice/
  │   ├── 001.jpg ~ 060.jpg
  ├── bob/
  │   └── ...
  └── ...（共 35 人，每人 60 张）
  ```
  - 50 张作训练 / 模板生成
  - 10 张作测试集（用 M2 `data_split` 自动切分，固定 seed=42）

- [ ] **A2. 下载 LFW 子集**

  ```bash
  uv run python -c "
  from sklearn.datasets import fetch_lfw_people
  fetch_lfw_people(min_faces_per_person=1, color=True, download_if_missing=True)
  "
  ```
  默认存到 `~/scikit_learn_data/`。M2 `lfw_loader` 会自动找到。

- [ ] **A3. 跑全套消融实验**

  ```bash
  uv run python -m face_recognition.evaluation.run_ablation \
      --dataset data/private_dataset/ \
      --output reports/m6-ablation/
  ```
  应产出 5 个策略 × {ROC 曲线 PNG, EER, TAR@FAR=1e-3, 各阈值下 FAR/FRR}。
  时间：约 5~15 分钟（取决于 GPU）。

- [ ] **A4. 数据健康度检查**

  打开 `reports/m6-ablation/summary.csv`，确认：
  - [ ] 每个策略的 EER 都在合理范围（ArcFace 35 人通常 1%~5%）
  - [ ] 5 策略的指标有差异（如果完全一样说明实验有 bug）
  - [ ] ROC 曲线视觉上区分度合理

  如果发现 bug，回到 M2 修代码再跑。

---

### B. 答辩报告

- [ ] **B1. 创建报告骨架**

  ```bash
  mkdir -p docs/report
  ```

  在 `docs/report/answer-defense.md` 中按以下章节写：

  ```markdown
  # 基于 ArcFace 的开放集人脸识别系统

  ## 1. 项目背景与问题定义
  - 开放集 vs 封闭集
  - 为什么用预训练 ArcFace 而非自训
  - 应用场景（35 人门禁场景）

  ## 2. 系统架构
  - 清洁架构四层图（贴 spec 的图）
  - 数据流：注册 / 实时识别两条路径
  - 技术栈选型理由（YAGNI 决策表）

  ## 3. 多模板策略消融实验
  - 5 策略定义
  - 评估方案（Genuine + 库内 Impostor + 库外 Impostor）
  - ROC 曲线（贴 reports/m6-ablation/*.png）
  - EER / TAR@FAR 表格
  - 结论：哪个策略最优 + 推测原因

  ## 4. 实时识别系统
  - 跟踪 + 按需识别架构
  - 端到端延迟分析（capture → recog → ws → render）
  - 屏幕截图（贴 docs/screenshots/m5/*.png）

  ## 5. 工程化思考
  - TDD 实践经验
  - 测试金字塔执行情况（单元/集成/E2E 数量）
  - 哪些"过度工程"被主动拒绝（spec §2 严禁清单）

  ## 6. 局限与未来工作
  - 当前 35 人，扩展到 1000+ 怎么改
  - 活体检测缺失
  - 光照鲁棒性
  ```

- [ ] **B2. 数据填充**

  把 A3 的实验结果填进 §3。CSV 转 Markdown 表格可以用：
  ```bash
  uv run python -c "
  import pandas as pd
  df = pd.read_csv('reports/m6-ablation/summary.csv')
  print(df.to_markdown(index=False))
  "
  ```

- [ ] **B3. 自审清单**

  - [ ] 每个章节都有"我做了什么 + 为什么这样做"
  - [ ] 有图表的章节都引用了图（不只贴图不解释）
  - [ ] 参考文献引用 ArcFace 原论文 / InsightFace GitHub
  - [ ] 没有空话（"本系统具有强大的扩展性"这种）

---

### C. 演示脚本与录屏

- [ ] **C1. 写演示稿（5 分钟版本）**

  在 `docs/report/demo-script.md`：

  ```markdown
  ## 演示流程（5 分钟）
  
  1. （30s）打开 http://localhost:8000 → 介绍 Tab 结构
  2. （1m）人员管理：注册 3 个人 → 切策略对比
  3. （1m）切到实时识别 → 出镜识别 → 离镜
  4. （1m）展示 unknown：找路人入镜 → 显示"未知"
  5. （1m）切回人员管理 → 删除一人 → 实时识别立即生效（不需重启）
  6. （30s）打开 reports/m6-ablation/*.png → 速览消融数据
  ```

- [ ] **C2. 录屏**

  - macOS：QuickTime Player → 文件 → 新建屏幕录制
  - 录前先彩排 1~2 次，避免现场卡壳
  - 输出 `docs/report/demo-video.mp4`（提交时压缩到 < 100MB）

---

### D. 仓库清理

- [ ] **D1. 检查 .gitignore 完整**

  必须包含：
  ```
  data/
  *.db
  *.pyc
  __pycache__/
  .venv/
  reports/*
  !reports/.gitkeep
  logs/
  CLAUDE.md
  ```

- [ ] **D2. 检查没有泄露**

  ```bash
  git ls-files | grep -E "(data/|\.db$|\.env$)" && echo "❌ 发现私有文件"
  uv run python -c "
  import pathlib
  for p in pathlib.Path('.').rglob('*'):
      if p.is_file() and p.stat().st_size > 5_000_000 and '.git' not in str(p):
          print(f'⚠️ 大文件: {p} ({p.stat().st_size//1024} KB)')
  "
  ```

- [ ] **D3. README.md 终版**

  确保 README 含：
  - 一句话项目简介
  - 截图（最有效果的一张实时识别画面）
  - 安装 + 启动 5 行命令
  - 项目结构概览（链接到 spec）
  - 评估实验摘要表
  - License

- [ ] **D4. 跑全套测试最后一遍**

  ```bash
  uv run pytest                       # 默认所有非 GPU 测试
  uv run pytest -m gpu                # GPU 集成测试
  uv run mypy src/face_recognition/domain src/face_recognition/application
  uv run ruff check .
  ```

  全绿才打 release tag。

- [ ] **D5. 打 final tag**

  ```bash
  git add -A
  git commit -m "chore: M6 final cleanup + report"
  git tag v1.0.0
  git push origin main --tags
  ```

---

### E. 答辩素材清单（提交前 self-check）

- [ ] 报告 PDF（从 markdown 转，pandoc 或 typora 都行）
- [ ] 演示视频（mp4，<100MB）
- [ ] 源码仓库链接（已 push 到 GitHub）
- [ ] 实验数据 CSV（reports/m6-ablation/summary.csv）
- [ ] ROC 曲线 5 张 PNG
- [ ] 关键架构图 1 张（清洁架构四层）
- [ ] PPT（可选，老师要求才做）

---

## M6 完成标准

- ✅ 5 策略消融实验全跑完，ROC/EER/TAR 数据全部入仓
- ✅ 答辩报告 markdown 6 章节齐全
- ✅ 演示视频已录制
- ✅ 仓库无私有数据 / 大文件泄露
- ✅ `git tag v1.0.0` + push 完成

至此项目完结。下一步是答辩。
