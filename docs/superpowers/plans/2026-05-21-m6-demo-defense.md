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

  > **网络与磁盘提醒**:这次下载约 **200MB**(完整 LFW 是 ~233MB, color=True 不会变更小,
  > 只是返回 RGB 三通道而非灰度)。首次拉会挂十几分钟到半小时——
  > 1) **要联网**,且能访问 `vis-www.cs.umass.edu`(国内经常超时——可挂代理或换镜像);
  > 2) **磁盘**留 ~500MB(下载临时文件 + 解压);
  > 3) 中断后重跑会断点续传,放心 Ctrl+C;
  > 4) 不想占用户家目录可设 `SCIKIT_LEARN_DATA=/path/to/cache` 环境变量。

- [ ] **A3. 跑全套消融实验**

  ```bash
  uv run python -m face_recognition.evaluation.run_ablation \
      --dataset data/private_dataset/ \
      --output reports/m6-ablation/
  ```
  应产出 5 个策略 × {ROC 曲线 PNG, EER, TAR@FAR=1e-3, 各阈值下 FAR/FRR}。

  **预计时间**(35 人 × 60 张 = 2100 张照片 + ~500 LFW 库外人):
  - **GPU**(RTX 3060 / Apple M1+ MPS):**5~10 分钟**——瓶颈在 buffalo_l 检测+编码 ~30ms/图
  - **纯 CPU**(笔记本 i5/i7):**40~80 分钟**——CPU 上 buffalo_l 单图 ~500ms
  - 第二次跑会快很多:特征向量已落库 SQLite,只需重做"组模板 + 评估"两步,不重抽特征
  - 如果想强制 CPU(没 GPU 或 GPU 显存不够),`config.yaml` 把 `pipeline.device = "cpu"`

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
  # ⚠️ to_markdown 内部依赖 `tabulate` 包,pandas 不自带。先装一次:
  uv add --dev tabulate
  uv run python -c "
  import pandas as pd
  df = pd.read_csv('reports/m6-ablation/summary.csv')
  print(df.to_markdown(index=False))
  "
  ```
  > 如果忘了装会报 `ImportError: Missing optional dependency 'tabulate'`——按提示装即可。
  > 也可以用 `df.to_string()` 出对齐文本,粘进 markdown 包 ` ``` ` 当代码块凑合用。

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

  > **压缩到 < 100MB 的 ffmpeg 食谱**(QuickTime 原片通常 5 分钟 ~500MB):
  > ```bash
  > # H.264 + CRF 28(质量良好,文件小);scale=-2:720 把高度限到 720p,宽度按比例
  > # -2 = 自动选偶数(H.264 要求宽高都是偶数,否则报错)
  > ffmpeg -i input.mov \
  >        -vcodec libx264 -crf 28 -preset slow \
  >        -vf "scale=-2:720" \
  >        -acodec aac -b:a 96k \
  >        docs/report/demo-video.mp4
  > ```
  > 如果还超 100MB:
  > - 提高 CRF(数字越大文件越小,画质越糊)——CRF 30~32 仍可看清画面
  > - 改 `scale=-2:480` 进一步缩到 480p
  > - 或裁剪开头/结尾静止段:`-ss 00:00:05 -to 00:04:30`(从第 5 秒到 4 分 30 秒)
  > 没装 ffmpeg:`brew install ffmpeg`(macOS)。

---

### D. 仓库清理

- [ ] **D1. 检查 .gitignore 完整**

  必须包含:
  ```
  data/
  *.db
  *.pyc
  __pycache__/
  .venv/
  reports/*
  !reports/.gitkeep
  logs/
  *.local.yaml
  *.mp4
  docs/screenshots/m5/raw/
  CLAUDE.md
  ```

  > **CLAUDE.md 不入仓**——这是用户的本地 AI 协作指令，不属于项目交付物。
  > 任何在仓库工作的 AI 助手会自动加载本地的 CLAUDE.md（Claude Code 的机制），
  > 不需要走 git 分发。
  >
  > 新增项解释:
  > - `*.local.yaml`:本地覆盖配置(开发者临时改阈值/模型路径)。`config.yaml`
  >   主配置入仓,本地变体不入仓。
  > - `*.mp4`:M6 录的演示视频通常 50~100MB,不适合走 git;若必须分发用 release
  >   附件或外链。
  > - `docs/screenshots/m5/raw/`:M5 截图未脱敏原图(含真人脸),只允许保留在本
  >   地;脱敏后的发布版放在 `docs/screenshots/m5/` 根目录,可入仓。

  > **⚠️ 已 tracked 文件不会被新增的 .gitignore 规则忽略**——`.gitignore` 只对**未 tracked**的文件生效。
  > 比如 `CLAUDE.md` 之前已经被 `git add` 过(历史 commit 里有),那即使现在加进 .gitignore,
  > git status 仍会跟踪它的修改。需要先**从索引移除**(保留磁盘文件):
  > ```bash
  > # 检查哪些文件已 tracked 但应当 ignore
  > git ls-files | grep -E "^(CLAUDE\.md|.*\.local\.yaml|.*\.mp4|data/)"
  >
  > # 从 git 索引删除但保留本地文件(--cached 关键)
  > git rm --cached CLAUDE.md
  > git rm --cached -r data/  # 若误入仓
  > git commit -m "chore: untrack files now in gitignore"
  > ```
  > 不带 `--cached` 会**真的删本地文件**——切记。

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
  uv run pytest                            # 默认：单元测试（毫秒级）
  uv run pytest -m integration             # 集成测试（真模型 + 真 SQLite，~分钟级）
  uv run mypy src/face_recognition/domain src/face_recognition/application
  uv run ruff check .
  ```

  > **marker 命名说明**：M1~M5 历史代码可能写过 `@pytest.mark.gpu`——但实际我们的集成测试在 CPU
  > 上也能跑（`ctx_id=-1` 强制 CPU），名字误导。统一改用 `integration`。
  > 自检：`grep -rn "pytest.mark.gpu\|pytest -m gpu" src tests docs` 应当无结果；如有，统一替换为 `integration`。
  > `pyproject.toml` 的 `[tool.pytest.ini_options].markers` 应包含：
  > ```toml
  > markers = [
  >     "integration: 集成测试（真模型 + 真 SQLite，需要本地下载的 buffalo_l 权重；跑 -m integration）",
  >     "slow: 耗时 > 1 秒的测试",
  > ]
  > ```
  > 如旧版还留着 `"gpu: ..."` 这一行，删除（`--strict-markers` 下未注册标记会报错——这正好帮我们抓漏网之鱼）。

  全绿才打 release tag。

- [ ] **D5. 打 final tag**

  ```bash
  # 1) 先确认远端配置 + 本地领先关系
  git remote -v                          # 应看到 origin → 你的 GitHub 仓库 URL
  git status                             # 工作区干净(或只剩 D2 中允许的文件)
  git log origin/main..HEAD --oneline    # 看本地领先了哪些 commit;空 = 已同步

  # 2) 提交 + 打 tag(顺序重要:先 commit 再 tag,否则 tag 会指向旧 commit)
  git add -A
  git commit -m "chore: M6 final cleanup + report"
  git tag -a v1.0.0 -m "Release v1.0.0: ArcFace 人脸识别系统(本科期末项目)"
  #  ↑ -a 创建带说明的 annotated tag(推荐),纯 `git tag v1.0.0` 是 lightweight tag

  # 3) 推送(必须分两步,或用 --follow-tags 一次推)
  git push origin main                   # 推 commit
  git push origin v1.0.0                 # 推 tag(默认 push 不带 tag,得显式)
  # 等价单行:git push --follow-tags origin main
  ```

  > **如果 push 被拒绝**(`! [rejected]` 或 `non-fast-forward`):说明远端有你本地没有的 commit。
  > **不要直接 `--force`!** 先 `git pull --rebase origin main` 把远端改动拉下来再推。
  > 只有"刚刚 force-push 完确认无误"才用 force,否则有覆盖队友工作的风险。
  >
  > **如果 tag 打错位置**:`git tag -d v1.0.0` 删本地, `git push origin :refs/tags/v1.0.0` 删远端,
  > 然后重新打 + push。

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
