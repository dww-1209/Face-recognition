from pathlib import Path

# matplotlib 在测试环境里默认会试图打开窗口，no-display 服务器上会报错。
# 'Agg' 后端 = "anti-grain geometry"，纯写 PNG 不开 GUI。
# ── 给小白：为什么 use('Agg') 必须写在 import pyplot 之前 ──
# matplotlib 在 `import matplotlib.pyplot` 这一刻会"锁定"当前后端——它内部初始化
# 了图形管线、字体缓存、事件循环钩子等等，跟所选后端绑死。如果先 `import pyplot`
# 再 `matplotlib.use("Agg")`，use() 只会发一行 UserWarning（不报错！）然后失效，
# 服务器上跑测试会**莫名 segfault**（因为 GUI 后端找不到 DISPLAY）。
# 安全做法：永远在文件最顶端、所有 pyplot 相关 import 之**前**调用 use()。
# noqa: E402（如果 ruff 抱怨 use 后面有 import）也可以加，但顺序不能变。
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
            "top1_with_threshold": m.top1_with_threshold,
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
    lines.append(
        "| Strategy | EER | TAR\\@FAR=1e-3 | Top-1 (no τ) | Top-1 (w/ τ) "
        "| EER thresh | n(Gen) | n(Imp) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in metrics:
        lines.append(
            f"| {m.strategy_name} "
            f"| {m.eer:.4f} "
            f"| {m.tar_at_far_1e3:.4f} "
            f"| {m.top1_accuracy:.4f} "
            f"| {m.top1_with_threshold:.4f} "
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
