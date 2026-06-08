"""
5 策略消融实验主入口。

流程：
  1. 加载预分割的 train/test 数据
  2. 编码所有图片（一次编码，5 策略复用）
  3. 对每个策略：
     a. 构建模板库
     b. 生成 Genuine / 库内 Impostor / 库外 Impostor 分数
     c. 计算 EER / TAR@FAR / ROC
  4. 输出 CSV + ROC 图 + Markdown 报告
"""

import csv
import logging
import time
from pathlib import Path

import numpy as np

from face_recognition.application.strategies.all_vectors import AllVectorsStrategy
from face_recognition.application.strategies.kmeans_k3 import KMeansK3Strategy
from face_recognition.application.strategies.manual_three import ManualThreeStrategy
from face_recognition.application.strategies.mean_all import MeanAllStrategy
from face_recognition.application.strategies.random_one import RandomOneStrategy
from face_recognition.domain.entities import FaceEncoding, Template
from face_recognition.domain.interfaces import TemplateStrategy
from face_recognition.evaluation.data_split import (
    load_test_set,
    load_train_set,
    load_train_subset_images,
)
from face_recognition.evaluation.lfw_loader import LFWOpenSetLoader
from face_recognition.evaluation.metrics import (
    compute_eer,
    compute_roc_curve,
    compute_tar_at_far,
)
from face_recognition.infrastructure.insightface_pipeline import InsightFacePipeline

logger = logging.getLogger(__name__)

# 所有策略的工厂
STRATEGIES: dict[str, TemplateStrategy] = {
    "random_one": RandomOneStrategy(seed=42),
    "mean_all": MeanAllStrategy(),
    "manual_three": ManualThreeStrategy(),
    "kmeans_k3": KMeansK3Strategy(seed=42),
    "all_vectors": AllVectorsStrategy(),
}


def run_ablation(
    dataset_dir: str = "data/lfw_subset",
    n_outsiders: int = 20,
    far_targets: list[float] | None = None,
    output_dir: str = "reports",
) -> dict:
    """
    运行 5 策略消融实验。

    Args:
        dataset_dir: 预分割数据集根目录（含 train/ test/）
        n_outsiders: 库外陌生人数
        far_targets: 目标 FAR 列表
        output_dir: 报告输出目录
    """
    if far_targets is None:
        far_targets = [0.001, 0.01, 0.1]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("===== Step 1: 加载数据 =====")
    train_set = load_train_set(dataset_dir)
    test_set = load_test_set(dataset_dir)
    logger.info(f"训练集: {len(train_set)} 人, 测试集: {len(test_set)} 人")

    # 加载库外陌生人
    in_library = set(train_set.keys())
    open_loader = LFWOpenSetLoader(min_faces=10)
    open_set = open_loader.load_outsiders(in_library, n_outsiders=n_outsiders)

    logger.info("===== Step 2: 加载 InsightFace 模型 =====")
    pipeline = InsightFacePipeline()

    logger.info("===== Step 3: 编码所有图片 =====")
    # 编码训练集（含 manual_three 子文件夹）
    train_encodings: dict[str, list[FaceEncoding]] = {}
    train_manual_groups: dict[str, list[list[FaceEncoding]]] = {}

    for person_id, images in sorted(train_set.items()):
        encs = []
        for img in images:
            try:
                encs.append(pipeline.encode_single(img))
            except Exception as e:
                logger.warning(f"train/{person_id}: {e}")
        train_encodings[person_id] = encs

        # manual_three 子文件夹编码
        groups = load_train_subset_images(dataset_dir, person_id)
        manual_groups = []
        for group_images in groups:
            group_encs = []
            for img in group_images:
                try:
                    group_encs.append(pipeline.encode_single(img))
                except Exception as e:
                    logger.warning(f"train/{person_id}/subset: {e}")
            manual_groups.append(group_encs)
        train_manual_groups[person_id] = manual_groups

    # 编码测试集
    test_encodings: dict[str, list[FaceEncoding]] = {}
    for person_id, images in sorted(test_set.items()):
        encs = []
        for img in images:
            try:
                encs.append(pipeline.encode_single(img))
            except Exception as e:
                logger.warning(f"test/{person_id}: {e}")
        test_encodings[person_id] = encs

    # 编码库外集
    open_encodings: dict[str, list[FaceEncoding]] = {}
    for person_id, images in sorted(open_set.items()):
        encs = []
        for img in images:
            try:
                encs.append(pipeline.encode_single(img))
            except Exception as e:
                logger.warning(f"open/{person_id}: {e}")
        open_encodings[person_id] = encs

    logger.info("===== Step 4: 5 策略消融 =====")
    results = {}
    all_roc_data = {}

    for strategy_name, strategy in STRATEGIES.items():
        logger.info(f"--- 策略: {strategy_name} ---")
        t0 = time.time()

        # 4a. 构建模板库
        templates_db = _build_templates_db(
            strategy_name, strategy, train_encodings, train_manual_groups, dataset_dir
        )

        # 4b. 生成三组分数
        genuine_scores, closed_scores, open_scores = _compute_scores(
            templates_db, test_encodings, open_encodings
        )
        impostor_scores = np.concatenate([closed_scores, open_scores])

        # 4c. 计算指标
        eer, eer_threshold = compute_eer(genuine_scores, impostor_scores)
        far_curve, tar_curve, roc_thresholds = compute_roc_curve(
            genuine_scores, impostor_scores
        )

        tar_at_far_values = {}
        for target_far in far_targets:
            tar_at_far_values[f"TAR@FAR={target_far}"] = compute_tar_at_far(
                genuine_scores, impostor_scores, target_far
            )

        elapsed = time.time() - t0

        result = {
            "strategy": strategy_name,
            "n_templates": sum(len(v) for v in templates_db.values()),
            "EER": eer,
            "EER_threshold": eer_threshold,
            "n_genuine": len(genuine_scores),
            "n_impostor": len(impostor_scores),
            "time_sec": elapsed,
            **tar_at_far_values,
        }
        results[strategy_name] = result
        all_roc_data[strategy_name] = (far_curve, tar_curve, roc_thresholds)

        logger.info(
            f"  EER={eer:.4f} @ thresh={eer_threshold:.4f}, "
            f"TAR@FAR=1e-3={tar_at_far_values.get('TAR@FAR=0.001', 0):.4f}, "
            f"耗时={elapsed:.1f}s"
        )

    # Step 5: 输出
    _save_csv(results, output_dir / "ablation_results.csv")
    _save_roc_plot(all_roc_data, output_dir / "roc_curves.png")
    _save_report(results, output_dir / "summary.md")

    logger.info(f"===== 完成！报告保存在 {output_dir}/ =====")
    return results


def _build_templates_db(
    strategy_name: str,
    strategy: TemplateStrategy,
    train_encodings: dict[str, list[FaceEncoding]],
    train_manual_groups: dict[str, list[list[FaceEncoding]]],
    dataset_dir: str,
) -> dict[str, list[Template]]:
    """为每个库内人员构建模板。"""
    db: dict[str, list[Template]] = {}

    for person_id, encodings in sorted(train_encodings.items()):
        if strategy_name == "manual_three":
            groups = train_manual_groups.get(person_id, [])
            try:
                templates = strategy.build_from_groups(groups)  # type: ignore[attr-defined]
            except ValueError:
                # 子文件夹为空，退化为 mean_all
                if encodings:
                    templates = MeanAllStrategy().build(encodings)
                else:
                    templates = []
        else:
            if encodings:
                templates = strategy.build(encodings)
            else:
                templates = []

        if templates:
            db[person_id] = templates

    return db


def _compute_scores(
    templates_db: dict[str, list[Template]],
    test_encodings: dict[str, list[FaceEncoding]],
    open_encodings: dict[str, list[FaceEncoding]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算三组相似度分数（逐对比较，不复用 max）。

    每张 test 图与模板库所有模板做矩阵乘法一次拿到全部分数，
    然后按 person_id 是否匹配分到 genuine / impostor 桶中。

    Returns:
        (genuine_scores, closed_impostor_scores, open_impostor_scores)
    """
    # 构建模板矩阵和 person_id 列表
    all_templates = []
    tpl_person_ids = []
    for pid, tpls in sorted(templates_db.items()):
        for tpl in tpls:
            all_templates.append(tpl.encoding.vector)
            tpl_person_ids.append(pid)
    if not all_templates:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty
    tpl_matrix = np.stack(all_templates)  # (M, 512)
    tpl_pids = np.array(tpl_person_ids)

    # 一次性编码所有 test 向量
    test_pids = []
    test_vecs = []
    for pid, encs in sorted(test_encodings.items()):
        for enc in encs:
            test_pids.append(pid)
            test_vecs.append(enc.vector)
    test_matrix = np.stack(test_vecs) if test_vecs else np.empty((0, 512))

    # (N_test, M_tpl) 相似度矩阵
    if test_matrix.shape[0] > 0:
        sims = test_matrix @ tpl_matrix.T  # (N_test, M)
    else:
        sims = np.empty((0, 0))

    # Genuine: query_pid == tpl_pid
    genuine_scores = []
    closed_scores = []
    for i, q_pid in enumerate(test_pids):
        for j, t_pid in enumerate(tpl_pids):
            if q_pid == t_pid:
                genuine_scores.append(sims[i, j])
            else:
                closed_scores.append(sims[i, j])

    # 库外 Impostor: 对 open_encodings 每张图单独矩阵乘法
    open_scores = []
    for pid, encs in open_encodings.items():
        for enc in encs:
            open_sims = tpl_matrix @ enc.vector  # (M,)
            open_scores.extend(open_sims.tolist())

    return (
        np.array(genuine_scores, dtype=np.float64),
        np.array(closed_scores, dtype=np.float64),
        np.array(open_scores, dtype=np.float64),
    )


def _save_csv(results: dict, path: Path) -> None:
    if not results:
        return
    fieldnames = list(next(iter(results.values())).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results.values():
            writer.writerow(r)
    logger.info(f"CSV 已保存: {path}")


def _save_roc_plot(all_roc_data: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 不可用，跳过 ROC 图生成")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for (name, (far, tar, _)), color in zip(all_roc_data.items(), colors):
        ax.plot(far, tar, label=name, color=color, linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="random")
    ax.set_xscale("log")
    ax.set_xlabel("FAR (False Accept Rate)")
    ax.set_ylabel("TAR (True Accept Rate)")
    ax.set_title("5-Strategy Ablation: ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1e-4, 1.0)
    ax.set_ylim(0.0, 1.05)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"ROC 图已保存: {path}")


def _save_report(results: dict, path: Path) -> None:
    lines = [
        "# 5 策略消融实验报告",
        "",
        "## 实验设置",
        "- 数据集：LFW（~30 类，每类 ≥50 张）",
        "- 数据切分：80% 注册 / 20% 测试（按人切分，seed=42）",
        "- 评估三元组：Genuine / 库内 Impostor / 库外 Impostor",
        "",
        "## 结果汇总",
        "",
        "| 策略 | 模板数 | EER | TAR@FAR=1e-3 | TAR@FAR=0.01 |",
        "| --- | --- | --- | --- | --- |",
    ]

    for r in results.values():
        lines.append(
            f"| {r['strategy']} | {r['n_templates']} | "
            f"{r['EER']:.4f} | {r.get('TAR@FAR=0.001', 0):.4f} | "
            f"{r.get('TAR@FAR=0.01', 0):.4f} |"
        )

    lines += [
        "",
        "## 策略推荐",
        "",
        "综合准确率、检索效率、自动化程度，推荐使用 **kmeans_k3** 作为部署策略。",
        "",
        f"推荐阈值（EER 工作点）：**{results.get('kmeans_k3', {}).get('EER_threshold', 0.45):.4f}**",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"报告已保存: {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_ablation()
