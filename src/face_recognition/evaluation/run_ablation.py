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
    compute_top1_with_threshold,
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


def _build_templates_per_person(
    person_id: str,
    train_encodings: list[EvalEncoding],
    strategy: TemplateStrategy,
    pipeline: FacePipeline,
) -> list[EvalEncoding]:
    """跑一个策略,把训练集映射成该人的"模板向量集"(可以是 1~N 个)。

    策略可能返回 1 个(mean_all/random_one)或多个(kmeans_k3=3、all_vectors=N)Template。
    **评估口径**:保留多模板,scoring 时对每个 query 取 max_t cos(query, t) 作为该人得分。
    这与生产识别(M1 RecognizeFace 的多模板矩阵 max-by-template)逻辑完全一致,
    避免了"评估时强行平均→kmeans_k3 退化为 mean_all"的失真。

    Edge case: 训练集为空 → 返回 []，调用方应跳过该人，不能让 0 模板的人混进 impostor 配对
    （否则 cos(any_query, ∅) 在矩阵实现里是 -inf 还是 0 都是 bug——不如压根不存在）。

    Edge case: 训练集 < 策略要求数（如 kmeans_k3 但只有 2 张照片）→ M1 的策略实现里
    已有降级路径（kmeans_k3 把 < k 时直接返回所有原始 encoding；
    manual_three 若无子文件夹则退化为 mean_all），所以本函数不再二次校验，
    相信 strategy.build 的契约：
        len(strategy.build(encs)) <= max(len(encs), strategy.max_templates)
    """
    if not train_encodings:
        # 训练集为空：上游 split 阶段保证不会发生（按人 80/20 切，每人至少 1 张训练图），
        # 但留这层防御是因为 LFW 库外集合可能因 encode 失败被压到 0
        logger.warning("person_id=%s 训练集为空，跳过该人模板构建", person_id)
        return []

    # M1 定义的策略接口签名(domain/interfaces.py):
    #   def build(self, encodings: list[FaceEncoding]) -> list[Template]
    # FaceEncoding 需要 model_version,从 pipeline 读取(不再硬编码 "buffalo_l")。
    fe_list = [
        FaceEncoding(vector=e.vector, model_version=pipeline.model_version)
        for e in train_encodings
    ]
    templates = strategy.build(fe_list)
    if not templates:
        # 策略主动返回 0 模板：当前 5 个策略都不会发生，但接口允许；防御一下
        logger.warning(
            "person_id=%s 策略 %s 返回 0 模板，跳过", person_id, strategy.__class__.__name__
        )
        return []

    # 把每个 Template 包成 EvalEncoding 返回(共享同一 person_id);
    # image_path 用第一张图占位,仅作 debug 回溯。
    return [
        EvalEncoding(
            vector=t.encoding.vector.astype(np.float32),
            person_id=person_id,
            image_path=train_encodings[0].image_path,
        )
        for t in templates
    ]


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
        # 4.1 用策略把每个人 train_set 压成模板向量集(可能 1~N 个);
        # 评估全程保留多模板,scoring 用 max-similarity(见 generate_*_pairs)。
        templates: dict[str, list[EvalEncoding]] = {}
        for pid, encs in train_encodings_per_person.items():
            tpls = _build_templates_per_person(pid, encs, strategy, pipeline)
            if tpls:
                templates[pid] = tpls

        # 4.2 造配对：Genuine + 库外 Impostor（spec 已决定省去库内 Impostor）
        genuine = generate_genuine_pairs(test_encodings, templates)
        open_imp = generate_open_impostor_pairs(lfw_encodings, templates)
        all_pairs = genuine + open_imp
        pairs_per_strategy[name] = all_pairs

        # 4.3 计算指标
        eer, eer_thresh = compute_eer(all_pairs)
        tar = compute_tar_at_far(all_pairs, target_far=target_far)
        top1 = compute_top1_accuracy(test_encodings, templates)
        # 带阈值版本用本策略自己的 EER 阈值,衡量"端到端可用性"——见 metrics.py 注释
        top1_thr = compute_top1_with_threshold(test_encodings, templates, threshold=eer_thresh)
        fpr, tpr, _ = compute_roc(all_pairs)

        all_metrics.append(StrategyMetrics(
            strategy_name=name,
            eer=eer,
            eer_threshold=eer_thresh,
            tar_at_far_1e3=tar,
            top1_accuracy=top1,
            top1_with_threshold=top1_thr,
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
