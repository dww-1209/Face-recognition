# 评估实体的最小冒烟测试：能造、字段齐、不可变
import numpy as np
import pytest

# 评估专用数据类：和 domain.entities 的区别在 Step 3 详细解释
from face_recognition.evaluation.types import (
    EvalEncoding,
    PairResult,
    StrategyMetrics,
)


def _unit_vec(seed: int) -> np.ndarray:
    """生成 L2 归一化随机 512 维向量。与 M1 的 _unit_vector 功能相同。

    注：本函数在下文 Task 2/3/4/5 的独立测试代码块中会重复定义——
    这是为了让每个 Task 的测试文件可以独立运行、不依赖其他 Task 的 import。
    读者跟着 plan 逐 Task 实现时，每个测试文件放到对应位置即可立即跑通。
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_eval_encoding_is_frozen():
    enc = EvalEncoding(
        vector=_unit_vec(0),
        person_id="alice",
        image_path="data/private_dataset/alice/000.jpg",
    )
    # frozen=True 在 M1 Task 1 已经解释过：试图改字段会抛 FrozenInstanceError
    # （dataclasses 模块自带的异常类型，标记"我是 frozen 但你想改我"）
    with pytest.raises(Exception):
        enc.person_id = "bob"  # type: ignore[misc]


def test_pair_result_carries_score_and_label():
    # PairResult 表示"一次比对的产出"：相似度分数 + 是否同人（ground truth）
    pair = PairResult(score=0.83, is_genuine=True, query_person="alice", template_person="alice")
    assert pair.score == 0.83
    assert pair.is_genuine is True


def test_strategy_metrics_holds_all_indicators():
    # StrategyMetrics 是给 reports 模块的统一交付物——5 策略每个一份。
    # ── 给小白：每个数到底是什么意思 ──
    m = StrategyMetrics(
        strategy_name="kmeans_k3",
        eer=0.04,                  # EER=4%：FAR 和 FRR 同时降到 4% 的最佳折中点。
                                   # ArcFace 在小库（35 人）上典型 1%~5%，4% 算正常水平。
        eer_threshold=0.62,        # 达到 EER 时的相似度阈值——余弦点积 ≥ 0.62 判同人。
                                   # 实际部署阈值常乘 1.05~1.10 让 FAR 更低（误识别比漏识别危险）。
        tar_at_far_1e3=0.91,       # 在 FAR=0.1%（每千次陌生人比对只放 1 个进来）这个安全约束下，
                                   # 91% 的本人能被正确识别。门禁场景常用这个指标——FAR 是硬约束。
        top1_accuracy=0.96,        # 闭集 Top-1 准确率 96%：只看"和谁最像"不看分数高低，最像的 96% 是本人。
        top1_with_threshold=0.93,  # 加上阈值后的 Top-1：相似度 ≥ eer_threshold 才接受，剩下 93%。
        roc_fpr=np.array([0.0, 0.5, 1.0]),  # ROC 曲线 X 轴 = FAR/FPR 数组（去重后阈值数）
        roc_tpr=np.array([0.0, 0.95, 1.0]),  # ROC 曲线 Y 轴 = TAR/TPR；FPR/TPR 是 sklearn 命名（与本项目 FAR/TAR 等价）
        n_genuine=200,             # 生成的"同人"配对数
        n_impostor=4500,           # 生成的"陌生人"配对数（通常远多于 genuine）
    )
    assert m.strategy_name == "kmeans_k3"
    assert 0.0 < m.eer < 1.0
