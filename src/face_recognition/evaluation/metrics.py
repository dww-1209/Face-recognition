import numpy as np
# sklearn.metrics.roc_curve(y_true, y_score) → (fpr, tpr, thresholds)
#   - y_true: 0/1 标签数组（True=正例，这里"同人"）
#   - y_score: 模型输出的"是正例的置信度"——我们用余弦分
#   - 返回值都是 (T,) 一维数组，T 是不同阈值数（去重后）
#   - thresholds 从大到小排，对应曲线从 (0, 0) 走到 (1, 1)
from sklearn.metrics import roc_curve

from face_recognition.evaluation.types import EvalEncoding, PairResult


def _pairs_to_arrays(pairs: list[PairResult]) -> tuple[np.ndarray, np.ndarray]:
    """把 PairResult 列表拆成 (scores, labels) 数组，喂给 sklearn。"""
    # 列表推导式两次扫一遍——50K 量级配对下可忽略；如要优化可改用 np.fromiter
    scores = np.array([p.score for p in pairs], dtype=np.float64)
    # ── 给小白：为什么 genuine（同人）= 1 而不是 0 ──
    # 二分类里 "positive class（正例）" 不是"好人"的意思，而是"我们关心的事件"。
    # 识别系统关心的是"本人来了能不能识别出来"——所以"同人"是正例，标 1。
    # sklearn `roc_curve` 也按这个口径：TPR (True Positive Rate) = 把 label=1
    # 的样本正确判为 1 的比例 = 本人被接受的比例 = 我们的 TAR。
    # 反过来 FPR (False Positive Rate) = 把 label=0 的样本错判为 1 的比例 =
    # 陌生人被错认为本人的比例 = 我们的 FAR。如果把 genuine=0 弄反了，TPR/FPR
    # 的物理含义就跟着颠倒，画出来的 ROC 是镜像的，所有阈值都失效。
    labels = np.array([1 if p.is_genuine else 0 for p in pairs], dtype=np.int64)
    return scores, labels


def compute_roc(pairs: list[PairResult]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 ROC 曲线，返回 (FPR, TPR, thresholds)。"""
    scores, labels = _pairs_to_arrays(pairs)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return fpr, tpr, thresholds


def compute_eer(pairs: list[PairResult]) -> tuple[float, float]:
    """计算 EER 和对应阈值。

    EER 定义：FAR = FRR 那个点。
      FRR = 1 - TPR
      条件 FAR = FRR → fpr = 1 - tpr → fpr + tpr = 1
    我们沿曲线找"|fpr - (1 - tpr)| 最小"的点，返回该点 fpr 当 EER。

    ── 给小白：为什么 FRR = 1 - TPR 而不是 1 - FPR ──
    新手最常混淆 FAR/FRR 的分母方向，记住一句话："两种错误率，分母不同"：
        FAR = False Accept Rate = 陌生人被错误接受的比例
              分母 = 所有陌生人配对总数（label=0 的样本）
              = sklearn 的 FPR
        FRR = False Reject Rate = 本人被错误拒绝的比例
              分母 = 所有本人配对总数（label=1 的样本）
              = 1 - TAR = 1 - TPR
    所以 FRR ≠ 1 - FAR——它们的分母都不一样，不能直接相减。1 - TPR 才是
    "label=1 里没被接受的那部分"，也就是 FRR。这个分母约定一旦理清，下面所有
    阈值优化、EER 求解、TAR@FAR 才说得通。
    """
    fpr, tpr, thresholds = compute_roc(pairs)
    # 全数组运算 fpr - (1 - tpr) = fpr + tpr - 1
    # np.argmin(arr) 返回最小元素的索引——绝对值最小 = 最接近"FAR=FRR"的那一格
    diffs = np.abs(fpr - (1.0 - tpr))
    idx = int(np.argmin(diffs))
    eer = float((fpr[idx] + (1.0 - tpr[idx])) / 2.0)  # 取两个误差的平均更稳
    return eer, float(thresholds[idx])


def compute_tar_at_far(pairs: list[PairResult], target_far: float = 1e-3) -> float:
    """卡定 FAR 时的 TAR。

    target_far 通常取 1e-3（千分之一），门禁场景的事实标准。
    实现：在 ROC 曲线上找 fpr ≤ target_far 的最右一格，返回该格的 tpr。
    """
    fpr, tpr, _ = compute_roc(pairs)
    # 布尔索引：fpr <= target_far 是 (T,) bool 数组
    # np.where(...)[0] 返回所有 True 位置的下标
    valid = np.where(fpr <= target_far)[0]
    if len(valid) == 0:
        # 极端情况：所有 fpr 都 > target_far（数据太烂或样本太少）
        return 0.0
    # 取最右那个——FAR 最接近上限的位置 TPR 最高
    return float(tpr[valid[-1]])


def _flatten_templates(
    templates: dict[str, list[EvalEncoding]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """把所有人的所有模板拍平成 (T, 512) 矩阵 + (T,) 所属人索引 + person_id 列表。

    返回的 flat_matrix 行顺序与 owner_arr / template_ids 严格对齐——多模板
    max-by-template 聚合时要按 owner_arr 把分数 reduce 回每个 person。

    ── 给小白：owner_arr 是干嘛用的（用 SQL 类比一下）──
    假设 alice 有 3 个模板、bob 有 2 个，拍平后 flat_matrix 形状是 (5, 512)，
    owner_arr = [0, 0, 0, 1, 1]——表示第 0/1/2 行属于第 0 个人 (alice)、
    第 3/4 行属于第 1 个人 (bob)。下面 `np.maximum.at(out, owner_arr, scores)`
    在做的事情，用 SQL 写就是：
        SELECT MAX(score) FROM rows GROUP BY owner_arr
    ——按 owner 分组、组内取最大相似度。这是"每人多模板取最大分"的向量化实现，
    比 Python 双循环快几十倍。
    """
    template_ids = list(templates.keys())
    flat_vectors: list[np.ndarray] = []
    owner_index: list[int] = []
    for pid_idx, pid in enumerate(template_ids):
        for tpl in templates[pid]:
            flat_vectors.append(tpl.vector)
            owner_index.append(pid_idx)
    if not flat_vectors:
        return np.zeros((0, 512), dtype=np.float32), np.zeros(0, dtype=np.int64), template_ids
    flat_matrix = np.stack(flat_vectors)
    owner_arr = np.asarray(owner_index, dtype=np.int64)
    return flat_matrix, owner_arr, template_ids


def compute_top1_accuracy(
    test_set: list[EvalEncoding],
    templates: dict[str, list[EvalEncoding]],
) -> float:
    """**无阈值** 闭集 Top-1:对所有 person 取 max-by-template 相似度,argmax 命中本人则算对。

    用途:衡量"如果库内必有人,哪个最像"的纯排序能力——这是学术界对比模板生成
    策略时的标准口径,不掺业务阈值。**不能**直接代表生产识别准确率,生产还会
    被阈值拦下"似但不够似"的情况——那种场景请用 compute_top1_with_threshold。

    只统计 template 里有的那些人;test 里"陌生人"应已被调用方过滤。
    """
    flat_matrix, owner_arr, template_ids = _flatten_templates(templates)
    if not template_ids or flat_matrix.shape[0] == 0:
        return 0.0
    n_persons = len(template_ids)

    correct = 0
    total = 0
    for q in test_set:
        if q.person_id not in templates:
            continue  # 不在库里的人不计入闭集 Top-1
        # ── 给小白：(T, 512) @ (512,) 形状广播魔法 ──
        # flat_matrix 形状 (T, 512)，q.vector 形状 (512,)。
        # numpy `@` 看到右边是 1D 向量时自动当列向量做矩阵乘 → 输出形状 (T,)。
        # 每个元素 scores_flat[i] = q.vector 与第 i 个模板的点积 = 余弦相似度
        # （前提：两个向量都已 L2 归一化）。
        # 等价但慢 50~100 倍的写法：[np.dot(q.vector, t) for t in flat_matrix]。
        # 用 @ 的版本走的是 numpy 的 BLAS 后端，单条指令跑完 T 次乘加。
        scores_flat = flat_matrix @ q.vector
        # ── 给小白：np.maximum.at 是什么 + 为什么要 -inf 初始化 ──
        # `np.maximum.at(out, idx, src)` 等价于：
        #     for i in range(len(idx)):
        #         out[idx[i]] = max(out[idx[i]], src[i])
        # 但用 C 实现，单次循环跑完几十倍快。它做的就是上面 owner_arr 注释里
        # 那个"按 owner 分组取 max"的操作。
        # 初始化用 -inf 而非 0 是因为：余弦相似度范围 [-1, 1]，可能为负；用 0
        # 当初值，未被任何模板覆盖的人槽位会留 0 → np.argmax 可能错选这种"假高
        # 分"的空人。-inf 比任何真实分数都小，没被覆盖到就永远是 -inf，argmax
        # 不会选它。
        per_person_max = np.full(n_persons, -np.inf, dtype=scores_flat.dtype)
        np.maximum.at(per_person_max, owner_arr, scores_flat)
        pred = template_ids[int(np.argmax(per_person_max))]
        if pred == q.person_id:
            correct += 1
        total += 1
    return correct / total if total else 0.0


def compute_top1_with_threshold(
    test_set: list[EvalEncoding],
    templates: dict[str, list[EvalEncoding]],
    threshold: float,
) -> float:
    """**带阈值** 的 Top-1——对齐生产 RecognizeFace 的真实判定。

    生产链路是:max-by-template 取最高分 → 若 < threshold 则判 unknown。
    评估时对每个 genuine 测试样本(query 是注册者本人):
      - 命中本人且分数 ≥ threshold → 算对
      - argmax 错人 / 分数不够阈值 → 算错(包括"应识别成自己却被拒")

    这个口径才是"用户实际在 M4 摄像头前看到的成功率"。报告里同时给两个 Top-1:
      - top1_accuracy:无阈值,衡量纯排序能力(模型本质)
      - top1_with_threshold:有阈值,衡量端到端可用性(部署后体感)
    两者差距大 = 模型排序对但分数被压低,提示阈值偏严或同人内方差大。

    阈值通常用对应策略的 EER 阈值或 TAR@FAR=1e-3 阈值——M2 主入口默认前者。
    """
    flat_matrix, owner_arr, template_ids = _flatten_templates(templates)
    if not template_ids or flat_matrix.shape[0] == 0:
        return 0.0
    n_persons = len(template_ids)

    correct = 0
    total = 0
    for q in test_set:
        if q.person_id not in templates:
            continue
        scores_flat = flat_matrix @ q.vector
        per_person_max = np.full(n_persons, -np.inf, dtype=scores_flat.dtype)
        np.maximum.at(per_person_max, owner_arr, scores_flat)
        best_idx = int(np.argmax(per_person_max))
        best_score = float(per_person_max[best_idx])
        # 阈值判拒:即便 argmax 选对人,分数太低也算"判 unknown"——错
        if best_score < threshold:
            total += 1
            continue
        if template_ids[best_idx] == q.person_id:
            correct += 1
        total += 1
    return correct / total if total else 0.0
