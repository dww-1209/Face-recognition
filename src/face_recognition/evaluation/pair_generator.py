# Genuine / 库内 Impostor / 库外 Impostor 三种配对的生成器。
# 共同模式：双重循环 + 算余弦 + 包成 PairResult。
import numpy as np

from face_recognition.evaluation.types import EvalEncoding, PairResult


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """两个**已 L2 归一化**的向量的余弦相似度 = 点积。
    M1 已经多次出现，这里只一行：np.dot(a, b) 标量返回。
    """
    # float() 把 np.float32 转成 Python float，避免下游 dataclass 字段类型不一致
    return float(np.dot(a, b))


def _max_cosine(query: np.ndarray, templates: list[EvalEncoding]) -> float:
    """对一组同人模板取最高相似度——评估口径与生产 RecognizeFace 一致。

    多模板策略(kmeans_k3 / all_vectors)的本意是"覆盖同一人的不同视角/光照",
    检索时只要 query 和**任何一个**模板足够近就视为命中,所以应该 max 而非 mean。
    """
    return max(_cosine(query, t.vector) for t in templates)


def generate_genuine_pairs(
    queries: list[EvalEncoding],
    templates: dict[str, list[EvalEncoding]],
) -> list[PairResult]:
    """同人配对:每个 query 对自己 person_id 的模板组取 max 相似度。

    queries: 测试集(每张测试图一个 EvalEncoding)
    templates: 每人 1~N 个模板向量(取决于策略),scoring 时取 max。
    """
    pairs: list[PairResult] = []
    for q in queries:
        # dict.get 在 key 不存在时返回 None,不抛 KeyError——
        # 允许"测试集出现库里没有的人",体现开放集场景的健壮性
        tpls = templates.get(q.person_id)
        if not tpls:
            continue
        pairs.append(PairResult(
            score=_max_cosine(q.vector, tpls),
            is_genuine=True,
            query_person=q.person_id,
            template_person=q.person_id,
        ))
    return pairs


def generate_closed_impostor_pairs(
    queries: list[EvalEncoding],
    templates: dict[str, list[EvalEncoding]],
) -> list[PairResult]:
    """库内异人配对:query 对**其他人**的模板组取 max 相似度。

    spec 已决定省去库内 Impostor,函数留着以备切回。
    """
    pairs: list[PairResult] = []
    for q in queries:
        for tpl_pid, tpls in templates.items():
            if tpl_pid == q.person_id:
                continue  # 跳过同人,那是 genuine 的活
            pairs.append(PairResult(
                score=_max_cosine(q.vector, tpls),
                is_genuine=False,
                query_person=q.person_id,
                template_person=tpl_pid,
            ))
    return pairs


def generate_open_impostor_pairs(
    lfw_queries: list[EvalEncoding],
    templates: dict[str, list[EvalEncoding]],
) -> list[PairResult]:
    """库外陌生人配对:LFW 人作为 query,对每个库内人的模板组取 max 相似度。

    所有产出 is_genuine=False(陌生人不可能"是"库里的任何人)。
    数量级:50 个 LFW × 35 个库内 → 1750 对。
    """
    pairs: list[PairResult] = []
    for q in lfw_queries:
        for tpl_pid, tpls in templates.items():
            pairs.append(PairResult(
                score=_max_cosine(q.vector, tpls),
                is_genuine=False,
                query_person=q.person_id,   # "LFW_xxx" 风格的伪 ID
                template_person=tpl_pid,
            ))
    return pairs
