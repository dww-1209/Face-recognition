import numpy as np
import pytest

from face_recognition.evaluation.pair_generator import (
    generate_genuine_pairs,
    generate_closed_impostor_pairs,
    generate_open_impostor_pairs,
)
from face_recognition.evaluation.types import EvalEncoding, PairResult


def _enc(person_id: str, seed: int, path: str = "x.jpg") -> EvalEncoding:
    """造一个单位向量 EvalEncoding，方便测试。"""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return EvalEncoding(vector=v / np.linalg.norm(v), person_id=person_id, image_path=path)


def test_genuine_pairs_only_same_person():
    """Genuine 应该全部 is_genuine=True 且 query/template 同人。"""
    queries = [_enc("alice", 0), _enc("alice", 1), _enc("bob", 2)]
    # templates 改为 dict[str, list[EvalEncoding]],对齐多模板评估口径。
    templates = {"alice": [_enc("alice", 10)], "bob": [_enc("bob", 11)]}
    pairs = generate_genuine_pairs(queries, templates)
    # 3 个 query 各自和"自己人"的模板组配对 → 3 个 PairResult
    assert len(pairs) == 3
    assert all(p.is_genuine for p in pairs)
    assert all(p.query_person == p.template_person for p in pairs)


def test_genuine_pairs_skip_query_without_template():
    """如果某人在 templates 里没有条目，他的 query 应该被跳过（不报错）。"""
    queries = [_enc("alice", 0), _enc("ghost", 1)]
    templates = {"alice": [_enc("alice", 10)]}
    pairs = generate_genuine_pairs(queries, templates)
    assert len(pairs) == 1
    assert pairs[0].query_person == "alice"


def test_closed_impostor_only_different_person():
    """库内 Impostor 必须 query_person != template_person。"""
    queries = [_enc("alice", 0)]
    templates = {
        "alice": [_enc("alice", 10)],
        "bob":   [_enc("bob", 11)],
        "carol": [_enc("carol", 12)],
    }
    pairs = generate_closed_impostor_pairs(queries, templates)
    # alice 的 query 配 bob、carol 的模板组 → 2 对
    assert len(pairs) == 2
    assert all(not p.is_genuine for p in pairs)
    assert all(p.query_person != p.template_person for p in pairs)


def test_open_impostor_pairs_all_negative():
    """库外陌生人 vs 库内任何 template → 全部 is_genuine=False。"""
    lfw_queries = [_enc("LFW_Stranger_1", 0, "lfw/x.jpg")]
    templates = {"alice": [_enc("alice", 10)], "bob": [_enc("bob", 11)]}
    pairs = generate_open_impostor_pairs(lfw_queries, templates)
    assert len(pairs) == 2
    assert all(not p.is_genuine for p in pairs)


def test_score_is_cosine_similarity():
    """分数 = 单位向量点积（余弦相似度，范围 [-1, 1]，同向接近 1）。"""
    # 自己跟自己点积 = 1
    same = _enc("alice", 0)
    pairs = generate_genuine_pairs([same], {"alice": [same]})
    # 浮点比较留余地（np.float32 累加误差通常 < 1e-6）
    assert pairs[0].score == pytest.approx(1.0, abs=1e-6)
