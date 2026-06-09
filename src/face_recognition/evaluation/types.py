# 评估专用值对象。全部 frozen：评估管线"产出 → 写报告"是单向流，没有"修改"语义。
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class EvalEncoding:
    """评估侧的"一张图的向量"。

    与 domain.entities.FaceEncoding 的区别：
      - FaceEncoding 不知道这张图属于谁、来自哪——它就是"模型输出向量+版本"
      - EvalEncoding 必须带 person_id（同/异人配对靠它）和 image_path（出错时回溯）
    把"评估的元数据"和"领域的纯向量"分离开，domain 层永远不知道 image_path。
    """
    # numpy ndarray 不是 hashable 类型，在 frozen dataclass 里要小心；
    # 但 @dataclass(frozen=True) 默认 eq=True 会让实例可哈希——只要你不 hash 它就没事。
    # 评估管线确实不会 hash EvalEncoding，所以 OK。
    vector: np.ndarray
    person_id: str
    image_path: str


@dataclass(frozen=True)
class PairResult:
    """一次"查询 vs 模板"比对的结果。

    is_genuine = True：query 和 template 是同一个人（应该高分通过）
    is_genuine = False：不同人（应该低分拒绝）
    评估指标的所有计算最终归结为一堆 PairResult 的统计。
    """
    score: float
    is_genuine: bool
    query_person: str
    template_person: str


@dataclass(frozen=True)
class StrategyMetrics:
    """单个策略跑完后的全部指标。reports.py 接收 list[StrategyMetrics] 写表/画图。"""
    strategy_name: str
    eer: float                 # Equal Error Rate
    eer_threshold: float       # 达到 EER 的相似度阈值
    tar_at_far_1e3: float      # 卡 FAR=0.1% 时的真员工接受率
    top1_accuracy: float       # 闭集 Top-1 无阈值版（衡量纯排序能力）
    top1_with_threshold: float # 闭集 Top-1 带阈值版（== 生产识别成功率;阈值默认用 EER 阈值）
    # field(default_factory=...) 是 dataclass 的"工厂默认值"语法。
    #   - 不能写 `roc_fpr: np.ndarray = np.array([])`——所有实例会**共享同一个数组**
    #     （和 list/dict 默认参数同样的坑）
    #   - default_factory 接受一个无参函数，每次构造时调用生成新实例
    roc_fpr: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    roc_tpr: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    n_genuine: int = 0
    n_impostor: int = 0
