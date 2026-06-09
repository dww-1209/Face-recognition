import numpy as np

from face_recognition.domain.interfaces import PersonRepository


class TemplateMatrixService:
    """把整个库的模板装进 (M, 512) 矩阵 + 长度 M 的 person_id 列表，
    支持 O(M) 矩阵乘法检索。M 在 35 人 × 平均 3 模板 ≈ 100 量级，毫秒返回。

    使用流程：
      svc = TemplateMatrixService(repo)
      svc.load()                           # 启动时一次
      pid, sim = svc.query(face_vector)    # 每次识别
      svc.reload()                         # POST/DELETE persons 后调
    """

    def __init__(self, repository: PersonRepository) -> None:
        self._repo = repository
        # 显式声明类型 + 初始化为空——load 之前 query 也能跑（返回空）
        self.matrix: np.ndarray = np.zeros((0, 512), dtype=np.float32)
        self.pid_list: list[str] = []

    def load(self) -> None:
        """从 repository 一次性加载所有人的所有模板到内存。"""
        persons = self._repo.list_all()
        # 列表推导展开"多人 × 多模板" → 一个长 list
        # 每个 Template 的 encoding.vector 已经是 L2 归一化的（domain 的不变量）
        rows: list[np.ndarray] = []
        pids: list[str] = []
        for p in persons:
            for tpl in p.templates:
                rows.append(tpl.encoding.vector)
                pids.append(p.person_id)
        if rows:
            # np.stack 把多个 (512,) 堆成 (M, 512)
            self.matrix = np.stack(rows).astype(np.float32)
        else:
            self.matrix = np.zeros((0, 512), dtype=np.float32)
        self.pid_list = pids

    def reload(self) -> None:
        """注册/删除人员后调用。等价 load——别名只是语义清晰。"""
        self.load()

    def query(self, vector: np.ndarray) -> tuple[str | None, float]:
        """单次识别：矩阵乘 → argmax → 返回 (person_id, similarity)。

        库为空时返回 (None, 0.0)；否则返回最相似那条模板对应的 person_id 和分数。
        阈值判断由调用方（RecognizeFrame 用例）做——这层只负责"找最像的"。
        """
        if self.matrix.shape[0] == 0:
            return None, 0.0
        # 矩阵乘 (M, 512) @ (512,) = (M,) 相似度向量；vector 必须已归一化
        scores = self.matrix @ vector
        best_idx = int(np.argmax(scores))
        return self.pid_list[best_idx], float(scores[best_idx])
