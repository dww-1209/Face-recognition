"""
策略 3：人工三分法。

与其它策略不同，本策略期望数据已按子文件夹分好三类
（如正光/侧光/逆光，或任意人工分组）。

对每个子文件夹内的图片各自取均值 → 生成 3 个模板。

目录结构约定：
  <person_dir>/
      subset_0/    ← 第一组
      subset_1/    ← 第二组
      subset_2/    ← 第三组
"""

from datetime import UTC, datetime

import numpy as np

from face_recognition.application.strategies.base import l2_normalize
from face_recognition.domain.entities import FaceEncoding, Template


class ManualThreeStrategy:
    """策略 3：三个子文件夹各取平均，生成 3 个模板。"""

    name = "manual_three"

    def build_from_groups(self, groups: list[list[FaceEncoding]]) -> list[Template]:
        """
        从分组编码构建模板。

        Args:
            groups: 3 个分组，每组分该文件夹下所有图片的编码列表。
                    允许某组为空（该组不生成模板）。

        Returns:
            模板列表（最多 3 个）。
        """
        templates = []
        for i, group_encodings in enumerate(groups):
            if not group_encodings:
                continue
            vectors = np.stack([e.vector for e in group_encodings])
            mean = np.mean(vectors, axis=0)
            mean = l2_normalize(mean)
            templates.append(
                Template(
                    encoding=FaceEncoding(vector=mean.astype(np.float32)),
                    source=f"subset_{i}_mean_n{len(group_encodings)}",
                    created_at=datetime.now(UTC),
                )
            )
        if not templates:
            raise ValueError("所有分组均为空，无法生成模板")
        return templates

    def build(self, encodings: list[FaceEncoding]) -> list[Template]:
        """
        兼容性方法：当没有子文件夹分组时，退化为 mean_all。
        实际使用时优先调用 build_from_groups。
        """
        from face_recognition.application.strategies.mean_all import MeanAllStrategy

        return MeanAllStrategy().build(encodings)
